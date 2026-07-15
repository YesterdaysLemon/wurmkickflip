import type { EnvironmentConfig } from '../creature/types'
import { ReplayRecorder, type ReplayRecorderFrame, type ReplayRecorderOptions } from './replayRecorder'
import type { RecordedReplayArtifact, ReplayEnvironmentSample, ReplayTaskMetrics } from './types'

/**
 * Browser-facing capture accumulator. ReplayRecorder owns the artifact
 * contract; this class derives deterministic task metrics from the exact
 * fixed-step frames sent by the terrarium.
 */
export class LiveReplayCapture {
  private readonly recorder: ReplayRecorder
  private readonly timestep: number
  private contactTotal = 0
  private energy = 0
  private firstContactAt: number | null = null
  private skateboardDiscoveredAt: number | null = null
  private mountTime = 0
  private rollingDistance = 0
  private maxRollingVelocity = 0

  constructor(options: ReplayRecorderOptions) {
    this.recorder = new ReplayRecorder(options)
    this.timestep = options.timestep
  }

  get frameCount() {
    return this.recorder.frameCount
  }

  get durationSeconds() {
    return this.frameCount * this.timestep
  }

  record(frame: ReplayRecorderFrame): number {
    // Capture time deliberately starts at zero even when recording begins in
    // the middle of a live rollout. ReplayRecorder assigns the grid time.
    const skateboardDiscovered = this.skateboardDiscoveredAt !== null || frame.skateboardDiscovered
    const index = this.recorder.record({ ...frame, time: undefined, skateboardDiscovered })
    const time = index * this.timestep
    this.contactTotal += frame.contactRatio
    this.rollingDistance += Math.abs(frame.rollingVelocity) * this.timestep
    this.maxRollingVelocity = Math.max(this.maxRollingVelocity, Math.abs(frame.rollingVelocity))

    let activationMagnitude = 0
    for (const activation of Array.from(frame.muscleActivations)) {
      activationMagnitude += Math.abs(activation)
    }
    this.energy += (activationMagnitude / Math.max(1, frame.muscleActivations.length)) * this.timestep

    if (skateboardDiscovered && this.skateboardDiscoveredAt === null) {
      this.skateboardDiscoveredAt = time
    }
    if (skateboardDiscovered && frame.contactRatio > 0.01 && this.firstContactAt === null) {
      this.firstContactAt = time
    }
    if (skateboardDiscovered && frame.contactRatio >= 0.5) this.mountTime += this.timestep
    return index
  }

  finalize(): RecordedReplayArtifact {
    if (this.frameCount === 0) {
      throw new Error('No fixed-step frames were captured. Play the live terrarium, then finish capture.')
    }
    return this.recorder.finalize(this.taskMetrics())
  }

  private taskMetrics(): ReplayTaskMetrics {
    const durationSeconds = this.durationSeconds
    return {
      skateboardDiscoveredAt: this.skateboardDiscoveredAt,
      firstContactAt: this.firstContactAt,
      mountTime: Math.min(this.mountTime, durationSeconds),
      rollingDistance: this.rollingDistance,
      maxRollingVelocity: this.maxRollingVelocity,
      averageContactRatio: this.contactTotal / this.frameCount,
      energy: this.energy,
      survivalTime: durationSeconds,
      fallReason: 'none',
    }
  }
}

/** The browser showcase uses the selected config at its nominal scale. */
export function replayEnvironmentSampleFor(environment: EnvironmentConfig): ReplayEnvironmentSample {
  return {
    seed: environment.seed,
    gravityScale: Math.abs(environment.world.gravity[1]) / 9.81,
    frictionScale: 1,
    dragScale: 1,
    slopeDegrees: environment.terrain.slopeDegrees,
    roughness: environment.terrain.roughness,
    obstacleDensity: environment.terrain.obstacleDensity,
    skateboardSpawn: [environment.skateboard.spawnPosition[0], environment.skateboard.spawnPosition[2]],
    skateboardMass: environment.skateboard.mass,
    wheelFriction: environment.skateboard.wheelFriction,
  }
}
