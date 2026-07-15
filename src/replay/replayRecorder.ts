import { cloneRecordedReplayArtifact, cloneReplayEnvironmentSample, cloneReplayFrame } from './replayClone'
import { replayIntegrityFor } from './replayIntegrity'
import {
  REPLAY_CHANNELS_PER_SEGMENT,
  REPLAY_CORE_VERSION,
  REPLAY_INTERPOLATION,
  REPLAY_KIND,
  REPLAY_MUSCLE_CHANNEL_COUNT,
  REPLAY_SCHEMA_VERSION,
  REPLAY_SEGMENT_COUNT,
  type RecordedReplayArtifact,
  type RecordedReplayFrame,
  type ReplayEnvironmentSample,
  type ReplayFrame,
  type ReplayPolicyBackend,
  type ReplayPose,
  type ReplayTaskMetrics,
} from './types'
import { validateReplayArtifact } from './replayValidation'

export type ReplayRecorderOptions = {
  replayId: string
  createdAt: string
  source: {
    creatureId: string
    environmentId: string
    policyBackend: ReplayPolicyBackend
    modelVersion: string
  }
  timestep: number
  environmentSample: ReplayEnvironmentSample
}

export type ReplayRecorderFrame = {
  time?: number
  board: ReplayPose
  creatureRoot: ReplayPose
  contactRatio: number
  skateboardDiscovered: boolean
  rollingVelocity: number
  reward: number
  muscleActivations: ArrayLike<number>
}

/** A deterministic, single-use builder for browser or headless replay capture. */
export class ReplayRecorder {
  private readonly options: ReplayRecorderOptions
  private readonly recordedFrames: RecordedReplayFrame[] = []
  private finalized = false

  constructor(options: ReplayRecorderOptions) {
    this.options = {
      ...options,
      source: { ...options.source },
      environmentSample: cloneReplayEnvironmentSample(options.environmentSample),
    }
    assertPositiveFinite(options.timestep, 'Replay recorder timestep')
  }

  get frameCount() {
    return this.recordedFrames.length
  }

  record(frame: ReplayRecorderFrame): number {
    if (this.finalized) throw new Error('Replay recorder has already been finalized.')
    const time = frame.time ?? this.recordedFrames.length * this.options.timestep
    assertNonNegativeFinite(time, 'Replay frame time')
    const previous = this.recordedFrames.at(-1)
    if (previous && time <= previous.time) {
      throw new Error('Replay frame time must be strictly greater than the previous frame time.')
    }

    const muscleActivations = Array.from(frame.muscleActivations)
    if (muscleActivations.length !== REPLAY_MUSCLE_CHANNEL_COUNT) {
      throw new Error(`Replay frame requires exactly ${REPLAY_MUSCLE_CHANNEL_COUNT} muscle channels.`)
    }
    for (let index = 0; index < muscleActivations.length; index += 1) {
      const value = muscleActivations[index]
      if (!Number.isFinite(value) || value < -1 || value > 1) {
        throw new Error(`Replay muscle channel ${index} must be finite and between -1 and 1.`)
      }
    }

    const recorded = cloneReplayFrame({
      ...frame,
      time,
      muscleActivations,
    } as ReplayFrame) as RecordedReplayFrame
    this.recordedFrames.push(recorded)
    return this.recordedFrames.length - 1
  }

  finalize(taskMetrics: ReplayTaskMetrics): RecordedReplayArtifact {
    if (this.finalized) throw new Error('Replay recorder has already been finalized.')
    if (this.recordedFrames.length === 0) throw new Error('Replay recorder cannot finalize without frames.')
    const lastFrame = this.recordedFrames.at(-1)!
    const payload: Omit<RecordedReplayArtifact, 'integrity'> = {
      schemaVersion: REPLAY_SCHEMA_VERSION,
      kind: REPLAY_KIND,
      replayId: this.options.replayId,
      createdAt: this.options.createdAt,
      source: { ...this.options.source },
      timestep: this.options.timestep,
      durationSeconds: lastFrame.time + this.options.timestep,
      frameCount: this.recordedFrames.length,
      environmentSample: cloneReplayEnvironmentSample(this.options.environmentSample),
      taskMetrics: { ...taskMetrics },
      frames: this.recordedFrames.map(frame => cloneReplayFrame(frame) as RecordedReplayFrame),
      playback: {
        coreVersion: REPLAY_CORE_VERSION,
        interpolation: REPLAY_INTERPOLATION,
        actuatorLayout: {
          segmentCount: REPLAY_SEGMENT_COUNT,
          channelsPerSegment: REPLAY_CHANNELS_PER_SEGMENT,
          muscleChannelCount: REPLAY_MUSCLE_CHANNEL_COUNT,
          channelOrder: ['dorsal', 'ventral'],
        },
      },
    }
    const artifact: RecordedReplayArtifact = {
      ...payload,
      integrity: replayIntegrityFor(payload),
    }
    const validation = validateReplayArtifact(artifact)
    if (!validation.ok) {
      throw new Error(`Replay recorder produced an invalid artifact: ${validation.errors.join(' ')}`)
    }
    this.finalized = true
    return cloneRecordedReplayArtifact(artifact)
  }
}

export function recordReplay(
  options: ReplayRecorderOptions,
  frames: readonly ReplayRecorderFrame[],
  taskMetrics: ReplayTaskMetrics,
): RecordedReplayArtifact {
  const recorder = new ReplayRecorder(options)
  for (const frame of frames) recorder.record(frame)
  return recorder.finalize(taskMetrics)
}

function assertPositiveFinite(value: number, label: string) {
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${label} must be finite and greater than 0.`)
}

function assertNonNegativeFinite(value: number, label: string) {
  if (!Number.isFinite(value) || value < 0) throw new Error(`${label} must be finite and at least 0.`)
}
