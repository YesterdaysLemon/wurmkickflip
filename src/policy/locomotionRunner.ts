import {
  EvolvedLocomotionPolicy,
  LOCOMOTION_POLICY_PATH,
  parseLocomotionPolicy,
  type LocomotionPolicyTelemetry,
  type LocomotionSensors,
} from './locomotionPolicy'
import { ACTION_SIZE, SEGMENT_COUNT, type NeuralGaitPerturbationKind, type PolicyAction } from './types'

export type LocomotionLoadStatus = {
  loaded: boolean
  modelVersion: string
  message: string
}

export class LocomotionPolicyRunner {
  private policy: EvolvedLocomotionPolicy | null = null
  private loadPromise: Promise<LocomotionLoadStatus> | null = null
  private status: LocomotionLoadStatus = {
    loaded: false,
    modelVersion: 'locomotion-unavailable',
    message: 'Evolved locomotion brain is waking.',
  }

  getStatus() {
    return this.status
  }

  load() {
    if (!this.loadPromise) this.loadPromise = this.loadOnce()
    return this.loadPromise
  }

  reset() {
    this.policy?.reset()
  }

  applyPerturbation(kind: NeuralGaitPerturbationKind, segment: number | null, durationSeconds: number) {
    this.policy?.applyPerturbation(kind, segment, durationSeconds)
  }

  clearPerturbation() {
    this.policy?.clearPerturbation()
  }

  hasActivePerturbation() {
    return this.policy?.hasActivePerturbation() ?? false
  }

  getTelemetry(): LocomotionPolicyTelemetry {
    return (
      this.policy?.getTelemetry() ?? {
        hidden: new Array(SEGMENT_COUNT).fill(0),
        drives: new Array(SEGMENT_COUNT).fill(0),
        commands: new Array(SEGMENT_COUNT).fill(0),
        requestedCommands: new Array(SEGMENT_COUNT).fill(0),
        sensedBends: new Array(SEGMENT_COUNT).fill(0),
        sensedBendVelocities: new Array(SEGMENT_COUNT).fill(0),
        sensedContactLoads: new Array(SEGMENT_COUNT).fill(0),
        sensedSlipSpeeds: new Array(SEGMENT_COUNT).fill(0),
        sensedObstacleForward: new Array(SEGMENT_COUNT).fill(0),
        sensedObstacleRight: new Array(SEGMENT_COUNT).fill(0),
        activePerturbation: null,
      }
    )
  }

  run(
    sensors: LocomotionSensors,
    segmentBends: ArrayLike<number>,
    segmentBendVelocities: ArrayLike<number>,
  ): PolicyAction {
    return this.policy?.run(sensors, segmentBends, segmentBendVelocities) ?? new Float32Array(ACTION_SIZE)
  }

  private async loadOnce(): Promise<LocomotionLoadStatus> {
    try {
      const response = await fetch(LOCOMOTION_POLICY_PATH)
      if (!response.ok) throw new Error(`model request returned ${response.status}`)
      const artifact = parseLocomotionPolicy(await response.json())
      this.policy = new EvolvedLocomotionPolicy(artifact)
      const generations = finiteTrainingMetric(artifact.training.generations)
      const generationCount = Math.round(generations)
      const generationLabel = `${generationCount} generation${generationCount === 1 ? '' : 's'}`
      const warmStart = readWarmStartVersion(artifact.training.warmStart)
      const guardedSelection =
        artifact.training.allSelectionMarginsFeasible === true
          ? ' Guarded causal, head-leading, and recovery margins passed.'
          : ''
      this.status = {
        loaded: true,
        modelVersion: artifact.modelVersion,
        message: warmStart
          ? `Clock-free recurrent crawl brain verified through ${generationLabel} from ${warmStart}.${guardedSelection} Kickflips remain scripted.`
          : `Clock-free recurrent crawl brain evolved for ${generationLabel}.${guardedSelection} Kickflips remain scripted.`,
      }
    } catch (error) {
      this.policy = null
      this.status = {
        loaded: false,
        modelVersion: 'locomotion-unavailable',
        message:
          error instanceof Error
            ? `Locomotion artifact unavailable: ${error.message}`
            : 'Locomotion artifact unavailable.',
      }
    }
    return this.status
  }
}

function finiteTrainingMetric(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function readWarmStartVersion(value: unknown) {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return null
  const modelVersion = (value as Record<string, unknown>).modelVersion
  return typeof modelVersion === 'string' && modelVersion.length > 0 ? modelVersion : null
}
