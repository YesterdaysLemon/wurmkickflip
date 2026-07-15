import {
  EvolvedLocomotionPolicy,
  LOCOMOTION_POLICY_PATH,
  parseLocomotionPolicy,
  type LocomotionSensors,
} from './locomotionPolicy'
import { ACTION_SIZE, type PolicyAction } from './types'

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
      const progress = finiteTrainingMetric(artifact.training.meanTargetProgress)
      const generations = finiteTrainingMetric(artifact.training.generations)
      const warmStart = readWarmStartVersion(artifact.training.warmStart)
      this.status = {
        loaded: true,
        modelVersion: artifact.modelVersion,
        message: warmStart
          ? `Clock-free recurrent crawl brain risk-refined for ${generations.toFixed(0)} generations from ${warmStart}; ${progress.toFixed(2)} m mean target progress. Kickflips remain scripted.`
          : `Clock-free recurrent crawl brain evolved for ${generations.toFixed(0)} generations; ${progress.toFixed(2)} m mean target progress. Kickflips remain scripted.`,
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
