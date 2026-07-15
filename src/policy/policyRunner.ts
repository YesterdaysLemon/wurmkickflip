import { NeuralStuntPolicy, parseStuntPolicy, STUNT_POLICY_PATH } from './neuralPolicy'
import { ScriptedMusclePolicy } from './scriptedPolicy'
import { ACTION_SIZE, type PolicyAction, type PolicyObservation, type PolicyStatus } from './types'

type RequestedProvider = 'neural' | 'scripted' | 'retired-onnx'

/**
 * Runs the compact tracked mounted-stunt brain. Detached locomotion has its own
 * evolved recurrent runner; the scripted policy remains an explicit diagnostic
 * fallback rather than a hidden substitute for the crawl controller.
 */
export class PolicyRunner {
  private status: PolicyStatus = {
    backend: 'loading',
    message: 'Policy runtime is starting.',
    modelVersion: 'scripted-v0',
  }

  private readonly scripted = new ScriptedMusclePolicy()
  private neural: NeuralStuntPolicy | null = null
  private loadPromise: Promise<PolicyStatus> | null = null

  getStatus(): PolicyStatus {
    return this.status
  }

  async load(): Promise<PolicyStatus> {
    if (!this.loadPromise) this.loadPromise = this.loadOnce()
    return this.loadPromise
  }

  private async loadOnce(): Promise<PolicyStatus> {
    const provider = getRequestedProvider()
    if (provider === 'scripted') {
      this.status = {
        backend: 'scripted',
        message: 'Using the deterministic mounted diagnostic by request.',
        modelVersion: 'scripted-v0',
      }
      return this.status
    }

    if (provider === 'retired-onnx') {
      this.status = {
        backend: 'scripted',
        message:
          'The legacy browser ONNX backend has been retired; using the deterministic mounted diagnostic.',
        modelVersion: 'scripted-v0',
      }
      return this.status
    }

    try {
      const response = await fetch(STUNT_POLICY_PATH)
      if (!response.ok) throw new Error(`model request returned ${response.status}`)
      const artifact = parseStuntPolicy(await response.json())
      this.neural = new NeuralStuntPolicy(artifact)
      this.status = {
        backend: 'neural-js',
        message: `Distilled stunt brain live — ${artifact.training.teacherAgreement.toLocaleString(
          undefined,
          {
            style: 'percent',
            maximumFractionDigits: 1,
          },
        )} teacher agreement.`,
        modelVersion: artifact.modelVersion,
      }
    } catch (error) {
      this.neural = null
      this.status = {
        backend: 'scripted',
        message:
          error instanceof Error
            ? `Neural artifact unavailable: ${error.message}`
            : 'Neural artifact unavailable.',
        modelVersion: 'scripted-v0',
      }
    }
    return this.status
  }

  async run(observation: PolicyObservation): Promise<PolicyAction> {
    return this.neural?.run(observation) ?? this.scripted.run(observation)
  }

  reset() {
    this.scripted.reset()
  }
}

export function sanitizePolicyAction(data: ArrayLike<number>): PolicyAction | null {
  if (data.length !== ACTION_SIZE) return null

  const action = new Float32Array(ACTION_SIZE)
  for (let index = 0; index < ACTION_SIZE; index += 1) {
    const value = data[index]
    if (!Number.isFinite(value)) return null
    action[index] = Math.max(-1, Math.min(1, value))
  }
  return action
}

function getRequestedProvider(): RequestedProvider {
  const params = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null
  const forced = params?.get('policyBackend')
  if (forced === 'wasm' || forced === 'webgpu') return 'retired-onnx'
  if (forced === 'scripted') return 'scripted'
  return 'neural'
}
