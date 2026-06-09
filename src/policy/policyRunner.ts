import { ScriptedMusclePolicy } from './scriptedPolicy'
import {
  ACTION_SIZE,
  OBSERVATION_SIZE,
  type PolicyAction,
  type PolicyMeta,
  type PolicyObservation,
  type PolicyStatus,
} from './types'

type OrtModule = typeof import('onnxruntime-web/webgpu')
type OrtSession = Awaited<ReturnType<OrtModule['InferenceSession']['create']>>
type OnnxProvider = 'webgpu' | 'wasm'
type RequestedProvider = OnnxProvider | 'scripted'

export class PolicyRunner {
  private status: PolicyStatus = {
    backend: 'loading',
    message: 'Policy runtime is starting.',
    modelVersion: 'scripted-v0',
  }

  private session: OrtSession | null = null
  private ort: OrtModule | null = null
  private inputName = ''
  private outputName = ''
  private scripted = new ScriptedMusclePolicy()
  private loadPromise: Promise<PolicyStatus> | null = null

  getStatus(): PolicyStatus {
    return this.status
  }

  async load(): Promise<PolicyStatus> {
    if (this.loadPromise) {
      return this.loadPromise
    }

    this.loadPromise = this.loadOnce()
    return this.loadPromise
  }

  private async loadOnce(): Promise<PolicyStatus> {
    let meta: PolicyMeta | null = null

    try {
      const metaResponse = await fetch('/models/wurmkickflip_policy.meta.json')
      meta = (await metaResponse.json()) as PolicyMeta
    } catch {
      this.status = {
        backend: 'scripted',
        message: 'No policy metadata found; using scripted muscle waves.',
        modelVersion: 'scripted-v0',
      }
      return this.status
    }

    if (meta.observationSize !== OBSERVATION_SIZE || meta.actionSize !== ACTION_SIZE) {
      this.status = {
        backend: 'scripted',
        message: 'Policy metadata shape mismatch; using scripted muscle waves.',
        modelVersion: meta.modelVersion,
      }
      return this.status
    }

    const provider = getRequestedProvider()
    if (provider === 'scripted') {
      this.status = {
        backend: 'scripted',
        message: 'Using scripted control. Add ?policyBackend=webgpu or ?policyBackend=wasm to load ONNX.',
        modelVersion: meta.modelVersion,
      }
      return this.status
    }

    try {
      const modelResponse = await fetch(meta.modelPath, { method: 'HEAD' })
      const contentType = modelResponse.headers.get('content-type') ?? ''
      if (!modelResponse.ok || contentType.includes('text/html')) {
        throw new Error(`missing model: ${meta.modelPath}`)
      }

      const ort = await import('onnxruntime-web/webgpu')
      this.ort = ort

      this.session = await ort.InferenceSession.create(meta.modelPath, {
        executionProviders: [provider],
      })

      this.inputName = this.session.inputNames[0] ?? 'observation'
      this.outputName = this.session.outputNames[0] ?? 'action'
      this.status = {
        backend: provider === 'webgpu' ? 'onnx-webgpu' : 'onnx-wasm',
        message: `Loaded ${meta.modelVersion} with ${provider.toUpperCase()}.`,
        modelVersion: meta.modelVersion,
      }
    } catch (error) {
      this.session = null
      this.ort = null
      this.status = {
        backend: 'scripted',
        message: error instanceof Error ? `ONNX unavailable: ${error.message}` : 'ONNX unavailable.',
        modelVersion: meta.modelVersion,
      }
    }

    return this.status
  }

  async run(observation: PolicyObservation): Promise<PolicyAction> {
    if (!this.session || !this.ort) {
      return this.scripted.run(observation)
    }

    const tensor = new this.ort.Tensor('float32', observation, [1, OBSERVATION_SIZE])
    const output = await this.session.run({ [this.inputName]: tensor })
    const actionTensor = output[this.outputName]

    if (!actionTensor || !(actionTensor.data instanceof Float32Array)) {
      return this.scripted.run(observation)
    }

    return sanitizePolicyAction(actionTensor.data) ?? this.scripted.run(observation)
  }

  reset() {
    this.scripted.reset()
  }
}

export function sanitizePolicyAction(data: ArrayLike<number>): PolicyAction | null {
  if (data.length !== ACTION_SIZE) {
    return null
  }

  const action = new Float32Array(ACTION_SIZE)
  for (let index = 0; index < ACTION_SIZE; index += 1) {
    const value = data[index]
    if (!Number.isFinite(value)) {
      return null
    }
    action[index] = Math.max(-1, Math.min(1, value))
  }

  return action
}

function getRequestedProvider(): RequestedProvider {
  const params = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null
  const forced = params?.get('policyBackend')
  if (forced === 'wasm') {
    return 'wasm'
  }
  if (forced === 'webgpu') {
    return 'webgpu'
  }
  if (forced === 'scripted') {
    return 'scripted'
  }

  return 'scripted'
}
