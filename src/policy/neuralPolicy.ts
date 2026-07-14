import { ACTION_SIZE, OBSERVATION_SIZE, type PolicyAction, type PolicyObservation } from './types'

export const STUNT_POLICY_PATH = '/models/wurmkickflip_stunt_policy.json'

export type StuntPolicyTraining = {
  seed: number
  samples: number
  epochs: number
  validationMse: number
  teacherAgreement: number
}

export type StuntPolicyArtifact = {
  schemaVersion: 1
  kind: 'wurmkickflip.stuntPolicy'
  modelVersion: string
  inputSize: number
  hiddenSize: number
  outputSize: number
  activation: 'tanh'
  hiddenWeights: number[][]
  hiddenBias: number[]
  outputWeights: number[][]
  outputBias: number[]
  training: StuntPolicyTraining
}

export class NeuralStuntPolicy {
  readonly modelVersion: string
  readonly training: StuntPolicyTraining
  private readonly hidden = new Float64Array(0)
  private readonly artifact: StuntPolicyArtifact

  constructor(artifact: StuntPolicyArtifact) {
    this.artifact = artifact
    this.modelVersion = artifact.modelVersion
    this.training = artifact.training
    this.hidden = new Float64Array(artifact.hiddenSize)
  }

  run(observation: PolicyObservation): PolicyAction {
    const { hiddenWeights, hiddenBias, outputWeights, outputBias } = this.artifact

    for (let row = 0; row < hiddenWeights.length; row += 1) {
      const weights = hiddenWeights[row]
      let value = hiddenBias[row]
      for (let column = 0; column < observation.length; column += 1) {
        value += weights[column] * observation[column]
      }
      this.hidden[row] = Math.tanh(value)
    }

    const action = new Float32Array(ACTION_SIZE)
    for (let row = 0; row < outputWeights.length; row += 1) {
      const weights = outputWeights[row]
      let value = outputBias[row]
      for (let column = 0; column < this.hidden.length; column += 1) {
        value += weights[column] * this.hidden[column]
      }
      action[row] = Math.tanh(value)
    }
    return action
  }
}

export function parseStuntPolicy(value: unknown): StuntPolicyArtifact {
  if (!isRecord(value)) throw new Error('artifact is not an object')
  if (value.schemaVersion !== 1) throw new Error('schemaVersion must be 1')
  if (value.kind !== 'wurmkickflip.stuntPolicy') throw new Error('unexpected artifact kind')
  if (typeof value.modelVersion !== 'string' || value.modelVersion.length === 0) {
    throw new Error('modelVersion must be non-empty')
  }
  if (value.inputSize !== OBSERVATION_SIZE || value.outputSize !== ACTION_SIZE) {
    throw new Error(`expected ${OBSERVATION_SIZE} inputs and ${ACTION_SIZE} outputs`)
  }
  if (!Number.isInteger(value.hiddenSize) || (value.hiddenSize as number) <= 0) {
    throw new Error('hiddenSize must be a positive integer')
  }
  if (value.activation !== 'tanh') throw new Error('activation must be tanh')

  const hiddenSize = value.hiddenSize as number
  const hiddenWeights = finiteMatrix(value.hiddenWeights, hiddenSize, OBSERVATION_SIZE, 'hiddenWeights')
  const hiddenBias = finiteVector(value.hiddenBias, hiddenSize, 'hiddenBias')
  const outputWeights = finiteMatrix(value.outputWeights, ACTION_SIZE, hiddenSize, 'outputWeights')
  const outputBias = finiteVector(value.outputBias, ACTION_SIZE, 'outputBias')
  const training = parseTraining(value.training)

  return {
    schemaVersion: 1,
    kind: 'wurmkickflip.stuntPolicy',
    modelVersion: value.modelVersion,
    inputSize: OBSERVATION_SIZE,
    hiddenSize,
    outputSize: ACTION_SIZE,
    activation: 'tanh',
    hiddenWeights,
    hiddenBias,
    outputWeights,
    outputBias,
    training,
  }
}

function finiteMatrix(value: unknown, rows: number, columns: number, label: string): number[][] {
  if (!Array.isArray(value) || value.length !== rows) {
    throw new Error(`${label} must contain ${rows} rows`)
  }
  return value.map((row, index) => finiteVector(row, columns, `${label}[${index}]`))
}

function finiteVector(value: unknown, length: number, label: string): number[] {
  if (!Array.isArray(value) || value.length !== length) {
    throw new Error(`${label} must contain ${length} values`)
  }
  return value.map((item, index) => {
    if (typeof item !== 'number' || !Number.isFinite(item)) {
      throw new Error(`${label}[${index}] must be finite`)
    }
    return item
  })
}

function parseTraining(value: unknown): StuntPolicyTraining {
  if (!isRecord(value)) throw new Error('training must be an object')
  const seed = finiteNumber(value.seed, 'training.seed')
  const samples = finiteNumber(value.samples, 'training.samples')
  const epochs = finiteNumber(value.epochs, 'training.epochs')
  const validationMse = finiteNumber(value.validationMse, 'training.validationMse')
  const teacherAgreement = finiteNumber(value.teacherAgreement, 'training.teacherAgreement')
  if (!Number.isInteger(seed) || !Number.isInteger(samples) || !Number.isInteger(epochs)) {
    throw new Error('training seed, samples, and epochs must be integers')
  }
  if (samples <= 0 || epochs <= 0 || validationMse < 0 || teacherAgreement < 0 || teacherAgreement > 1) {
    throw new Error('training metrics are out of range')
  }
  return { seed, samples, epochs, validationMse, teacherAgreement }
}

function finiteNumber(value: unknown, label: string) {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${label} must be finite`)
  return value
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
