import { ACTION_SIZE, POLICY_TIMESTEP, SEGMENT_COUNT, type PolicyAction } from './types'

export const LOCOMOTION_POLICY_PATH = '/models/wurmkickflip_locomotion_policy.json'

/** Fixed plant contract used by evolution, artifact validation, and the browser. */
export const LOCOMOTION_PLANT_CONTRACT = {
  timestep: POLICY_TIMESTEP,
  jointStiffness: 22,
  jointDamping: 7,
  jointLimit: 1.15,
  waveGain: -1.35,
  maximumForwardAcceleration: 2.4,
  linearDrag: 0.75,
  inverseTractionDrag: 0.18,
  tractionMinimum: 0,
  tractionMaximum: 1.2,
  inverseTractionFloor: 0.12,
  minimumForwardSpeed: -1.45,
  maximumForwardSpeed: 1.45,
  frontTurnBase: 1.15,
  frontTurnSpeedGain: 1.65,
  angularDrag: 2.8,
  minimumAngularSpeed: -3,
  maximumAngularSpeed: 3,
} as const

export const LOCOMOTION_SENSOR_NAMES = [
  'targetForward',
  'targetRight',
  'targetDistance',
  'forwardSpeed',
  'angularSpeed',
  'terrainFriction',
  'urgency',
] as const

const INPUT_WEIGHT_NAMES = [
  'bias',
  'segmentPosition',
  'targetForward',
  'targetRight',
  'targetRightByPosition',
  'targetDistance',
  'forwardSpeed',
  'angularSpeed',
  'terrainFriction',
  'urgency',
  'segmentBend',
  'segmentBendVelocity',
  'previousCommand',
] as const

const RECURRENT_WEIGHT_NAMES = ['self', 'anteriorNeighbor', 'posteriorNeighbor'] as const

const OUTPUT_WEIGHT_NAMES = [
  'bias',
  'self',
  'anteriorNeighbor',
  'posteriorNeighbor',
  'targetRight',
  'targetRightByPosition',
  'segmentBend',
] as const

export type LocomotionSensors = {
  targetForward: number
  targetRight: number
  targetDistance: number
  forwardSpeed: number
  angularSpeed: number
  terrainFriction: number
  urgency: number
}

export type LocomotionPolicyArtifact = {
  schemaVersion: 1
  kind: 'wurmkickflip.locomotionPolicy'
  modelVersion: string
  architecture: 'segmental-recurrent-tanh'
  segmentCount: number
  actionSize: number
  sensorNames: string[]
  segmentPositions: number[]
  initialState: number[]
  weights: {
    inputNames: string[]
    input: number[]
    recurrentNames: string[]
    recurrent: number[]
    outputNames: string[]
    output: number[]
  }
  plant: {
    timestep: number
    jointStiffness: number
    jointDamping: number
    jointLimit: number
    waveGain: number
    maximumForwardAcceleration: number
    linearDrag: number
    inverseTractionDrag: number
    tractionMinimum: number
    tractionMaximum: number
    inverseTractionFloor: number
    minimumForwardSpeed: number
    maximumForwardSpeed: number
    frontTurnBase: number
    frontTurnSpeedGain: number
    angularDrag: number
    minimumAngularSpeed: number
    maximumAngularSpeed: number
  }
  training: Record<string, unknown>
}

/**
 * A clock-free chain of locally coupled recurrent neurons. Every hidden unit owns
 * one anatomical segment and produces that segment's antagonistic actuator pair.
 */
export class EvolvedLocomotionPolicy {
  readonly modelVersion: string
  private readonly artifact: LocomotionPolicyArtifact
  private readonly hidden = new Float64Array(SEGMENT_COUNT)
  private readonly nextHidden = new Float64Array(SEGMENT_COUNT)
  private readonly previousCommand = new Float64Array(SEGMENT_COUNT)

  constructor(artifact: LocomotionPolicyArtifact) {
    this.artifact = artifact
    this.modelVersion = artifact.modelVersion
    this.reset()
  }

  reset() {
    this.hidden.set(this.artifact.initialState)
    this.nextHidden.fill(0)
    this.previousCommand.fill(0)
  }

  run(
    sensors: LocomotionSensors,
    segmentBends: ArrayLike<number>,
    segmentBendVelocities: ArrayLike<number>,
  ): PolicyAction {
    if (segmentBends.length !== SEGMENT_COUNT || segmentBendVelocities.length !== SEGMENT_COUNT) {
      throw new Error(`locomotion policy requires ${SEGMENT_COUNT} segment bend values and velocities`)
    }

    const input = this.artifact.weights.input
    const recurrent = this.artifact.weights.recurrent
    const output = this.artifact.weights.output
    const positions = this.artifact.segmentPositions
    const safeSensors = sanitizeSensors(sensors)

    for (let segment = 0; segment < SEGMENT_COUNT; segment += 1) {
      const position = positions[segment]
      const bend = finiteOrZero(segmentBends[segment])
      const bendVelocity = finiteOrZero(segmentBendVelocities[segment])
      const anterior = segment > 0 ? this.hidden[segment - 1] : 0
      const posterior = segment + 1 < SEGMENT_COUNT ? this.hidden[segment + 1] : 0
      const drive =
        input[0] +
        input[1] * position +
        input[2] * safeSensors.targetForward +
        input[3] * safeSensors.targetRight +
        input[4] * safeSensors.targetRight * position +
        input[5] * safeSensors.targetDistance +
        input[6] * safeSensors.forwardSpeed +
        input[7] * safeSensors.angularSpeed +
        input[8] * safeSensors.terrainFriction +
        input[9] * safeSensors.urgency +
        input[10] * bend +
        input[11] * bendVelocity +
        input[12] * this.previousCommand[segment] +
        recurrent[0] * this.hidden[segment] +
        recurrent[1] * anterior +
        recurrent[2] * posterior
      this.nextHidden[segment] = Math.tanh(drive)
    }

    const action = new Float32Array(ACTION_SIZE)
    for (let segment = 0; segment < SEGMENT_COUNT; segment += 1) {
      const position = positions[segment]
      const bend = finiteOrZero(segmentBends[segment])
      const anterior = segment > 0 ? this.nextHidden[segment - 1] : 0
      const posterior = segment + 1 < SEGMENT_COUNT ? this.nextHidden[segment + 1] : 0
      const command = Math.tanh(
        output[0] +
          output[1] * this.nextHidden[segment] +
          output[2] * anterior +
          output[3] * posterior +
          output[4] * safeSensors.targetRight +
          output[5] * safeSensors.targetRight * position +
          output[6] * bend,
      )
      this.previousCommand[segment] = command
      action[segment * 2] = command
      action[segment * 2 + 1] = -command
    }
    this.hidden.set(this.nextHidden)
    return action
  }
}

export function parseLocomotionPolicy(value: unknown): LocomotionPolicyArtifact {
  if (!isRecord(value)) throw new Error('locomotion artifact is not an object')
  if (value.schemaVersion !== 1) throw new Error('locomotion schemaVersion must be 1')
  if (value.kind !== 'wurmkickflip.locomotionPolicy') throw new Error('unexpected locomotion artifact kind')
  if (value.architecture !== 'segmental-recurrent-tanh') throw new Error('unexpected locomotion architecture')
  if (typeof value.modelVersion !== 'string' || value.modelVersion.length === 0) {
    throw new Error('locomotion modelVersion must be non-empty')
  }
  if (value.segmentCount !== SEGMENT_COUNT || value.actionSize !== ACTION_SIZE) {
    throw new Error(`locomotion artifact requires ${SEGMENT_COUNT} segments and ${ACTION_SIZE} actions`)
  }
  const sensorNames = exactStringVector(value.sensorNames, LOCOMOTION_SENSOR_NAMES, 'sensorNames')
  const segmentPositions = finiteVector(value.segmentPositions, SEGMENT_COUNT, 'segmentPositions')
  const initialState = finiteVector(value.initialState, SEGMENT_COUNT, 'initialState')
  if (!isRecord(value.weights)) throw new Error('locomotion weights must be an object')
  const inputNames = exactStringVector(value.weights.inputNames, INPUT_WEIGHT_NAMES, 'weights.inputNames')
  const input = finiteVector(value.weights.input, INPUT_WEIGHT_NAMES.length, 'weights.input')
  const recurrentNames = exactStringVector(
    value.weights.recurrentNames,
    RECURRENT_WEIGHT_NAMES,
    'weights.recurrentNames',
  )
  const recurrent = finiteVector(
    value.weights.recurrent,
    RECURRENT_WEIGHT_NAMES.length,
    'weights.recurrent',
  )
  const outputNames = exactStringVector(value.weights.outputNames, OUTPUT_WEIGHT_NAMES, 'weights.outputNames')
  const output = finiteVector(value.weights.output, OUTPUT_WEIGHT_NAMES.length, 'weights.output')
  const plant = parsePlant(value.plant)
  if (!isRecord(value.training)) throw new Error('locomotion training metadata must be an object')

  return {
    schemaVersion: 1,
    kind: 'wurmkickflip.locomotionPolicy',
    modelVersion: value.modelVersion,
    architecture: 'segmental-recurrent-tanh',
    segmentCount: SEGMENT_COUNT,
    actionSize: ACTION_SIZE,
    sensorNames,
    segmentPositions,
    initialState,
    weights: { inputNames, input, recurrentNames, recurrent, outputNames, output },
    plant,
    training: { ...value.training },
  }
}

function parsePlant(value: unknown): LocomotionPolicyArtifact['plant'] {
  if (!isRecord(value)) throw new Error('locomotion plant metadata must be an object')
  const plant = {
    timestep: positiveFinite(value.timestep, 'plant.timestep'),
    jointStiffness: positiveFinite(value.jointStiffness, 'plant.jointStiffness'),
    jointDamping: positiveFinite(value.jointDamping, 'plant.jointDamping'),
    jointLimit: positiveFinite(value.jointLimit, 'plant.jointLimit'),
    waveGain: finiteNumber(value.waveGain, 'plant.waveGain'),
    maximumForwardAcceleration: positiveFinite(
      value.maximumForwardAcceleration,
      'plant.maximumForwardAcceleration',
    ),
    linearDrag: positiveFinite(value.linearDrag, 'plant.linearDrag'),
    inverseTractionDrag: positiveFinite(value.inverseTractionDrag, 'plant.inverseTractionDrag'),
    tractionMinimum: finiteNumber(value.tractionMinimum, 'plant.tractionMinimum'),
    tractionMaximum: positiveFinite(value.tractionMaximum, 'plant.tractionMaximum'),
    inverseTractionFloor: positiveFinite(value.inverseTractionFloor, 'plant.inverseTractionFloor'),
    minimumForwardSpeed: finiteNumber(value.minimumForwardSpeed, 'plant.minimumForwardSpeed'),
    maximumForwardSpeed: positiveFinite(value.maximumForwardSpeed, 'plant.maximumForwardSpeed'),
    frontTurnBase: positiveFinite(value.frontTurnBase, 'plant.frontTurnBase'),
    frontTurnSpeedGain: positiveFinite(value.frontTurnSpeedGain, 'plant.frontTurnSpeedGain'),
    angularDrag: positiveFinite(value.angularDrag, 'plant.angularDrag'),
    minimumAngularSpeed: finiteNumber(value.minimumAngularSpeed, 'plant.minimumAngularSpeed'),
    maximumAngularSpeed: positiveFinite(value.maximumAngularSpeed, 'plant.maximumAngularSpeed'),
  }
  for (const key of Object.keys(LOCOMOTION_PLANT_CONTRACT) as Array<keyof typeof LOCOMOTION_PLANT_CONTRACT>) {
    if (Math.abs(plant[key] - LOCOMOTION_PLANT_CONTRACT[key]) > 1e-9) {
      throw new Error(`plant.${key} does not match the browser locomotion contract`)
    }
  }
  return plant
}

function sanitizeSensors(sensors: LocomotionSensors): LocomotionSensors {
  return {
    targetForward: clamp(finiteOrZero(sensors.targetForward), -1, 1),
    targetRight: clamp(finiteOrZero(sensors.targetRight), -1, 1),
    targetDistance: clamp(finiteOrZero(sensors.targetDistance), 0, 1.5),
    forwardSpeed: clamp(finiteOrZero(sensors.forwardSpeed), -2, 2),
    angularSpeed: clamp(finiteOrZero(sensors.angularSpeed), -3, 3),
    terrainFriction: clamp(finiteOrZero(sensors.terrainFriction), 0, 1.2),
    urgency: clamp(finiteOrZero(sensors.urgency), 0, 1),
  }
}

function exactStringVector(
  value: unknown,
  expected: readonly string[],
  label: string,
): string[] {
  if (
    !Array.isArray(value) ||
    value.length !== expected.length ||
    !value.every((item, index) => item === expected[index])
  ) {
    throw new Error(`${label} must match the documented ordering`)
  }
  return [...expected]
}

function finiteVector(value: unknown, length: number, label: string): number[] {
  if (!Array.isArray(value) || value.length !== length) {
    throw new Error(`${label} must contain ${length} values`)
  }
  return value.map((item, index) => finiteNumber(item, `${label}[${index}]`))
}

function positiveFinite(value: unknown, label: string): number {
  const parsed = finiteNumber(value, label)
  if (parsed <= 0) throw new Error(`${label} must be positive`)
  return parsed
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${label} must be finite`)
  return value
}

function finiteOrZero(value: number | undefined): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function clamp(value: number, lower: number, upper: number): number {
  return Math.max(lower, Math.min(upper, value))
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
