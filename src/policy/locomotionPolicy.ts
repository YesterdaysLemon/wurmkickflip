import { ACTION_SIZE, POLICY_TIMESTEP, SEGMENT_COUNT, type PolicyAction } from './types'
import { LOCOMOTION_CONTRACT } from './locomotionContract'

export const LOCOMOTION_POLICY_PATH = '/models/wurmkickflip_locomotion_policy.json'

/** Versioned articulated plant contract shared by evolution and the browser. */
export const LOCOMOTION_PLANT_CONTRACT = {
  version: LOCOMOTION_CONTRACT.plantVersion,
  timestep: POLICY_TIMESTEP,
  jointStiffness: LOCOMOTION_CONTRACT.joint.stiffness,
  jointDamping: LOCOMOTION_CONTRACT.joint.damping,
  jointLimit: LOCOMOTION_CONTRACT.joint.limit,
  ...LOCOMOTION_CONTRACT.dynamics,
} as const

export const LOCOMOTION_SENSOR_NAMES = LOCOMOTION_CONTRACT.sensorNames
export const LOCOMOTION_INPUT_WEIGHT_NAMES = LOCOMOTION_CONTRACT.inputWeightNames
export const LOCOMOTION_RECURRENT_WEIGHT_NAMES = LOCOMOTION_CONTRACT.recurrentWeightNames
export const LOCOMOTION_OUTPUT_WEIGHT_NAMES = LOCOMOTION_CONTRACT.outputWeightNames

const LEGACY_SENSOR_NAMES = LOCOMOTION_SENSOR_NAMES.slice(0, 7)
const LEGACY_INPUT_WEIGHT_NAMES = LOCOMOTION_INPUT_WEIGHT_NAMES.slice(0, 13)
const LEGACY_OUTPUT_WEIGHT_NAMES = LOCOMOTION_OUTPUT_WEIGHT_NAMES.slice(0, 7)

export type LocomotionSensors = {
  targetForward: number
  targetRight: number
  targetDistance: number
  forwardSpeed: number
  angularSpeed: number
  terrainFriction: number
  urgency: number
  /** Segment-local feedback arrays are anterior-to-posterior. */
  contactLoads?: ArrayLike<number>
  slipSpeeds?: ArrayLike<number>
  obstacleForward?: ArrayLike<number>
  obstacleRight?: ArrayLike<number>
}

export type LocomotionPolicyArtifact = {
  schemaVersion: 2
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
    version: string
    spacing: number
    actuatorStiffness: number
    actuatorDamping: number
    shapeBendScale: number
    constraintIterations: number
    constraintCompliance: number
    longitudinalFriction: number
    lateralFriction: number
    freeVelocityDrag: number
    maximumSpeed: number
    maximumVerticalSpeed: number
    baseRadius: number
    baseGroundClearance: number
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
      const contactLoad = localSensor(safeSensors.contactLoads, segment, 1, 0, 1)
      const slipSpeed = localSensor(safeSensors.slipSpeeds, segment, 0, 0, 2)
      const obstacleForward = localSensor(safeSensors.obstacleForward, segment, 0, -1, 1)
      const obstacleRight = localSensor(safeSensors.obstacleRight, segment, 0, -1, 1)
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
        input[13] * contactLoad +
        input[14] * slipSpeed +
        input[15] * obstacleForward +
        input[16] * obstacleRight +
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
      const contactLoad = localSensor(safeSensors.contactLoads, segment, 1, 0, 1)
      const obstacleRight = localSensor(safeSensors.obstacleRight, segment, 0, -1, 1)
      const command = Math.tanh(
        output[0] +
          output[1] * this.nextHidden[segment] +
          output[2] * anterior +
          output[3] * posterior +
          output[4] * safeSensors.targetRight +
          output[5] * safeSensors.targetRight * position +
          output[6] * bend +
          output[7] * obstacleRight +
          output[8] * contactLoad,
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
  if (value.schemaVersion !== 1 && value.schemaVersion !== 2) {
    throw new Error('locomotion schemaVersion must be 1 or 2')
  }
  const legacy = value.schemaVersion === 1
  if (value.kind !== 'wurmkickflip.locomotionPolicy') throw new Error('unexpected locomotion artifact kind')
  if (value.architecture !== 'segmental-recurrent-tanh') throw new Error('unexpected locomotion architecture')
  if (typeof value.modelVersion !== 'string' || value.modelVersion.length === 0) {
    throw new Error('locomotion modelVersion must be non-empty')
  }
  if (value.segmentCount !== SEGMENT_COUNT || value.actionSize !== ACTION_SIZE) {
    throw new Error(`locomotion artifact requires ${SEGMENT_COUNT} segments and ${ACTION_SIZE} actions`)
  }
  exactStringVector(value.sensorNames, legacy ? LEGACY_SENSOR_NAMES : LOCOMOTION_SENSOR_NAMES, 'sensorNames')
  const segmentPositions = finiteVector(value.segmentPositions, SEGMENT_COUNT, 'segmentPositions')
  const initialState = finiteVector(value.initialState, SEGMENT_COUNT, 'initialState')
  if (!isRecord(value.weights)) throw new Error('locomotion weights must be an object')
  const expectedInputNames = legacy ? LEGACY_INPUT_WEIGHT_NAMES : LOCOMOTION_INPUT_WEIGHT_NAMES
  exactStringVector(value.weights.inputNames, expectedInputNames, 'weights.inputNames')
  const parsedInput = finiteVector(value.weights.input, expectedInputNames.length, 'weights.input')
  const input = legacy
    ? [...parsedInput, ...new Array(LOCOMOTION_INPUT_WEIGHT_NAMES.length - parsedInput.length).fill(0)]
    : parsedInput
  const recurrentNames = exactStringVector(
    value.weights.recurrentNames,
    LOCOMOTION_RECURRENT_WEIGHT_NAMES,
    'weights.recurrentNames',
  )
  const recurrent = finiteVector(
    value.weights.recurrent,
    LOCOMOTION_RECURRENT_WEIGHT_NAMES.length,
    'weights.recurrent',
  )
  const expectedOutputNames = legacy ? LEGACY_OUTPUT_WEIGHT_NAMES : LOCOMOTION_OUTPUT_WEIGHT_NAMES
  exactStringVector(value.weights.outputNames, expectedOutputNames, 'weights.outputNames')
  const parsedOutput = finiteVector(value.weights.output, expectedOutputNames.length, 'weights.output')
  const output = legacy
    ? [...parsedOutput, ...new Array(LOCOMOTION_OUTPUT_WEIGHT_NAMES.length - parsedOutput.length).fill(0)]
    : parsedOutput
  const plant = legacy ? parseLegacyPlant(value.plant) : parsePlant(value.plant)
  if (!isRecord(value.training)) throw new Error('locomotion training metadata must be an object')

  return {
    schemaVersion: 2,
    kind: 'wurmkickflip.locomotionPolicy',
    modelVersion: value.modelVersion,
    architecture: 'segmental-recurrent-tanh',
    segmentCount: SEGMENT_COUNT,
    actionSize: ACTION_SIZE,
    sensorNames: [...LOCOMOTION_SENSOR_NAMES],
    segmentPositions,
    initialState,
    weights: {
      inputNames: [...LOCOMOTION_INPUT_WEIGHT_NAMES],
      input,
      recurrentNames,
      recurrent,
      outputNames: [...LOCOMOTION_OUTPUT_WEIGHT_NAMES],
      output,
    },
    plant,
    training: legacy
      ? { ...value.training, runtimeMigration: 'schema-1-zero-contact-weights' }
      : { ...value.training },
  }
}

function parsePlant(value: unknown): LocomotionPolicyArtifact['plant'] {
  if (!isRecord(value)) throw new Error('locomotion plant metadata must be an object')
  const plant = {
    version: exactString(value.version, 'plant.version'),
    timestep: positiveFinite(value.timestep, 'plant.timestep'),
    jointStiffness: positiveFinite(value.jointStiffness, 'plant.jointStiffness'),
    jointDamping: positiveFinite(value.jointDamping, 'plant.jointDamping'),
    jointLimit: positiveFinite(value.jointLimit, 'plant.jointLimit'),
    spacing: positiveFinite(value.spacing, 'plant.spacing'),
    actuatorStiffness: positiveFinite(value.actuatorStiffness, 'plant.actuatorStiffness'),
    actuatorDamping: positiveFinite(value.actuatorDamping, 'plant.actuatorDamping'),
    shapeBendScale: positiveFinite(value.shapeBendScale, 'plant.shapeBendScale'),
    constraintIterations: positiveFinite(value.constraintIterations, 'plant.constraintIterations'),
    constraintCompliance: positiveFinite(value.constraintCompliance, 'plant.constraintCompliance'),
    longitudinalFriction: positiveFinite(value.longitudinalFriction, 'plant.longitudinalFriction'),
    lateralFriction: positiveFinite(value.lateralFriction, 'plant.lateralFriction'),
    freeVelocityDrag: positiveFinite(value.freeVelocityDrag, 'plant.freeVelocityDrag'),
    maximumSpeed: positiveFinite(value.maximumSpeed, 'plant.maximumSpeed'),
    maximumVerticalSpeed:
      value.maximumVerticalSpeed === undefined
        ? LOCOMOTION_PLANT_CONTRACT.maximumVerticalSpeed
        : positiveFinite(value.maximumVerticalSpeed, 'plant.maximumVerticalSpeed'),
    baseRadius: positiveFinite(value.baseRadius, 'plant.baseRadius'),
    baseGroundClearance: positiveFinite(value.baseGroundClearance, 'plant.baseGroundClearance'),
  }
  for (const key of Object.keys(LOCOMOTION_PLANT_CONTRACT) as Array<keyof typeof LOCOMOTION_PLANT_CONTRACT>) {
    const actual = plant[key]
    const expected = LOCOMOTION_PLANT_CONTRACT[key]
    if (
      typeof expected === 'string'
        ? actual !== expected
        : typeof actual !== 'number' || Math.abs(actual - expected) > 1e-9
    ) {
      throw new Error(`plant.${key} does not match the browser locomotion contract`)
    }
  }
  return plant
}

function parseLegacyPlant(value: unknown): LocomotionPolicyArtifact['plant'] {
  if (!isRecord(value)) throw new Error('legacy locomotion plant metadata must be an object')
  const expected = LOCOMOTION_PLANT_CONTRACT
  const checks = [
    ['timestep', expected.timestep],
    ['jointStiffness', expected.jointStiffness],
    ['jointDamping', expected.jointDamping],
    ['jointLimit', expected.jointLimit],
  ] as const
  for (const [key, expectedValue] of checks) {
    const actual = finiteNumber(value[key], `plant.${key}`)
    if (Math.abs(actual - expectedValue) > 1e-9) {
      throw new Error(`legacy plant.${key} does not match the joint contract`)
    }
  }
  return { ...LOCOMOTION_PLANT_CONTRACT }
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
    contactLoads: sensors.contactLoads,
    slipSpeeds: sensors.slipSpeeds,
    obstacleForward: sensors.obstacleForward,
    obstacleRight: sensors.obstacleRight,
  }
}

function localSensor(
  values: ArrayLike<number> | undefined,
  index: number,
  fallback: number,
  minimum: number,
  maximum: number,
) {
  return clamp(values ? finiteOr(values[index], fallback) : fallback, minimum, maximum)
}

function exactString(value: unknown, label: string) {
  if (typeof value !== 'string' || value.length === 0) throw new Error(`${label} must be non-empty`)
  return value
}

function exactStringVector(value: unknown, expected: readonly string[], label: string): string[] {
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

function finiteOr(value: number | undefined, fallback: number) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function clamp(value: number, lower: number, upper: number): number {
  return Math.max(lower, Math.min(upper, value))
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
