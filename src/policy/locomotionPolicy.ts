import {
  ACTION_SIZE,
  POLICY_TIMESTEP,
  SEGMENT_COUNT,
  type ActiveGaitExperiment,
  type NeuralGaitPerturbationKind,
  type PolicyAction,
} from './types'
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
const V3_SELECTION_GATE_KEYS = [
  'nominalSignedHeadLeading',
  'nominalLateralMotion',
  'nominalTargetAlignment',
  'combinedControlProgress',
  'combinedPerturbedProgress',
  'combinedProgressRetention',
  'combinedLateSpeedRetention',
  'combinedControlLateSpeed',
  'recoveredSignedHeadLeading',
  'recoveredLateralMotion',
  'recoveredTargetAlignment',
  'causalOrderingGap',
  'causalProgressGap',
  'localCausalInitial',
  'localCausalRadiusTwo',
  'localCausalRadiusThree',
  'localCausalSettled',
] as const
const V3_ABLATION_GATE_KEYS = [
  'fullProgress',
  'zeroActionStationary',
  'fullBeatsFrozen',
  'fullBeatsShuffled',
  'noFrictionConservesCom',
] as const

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

export type LocomotionPolicyTelemetry = {
  hidden: number[]
  drives: number[]
  commands: number[]
  requestedCommands: number[]
  sensedBends: number[]
  sensedBendVelocities: number[]
  sensedContactLoads: number[]
  sensedSlipSpeeds: number[]
  sensedObstacleForward: number[]
  sensedObstacleRight: number[]
  activePerturbation: ActiveGaitExperiment | null
}

type NeuralPerturbationState = {
  kind: NeuralGaitPerturbationKind
  segment: number | null
  remainingSteps: number
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
  private readonly drives = new Float64Array(SEGMENT_COUNT)
  private readonly requestedCommands = new Float64Array(SEGMENT_COUNT)
  private readonly sensedBends = new Float64Array(SEGMENT_COUNT)
  private readonly sensedBendVelocities = new Float64Array(SEGMENT_COUNT)
  private readonly sensedContactLoads = new Float64Array(SEGMENT_COUNT)
  private readonly sensedSlipSpeeds = new Float64Array(SEGMENT_COUNT)
  private readonly sensedObstacleForward = new Float64Array(SEGMENT_COUNT)
  private readonly sensedObstacleRight = new Float64Array(SEGMENT_COUNT)
  private perturbation: NeuralPerturbationState | null = null

  constructor(artifact: LocomotionPolicyArtifact) {
    this.artifact = artifact
    this.modelVersion = artifact.modelVersion
    this.reset()
  }

  reset() {
    this.hidden.set(this.artifact.initialState)
    this.nextHidden.fill(0)
    this.previousCommand.fill(0)
    this.drives.fill(0)
    this.requestedCommands.fill(0)
    this.sensedBends.fill(0)
    this.sensedBendVelocities.fill(0)
    this.sensedContactLoads.fill(0)
    this.sensedSlipSpeeds.fill(0)
    this.sensedObstacleForward.fill(0)
    this.sensedObstacleRight.fill(0)
    this.perturbation = null
  }

  applyPerturbation(kind: NeuralGaitPerturbationKind, segment: number | null, durationSeconds: number) {
    if (
      kind === 'numb-neuron' &&
      (!Number.isInteger(segment) || segment === null || segment < 0 || segment >= SEGMENT_COUNT)
    ) {
      throw new Error(`neural perturbation segment must be between 0 and ${SEGMENT_COUNT - 1}`)
    }
    if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
      throw new Error('neural perturbation duration must be positive and finite')
    }
    this.perturbation = {
      kind,
      segment: kind === 'numb-neuron' ? segment : null,
      remainingSteps: Math.max(1, Math.round(durationSeconds / POLICY_TIMESTEP)),
    }
    if (kind === 'numb-neuron' && segment !== null) {
      this.hidden[segment] = 0
      this.nextHidden[segment] = 0
      this.previousCommand[segment] = 0
    }
  }

  clearPerturbation() {
    this.perturbation = null
  }

  hasActivePerturbation() {
    return this.perturbation !== null
  }

  /**
   * Replace the recurrent command feedback after an external actuator
   * intervention. Ordinary runtime inference does not need this: run() already
   * commits the controller's effective command. Verifiers use it when they
   * deliberately zero, freeze, or reassign commands before the plant sees them.
   *
   * Keep this vector at controller precision. The browser plant separately
   * receives the Float32 PolicyAction, matching the trainer's precision split.
   */
  commitCommandFeedback(commands: ArrayLike<number>) {
    if (commands.length !== SEGMENT_COUNT) {
      throw new Error(`locomotion command feedback requires ${SEGMENT_COUNT} segment values`)
    }
    const committed = new Float64Array(SEGMENT_COUNT)
    for (let segment = 0; segment < SEGMENT_COUNT; segment += 1) {
      committed[segment] = clamp(
        finiteNumber(commands[segment], `locomotion command feedback segment ${segment}`),
        -1,
        1,
      )
    }
    this.previousCommand.set(committed)
  }

  getTelemetry(): LocomotionPolicyTelemetry {
    return {
      hidden: Array.from(this.hidden),
      drives: Array.from(this.drives),
      commands: Array.from(this.previousCommand),
      requestedCommands: Array.from(this.requestedCommands),
      sensedBends: Array.from(this.sensedBends),
      sensedBendVelocities: Array.from(this.sensedBendVelocities),
      sensedContactLoads: Array.from(this.sensedContactLoads),
      sensedSlipSpeeds: Array.from(this.sensedSlipSpeeds),
      sensedObstacleForward: Array.from(this.sensedObstacleForward),
      sensedObstacleRight: Array.from(this.sensedObstacleRight),
      activePerturbation: this.perturbation
        ? {
            kind: this.perturbation.kind,
            segment: this.perturbation.segment,
            remainingSeconds: this.perturbation.remainingSteps * POLICY_TIMESTEP,
          }
        : null,
    }
  }

  run(
    sensors: LocomotionSensors,
    segmentBends: ArrayLike<number>,
    segmentBendVelocities: ArrayLike<number>,
  ): PolicyAction {
    if (segmentBends.length !== SEGMENT_COUNT || segmentBendVelocities.length !== SEGMENT_COUNT) {
      throw new Error(`locomotion policy requires ${SEGMENT_COUNT} segment bend values and velocities`)
    }

    // Keep a zero-second perturbation observable after its final affected tick,
    // then retire it immediately before the next inference step.
    if (this.perturbation?.remainingSteps === 0) this.perturbation = null
    const input = this.artifact.weights.input
    const recurrent = this.artifact.weights.recurrent
    const output = this.artifact.weights.output
    const positions = this.artifact.segmentPositions
    const safeSensors = sanitizeSensors(sensors)
    const perturbation = this.perturbation

    for (let segment = 0; segment < SEGMENT_COUNT; segment += 1) {
      const position = positions[segment]
      const sensorIndex = perturbation?.kind === 'reverse-sensors' ? SEGMENT_COUNT - 1 - segment : segment
      const bend = finiteOrZero(segmentBends[sensorIndex])
      const bendVelocity = finiteOrZero(segmentBendVelocities[sensorIndex])
      const contactLoad = localSensor(safeSensors.contactLoads, sensorIndex, 1, 0, 1)
      const slipSpeed = localSensor(safeSensors.slipSpeeds, sensorIndex, 0, 0, 2)
      const obstacleForward = localSensor(safeSensors.obstacleForward, sensorIndex, 0, -1, 1)
      const obstacleRight = localSensor(safeSensors.obstacleRight, sensorIndex, 0, -1, 1)
      this.sensedBends[segment] = bend
      this.sensedBendVelocities[segment] = bendVelocity
      this.sensedContactLoads[segment] = contactLoad
      this.sensedSlipSpeeds[segment] = slipSpeed
      this.sensedObstacleForward[segment] = obstacleForward
      this.sensedObstacleRight[segment] = obstacleRight
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
      this.drives[segment] = drive
      this.nextHidden[segment] =
        perturbation?.kind === 'numb-neuron' && perturbation.segment === segment ? 0 : Math.tanh(drive)
    }

    const action = new Float32Array(ACTION_SIZE)
    for (let segment = 0; segment < SEGMENT_COUNT; segment += 1) {
      const position = positions[segment]
      const bend = this.sensedBends[segment]
      const anterior = segment > 0 ? this.nextHidden[segment - 1] : 0
      const posterior = segment + 1 < SEGMENT_COUNT ? this.nextHidden[segment + 1] : 0
      const contactLoad = this.sensedContactLoads[segment]
      const obstacleRight = this.sensedObstacleRight[segment]
      const requestedCommand = Math.tanh(
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
      const command =
        perturbation?.kind === 'numb-neuron' && perturbation.segment === segment ? 0 : requestedCommand
      this.requestedCommands[segment] = requestedCommand
      this.previousCommand[segment] = command
      action[segment * 2] = command
      action[segment * 2 + 1] = -command
    }
    this.hidden.set(this.nextHidden)
    if (this.perturbation) {
      this.perturbation.remainingSteps -= 1
    }
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
  validatePublicationMetadata(value.training, value.modelVersion)

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

function validatePublicationMetadata(training: Record<string, unknown>, modelVersion: string) {
  const objective = training.objectiveVersion
  const v3Objective = 'articulated-head-leading-transient-recovery-v3'
  const publishedV3Model = 'locomotion-articulated-head-leading-es-v3'
  if (modelVersion === publishedV3Model && objective !== v3Objective) {
    throw new Error('published v3 locomotion identity requires the v3 objective contract')
  }
  if (objective !== v3Objective) return

  for (const key of [
    'allSelectionGatesPassed',
    'allSelectionMarginsFeasible',
    'allSelectionGuardBandsPassed',
    'allAblationGatesPassed',
    'allAblationGuardBandsPassed',
    'allGateGuardBandsPassed',
    'allPublicationGatesPassed',
  ]) {
    if (training[key] !== true) throw new Error(`v3 training.${key} must be true before runtime loading`)
  }
  requireExactPassingGates(training.selectionGateResults, 'selectionGateResults', V3_SELECTION_GATE_KEYS)
  requireExactPassingGates(
    training.selectionGateGuardBandResults,
    'selectionGateGuardBandResults',
    V3_SELECTION_GATE_KEYS,
  )
  requireExactPassingGates(training.ablationGateResults, 'ablationGateResults', V3_ABLATION_GATE_KEYS)
  requireExactPassingGates(
    training.ablationGateGuardBandResults,
    'ablationGateGuardBandResults',
    V3_ABLATION_GATE_KEYS,
  )
  if (!/^[a-f\d]{64}$/u.test(String(training.canonicalGenomeRecipeHash))) {
    throw new Error('v3 training.canonicalGenomeRecipeHash must be a lowercase SHA-256 digest')
  }
  const pairedHost = training.pairedHostReproduction
  if (
    !isRecord(pairedHost) ||
    pairedHost.requiredForPublication !== true ||
    pairedHost.trainerAttestation !== 'none' ||
    !Array.isArray(pairedHost.compared) ||
    pairedHost.compared.length !== 2 ||
    pairedHost.compared[0] !== 'serializedGenome' ||
    pairedHost.compared[1] !== 'artifactBytes'
  ) {
    throw new Error('v3 training.pairedHostReproduction must require external genome and byte comparison')
  }
}

function requireExactPassingGates(value: unknown, label: string, expectedKeys: readonly string[]) {
  if (!isRecord(value)) throw new Error(`v3 training.${label} must be a gate record`)
  const actualKeys = Object.keys(value).sort()
  const canonicalKeys = [...expectedKeys].sort()
  if (
    actualKeys.length !== canonicalKeys.length ||
    actualKeys.some((key, index) => key !== canonicalKeys[index])
  ) {
    throw new Error(`v3 training.${label} keys do not match the publication contract`)
  }
  if (expectedKeys.some(key => value[key] !== true)) {
    throw new Error(`v3 training.${label} must contain only passing gates`)
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
