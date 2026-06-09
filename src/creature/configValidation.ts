import type {
  CreatureBodyPart,
  CreatureGenome,
  CreatureJoint,
  CreatureSensor,
  EnvironmentConfig,
  PrimitiveShape,
  Range,
  Vec3,
} from './types'

export type ConfigValidationResult<T> =
  | { ok: true; value: T }
  | { ok: false; errors: string[] }

const primitiveShapes = ['capsule', 'box', 'sphere', 'cylinder'] as const satisfies readonly PrimitiveShape[]
const jointTypes = ['hinge', 'ball', 'fixed'] as const
const symmetries = ['none', 'bilateral', 'radial'] as const
const controllerKinds = ['cpg', 'onnx_policy', 'hybrid'] as const
const sensors = [
  'body_pose',
  'body_velocity',
  'joint_angles',
  'joint_velocities',
  'contacts',
  'target_direction',
  'environment_probe',
] as const satisfies readonly CreatureSensor[]
const terrainKinds = ['flat', 'slope', 'bumps', 'obstacle_field'] as const
const taskKinds = ['forward_locomotion', 'target_reach', 'skateboard_balance', 'find_and_ride_skateboard'] as const

export function validateCreatureGenome(value: unknown): ConfigValidationResult<CreatureGenome> {
  const errors: string[] = []
  const root = asRecord(value, 'creature', errors)
  const morphology = root ? asRecord(root.morphology, 'creature.morphology', errors) : null
  const controller = root ? asRecord(root.controller, 'creature.controller', errors) : null
  const mutation = root ? asRecord(root.mutation, 'creature.mutation', errors) : null
  const bodyPartsInput = morphology ? asArray(morphology.bodyParts, 'creature.morphology.bodyParts', errors) : []
  const jointsInput = morphology ? asArray(morphology.joints, 'creature.morphology.joints', errors) : []
  const sensorsInput = root ? asArray(root.sensors, 'creature.sensors', errors) : []

  const bodyParts = bodyPartsInput.map((part, index) => parseBodyPart(part, `creature.morphology.bodyParts[${index}]`, errors))
  const joints = jointsInput.map((joint, index) => parseJoint(joint, `creature.morphology.joints[${index}]`, errors))
  const bodyPartIds = new Set(bodyParts.map((part) => part.id).filter(Boolean))
  const jointIds = new Set(joints.map((joint) => joint.id).filter(Boolean))

  if (bodyParts.length === 0) {
    errors.push('creature.morphology.bodyParts must include at least one part.')
  }
  if (bodyPartIds.size !== bodyParts.length) {
    errors.push('creature.morphology.bodyParts ids must be unique and non-empty.')
  }
  if (jointIds.size !== joints.length) {
    errors.push('creature.morphology.joints ids must be unique and non-empty.')
  }

  for (const part of bodyParts) {
    if (part.parentId !== null && !bodyPartIds.has(part.parentId)) {
      errors.push(`creature body part ${part.id} references missing parentId ${part.parentId}.`)
    }
  }
  for (const joint of joints) {
    if (!bodyPartIds.has(joint.parentId)) {
      errors.push(`creature joint ${joint.id} references missing parentId ${joint.parentId}.`)
    }
    if (!bodyPartIds.has(joint.childId)) {
      errors.push(`creature joint ${joint.id} references missing childId ${joint.childId}.`)
    }
  }

  const parsedSensors = sensorsInput.map((sensor, index) =>
    asEnum(sensor, sensors, `creature.sensors[${index}]`, errors),
  )

  const creature: CreatureGenome = {
    schemaVersion: asSchemaVersion(root?.schemaVersion, 'creature.schemaVersion', errors),
    id: asNonEmptyString(root?.id, 'creature.id', errors),
    name: asNonEmptyString(root?.name, 'creature.name', errors),
    description: asString(root?.description, 'creature.description', errors),
    morphology: {
      symmetry: asEnum(morphology?.symmetry, symmetries, 'creature.morphology.symmetry', errors),
      bodyParts,
      joints,
    },
    controller: {
      kind: asEnum(controller?.kind, controllerKinds, 'creature.controller.kind', errors),
      actionSize: asPositiveNumber(controller?.actionSize, 'creature.controller.actionSize', errors),
      parameters: parseNumberRecord(controller?.parameters, 'creature.controller.parameters', errors),
    },
    sensors: parsedSensors,
    mutation: {
      enabled: asBoolean(mutation?.enabled, 'creature.mutation.enabled', errors),
      partScaleRange: asRange(mutation?.partScaleRange, 'creature.mutation.partScaleRange', errors),
      massScaleRange: asRange(mutation?.massScaleRange, 'creature.mutation.massScaleRange', errors),
      jointLimitJitter: asNonNegativeNumber(mutation?.jointLimitJitter, 'creature.mutation.jointLimitJitter', errors),
      controllerJitter: asNonNegativeNumber(mutation?.controllerJitter, 'creature.mutation.controllerJitter', errors),
    },
  }

  return errors.length > 0 ? { ok: false, errors } : { ok: true, value: creature }
}

export function validateEnvironmentConfig(value: unknown): ConfigValidationResult<EnvironmentConfig> {
  const errors: string[] = []
  const root = asRecord(value, 'environment', errors)
  const world = root ? asRecord(root.world, 'environment.world', errors) : null
  const terrain = root ? asRecord(root.terrain, 'environment.terrain', errors) : null
  const task = root ? asRecord(root.task, 'environment.task', errors) : null
  const skateboard = root ? asRecord(root.skateboard, 'environment.skateboard', errors) : null
  const randomization = root ? asRecord(root.randomization, 'environment.randomization', errors) : null
  const rewardWeights = root ? asRecord(root.rewardWeights, 'environment.rewardWeights', errors) : null

  const environment: EnvironmentConfig = {
    schemaVersion: asSchemaVersion(root?.schemaVersion, 'environment.schemaVersion', errors),
    id: asNonEmptyString(root?.id, 'environment.id', errors),
    name: asNonEmptyString(root?.name, 'environment.name', errors),
    description: asString(root?.description, 'environment.description', errors),
    seed: asFiniteNumber(root?.seed, 'environment.seed', errors),
    world: {
      size: asVec3(world?.size, 'environment.world.size', errors),
      gravity: asVec3(world?.gravity, 'environment.world.gravity', errors),
      airDrag: asNonNegativeNumber(world?.airDrag, 'environment.world.airDrag', errors),
      solverIterations: asPositiveNumber(world?.solverIterations, 'environment.world.solverIterations', errors),
    },
    terrain: {
      kind: asEnum(terrain?.kind, terrainKinds, 'environment.terrain.kind', errors),
      baseFriction: asNonNegativeNumber(terrain?.baseFriction, 'environment.terrain.baseFriction', errors),
      restitution: asNonNegativeNumber(terrain?.restitution, 'environment.terrain.restitution', errors),
      slopeDegrees: asFiniteNumber(terrain?.slopeDegrees, 'environment.terrain.slopeDegrees', errors),
      roughness: asNonNegativeNumber(terrain?.roughness, 'environment.terrain.roughness', errors),
      obstacleDensity: asNonNegativeNumber(terrain?.obstacleDensity, 'environment.terrain.obstacleDensity', errors),
    },
    task: {
      kind: asEnum(task?.kind, taskKinds, 'environment.task.kind', errors),
      targetDirection: asVec3(task?.targetDirection, 'environment.task.targetDirection', errors),
      durationSeconds: asPositiveNumber(task?.durationSeconds, 'environment.task.durationSeconds', errors),
      failureHeight: asFiniteNumber(task?.failureHeight, 'environment.task.failureHeight', errors),
    },
    skateboard: {
      enabled: asBoolean(skateboard?.enabled, 'environment.skateboard.enabled', errors),
      spawnPosition: asVec3(skateboard?.spawnPosition, 'environment.skateboard.spawnPosition', errors),
      deckSize: asVec3(skateboard?.deckSize, 'environment.skateboard.deckSize', errors),
      mass: asPositiveNumber(skateboard?.mass, 'environment.skateboard.mass', errors),
      wheelRadius: asPositiveNumber(skateboard?.wheelRadius, 'environment.skateboard.wheelRadius', errors),
      wheelFriction: asNonNegativeNumber(skateboard?.wheelFriction, 'environment.skateboard.wheelFriction', errors),
      discoveryRadius: asPositiveNumber(skateboard?.discoveryRadius, 'environment.skateboard.discoveryRadius', errors),
      mountHeightTolerance: asNonNegativeNumber(
        skateboard?.mountHeightTolerance,
        'environment.skateboard.mountHeightTolerance',
        errors,
      ),
    },
    randomization: {
      gravityScale: asRange(randomization?.gravityScale, 'environment.randomization.gravityScale', errors),
      frictionScale: asRange(randomization?.frictionScale, 'environment.randomization.frictionScale', errors),
      dragScale: asRange(randomization?.dragScale, 'environment.randomization.dragScale', errors),
      slopeDegrees: asRange(randomization?.slopeDegrees, 'environment.randomization.slopeDegrees', errors),
      roughness: asRange(randomization?.roughness, 'environment.randomization.roughness', errors),
      obstacleDensity: asRange(randomization?.obstacleDensity, 'environment.randomization.obstacleDensity', errors),
      actuatorStrength: asRange(randomization?.actuatorStrength, 'environment.randomization.actuatorStrength', errors),
      actuatorLatencyMs: asRange(randomization?.actuatorLatencyMs, 'environment.randomization.actuatorLatencyMs', errors),
      sensorNoise: asRange(randomization?.sensorNoise, 'environment.randomization.sensorNoise', errors),
      spawnYawDegrees: asRange(randomization?.spawnYawDegrees, 'environment.randomization.spawnYawDegrees', errors),
      skateboardSpawnX: asRange(randomization?.skateboardSpawnX, 'environment.randomization.skateboardSpawnX', errors),
      skateboardSpawnZ: asRange(randomization?.skateboardSpawnZ, 'environment.randomization.skateboardSpawnZ', errors),
      skateboardMass: asRange(randomization?.skateboardMass, 'environment.randomization.skateboardMass', errors),
      wheelFriction: asRange(randomization?.wheelFriction, 'environment.randomization.wheelFriction', errors),
    },
    rewardWeights: {
      forwardProgress: asFiniteNumber(rewardWeights?.forwardProgress, 'environment.rewardWeights.forwardProgress', errors),
      stability: asFiniteNumber(rewardWeights?.stability, 'environment.rewardWeights.stability', errors),
      energy: asFiniteNumber(rewardWeights?.energy, 'environment.rewardWeights.energy', errors),
      survival: asFiniteNumber(rewardWeights?.survival, 'environment.rewardWeights.survival', errors),
      targetAlignment: asFiniteNumber(rewardWeights?.targetAlignment, 'environment.rewardWeights.targetAlignment', errors),
      skateboardDiscovery: asFiniteNumber(
        rewardWeights?.skateboardDiscovery,
        'environment.rewardWeights.skateboardDiscovery',
        errors,
      ),
      skateboardContact: asFiniteNumber(rewardWeights?.skateboardContact, 'environment.rewardWeights.skateboardContact', errors),
      skateboardVelocity: asFiniteNumber(
        rewardWeights?.skateboardVelocity,
        'environment.rewardWeights.skateboardVelocity',
        errors,
      ),
    },
  }

  return errors.length > 0 ? { ok: false, errors } : { ok: true, value: environment }
}

function parseBodyPart(value: unknown, path: string, errors: string[]): CreatureBodyPart {
  const record = asRecord(value, path, errors)
  const material = record ? asRecord(record.material, `${path}.material`, errors) : null
  const visual = record ? asRecord(record.visual, `${path}.visual`, errors) : null
  return {
    id: asNonEmptyString(record?.id, `${path}.id`, errors),
    shape: asEnum(record?.shape, primitiveShapes, `${path}.shape`, errors),
    parentId: asNullableString(record?.parentId, `${path}.parentId`, errors),
    position: asVec3(record?.position, `${path}.position`, errors),
    rotation: asVec3(record?.rotation, `${path}.rotation`, errors),
    size: asVec3(record?.size, `${path}.size`, errors),
    mass: asPositiveNumber(record?.mass, `${path}.mass`, errors),
    material: {
      friction: asNonNegativeNumber(material?.friction, `${path}.material.friction`, errors),
      restitution: asNonNegativeNumber(material?.restitution, `${path}.material.restitution`, errors),
    },
    visual: {
      color: asNonEmptyString(visual?.color, `${path}.visual.color`, errors),
    },
  }
}

function parseJoint(value: unknown, path: string, errors: string[]): CreatureJoint {
  const record = asRecord(value, path, errors)
  return {
    id: asNonEmptyString(record?.id, `${path}.id`, errors),
    parentId: asNonEmptyString(record?.parentId, `${path}.parentId`, errors),
    childId: asNonEmptyString(record?.childId, `${path}.childId`, errors),
    type: asEnum(record?.type, jointTypes, `${path}.type`, errors),
    axis: asVec3(record?.axis, `${path}.axis`, errors),
    anchor: asVec3(record?.anchor, `${path}.anchor`, errors),
    limits: asRange(record?.limits, `${path}.limits`, errors),
    stiffness: asNonNegativeNumber(record?.stiffness, `${path}.stiffness`, errors),
    damping: asNonNegativeNumber(record?.damping, `${path}.damping`, errors),
    motorStrength: asNonNegativeNumber(record?.motorStrength, `${path}.motorStrength`, errors),
  }
}

function parseNumberRecord(value: unknown, path: string, errors: string[]): Record<string, number> {
  const record = asRecord(value, path, errors)
  const result: Record<string, number> = {}
  if (!record) {
    return result
  }
  for (const [key, item] of Object.entries(record)) {
    result[key] = asFiniteNumber(item, `${path}.${key}`, errors)
  }
  return result
}

function asRecord(value: unknown, path: string, errors: string[]): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    errors.push(`${path} must be an object.`)
    return null
  }
  return value as Record<string, unknown>
}

function asArray(value: unknown, path: string, errors: string[]): unknown[] {
  if (!Array.isArray(value)) {
    errors.push(`${path} must be an array.`)
    return []
  }
  return value
}

function asString(value: unknown, path: string, errors: string[]): string {
  if (typeof value !== 'string') {
    errors.push(`${path} must be a string.`)
    return ''
  }
  return value
}

function asNonEmptyString(value: unknown, path: string, errors: string[]): string {
  const result = asString(value, path, errors)
  if (result.length === 0) {
    errors.push(`${path} must not be empty.`)
  }
  return result
}

function asNullableString(value: unknown, path: string, errors: string[]): string | null {
  if (value === null) {
    return null
  }
  return asNonEmptyString(value, path, errors)
}

function asBoolean(value: unknown, path: string, errors: string[]): boolean {
  if (typeof value !== 'boolean') {
    errors.push(`${path} must be a boolean.`)
    return false
  }
  return value
}

function asSchemaVersion(value: unknown, path: string, errors: string[]): 1 {
  if (value !== 1) {
    errors.push(`${path} must equal 1.`)
  }
  return 1
}

function asFiniteNumber(value: unknown, path: string, errors: string[]): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    errors.push(`${path} must be a finite number.`)
    return 0
  }
  return value
}

function asPositiveNumber(value: unknown, path: string, errors: string[]): number {
  const result = asFiniteNumber(value, path, errors)
  if (result <= 0) {
    errors.push(`${path} must be greater than 0.`)
  }
  return result
}

function asNonNegativeNumber(value: unknown, path: string, errors: string[]): number {
  const result = asFiniteNumber(value, path, errors)
  if (result < 0) {
    errors.push(`${path} must be at least 0.`)
  }
  return result
}

function asVec3(value: unknown, path: string, errors: string[]): Vec3 {
  const values = asTuple(value, path, 3, errors)
  return [values[0], values[1], values[2]]
}

function asRange(value: unknown, path: string, errors: string[]): Range {
  const values = asTuple(value, path, 2, errors)
  if (values[0] > values[1]) {
    errors.push(`${path} minimum must be less than or equal to maximum.`)
  }
  return [values[0], values[1]]
}

function asTuple(value: unknown, path: string, length: number, errors: string[]): number[] {
  if (!Array.isArray(value) || value.length !== length) {
    errors.push(`${path} must be an array of ${length} finite numbers.`)
    return new Array<number>(length).fill(0)
  }
  return value.map((item, index) => asFiniteNumber(item, `${path}[${index}]`, errors))
}

function asEnum<const T extends readonly string[]>(value: unknown, allowed: T, path: string, errors: string[]): T[number] {
  if (typeof value === 'string' && allowed.includes(value)) {
    return value
  }
  errors.push(`${path} must be one of: ${allowed.join(', ')}.`)
  return allowed[0]
}
