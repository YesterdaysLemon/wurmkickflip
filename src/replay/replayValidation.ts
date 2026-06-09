import {
  REPLAY_KIND,
  REPLAY_SCHEMA_VERSION,
  type ReplayArtifact,
  type ReplayFallReason,
  type ReplayPolicyBackend,
  type ReplayVec3,
} from './types'

export type ReplayValidationResult =
  | { ok: true; value: ReplayArtifact }
  | { ok: false; errors: string[] }

const policyBackends = ['scripted', 'onnx-webgpu', 'onnx-wasm', 'training'] as const satisfies readonly ReplayPolicyBackend[]
const fallReasons = ['none', 'lost_contact', 'fell_below_height', 'out_of_bounds', 'timeout'] as const satisfies readonly ReplayFallReason[]

export function validateReplayArtifact(value: unknown): ReplayValidationResult {
  const errors: string[] = []
  const root = asRecord(value, 'replay', errors)
  const source = root ? asRecord(root.source, 'replay.source', errors) : null
  const environmentSample = root ? asRecord(root.environmentSample, 'replay.environmentSample', errors) : null
  const taskMetrics = root ? asRecord(root.taskMetrics, 'replay.taskMetrics', errors) : null
  const framesInput = root ? asArray(root.frames, 'replay.frames', errors) : []

  const frameCount = asNonNegativeInteger(root?.frameCount, 'replay.frameCount', errors)
  const frames = framesInput.map((frame, index) => {
    const record = asRecord(frame, `replay.frames[${index}]`, errors)
    const board = record ? parsePose(record.board, `replay.frames[${index}].board`, errors) : emptyPose()
    const creatureRoot = record
      ? parsePose(record.creatureRoot, `replay.frames[${index}].creatureRoot`, errors)
      : emptyPose()
    return {
      time: asNonNegativeNumber(record?.time, `replay.frames[${index}].time`, errors),
      board,
      creatureRoot,
      contactRatio: asUnitNumber(record?.contactRatio, `replay.frames[${index}].contactRatio`, errors),
      skateboardDiscovered: asBoolean(record?.skateboardDiscovered, `replay.frames[${index}].skateboardDiscovered`, errors),
      rollingVelocity: asFiniteNumber(record?.rollingVelocity, `replay.frames[${index}].rollingVelocity`, errors),
      reward: asFiniteNumber(record?.reward, `replay.frames[${index}].reward`, errors),
    }
  })

  if (frameCount !== frames.length) {
    errors.push(`replay.frameCount must equal replay.frames length ${frames.length}.`)
  }
  for (let index = 1; index < frames.length; index += 1) {
    if (frames[index].time < frames[index - 1].time) {
      errors.push(`replay.frames[${index}].time must be greater than or equal to the previous frame time.`)
    }
  }

  const replay: ReplayArtifact = {
    schemaVersion: asLiteral(root?.schemaVersion, REPLAY_SCHEMA_VERSION, 'replay.schemaVersion', errors),
    kind: asLiteral(root?.kind, REPLAY_KIND, 'replay.kind', errors),
    replayId: asNonEmptyString(root?.replayId, 'replay.replayId', errors),
    createdAt: asNonEmptyString(root?.createdAt, 'replay.createdAt', errors),
    source: {
      creatureId: asNonEmptyString(source?.creatureId, 'replay.source.creatureId', errors),
      environmentId: asNonEmptyString(source?.environmentId, 'replay.source.environmentId', errors),
      policyBackend: asEnum(source?.policyBackend, policyBackends, 'replay.source.policyBackend', errors),
      modelVersion: asNonEmptyString(source?.modelVersion, 'replay.source.modelVersion', errors),
    },
    timestep: asPositiveNumber(root?.timestep, 'replay.timestep', errors),
    durationSeconds: asNonNegativeNumber(root?.durationSeconds, 'replay.durationSeconds', errors),
    frameCount,
    environmentSample: {
      seed: asFiniteNumber(environmentSample?.seed, 'replay.environmentSample.seed', errors),
      gravityScale: asFiniteNumber(environmentSample?.gravityScale, 'replay.environmentSample.gravityScale', errors),
      frictionScale: asFiniteNumber(environmentSample?.frictionScale, 'replay.environmentSample.frictionScale', errors),
      dragScale: asFiniteNumber(environmentSample?.dragScale, 'replay.environmentSample.dragScale', errors),
      slopeDegrees: asFiniteNumber(environmentSample?.slopeDegrees, 'replay.environmentSample.slopeDegrees', errors),
      roughness: asNonNegativeNumber(environmentSample?.roughness, 'replay.environmentSample.roughness', errors),
      obstacleDensity: asNonNegativeNumber(
        environmentSample?.obstacleDensity,
        'replay.environmentSample.obstacleDensity',
        errors,
      ),
      skateboardSpawn: asVec2(environmentSample?.skateboardSpawn, 'replay.environmentSample.skateboardSpawn', errors),
      skateboardMass: asPositiveNumber(environmentSample?.skateboardMass, 'replay.environmentSample.skateboardMass', errors),
      wheelFriction: asNonNegativeNumber(environmentSample?.wheelFriction, 'replay.environmentSample.wheelFriction', errors),
    },
    taskMetrics: {
      skateboardDiscoveredAt: asNullableNonNegativeNumber(
        taskMetrics?.skateboardDiscoveredAt,
        'replay.taskMetrics.skateboardDiscoveredAt',
        errors,
      ),
      firstContactAt: asNullableNonNegativeNumber(taskMetrics?.firstContactAt, 'replay.taskMetrics.firstContactAt', errors),
      mountTime: asNonNegativeNumber(taskMetrics?.mountTime, 'replay.taskMetrics.mountTime', errors),
      rollingDistance: asNonNegativeNumber(taskMetrics?.rollingDistance, 'replay.taskMetrics.rollingDistance', errors),
      maxRollingVelocity: asNonNegativeNumber(taskMetrics?.maxRollingVelocity, 'replay.taskMetrics.maxRollingVelocity', errors),
      averageContactRatio: asUnitNumber(taskMetrics?.averageContactRatio, 'replay.taskMetrics.averageContactRatio', errors),
      energy: asNonNegativeNumber(taskMetrics?.energy, 'replay.taskMetrics.energy', errors),
      survivalTime: asNonNegativeNumber(taskMetrics?.survivalTime, 'replay.taskMetrics.survivalTime', errors),
      fallReason: asEnum(taskMetrics?.fallReason, fallReasons, 'replay.taskMetrics.fallReason', errors),
    },
    frames,
  }

  return errors.length > 0 ? { ok: false, errors } : { ok: true, value: replay }
}

function parsePose(value: unknown, path: string, errors: string[]) {
  const record = asRecord(value, path, errors)
  return {
    position: asVec3(record?.position, `${path}.position`, errors),
    velocity: asVec3(record?.velocity, `${path}.velocity`, errors),
    rotation: asVec3(record?.rotation, `${path}.rotation`, errors),
  }
}

function emptyPose() {
  return {
    position: [0, 0, 0] as ReplayVec3,
    velocity: [0, 0, 0] as ReplayVec3,
    rotation: [0, 0, 0] as ReplayVec3,
  }
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

function asNonEmptyString(value: unknown, path: string, errors: string[]): string {
  if (typeof value !== 'string' || value.length === 0) {
    errors.push(`${path} must be a non-empty string.`)
    return ''
  }
  return value
}

function asBoolean(value: unknown, path: string, errors: string[]): boolean {
  if (typeof value !== 'boolean') {
    errors.push(`${path} must be a boolean.`)
    return false
  }
  return value
}

function asFiniteNumber(value: unknown, path: string, errors: string[]): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    errors.push(`${path} must be a finite number.`)
    return 0
  }
  return value
}

function asNonNegativeInteger(value: unknown, path: string, errors: string[]): number {
  const result = asFiniteNumber(value, path, errors)
  if (!Number.isInteger(result) || result < 0) {
    errors.push(`${path} must be a non-negative integer.`)
  }
  return result
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

function asNullableNonNegativeNumber(value: unknown, path: string, errors: string[]): number | null {
  if (value === null) {
    return null
  }
  return asNonNegativeNumber(value, path, errors)
}

function asUnitNumber(value: unknown, path: string, errors: string[]): number {
  const result = asFiniteNumber(value, path, errors)
  if (result < 0 || result > 1) {
    errors.push(`${path} must be between 0 and 1.`)
  }
  return result
}

function asVec3(value: unknown, path: string, errors: string[]): ReplayVec3 {
  const values = asTuple(value, path, 3, errors)
  return [values[0], values[1], values[2]]
}

function asVec2(value: unknown, path: string, errors: string[]): [number, number] {
  const values = asTuple(value, path, 2, errors)
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

function asLiteral<const T extends string | number>(value: unknown, expected: T, path: string, errors: string[]): T {
  if (value !== expected) {
    errors.push(`${path} must equal ${String(expected)}.`)
  }
  return expected
}
