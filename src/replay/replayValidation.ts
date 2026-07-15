import {
  REPLAY_CHANNELS_PER_SEGMENT,
  REPLAY_CORE_VERSION,
  REPLAY_INTEGRITY_ALGORITHM,
  REPLAY_INTERPOLATION,
  REPLAY_KIND,
  REPLAY_MUSCLE_CHANNEL_COUNT,
  REPLAY_SCHEMA_VERSION,
  REPLAY_SEGMENT_COUNT,
  type ReplayArtifact,
  type ReplayFallReason,
  type ReplayIntegrity,
  type ReplayPlaybackContract,
  type ReplayPolicyBackend,
  type ReplayVec3,
} from './types'
import { replayIntegrityMatches } from './replayIntegrity'

export type ReplayValidationResult = { ok: true; value: ReplayArtifact } | { ok: false; errors: string[] }

const policyBackends = [
  'scripted',
  'neural-js',
  'onnx-webgpu',
  'onnx-wasm',
  'training',
] as const satisfies readonly ReplayPolicyBackend[]
const fallReasons = [
  'none',
  'lost_contact',
  'fell_below_height',
  'out_of_bounds',
  'timeout',
] as const satisfies readonly ReplayFallReason[]

export function validateReplayArtifact(value: unknown): ReplayValidationResult {
  const errors: string[] = []
  const root = asRecord(value, 'replay', errors)
  const source = root ? asRecord(root.source, 'replay.source', errors) : null
  const environmentSample = root ? asRecord(root.environmentSample, 'replay.environmentSample', errors) : null
  const taskMetrics = root ? asRecord(root.taskMetrics, 'replay.taskMetrics', errors) : null
  const framesInput = root ? asArray(root.frames, 'replay.frames', errors) : []
  const hasPlayback = Boolean(root && Object.hasOwn(root, 'playback'))
  const hasIntegrity = Boolean(root && Object.hasOwn(root, 'integrity'))
  const playbackInput = hasPlayback ? asRecord(root?.playback, 'replay.playback', errors) : null
  const integrityInput = hasIntegrity ? asRecord(root?.integrity, 'replay.integrity', errors) : null

  if (hasPlayback !== hasIntegrity) {
    errors.push('replay.playback and replay.integrity must either both be present or both be absent.')
  }

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
      skateboardDiscovered: asBoolean(
        record?.skateboardDiscovered,
        `replay.frames[${index}].skateboardDiscovered`,
        errors,
      ),
      rollingVelocity: asFiniteNumber(
        record?.rollingVelocity,
        `replay.frames[${index}].rollingVelocity`,
        errors,
      ),
      reward: asFiniteNumber(record?.reward, `replay.frames[${index}].reward`, errors),
      ...(record && Object.hasOwn(record, 'muscleActivations')
        ? {
            muscleActivations: asMuscleActivations(
              record.muscleActivations,
              `replay.frames[${index}].muscleActivations`,
              errors,
            ),
          }
        : {}),
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

  const playback = playbackInput ? parsePlaybackContract(playbackInput, errors) : undefined
  const integrity = integrityInput ? parseIntegrity(integrityInput, errors) : undefined

  const replay: ReplayArtifact = {
    schemaVersion: asLiteral(root?.schemaVersion, REPLAY_SCHEMA_VERSION, 'replay.schemaVersion', errors),
    kind: asLiteral(root?.kind, REPLAY_KIND, 'replay.kind', errors),
    replayId: asNonEmptyString(root?.replayId, 'replay.replayId', errors),
    createdAt: asIsoTimestamp(root?.createdAt, 'replay.createdAt', errors),
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
      gravityScale: asFiniteNumber(
        environmentSample?.gravityScale,
        'replay.environmentSample.gravityScale',
        errors,
      ),
      frictionScale: asFiniteNumber(
        environmentSample?.frictionScale,
        'replay.environmentSample.frictionScale',
        errors,
      ),
      dragScale: asFiniteNumber(environmentSample?.dragScale, 'replay.environmentSample.dragScale', errors),
      slopeDegrees: asFiniteNumber(
        environmentSample?.slopeDegrees,
        'replay.environmentSample.slopeDegrees',
        errors,
      ),
      roughness: asNonNegativeNumber(
        environmentSample?.roughness,
        'replay.environmentSample.roughness',
        errors,
      ),
      obstacleDensity: asNonNegativeNumber(
        environmentSample?.obstacleDensity,
        'replay.environmentSample.obstacleDensity',
        errors,
      ),
      skateboardSpawn: asVec2(
        environmentSample?.skateboardSpawn,
        'replay.environmentSample.skateboardSpawn',
        errors,
      ),
      skateboardMass: asPositiveNumber(
        environmentSample?.skateboardMass,
        'replay.environmentSample.skateboardMass',
        errors,
      ),
      wheelFriction: asNonNegativeNumber(
        environmentSample?.wheelFriction,
        'replay.environmentSample.wheelFriction',
        errors,
      ),
    },
    taskMetrics: {
      skateboardDiscoveredAt: asNullableNonNegativeNumber(
        taskMetrics?.skateboardDiscoveredAt,
        'replay.taskMetrics.skateboardDiscoveredAt',
        errors,
      ),
      firstContactAt: asNullableNonNegativeNumber(
        taskMetrics?.firstContactAt,
        'replay.taskMetrics.firstContactAt',
        errors,
      ),
      mountTime: asNonNegativeNumber(taskMetrics?.mountTime, 'replay.taskMetrics.mountTime', errors),
      rollingDistance: asNonNegativeNumber(
        taskMetrics?.rollingDistance,
        'replay.taskMetrics.rollingDistance',
        errors,
      ),
      maxRollingVelocity: asNonNegativeNumber(
        taskMetrics?.maxRollingVelocity,
        'replay.taskMetrics.maxRollingVelocity',
        errors,
      ),
      averageContactRatio: asUnitNumber(
        taskMetrics?.averageContactRatio,
        'replay.taskMetrics.averageContactRatio',
        errors,
      ),
      energy: asNonNegativeNumber(taskMetrics?.energy, 'replay.taskMetrics.energy', errors),
      survivalTime: asNonNegativeNumber(taskMetrics?.survivalTime, 'replay.taskMetrics.survivalTime', errors),
      fallReason: asEnum(taskMetrics?.fallReason, fallReasons, 'replay.taskMetrics.fallReason', errors),
    },
    frames,
    ...(playback ? { playback } : {}),
    ...(integrity ? { integrity } : {}),
  }

  if (playback) validateRecorderCoreContract(replay, errors)
  if (integrity) {
    try {
      if (!replayIntegrityMatches(value, integrity)) {
        errors.push('replay.integrity.digest does not match the replay payload.')
      }
    } catch (error) {
      errors.push(
        error instanceof Error
          ? `replay.integrity could not be computed: ${error.message}`
          : 'replay.integrity could not be computed.',
      )
    }
  }

  return errors.length > 0 ? { ok: false, errors } : { ok: true, value: replay }
}

function parsePlaybackContract(value: Record<string, unknown>, errors: string[]): ReplayPlaybackContract {
  const actuatorLayout = asRecord(value.actuatorLayout, 'replay.playback.actuatorLayout', errors)
  const channelOrder = asStringTuple(
    actuatorLayout?.channelOrder,
    ['dorsal', 'ventral'],
    'replay.playback.actuatorLayout.channelOrder',
    errors,
  )
  return {
    coreVersion: asLiteral(value.coreVersion, REPLAY_CORE_VERSION, 'replay.playback.coreVersion', errors),
    interpolation: asLiteral(
      value.interpolation,
      REPLAY_INTERPOLATION,
      'replay.playback.interpolation',
      errors,
    ),
    actuatorLayout: {
      segmentCount: asLiteral(
        actuatorLayout?.segmentCount,
        REPLAY_SEGMENT_COUNT,
        'replay.playback.actuatorLayout.segmentCount',
        errors,
      ),
      channelsPerSegment: asLiteral(
        actuatorLayout?.channelsPerSegment,
        REPLAY_CHANNELS_PER_SEGMENT,
        'replay.playback.actuatorLayout.channelsPerSegment',
        errors,
      ),
      muscleChannelCount: asLiteral(
        actuatorLayout?.muscleChannelCount,
        REPLAY_MUSCLE_CHANNEL_COUNT,
        'replay.playback.actuatorLayout.muscleChannelCount',
        errors,
      ),
      channelOrder,
    },
  }
}

function parseIntegrity(value: Record<string, unknown>, errors: string[]): ReplayIntegrity {
  const digest = asNonEmptyString(value.digest, 'replay.integrity.digest', errors)
  if (!/^[0-9a-f]{16}$/.test(digest)) {
    errors.push('replay.integrity.digest must be 16 lowercase hexadecimal characters.')
  }
  return {
    algorithm: asLiteral(value.algorithm, REPLAY_INTEGRITY_ALGORITHM, 'replay.integrity.algorithm', errors),
    digest,
  }
}

function validateRecorderCoreContract(replay: ReplayArtifact, errors: string[]) {
  if (replay.frames.length === 0) {
    errors.push('replay recorder-core artifacts must contain at least one frame.')
    return
  }

  const tolerance = Math.max(1e-9, replay.timestep * 1e-6)
  if (Math.abs(replay.frames[0].time) > tolerance) {
    errors.push('replay recorder-core artifacts must start at time 0.')
  }

  for (let index = 0; index < replay.frames.length; index += 1) {
    const frame = replay.frames[index]
    if (!frame.muscleActivations) {
      errors.push(`replay.frames[${index}].muscleActivations is required by the recorder core.`)
    }
    if (frame.rollingVelocity < 0) {
      errors.push(`replay.frames[${index}].rollingVelocity must be at least 0.`)
    }
    const gridTime = Math.round(frame.time / replay.timestep) * replay.timestep
    if (Math.abs(frame.time - gridTime) > tolerance) {
      errors.push(`replay.frames[${index}].time must lie on the replay timestep grid.`)
    }
    if (index > 0 && frame.time - replay.frames[index - 1].time <= tolerance) {
      errors.push(
        `replay.frames[${index}].time must be strictly greater than the previous recorder-core frame time.`,
      )
    }
  }

  const expectedDuration = replay.frames.at(-1)!.time + replay.timestep
  if (Math.abs(replay.durationSeconds - expectedDuration) > tolerance) {
    errors.push(
      `replay.durationSeconds must equal the final frame time plus one timestep (${expectedDuration}).`,
    )
  }

  validateMetricTime(
    replay.taskMetrics.skateboardDiscoveredAt,
    'skateboardDiscoveredAt',
    replay.durationSeconds,
    tolerance,
    errors,
  )
  validateMetricTime(
    replay.taskMetrics.firstContactAt,
    'firstContactAt',
    replay.durationSeconds,
    tolerance,
    errors,
  )
  validateMetricTime(replay.taskMetrics.mountTime, 'mountTime', replay.durationSeconds, tolerance, errors)
  validateMetricTime(
    replay.taskMetrics.survivalTime,
    'survivalTime',
    replay.durationSeconds,
    tolerance,
    errors,
  )

  const { skateboardDiscoveredAt, firstContactAt, maxRollingVelocity } = replay.taskMetrics
  if (
    skateboardDiscoveredAt !== null &&
    firstContactAt !== null &&
    skateboardDiscoveredAt - firstContactAt > tolerance
  ) {
    errors.push('replay.taskMetrics.skateboardDiscoveredAt must not be after firstContactAt.')
  }
  const observedMaximum = replay.frames.reduce(
    (maximum, frame) => Math.max(maximum, frame.rollingVelocity),
    0,
  )
  if (maxRollingVelocity + tolerance < observedMaximum) {
    errors.push('replay.taskMetrics.maxRollingVelocity must cover every recorded frame.')
  }
}

function validateMetricTime(
  value: number | null,
  label: string,
  duration: number,
  tolerance: number,
  errors: string[],
) {
  if (value !== null && value - duration > tolerance) {
    errors.push(`replay.taskMetrics.${label} must not exceed replay.durationSeconds.`)
  }
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

function asIsoTimestamp(value: unknown, path: string, errors: string[]): string {
  const result = asNonEmptyString(value, path, errors)
  const timestamp = Date.parse(result)
  if (!Number.isFinite(timestamp) || new Date(timestamp).toISOString() !== result) {
    errors.push(`${path} must be a canonical ISO-8601 timestamp.`)
  }
  return result
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

function asMuscleActivations(value: unknown, path: string, errors: string[]): number[] {
  const values = asTuple(value, path, REPLAY_MUSCLE_CHANNEL_COUNT, errors)
  for (let index = 0; index < values.length; index += 1) {
    if (values[index] < -1 || values[index] > 1) {
      errors.push(`${path}[${index}] must be between -1 and 1.`)
    }
  }
  return values
}

function asStringTuple<const T extends readonly string[]>(
  value: unknown,
  expected: T,
  path: string,
  errors: string[],
): [...T] {
  if (!Array.isArray(value) || value.length !== expected.length) {
    errors.push(`${path} must equal [${expected.join(', ')}].`)
    return [...expected]
  }
  for (let index = 0; index < expected.length; index += 1) {
    if (value[index] !== expected[index]) {
      errors.push(`${path}[${index}] must equal ${expected[index]}.`)
    }
  }
  return [...expected]
}

function asTuple(value: unknown, path: string, length: number, errors: string[]): number[] {
  if (!Array.isArray(value) || value.length !== length) {
    errors.push(`${path} must be an array of ${length} finite numbers.`)
    return new Array<number>(length).fill(0)
  }
  return value.map((item, index) => asFiniteNumber(item, `${path}[${index}]`, errors))
}

function asEnum<const T extends readonly string[]>(
  value: unknown,
  allowed: T,
  path: string,
  errors: string[],
): T[number] {
  if (typeof value === 'string' && allowed.includes(value)) {
    return value
  }
  errors.push(`${path} must be one of: ${allowed.join(', ')}.`)
  return allowed[0]
}

function asLiteral<const T extends string | number>(
  value: unknown,
  expected: T,
  path: string,
  errors: string[],
): T {
  if (value !== expected) {
    errors.push(`${path} must equal ${String(expected)}.`)
  }
  return expected
}
