import type { EnvironmentConfig, Range } from '../creature/types'

export const DOMAIN_SAMPLE_SCHEMA_VERSION = 1 as const
export const DOMAIN_SAMPLE_KIND = 'wurmkickflip.domainSample' as const
const DOMAIN_QUANTIZATION_SCALE = 1_000_000

export const DOMAIN_PARAMETER_KEYS = [
  'gravityScale',
  'frictionScale',
  'dragScale',
  'slopeDegrees',
  'roughness',
  'obstacleDensity',
  'actuatorStrength',
  'actuatorLatencyMs',
  'sensorNoise',
  'spawnYawDegrees',
  'skateboardSpawnX',
  'skateboardSpawnZ',
  'skateboardMass',
  'wheelFriction',
] as const

export type DomainParameterKey = (typeof DOMAIN_PARAMETER_KEYS)[number]

export type DomainSample = {
  schemaVersion: typeof DOMAIN_SAMPLE_SCHEMA_VERSION
  kind: typeof DOMAIN_SAMPLE_KIND
  seed: number
} & Record<DomainParameterKey, number>

export type DomainLocks = Record<DomainParameterKey, boolean>

export type DomainParameterSpec = {
  key: DomainParameterKey
  label: string
  shortLabel: string
  unit: string
  decimals: number
  displayScale?: number
}

export const SEED_FORGE_PARAMETER_SPECS: readonly DomainParameterSpec[] = [
  { key: 'gravityScale', label: 'Gravity scale', shortLabel: 'Gravity', unit: '×', decimals: 2 },
  { key: 'frictionScale', label: 'Ground friction scale', shortLabel: 'Ground grip', unit: '×', decimals: 2 },
  { key: 'dragScale', label: 'Air drag scale', shortLabel: 'Air drag', unit: '×', decimals: 2 },
  { key: 'slopeDegrees', label: 'Terrain slope', shortLabel: 'Slope', unit: '°', decimals: 1 },
  { key: 'roughness', label: 'Terrain roughness', shortLabel: 'Roughness', unit: '', decimals: 2 },
  {
    key: 'obstacleDensity',
    label: 'Obstacle density',
    shortLabel: 'Obstacles',
    unit: '%',
    decimals: 0,
    displayScale: 100,
  },
  { key: 'actuatorStrength', label: 'Actuator strength', shortLabel: 'Actuators', unit: '×', decimals: 2 },
  { key: 'actuatorLatencyMs', label: 'Actuator latency', shortLabel: 'Latency', unit: ' ms', decimals: 0 },
  {
    key: 'sensorNoise',
    label: 'Sensor noise',
    shortLabel: 'Noise',
    unit: '%',
    decimals: 1,
    displayScale: 100,
  },
  { key: 'spawnYawDegrees', label: 'Spawn yaw', shortLabel: 'Spawn yaw', unit: '°', decimals: 1 },
  { key: 'skateboardSpawnX', label: 'Skateboard spawn X', shortLabel: 'Board X', unit: ' m', decimals: 2 },
  { key: 'skateboardSpawnZ', label: 'Skateboard spawn Z', shortLabel: 'Board Z', unit: ' m', decimals: 2 },
  { key: 'skateboardMass', label: 'Skateboard mass', shortLabel: 'Board mass', unit: ' kg', decimals: 2 },
  { key: 'wheelFriction', label: 'Wheel friction', shortLabel: 'Wheel grip', unit: '', decimals: 2 },
] as const

export const SEED_FORGE_PRESETS = [
  {
    id: 'nominal',
    label: 'Nominal',
    description: 'Authored environment values with no sensor or actuator perturbation.',
  },
  {
    id: 'ice-rink',
    label: 'Ice rink',
    description: 'Minimum substrate and wheel grip with a light, clean course.',
  },
  {
    id: 'moon-dirt',
    label: 'Moon dirt',
    description: 'Low gravity, dusty relief, sparse sensing error, and a light deck.',
  },
  {
    id: 'cinderblock-deck',
    label: 'Cinderblock deck',
    description: 'A heavy board in the roughest, busiest, highest-latency domain.',
  },
] as const

export type SeedForgePresetId = (typeof SEED_FORGE_PRESETS)[number]['id']

export type DomainSampleOptions = {
  locks?: Partial<DomainLocks>
  previous?: DomainSample | null
}

type DomainConstraint = {
  minimum?: number
  maximum?: number
  exclusiveMinimum?: boolean
}

const DOMAIN_CONSTRAINTS: Partial<Record<DomainParameterKey, DomainConstraint>> = {
  gravityScale: { minimum: 0 },
  frictionScale: { minimum: 0 },
  dragScale: { minimum: 0 },
  roughness: { minimum: 0 },
  obstacleDensity: { minimum: 0, maximum: 1 },
  actuatorStrength: { minimum: 0, exclusiveMinimum: true },
  actuatorLatencyMs: { minimum: 0 },
  sensorNoise: { minimum: 0, maximum: 1 },
  spawnYawDegrees: { minimum: -180, maximum: 180 },
  skateboardMass: { minimum: 0, exclusiveMinimum: true },
  wheelFriction: { minimum: 0 },
}

export function normalizeDomainSeed(value: number) {
  if (!Number.isFinite(value)) {
    throw new RangeError('Domain seeds must be finite before uint32 normalization.')
  }
  const truncated = Math.trunc(value)
  if (!Number.isSafeInteger(truncated)) {
    throw new RangeError('Truncated domain seeds must be safe integers before uint32 normalization.')
  }
  return truncated >>> 0
}

export function nextDomainSeed(seed: number) {
  return (normalizeDomainSeed(seed) + 0x9e3779b9) >>> 0
}

export function createDomainLocks(locked = false): DomainLocks {
  return Object.fromEntries(DOMAIN_PARAMETER_KEYS.map(key => [key, locked])) as DomainLocks
}

export function sampleDomain(
  environment: EnvironmentConfig,
  seed: number,
  options: DomainSampleOptions = {},
): DomainSample {
  assertValidDomainEnvironment(environment)
  const normalizedSeed = normalizeDomainSeed(seed)
  const random = mulberry32(normalizedSeed)
  const values = {} as Record<DomainParameterKey, number>

  for (const key of DOMAIN_PARAMETER_KEYS) {
    const range = domainRangeFor(environment, key)
    const previous = options.previous?.[key]
    const sampled = quantizeToRange(lerp(range[0], range[1], random()), range[0], range[1])
    values[key] =
      options.locks?.[key] && typeof previous === 'number' && Number.isFinite(previous)
        ? quantizeToRange(previous, range[0], range[1])
        : sampled
  }

  return {
    schemaVersion: DOMAIN_SAMPLE_SCHEMA_VERSION,
    kind: DOMAIN_SAMPLE_KIND,
    seed: normalizedSeed,
    ...values,
  }
}

export function createNominalDomainSample(
  environment: EnvironmentConfig,
  seed = environment.seed,
): DomainSample {
  assertValidDomainEnvironment(environment)
  return makeDomainSample(environment, seed, nominalDomainValues(environment))
}

export function createPresetDomainSample(
  environment: EnvironmentConfig,
  presetId: SeedForgePresetId,
  seed = environment.seed,
): DomainSample {
  const nominal = createNominalDomainSample(environment, seed)
  if (presetId === 'nominal') return nominal

  const at = (key: DomainParameterKey, amount: number) => {
    const range = domainRangeFor(environment, key)
    return lerp(range[0], range[1], clamp(amount, 0, 1))
  }
  const values: Record<DomainParameterKey, number> = { ...pickDomainValues(nominal) }

  if (presetId === 'ice-rink') {
    Object.assign(values, {
      frictionScale: at('frictionScale', 0),
      dragScale: at('dragScale', 0.15),
      slopeDegrees: clamp(0, ...domainRangeFor(environment, 'slopeDegrees')),
      roughness: at('roughness', 0),
      obstacleDensity: at('obstacleDensity', 0),
      actuatorStrength: at('actuatorStrength', 0.42),
      actuatorLatencyMs: at('actuatorLatencyMs', 0),
      sensorNoise: at('sensorNoise', 0),
      skateboardMass: at('skateboardMass', 0.18),
      wheelFriction: at('wheelFriction', 0),
    })
  } else if (presetId === 'moon-dirt') {
    Object.assign(values, {
      gravityScale: at('gravityScale', 0),
      frictionScale: at('frictionScale', 0.45),
      dragScale: at('dragScale', 0),
      slopeDegrees: at('slopeDegrees', 0.58),
      roughness: at('roughness', 0.88),
      obstacleDensity: at('obstacleDensity', 0.52),
      actuatorStrength: at('actuatorStrength', 0.68),
      actuatorLatencyMs: at('actuatorLatencyMs', 0.16),
      sensorNoise: at('sensorNoise', 0.28),
      spawnYawDegrees: at('spawnYawDegrees', 0.32),
      skateboardMass: at('skateboardMass', 0),
      wheelFriction: at('wheelFriction', 0.55),
    })
  } else if (presetId === 'cinderblock-deck') {
    Object.assign(values, {
      gravityScale: at('gravityScale', 1),
      frictionScale: at('frictionScale', 1),
      dragScale: at('dragScale', 1),
      slopeDegrees: at('slopeDegrees', 1),
      roughness: at('roughness', 1),
      obstacleDensity: at('obstacleDensity', 1),
      actuatorStrength: at('actuatorStrength', 1),
      actuatorLatencyMs: at('actuatorLatencyMs', 1),
      sensorNoise: at('sensorNoise', 1),
      spawnYawDegrees: at('spawnYawDegrees', 1),
      skateboardSpawnX: at('skateboardSpawnX', 1),
      skateboardSpawnZ: at('skateboardSpawnZ', 1),
      skateboardMass: at('skateboardMass', 1),
      wheelFriction: at('wheelFriction', 1),
    })
  } else {
    throw new Error(`Unknown Seed Forge preset: ${String(presetId)}`)
  }

  return makeDomainSample(environment, seed, values)
}

export function materializeDomainEnvironment(
  environment: EnvironmentConfig,
  sample: DomainSample,
): EnvironmentConfig {
  const errors = validateDomainSample(sample, environment)
  if (errors.length > 0) throw new Error(`Invalid domain sample: ${errors.join(' ')}`)

  return {
    ...environment,
    seed: sample.seed,
    world: {
      ...environment.world,
      gravity: environment.world.gravity.map(value => quantize(value * sample.gravityScale)) as [
        number,
        number,
        number,
      ],
      airDrag: quantize(environment.world.airDrag * sample.dragScale),
    },
    terrain: {
      ...environment.terrain,
      baseFriction: quantize(environment.terrain.baseFriction * sample.frictionScale),
      slopeDegrees: sample.slopeDegrees,
      roughness: sample.roughness,
      obstacleDensity: sample.obstacleDensity,
    },
    skateboard: {
      ...environment.skateboard,
      spawnPosition: [
        sample.skateboardSpawnX,
        environment.skateboard.spawnPosition[1],
        sample.skateboardSpawnZ,
      ],
      mass: sample.skateboardMass,
      wheelFriction: sample.wheelFriction,
    },
  }
}

export function validateDomainSample(sample: DomainSample, environment: EnvironmentConfig) {
  const errors = validateDomainEnvironment(environment)
  if (sample.schemaVersion !== DOMAIN_SAMPLE_SCHEMA_VERSION) {
    errors.push(`schemaVersion must be ${DOMAIN_SAMPLE_SCHEMA_VERSION}.`)
  }
  if (sample.kind !== DOMAIN_SAMPLE_KIND) errors.push(`kind must be ${DOMAIN_SAMPLE_KIND}.`)
  if (!Number.isInteger(sample.seed) || sample.seed < 0 || sample.seed > 0xffffffff) {
    errors.push('seed must be an unsigned 32-bit integer.')
  }
  for (const key of DOMAIN_PARAMETER_KEYS) {
    const value = sample[key]
    const [minimum, maximum] = domainRangeFor(environment, key)
    if (!Number.isFinite(value)) {
      errors.push(`${key} must be finite.`)
    } else {
      if (!isQuantizedDomainNumber(value)) {
        errors.push(`${key} must use at most 6 decimal places within the exact quantization range.`)
      }
      if (value < minimum || value > maximum) {
        errors.push(`${key} must be between ${minimum} and ${maximum}.`)
        continue
      }
      const constraint = DOMAIN_CONSTRAINTS[key]
      if (
        constraint?.minimum !== undefined &&
        (constraint.exclusiveMinimum ? value <= constraint.minimum : value < constraint.minimum)
      ) {
        errors.push(
          constraint.exclusiveMinimum
            ? `${key} must be greater than ${constraint.minimum}.`
            : `${key} must be at least ${constraint.minimum}.`,
        )
      }
      if (constraint?.maximum !== undefined && value > constraint.maximum) {
        errors.push(`${key} must be at most ${constraint.maximum}.`)
      }
    }
  }
  return errors
}

export function validateDomainEnvironment(environment: EnvironmentConfig) {
  const errors: string[] = []
  if (!Number.isSafeInteger(environment.seed)) {
    errors.push('environment.seed must be a safe integer before uint32 normalization.')
  }
  for (const key of DOMAIN_PARAMETER_KEYS) {
    const path = `environment.randomization.${key}`
    const range = environment.randomization[key]
    if (
      !Array.isArray(range) ||
      range.length !== 2 ||
      !Number.isFinite(range[0]) ||
      !Number.isFinite(range[1])
    ) {
      errors.push(`${path} must be an array of 2 finite numbers.`)
      continue
    }
    for (const [index, endpoint] of range.entries()) {
      if (!isQuantizedDomainNumber(endpoint)) {
        errors.push(
          `${path}[${index}] must use at most 6 decimal places within the exact quantization range.`,
        )
      }
    }
    if (range[0] > range[1]) errors.push(`${path} minimum must be less than or equal to maximum.`)
    const constraint = DOMAIN_CONSTRAINTS[key]
    if (
      constraint?.minimum !== undefined &&
      (constraint.exclusiveMinimum ? range[0] <= constraint.minimum : range[0] < constraint.minimum)
    ) {
      errors.push(
        constraint.exclusiveMinimum
          ? `${path} minimum must be greater than ${constraint.minimum}.`
          : `${path} minimum must be at least ${constraint.minimum}.`,
      )
    }
    if (constraint?.maximum !== undefined && range[1] > constraint.maximum) {
      errors.push(`${path} maximum must be at most ${constraint.maximum}.`)
    }
  }

  let nominal: Record<DomainParameterKey, number>
  try {
    nominal = nominalDomainValues(environment)
  } catch {
    errors.push('environment authored nominal values must be available.')
    return errors
  }
  for (const key of DOMAIN_PARAMETER_KEYS) {
    const range = environment.randomization[key]
    if (
      !Array.isArray(range) ||
      range.length !== 2 ||
      !Number.isFinite(range[0]) ||
      !Number.isFinite(range[1])
    ) {
      continue
    }
    if (!Number.isFinite(nominal[key])) {
      errors.push(`environment authored nominal value for ${key} must be finite.`)
      continue
    }
    if (!isQuantizedDomainNumber(nominal[key])) {
      errors.push(
        `environment authored nominal value for ${key} must use at most 6 decimal places within the exact quantization range.`,
      )
    }
    const [minimum, maximum] = range
    if (nominal[key] < minimum || nominal[key] > maximum) {
      errors.push(`environment.randomization.${key} must contain its authored nominal value ${nominal[key]}.`)
    }
  }
  return errors
}

export function domainRangeFor(environment: EnvironmentConfig, key: DomainParameterKey): Range {
  return environment.randomization[key]
}

export function domainSampleFingerprint(sample: DomainSample) {
  return [sample.seed, ...DOMAIN_PARAMETER_KEYS.map(key => sample[key].toFixed(6))].join(':')
}

function makeDomainSample(
  environment: EnvironmentConfig,
  seed: number,
  values: Record<DomainParameterKey, number>,
): DomainSample {
  const clamped = {} as Record<DomainParameterKey, number>
  for (const key of DOMAIN_PARAMETER_KEYS) {
    const [minimum, maximum] = domainRangeFor(environment, key)
    clamped[key] = quantizeToRange(values[key], minimum, maximum)
  }
  return {
    schemaVersion: DOMAIN_SAMPLE_SCHEMA_VERSION,
    kind: DOMAIN_SAMPLE_KIND,
    seed: normalizeDomainSeed(seed),
    ...clamped,
  }
}

function nominalDomainValues(environment: EnvironmentConfig): Record<DomainParameterKey, number> {
  return {
    gravityScale: 1,
    frictionScale: 1,
    dragScale: 1,
    slopeDegrees: environment.terrain.slopeDegrees,
    roughness: environment.terrain.roughness,
    obstacleDensity: environment.terrain.obstacleDensity,
    actuatorStrength: 1,
    actuatorLatencyMs: 0,
    sensorNoise: 0,
    spawnYawDegrees: 0,
    skateboardSpawnX: environment.skateboard.spawnPosition[0],
    skateboardSpawnZ: environment.skateboard.spawnPosition[2],
    skateboardMass: environment.skateboard.mass,
    wheelFriction: environment.skateboard.wheelFriction,
  }
}

function assertValidDomainEnvironment(environment: EnvironmentConfig) {
  const errors = validateDomainEnvironment(environment)
  if (errors.length > 0) throw new Error(`Invalid domain environment: ${errors.join(' ')}`)
}

function pickDomainValues(sample: DomainSample): Record<DomainParameterKey, number> {
  return Object.fromEntries(DOMAIN_PARAMETER_KEYS.map(key => [key, sample[key]])) as Record<
    DomainParameterKey,
    number
  >
}

function mulberry32(seed: number) {
  let state = normalizeDomainSeed(seed)
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let value = state
    value = Math.imul(value ^ (value >>> 15), value | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
    return ((value ^ (value >>> 14)) >>> 0) / 0x100000000
  }
}

function quantize(value: number) {
  return Math.floor(value * DOMAIN_QUANTIZATION_SCALE + 0.5) / DOMAIN_QUANTIZATION_SCALE
}

function lerp(minimum: number, maximum: number, amount: number) {
  if (amount <= 0) return minimum
  if (amount >= 1) return maximum
  return minimum * (1 - amount) + maximum * amount
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value))
}

function quantizeToRange(value: number, minimum: number, maximum: number) {
  return clamp(quantize(value), minimum, maximum)
}

function isQuantizedDomainNumber(value: number) {
  const scaled = value * DOMAIN_QUANTIZATION_SCALE
  return Number.isFinite(scaled) && Math.abs(scaled) <= Number.MAX_SAFE_INTEGER && quantize(value) === value
}
