import { spawnSync } from 'node:child_process'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import type { EnvironmentConfig } from '../src/creature/types'
import {
  createDomainLocks,
  createNominalDomainSample,
  createPresetDomainSample,
  DOMAIN_PARAMETER_KEYS,
  domainSampleFingerprint,
  materializeDomainEnvironment,
  nextDomainSeed,
  normalizeDomainSeed,
  sampleDomain,
  SEED_FORGE_PRESETS,
  validateDomainSample,
  validateDomainEnvironment,
  type DomainSample,
  type SeedForgePresetId,
} from '../src/environment/seedForge'

const root = resolve(import.meta.dirname, '..')
const environmentPaths = ['adaptive-skate-terrarium.json', 'ripple-yard.json', 'tilt-basin.json'].map(name =>
  resolve(root, 'public/configs/environments', name),
)
const environments = await Promise.all(
  environmentPaths.map(async path => JSON.parse(await readFile(path, 'utf8')) as EnvironmentConfig),
)
const environment = environments[0]
const seeds = [0, 1, 1337, 0x7fffffff, 0xffffffff]

expect(normalizeDomainSeed(-1) === 0xffffffff, 'negative seed normalization is not uint32')
expect(normalizeDomainSeed(0x100000001) === 1, 'overflow seed normalization is not uint32')
expect(normalizeDomainSeed(1.9) === 1, 'fractional seed normalization no longer truncates')
expectThrows(
  () => normalizeDomainSeed(Number.MAX_SAFE_INTEGER + 1),
  'unsafe seed was accepted before uint32 normalization',
)
expectThrows(() => normalizeDomainSeed(Number.POSITIVE_INFINITY), 'non-finite seed was accepted')
expect(nextDomainSeed(7) === 0x9e3779c0, 'deterministic reroll seed changed unexpectedly')

for (const candidateEnvironment of environments) {
  for (const seed of seeds) {
    const first = sampleDomain(candidateEnvironment, seed)
    const second = sampleDomain(candidateEnvironment, seed)
    expectDeepEqual(first, second, `${candidateEnvironment.id} seed ${seed} reproducibility`)
    expect(
      validateDomainSample(first, candidateEnvironment).length === 0,
      `${candidateEnvironment.id} seed ${seed} escaped its declared ranges`,
    )
  }
}

const firstSample = sampleDomain(environment, 1337)
const nextSample = sampleDomain(environment, nextDomainSeed(firstSample.seed))
expect(
  domainSampleFingerprint(firstSample) !== domainSampleFingerprint(nextSample),
  'different reroll seeds produced the same sample',
)

const locks = createDomainLocks()
locks.gravityScale = true
locks.sensorNoise = true
locks.skateboardMass = true
const lockedSample = sampleDomain(environment, 987654321, {
  locks,
  previous: firstSample,
})
const unlockedSample = sampleDomain(environment, 987654321)
for (const key of ['gravityScale', 'sensorNoise', 'skateboardMass'] as const) {
  expect(lockedSample[key] === firstSample[key], `${key} did not survive a locked reroll`)
}
expect(lockedSample.frictionScale !== firstSample.frictionScale, 'an unlocked parameter did not reroll')
for (const key of DOMAIN_PARAMETER_KEYS.filter(key => !locks[key])) {
  expect(lockedSample[key] === unlockedSample[key], `${key} changed because an earlier parameter was locked`)
}

const baseSnapshot = JSON.stringify(environment)
const nominal = createNominalDomainSample(environment)
const materialized = materializeDomainEnvironment(environment, nominal)
expect(JSON.stringify(environment) === baseSnapshot, 'materialization mutated the source environment')
expect(materialized !== environment, 'materialization returned the source environment')
expect(materialized.world !== environment.world, 'materialization shared mutable world state')
expect(materialized.terrain !== environment.terrain, 'materialization shared mutable terrain state')
expect(materialized.skateboard !== environment.skateboard, 'materialization shared mutable board state')
expectClose(
  materialized.world.gravity[1],
  environment.world.gravity[1] * nominal.gravityScale,
  'materialized gravity',
)
expectClose(
  materialized.terrain.baseFriction,
  environment.terrain.baseFriction * nominal.frictionScale,
  'materialized friction',
)
expectClose(materialized.world.airDrag, environment.world.airDrag * nominal.dragScale, 'materialized drag')
expect(materialized.seed === nominal.seed, 'materialized environment seed does not match its sample')
expect(
  materialized.skateboard.spawnPosition[0] === nominal.skateboardSpawnX &&
    materialized.skateboard.spawnPosition[2] === nominal.skateboardSpawnZ,
  'materialized skateboard spawn does not match its sample',
)

const presets = SEED_FORGE_PRESETS.map(preset =>
  createPresetDomainSample(environment, preset.id, environment.seed),
)
for (const [index, preset] of presets.entries()) {
  expect(
    validateDomainSample(preset, environment).length === 0,
    `${SEED_FORGE_PRESETS[index].label} preset escaped its declared ranges`,
  )
}
expect(
  new Set(presets.map(domainSampleFingerprint)).size === presets.length,
  'Seed Forge presets are not behaviorally distinct',
)
expect(
  presets[1].frictionScale === environment.randomization.frictionScale[0] &&
    presets[1].wheelFriction === environment.randomization.wheelFriction[0],
  'Ice rink preset is not maximally slick',
)
expect(
  presets[3].skateboardMass === environment.randomization.skateboardMass[1] &&
    presets[3].actuatorLatencyMs === environment.randomization.actuatorLatencyMs[1],
  'Cinderblock deck preset is not the intended heavy/high-latency challenge',
)
expectThrows(
  () => createPresetDomainSample(environment, 'unknown' as SeedForgePresetId),
  'unknown TypeScript preset was accepted',
)
const unsafeEnvironment = structuredClone(environment)
unsafeEnvironment.randomization.actuatorStrength = [-2, -1]
expect(
  validateDomainEnvironment(unsafeEnvironment).includes(
    'environment.randomization.actuatorStrength minimum must be greater than 0.',
  ),
  'unsafe actuator range did not fail the shared domain contract',
)
expectThrows(
  () => sampleDomain(unsafeEnvironment, 7),
  'sampling accepted an environment that cannot produce replay-safe actuator values',
)

const offLatticeEnvironment = structuredClone(environment)
offLatticeEnvironment.terrain.roughness = 0.12345649
offLatticeEnvironment.randomization.roughness = [0.12345649, 0.12345649]
expect(
  validateDomainEnvironment(offLatticeEnvironment).some(error =>
    error.includes('must use at most 6 decimal places'),
  ),
  'off-lattice ranges and authored values passed validation',
)
expectThrows(
  () => sampleDomain(offLatticeEnvironment, 7),
  'sampling accepted a domain that is not closed under quantization',
)

const overflowingEnvironment = structuredClone(environment)
overflowingEnvironment.randomization.skateboardSpawnX = [-Number.MAX_VALUE, Number.MAX_VALUE]
expect(
  validateDomainEnvironment(overflowingEnvironment).some(error =>
    error.includes('within the exact quantization range'),
  ),
  'overflow-prone ranges passed validation',
)
expectThrows(
  () => sampleDomain(overflowingEnvironment, 7),
  'sampling accepted an overflow-prone interpolation range',
)

const invalid = { ...firstSample, sensorNoise: Number.NaN } as DomainSample
expect(validateDomainSample(invalid, environment).length > 0, 'non-finite sample passed validation')
expectThrows(
  () => materializeDomainEnvironment(environment, invalid),
  'materialization accepted a non-finite sample',
)
const outOfRange = {
  ...firstSample,
  skateboardMass: environment.randomization.skateboardMass[1] + 1,
}
expect(validateDomainSample(outOfRange, environment).length > 0, 'out-of-range sample passed validation')
const offLatticeSample = { ...firstSample, roughness: 0.12345649 }
expect(
  validateDomainSample(offLatticeSample, environment).some(error =>
    error.includes('must use at most 6 decimal places'),
  ),
  'off-lattice sample passed validation',
)

const unsafeSeedEnvironment = structuredClone(environment)
unsafeSeedEnvironment.seed = Number.MAX_SAFE_INTEGER + 1
expect(
  validateDomainEnvironment(unsafeSeedEnvironment).includes(
    'environment.seed must be a safe integer before uint32 normalization.',
  ),
  'unsafe authored seed passed validation',
)

const parityPayload = {
  environment,
  seeds,
  presets: SEED_FORGE_PRESETS.map(preset => preset.id),
}
const python = spawnSync('uv', ['run', '--locked', 'python', '-m', 'wurmkickflip_rl.seed_forge'], {
  cwd: resolve(root, 'training'),
  encoding: 'utf8',
  input: JSON.stringify(parityPayload),
})
if (python.status !== 0) {
  console.error(python.stdout)
  console.error(python.stderr)
  process.exit(python.status ?? 1)
}
const pythonOutput = JSON.parse(python.stdout) as {
  samples: DomainSample[]
  presets: DomainSample[]
}
expectDeepEqual(
  seeds.map(seed => sampleDomain(environment, seed)),
  pythonOutput.samples,
  'TypeScript/Python seeded sample parity',
)
expectDeepEqual(presets, pythonOutput.presets, 'TypeScript/Python preset parity')

const adversarialEnvironments = [
  (() => {
    const fixed = structuredClone(environment)
    fixed.terrain.roughness = 0.123456
    fixed.randomization.roughness = [0.123456, 0.123456]
    return fixed
  })(),
  (() => {
    const wide = structuredClone(environment)
    wide.skateboard.spawnPosition[0] = 0
    wide.randomization.skateboardSpawnX = [-9_000_000_000, 9_000_000_000]
    return wide
  })(),
]
for (const [index, adversarialEnvironment] of adversarialEnvironments.entries()) {
  expect(
    validateDomainEnvironment(adversarialEnvironment).length === 0,
    `valid adversarial environment ${index} failed validation`,
  )
  const adversarialPython = spawnSync(
    'uv',
    ['run', '--locked', 'python', '-m', 'wurmkickflip_rl.seed_forge'],
    {
      cwd: resolve(root, 'training'),
      encoding: 'utf8',
      input: JSON.stringify({
        environment: adversarialEnvironment,
        seeds: [0, 1.9, -1, Number.MAX_SAFE_INTEGER],
        presets: [],
      }),
    },
  )
  if (adversarialPython.status !== 0) {
    console.error(adversarialPython.stdout)
    console.error(adversarialPython.stderr)
    process.exit(adversarialPython.status ?? 1)
  }
  const adversarialOutput = JSON.parse(adversarialPython.stdout) as { samples: DomainSample[] }
  expectDeepEqual(
    [0, 1.9, -1, Number.MAX_SAFE_INTEGER].map(seed => sampleDomain(adversarialEnvironment, seed)),
    adversarialOutput.samples,
    `TypeScript/Python adversarial sample parity ${index}`,
  )
  for (const sample of adversarialOutput.samples) {
    expect(
      validateDomainSample(sample, adversarialEnvironment).length === 0,
      `adversarial sample ${index} escaped its validated environment`,
    )
  }
}

const invalidPythonPreset = spawnSync(
  'uv',
  ['run', '--locked', 'python', '-m', 'wurmkickflip_rl.seed_forge'],
  {
    cwd: resolve(root, 'training'),
    encoding: 'utf8',
    input: JSON.stringify({ environment, seeds: [], presets: ['unknown'] }),
  },
)
expect(invalidPythonPreset.status !== 0, 'unknown Python preset was accepted')

const stableProbe = sampleDomain(environment, 1337)
expect(
  domainSampleFingerprint(stableProbe) ===
    '1337:0.836179:0.630490:2.026180:2.299981:0.180925:0.083830:1.129219:43.886910:0.043520:2.181177:1.805952:-0.680474:1.159616:0.766109',
  'Seed Forge algorithm changed its canonical seed-1337 vector',
)

console.log(
  JSON.stringify({
    environments: environments.map(candidate => candidate.id),
    seeds,
    canonicalFingerprint: domainSampleFingerprint(stableProbe),
    presets: SEED_FORGE_PRESETS.map(preset => preset.id),
  }),
)
console.log('Seed Forge deterministic contract and Python parity verification passed.')

function expect(condition: boolean, message: string): asserts condition {
  if (!condition) throw new Error(message)
}

function expectClose(actual: number, expected: number, label: string) {
  expect(Math.abs(actual - expected) < 1e-6, `${label}: expected ${expected}, received ${actual}`)
}

function expectDeepEqual(actual: unknown, expected: unknown, label: string) {
  const actualJson = JSON.stringify(actual)
  const expectedJson = JSON.stringify(expected)
  expect(actualJson === expectedJson, `${label} mismatch.\nTS: ${actualJson}\nPY: ${expectedJson}`)
}

function expectThrows(run: () => unknown, message: string) {
  try {
    run()
  } catch {
    return
  }
  throw new Error(message)
}
