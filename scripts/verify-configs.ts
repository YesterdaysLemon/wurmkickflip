import { readdir, readFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  type ConfigValidationResult,
  validateCreatureGenome,
  validateEnvironmentConfig,
  validateRuntimeProfile,
} from '../src/creature/configValidation'
import {
  loadLabConfigResources,
  type LabConfigFetchResult,
  type LabConfigFetcher,
  type LabConfigPaths,
} from '../src/creature/labConfigLoader'
import {
  createCreatureRuntimeAdapter,
  FIXED_ARTICULATED_RUNTIME_PROFILE,
} from '../src/creature/runtimeProfile'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const creatureDir = resolve(root, 'public/configs/creatures')
const environmentDir = resolve(root, 'public/configs/environments')

const creatureFiles = await jsonFiles(creatureDir)
const environmentFiles = await jsonFiles(environmentDir)
const creatures = await Promise.all(creatureFiles.map(file => readJson(resolve(creatureDir, file))))
const environments = await Promise.all(environmentFiles.map(file => readJson(resolve(environmentDir, file))))

const creatureValidations = creatures.map(validateCreatureGenome)
const environmentValidations = environments.map(validateEnvironmentConfig)
const failures = [
  ...creatureValidations.flatMap((validation, index) =>
    validation.ok ? [] : validation.errors.map(error => `${creatureFiles[index]}: ${error}`),
  ),
  ...environmentValidations.flatMap((validation, index) =>
    validation.ok ? [] : validation.errors.map(error => `${environmentFiles[index]}: ${error}`),
  ),
]

if (failures.length > 0) {
  console.error('Config verification failed:')
  for (const error of failures) {
    console.error(`- ${error}`)
  }
  process.exit(1)
}

const starterCreature = creatureValidations.find(
  validation => validation.ok && validation.value.id === 'segmented-starter',
)
const starterEnvironment = environmentValidations.find(
  validation => validation.ok && validation.value.id === 'adaptive-skate-terrarium',
)

if (!starterCreature?.ok || !starterEnvironment?.ok) {
  console.error('Expected starter creature and environment configs to be present.')
  process.exit(1)
}

const invalidCreature = structuredClone(starterCreature.value)
invalidCreature.morphology.joints[0].childId = 'missing-part'
const invalidEnvironment = structuredClone(starterEnvironment.value)
invalidEnvironment.randomization.gravityScale = [1.4, 0.7]

assertRejects(
  validateCreatureGenome(invalidCreature),
  'creature joint j-head-mid-1 references missing childId missing-part.',
)
assertRejects(
  validateEnvironmentConfig(invalidEnvironment),
  'environment.randomization.gravityScale minimum must be less than or equal to maximum.',
)

const fixedProfileValidation = validateRuntimeProfile(FIXED_ARTICULATED_RUNTIME_PROFILE)
if (!fixedProfileValidation.ok) {
  fail(`Fixed runtime profile must validate:\n${fixedProfileValidation.errors.join('\n')}`)
}

const malformedProfile = structuredClone(FIXED_ARTICULATED_RUNTIME_PROFILE) as unknown as {
  policy: { artifactPath?: string }
}
delete malformedProfile.policy.artifactPath
assertRejects(
  validateRuntimeProfile(malformedProfile),
  'runtimeProfile.policy.artifactPath must be a string.',
)

const incompatibleProfile = {
  ...FIXED_ARTICULATED_RUNTIME_PROFILE,
  id: 'valid-but-unsupported-12-segment-profile',
  segmentCount: 12,
  actionSize: 24,
}
const incompatibleAdapter = createCreatureRuntimeAdapter(starterCreature.value, incompatibleProfile)
assertRejects(incompatibleAdapter, 'runtimeProfile.segmentCount 12 is incompatible; expected 16.')

const adapter = createCreatureRuntimeAdapter(starterCreature.value, FIXED_ARTICULATED_RUNTIME_PROFILE)
if (!adapter.ok) fail(`Fixed runtime adapter must succeed:\n${adapter.errors.join('\n')}`)
if (
  adapter.value.actuatorContract.antagonisticPairCount !== 16 ||
  adapter.value.actuatorContract.muscleChannelCount !== 32 ||
  adapter.value.appearanceProjection.declaredControllerExecuted !== false ||
  adapter.value.appearanceProjection.declaredControllerActionSize !==
    starterCreature.value.controller.actionSize
) {
  fail('Fixed runtime adapter did not preserve the explicit 16-pair/32-channel appearance-only contract.')
}

const partialPaths: LabConfigPaths = {
  creatures: ['/test/valid-creature.json', '/test/malformed-creature.json'],
  environments: ['/test/valid-environment.json', '/test/unavailable-environment.json'],
  generatedCreatureManifest: '/configs/evolved/manifest.json',
}
const partialResponses = new Map<string, LabConfigFetchResult>([
  ['/test/valid-creature.json', { ok: true, value: starterCreature.value }],
  ['/test/malformed-creature.json', { ok: true, value: invalidCreature }],
  ['/test/valid-environment.json', { ok: true, value: starterEnvironment.value }],
  ['/test/unavailable-environment.json', { ok: false, status: 503, error: 'Synthetic service outage.' }],
  ['/configs/evolved/manifest.json', { ok: false, status: 404, error: 'Optional manifest absent.' }],
])
const partialResult = await loadLabConfigResources(mapFetcher(partialResponses), partialPaths)
if (
  partialResult.status !== 'partial' ||
  partialResult.creatures.length !== 1 ||
  partialResult.environments.length !== 1
) {
  fail(`Partial config loading discarded valid data: ${JSON.stringify(partialResult)}`)
}
assertResourceError(partialResult.resources, '/test/malformed-creature.json', 'missing childId missing-part')
assertResourceError(
  partialResult.resources,
  '/test/unavailable-environment.json',
  'Synthetic service outage.',
)
const optionalManifest = partialResult.resources.find(
  resource => resource.path === '/configs/evolved/manifest.json',
)
if (optionalManifest?.state !== 'optional-missing') {
  fail('A missing optional generated-creature manifest must not become a load failure.')
}

const failedResult = await loadLabConfigResources(
  async path => ({ ok: false, status: 500, error: `Synthetic failure for ${path}.` }),
  {
    creatures: ['/test/no-creature.json'],
    environments: ['/test/no-environment.json'],
    generatedCreatureManifest: '/configs/evolved/manifest.json',
  },
)
if (
  failedResult.status !== 'error' ||
  failedResult.creatures.length !== 0 ||
  failedResult.environments.length !== 0
) {
  fail('Config loader must report error when it cannot preserve a runnable creature/environment pair.')
}

console.log('Config validation, partial loading, and fixed runtime adapter verification passed.')

async function jsonFiles(path: string): Promise<string[]> {
  return (await readdir(path)).filter(file => file.endsWith('.json')).sort()
}

async function readJson(path: string): Promise<unknown> {
  return JSON.parse(await readFile(path, 'utf8')) as unknown
}

function assertRejects<T>(result: ConfigValidationResult<T>, expectedError: string) {
  if (result.ok) {
    console.error(`Expected invalid config to fail with: ${expectedError}`)
    process.exit(1)
  }
  if (!result.errors.includes(expectedError)) {
    console.error(`Expected validation error not found: ${expectedError}`)
    console.error(result.errors.join('\n'))
    process.exit(1)
  }
}

function mapFetcher(responses: Map<string, LabConfigFetchResult>): LabConfigFetcher {
  return async path => responses.get(path) ?? { ok: false, status: 404, error: `No fixture for ${path}.` }
}

function assertResourceError(
  resources: Awaited<ReturnType<typeof loadLabConfigResources>>['resources'],
  path: string,
  expectedFragment: string,
) {
  const resource = resources.find(candidate => candidate.path === path)
  if (resource?.state !== 'error' || !resource.errors.some(error => error.includes(expectedFragment))) {
    fail(`Expected ${path} to expose an error containing: ${expectedFragment}`)
  }
}

function fail(message: string): never {
  console.error(message)
  process.exit(1)
}
