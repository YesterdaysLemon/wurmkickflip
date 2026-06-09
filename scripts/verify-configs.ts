import { readdir, readFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { validateCreatureGenome, validateEnvironmentConfig } from '../src/creature/configValidation'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const creatureDir = resolve(root, 'public/configs/creatures')
const environmentDir = resolve(root, 'public/configs/environments')

const creatureFiles = await jsonFiles(creatureDir)
const environmentFiles = await jsonFiles(environmentDir)
const creatures = await Promise.all(creatureFiles.map((file) => readJson(resolve(creatureDir, file))))
const environments = await Promise.all(environmentFiles.map((file) => readJson(resolve(environmentDir, file))))

const creatureValidations = creatures.map(validateCreatureGenome)
const environmentValidations = environments.map(validateEnvironmentConfig)
const failures = [
  ...creatureValidations.flatMap((validation, index) =>
    validation.ok ? [] : validation.errors.map((error) => `${creatureFiles[index]}: ${error}`),
  ),
  ...environmentValidations.flatMap((validation, index) =>
    validation.ok ? [] : validation.errors.map((error) => `${environmentFiles[index]}: ${error}`),
  ),
]

if (failures.length > 0) {
  console.error('Config verification failed:')
  for (const error of failures) {
    console.error(`- ${error}`)
  }
  process.exit(1)
}

const starterCreature = creatureValidations.find((validation) => validation.ok && validation.value.id === 'segmented-starter')
const starterEnvironment = environmentValidations.find(
  (validation) => validation.ok && validation.value.id === 'adaptive-skate-terrarium',
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

console.log('Config verification passed.')

async function jsonFiles(path: string): Promise<string[]> {
  return (await readdir(path)).filter((file) => file.endsWith('.json')).sort()
}

async function readJson(path: string): Promise<unknown> {
  return JSON.parse(await readFile(path, 'utf8')) as unknown
}

function assertRejects(result: ReturnType<typeof validateCreatureGenome>, expectedError: string) {
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
