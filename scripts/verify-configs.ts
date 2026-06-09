import { readFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { validateCreatureGenome, validateEnvironmentConfig } from '../src/creature/configValidation'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')

const creature = JSON.parse(await readFile(resolve(root, 'public/configs/creatures/segmented-starter.json'), 'utf8')) as unknown
const environment = JSON.parse(
  await readFile(resolve(root, 'public/configs/environments/adaptive-skate-terrarium.json'), 'utf8'),
) as unknown

const creatureValidation = validateCreatureGenome(creature)
const environmentValidation = validateEnvironmentConfig(environment)

if (!creatureValidation.ok || !environmentValidation.ok) {
  console.error('Config verification failed:')
  for (const error of creatureValidation.ok ? [] : creatureValidation.errors) {
    console.error(`- ${error}`)
  }
  for (const error of environmentValidation.ok ? [] : environmentValidation.errors) {
    console.error(`- ${error}`)
  }
  process.exit(1)
}

const invalidCreature = structuredClone(creatureValidation.value)
invalidCreature.morphology.joints[0].childId = 'missing-part'
const invalidEnvironment = structuredClone(environmentValidation.value)
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
