import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const root = resolve(import.meta.dirname, '..')

const expected = {
  segmentCount: 16,
  muscleCount: 32,
  observationSize: 118,
  actionSize: 32,
  timestepTs: '1 / 60',
  timestepPy: '1.0 / 60.0',
  timestepMeta: 1 / 60,
}

const failures = []

const read = (path) => readFile(resolve(root, path), 'utf8')

const assert = (condition, message) => {
  if (!condition) {
    failures.push(message)
  }
}

const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

const includes = (text, value) => text.includes(String(value))

const types = await read('src/policy/types.ts')
const contracts = await read('training/wurmkickflip_rl/contracts.py')
const metadata = JSON.parse(await read('public/models/wurmkickflip_policy.meta.json'))
const requirements = await read('docs/requirements.md')
const policyContract = await read('docs/policy-contract.md')
const architecture = await read('docs/architecture.md')

assert(
  new RegExp(`export const SEGMENT_COUNT = ${expected.segmentCount}\\b`).test(types),
  'TypeScript SEGMENT_COUNT does not match expected value.',
)
assert(
  new RegExp(`export const MUSCLE_COUNT = SEGMENT_COUNT \\* 2\\b`).test(types),
  'TypeScript MUSCLE_COUNT must be derived from SEGMENT_COUNT.',
)
assert(
  new RegExp(`export const OBSERVATION_SIZE = ${expected.observationSize}\\b`).test(types),
  'TypeScript OBSERVATION_SIZE does not match expected value.',
)
assert(
  new RegExp(`export const ACTION_SIZE = MUSCLE_COUNT\\b`).test(types),
  'TypeScript ACTION_SIZE must be derived from MUSCLE_COUNT.',
)
assert(
  new RegExp(`export const POLICY_TIMESTEP = ${escapeRegExp(expected.timestepTs)}\\b`).test(types),
  'TypeScript POLICY_TIMESTEP does not match expected value.',
)

assert(
  new RegExp(`SEGMENT_COUNT = ${expected.segmentCount}\\b`).test(contracts),
  'Python SEGMENT_COUNT does not match expected value.',
)
assert(
  new RegExp(`ACTION_SIZE = SEGMENT_COUNT \\* 2\\b`).test(contracts),
  'Python ACTION_SIZE must be derived from SEGMENT_COUNT.',
)
assert(
  new RegExp(`OBSERVATION_SIZE = ${expected.observationSize}\\b`).test(contracts),
  'Python OBSERVATION_SIZE does not match expected value.',
)
assert(
  new RegExp(`POLICY_TIMESTEP = ${escapeRegExp(expected.timestepPy)}\\b`).test(contracts),
  'Python POLICY_TIMESTEP does not match expected value.',
)

assert(metadata.observationSize === expected.observationSize, 'Policy metadata observationSize mismatch.')
assert(metadata.actionSize === expected.actionSize, 'Policy metadata actionSize mismatch.')
assert(
  Math.abs(metadata.timestep - expected.timestepMeta) < Number.EPSILON,
  'Policy metadata timestep mismatch.',
)
assert(metadata.modelPath === '/models/wurmkickflip_policy.onnx', 'Policy metadata modelPath mismatch.')

const docs = [
  ['docs/requirements.md', requirements],
  ['docs/policy-contract.md', policyContract],
  ['docs/architecture.md', architecture],
]

for (const [name, text] of docs) {
  assert(includes(text, expected.segmentCount), `${name} does not mention segment count ${expected.segmentCount}.`)
  assert(includes(text, expected.observationSize), `${name} does not mention observation size ${expected.observationSize}.`)
  assert(includes(text, expected.actionSize), `${name} does not mention action size ${expected.actionSize}.`)
}

if (failures.length > 0) {
  console.error('Policy contract verification failed:')
  for (const failure of failures) {
    console.error(`- ${failure}`)
  }
  process.exit(1)
}

console.log('Policy contract verification passed.')
