import { spawnSync } from 'node:child_process'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { ACTION_SIZE, OBSERVATION_SIZE } from '../src/policy/types'

type StuntPolicy = {
  schemaVersion: number
  kind: string
  modelVersion: string
  inputSize: number
  hiddenSize: number
  outputSize: number
  activation: string
  hiddenWeights: unknown
  hiddenBias: unknown
  outputWeights: unknown
  outputBias: unknown
  training: unknown
}

const policyPath = resolve(import.meta.dirname, '../public/models/wurmkickflip_stunt_policy.json')
const failures: string[] = []
const teacherFeatureIndices = [
  0,
  8,
  9,
  10,
  11,
  ...Array.from({ length: 16 }, (_, segment) => [14 + segment * 8 + 6, 14 + segment * 8 + 7]).flat(),
]
let policy: StuntPolicy

try {
  policy = JSON.parse(await readFile(policyPath, 'utf8')) as StuntPolicy
} catch (error) {
  console.error(
    error instanceof Error
      ? `Unable to read stunt policy artifact: ${error.message}`
      : 'Unable to read stunt policy artifact.',
  )
  process.exit(1)
}

check(policy.schemaVersion === 1, 'schemaVersion must equal 1.')
check(policy.kind === 'wurmkickflip.stuntPolicy', 'kind must equal wurmkickflip.stuntPolicy.')
check(
  typeof policy.modelVersion === 'string' && policy.modelVersion.length > 0,
  'modelVersion must be non-empty.',
)
check(policy.inputSize === OBSERVATION_SIZE, `inputSize must equal ${OBSERVATION_SIZE}.`)
check(policy.outputSize === ACTION_SIZE, `outputSize must equal ${ACTION_SIZE}.`)
check(Number.isInteger(policy.hiddenSize) && policy.hiddenSize > 0, 'hiddenSize must be a positive integer.')
check(policy.activation === 'tanh', 'activation must equal tanh.')

checkMatrix(policy.hiddenWeights, policy.hiddenSize, policy.inputSize, 'hiddenWeights')
checkIgnoredColumns(policy.hiddenWeights)
checkVector(policy.hiddenBias, policy.hiddenSize, 'hiddenBias')
checkMatrix(policy.outputWeights, policy.outputSize, policy.hiddenSize, 'outputWeights')
checkVector(policy.outputBias, policy.outputSize, 'outputBias')
checkTraining(policy.training)

if (failures.length > 0) {
  console.error('Stunt policy verification failed:')
  for (const failure of failures) {
    console.error(`- ${failure}`)
  }
  process.exit(1)
}

const python = spawnSync('uv', ['run', 'python', '-m', 'wurmkickflip_rl.validate_stunt_policy'], {
  cwd: resolve(import.meta.dirname, '../training'),
  encoding: 'utf8',
})

if (python.status !== 0) {
  console.error(python.stdout)
  console.error(python.stderr)
  process.exit(python.status ?? 1)
}

if (python.stdout.trim()) {
  console.log(python.stdout.trim())
}
console.log('Stunt policy verification passed.')

function checkMatrix(value: unknown, rows: number, columns: number, path: string) {
  if (!Array.isArray(value)) {
    failures.push(`${path} must be an array.`)
    return
  }
  check(value.length === rows, `${path} must contain ${rows} rows.`)
  for (let row = 0; row < value.length; row += 1) {
    checkVector(value[row], columns, `${path}[${row}]`)
  }
}

function checkVector(value: unknown, length: number, path: string) {
  if (!Array.isArray(value)) {
    failures.push(`${path} must be an array.`)
    return
  }
  check(value.length === length, `${path} must contain ${length} values.`)
  for (let index = 0; index < value.length; index += 1) {
    if (typeof value[index] !== 'number' || !Number.isFinite(value[index])) {
      failures.push(`${path}[${index}] must be a finite number.`)
      return
    }
  }
}

function checkTraining(value: unknown) {
  if (!isRecord(value)) {
    failures.push('training must be an object.')
    return
  }
  check(typeof value.seed === 'number' && Number.isInteger(value.seed), 'training.seed must be an integer.')
  check(isPositiveInteger(value.samples), 'training.samples must be a positive integer.')
  check(isPositiveInteger(value.epochs), 'training.epochs must be a positive integer.')
  check(
    isNonNegativeFinite(value.validationMse),
    'training.validationMse must be a non-negative finite number.',
  )
  check(
    typeof value.teacherAgreement === 'number' &&
      Number.isFinite(value.teacherAgreement) &&
      value.teacherAgreement >= 0 &&
      value.teacherAgreement <= 1,
    'training.teacherAgreement must be a finite number in [0, 1].',
  )
  if (!Array.isArray(value.teacherFeatureIndices)) {
    failures.push('training.teacherFeatureIndices must be an array.')
  } else {
    check(
      value.teacherFeatureIndices.length === teacherFeatureIndices.length &&
        value.teacherFeatureIndices.every((index, position) => index === teacherFeatureIndices[position]),
      'training.teacherFeatureIndices must match the documented teacher feature mask.',
    )
  }
}

function checkIgnoredColumns(value: unknown) {
  if (!Array.isArray(value)) return
  const used = new Set(teacherFeatureIndices)
  let maximum = 0
  for (const row of value) {
    if (!Array.isArray(row)) continue
    for (let column = 0; column < Math.min(row.length, OBSERVATION_SIZE); column += 1) {
      if (used.has(column) || typeof row[column] !== 'number' || !Number.isFinite(row[column])) continue
      maximum = Math.max(maximum, Math.abs(row[column]))
    }
  }
  check(
    maximum === 0,
    `ignored observation columns must be exactly zero; max absolute weight was ${maximum}.`,
  )
}

function check(condition: boolean, message: string) {
  if (!condition) failures.push(message)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0
}

function isNonNegativeFinite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0
}
