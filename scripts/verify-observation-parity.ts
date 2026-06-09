import { spawnSync } from 'node:child_process'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { snapshotToObservation } from '../src/policy/simulationAdapter'
import { OBSERVATION_SIZE, type SimulationSnapshot } from '../src/policy/types'

const root = resolve(import.meta.dirname, '..')
const fixturePath = resolve(root, 'fixtures/policy-snapshot.json')
const fixture = JSON.parse(await readFile(fixturePath, 'utf8')) as SimulationSnapshot

const browserObservation = Array.from(snapshotToObservation({
  ...fixture,
  previousAction: new Float32Array(fixture.previousAction),
}))

const python = spawnSync(
  'uv',
  ['run', 'python', '-m', 'wurmkickflip_rl.fixture_observation', '../fixtures/policy-snapshot.json'],
  {
    cwd: resolve(root, 'training'),
    encoding: 'utf8',
  },
)

if (python.status !== 0) {
  console.error(python.stdout)
  console.error(python.stderr)
  process.exit(python.status ?? 1)
}

const pythonObservation = JSON.parse(python.stdout) as number[]

if (browserObservation.length !== OBSERVATION_SIZE) {
  console.error(`Browser observation length ${browserObservation.length} did not equal ${OBSERVATION_SIZE}.`)
  process.exit(1)
}

if (pythonObservation.length !== OBSERVATION_SIZE) {
  console.error(`Python observation length ${pythonObservation.length} did not equal ${OBSERVATION_SIZE}.`)
  process.exit(1)
}

const tolerance = 1e-6
const mismatches: string[] = []

for (let index = 0; index < OBSERVATION_SIZE; index += 1) {
  const browserValue = browserObservation[index]
  const pythonValue = pythonObservation[index]
  if (Math.abs(browserValue - pythonValue) > tolerance) {
    mismatches.push(
      `${index}: browser=${browserValue.toFixed(8)} python=${pythonValue.toFixed(8)}`,
    )
  }
}

if (mismatches.length > 0) {
  console.error('Observation parity verification failed:')
  for (const mismatch of mismatches.slice(0, 20)) {
    console.error(`- ${mismatch}`)
  }
  if (mismatches.length > 20) {
    console.error(`...and ${mismatches.length - 20} more mismatches.`)
  }
  process.exit(1)
}

console.log('Observation parity verification passed.')
