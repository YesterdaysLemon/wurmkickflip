import { spawnSync } from 'node:child_process'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { snapshotToObservation } from '../src/policy/simulationAdapter'
import {
  ACTION_SIZE,
  OBSERVATION_SIZE,
  SEGMENT_COUNT,
  type SegmentSnapshot,
  type SimulationSnapshot,
} from '../src/policy/types'

const OBSERVATION_HEADER_SIZE = 14
const SEGMENT_OBSERVATION_SIZE = 8
const PREVIOUS_ACTION_START = OBSERVATION_HEADER_SIZE + SEGMENT_COUNT * SEGMENT_OBSERVATION_SIZE
const TOLERANCE = 1e-6

type ObservationFixture = SimulationSnapshot & {
  observationSize: number
}

const root = resolve(import.meta.dirname, '..')
const fixturePath = resolve(root, 'fixtures/policy-snapshot.json')
const fixture = JSON.parse(await readFile(fixturePath, 'utf8')) as ObservationFixture
const browserSnapshot: SimulationSnapshot = {
  ...fixture,
  previousAction: new Float32Array(fixture.previousAction),
}
const browserObservation = Array.from(snapshotToObservation(browserSnapshot))

assert(
  fixture.observationSize === OBSERVATION_SIZE,
  `Fixture observationSize must equal ${OBSERVATION_SIZE}.`,
)
assert(
  PREVIOUS_ACTION_START + ACTION_SIZE === OBSERVATION_SIZE,
  `Observation layout arithmetic must fill exactly ${OBSERVATION_SIZE} values.`,
)

if (browserObservation.length !== OBSERVATION_SIZE) {
  fail(`Browser observation length ${browserObservation.length} did not equal ${OBSERVATION_SIZE}.`)
}

verifyCompleteLayout(browserObservation, browserSnapshot)
verifyMissingRecordPadding(browserSnapshot)

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

if (pythonObservation.length !== OBSERVATION_SIZE) {
  fail(`Python observation length ${pythonObservation.length} did not equal ${OBSERVATION_SIZE}.`)
}

const mismatches: string[] = []

for (let index = 0; index < OBSERVATION_SIZE; index += 1) {
  const browserValue = browserObservation[index]
  const pythonValue = pythonObservation[index]
  if (Math.abs(browserValue - pythonValue) > TOLERANCE) {
    mismatches.push(`${index}: browser=${browserValue.toFixed(8)} python=${pythonValue.toFixed(8)}`)
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

function verifyCompleteLayout(observation: number[], snapshot: SimulationSnapshot) {
  const tailIndex = SEGMENT_COUNT - 1
  const tailStart = OBSERVATION_HEADER_SIZE + tailIndex * SEGMENT_OBSERVATION_SIZE
  const tail = snapshot.segments[tailIndex]
  assert(tail !== undefined, `Fixture must include segment ${tailIndex}.`)
  assertSliceClose(observation, tailStart, segmentValues(tail, snapshot), 'tail segment')
  assertSliceClose(
    observation,
    PREVIOUS_ACTION_START,
    Array.from(snapshot.previousAction),
    'complete previous action',
  )
}

function verifyMissingRecordPadding(snapshot: SimulationSnapshot) {
  const missingTail = snapshotToObservation({
    ...snapshot,
    segments: snapshot.segments.slice(0, SEGMENT_COUNT - 1),
  })
  const tailStart = OBSERVATION_HEADER_SIZE + (SEGMENT_COUNT - 1) * SEGMENT_OBSERVATION_SIZE
  assertZeroSlice(missingTail, tailStart, SEGMENT_OBSERVATION_SIZE, 'missing tail segment')
  assertSliceClose(
    missingTail,
    PREVIOUS_ACTION_START,
    Array.from(snapshot.previousAction),
    'previous action after missing tail',
  )

  const missingIndex = 5
  const followingIndex = missingIndex + 1
  const sparseSegments = snapshot.segments.slice()
  Reflect.deleteProperty(sparseSegments, String(missingIndex))
  const sparseObservation = snapshotToObservation({ ...snapshot, segments: sparseSegments })
  const missingStart = OBSERVATION_HEADER_SIZE + missingIndex * SEGMENT_OBSERVATION_SIZE
  const followingStart = OBSERVATION_HEADER_SIZE + followingIndex * SEGMENT_OBSERVATION_SIZE
  assertZeroSlice(sparseObservation, missingStart, SEGMENT_OBSERVATION_SIZE, 'missing middle segment')
  assertSliceClose(
    sparseObservation,
    followingStart,
    segmentValues(snapshot.segments[followingIndex], snapshot),
    'segment after missing middle record',
  )

  const shortPreviousAction = new Float32Array(Array.from(snapshot.previousAction).slice(0, ACTION_SIZE - 1))
  const shortActionObservation = snapshotToObservation({ ...snapshot, previousAction: shortPreviousAction })
  assert(
    shortActionObservation[OBSERVATION_SIZE - 1] === 0,
    'A missing final previous-action value must be zero-filled at index 173.',
  )
}

function segmentValues(segment: SegmentSnapshot | undefined, snapshot: SimulationSnapshot): number[] {
  if (!segment) {
    fail('Expected segment record was missing from the fixture.')
  }
  return [
    segment.x - snapshot.board.x,
    segment.y - snapshot.board.y,
    segment.z - snapshot.board.z,
    segment.vx,
    segment.vy,
    segment.vz,
    segment.pitch,
    segment.yaw,
  ]
}

function assertSliceClose(
  actual: ArrayLike<number>,
  start: number,
  expected: ArrayLike<number>,
  label: string,
) {
  for (let offset = 0; offset < expected.length; offset += 1) {
    if (Math.abs(actual[start + offset] - expected[offset]) > TOLERANCE) {
      fail(
        `${label} mismatch at observation index ${start + offset}: ` +
          `actual=${actual[start + offset]} expected=${expected[offset]}.`,
      )
    }
  }
}

function assertZeroSlice(actual: ArrayLike<number>, start: number, length: number, label: string) {
  for (let offset = 0; offset < length; offset += 1) {
    if (actual[start + offset] !== 0) {
      fail(`${label} must zero-fill observation index ${start + offset}.`)
    }
  }
}

function assert(condition: boolean, message: string): asserts condition {
  if (!condition) {
    fail(message)
  }
}

function fail(message: string): never {
  console.error(message)
  process.exit(1)
}
