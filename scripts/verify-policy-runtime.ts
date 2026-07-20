import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { EnvironmentConfig } from '../src/creature/types'
import { sanitizePolicyAction } from '../src/policy/policyRunner'
import { ScriptedMusclePolicy } from '../src/policy/scriptedPolicy'
import { makeInitialAction, snapshotToObservation } from '../src/policy/simulationAdapter'
import { ACTION_SIZE, OBSERVATION_SIZE, POLICY_TIMESTEP, SEGMENT_COUNT } from '../src/policy/types'
import {
  advanceStunt,
  createStuntState,
  makeTerrariumDecor,
  smoothAction,
  toSnapshot,
  type StuntState,
} from '../src/scene/terrariumSimulation'
import { createTerrainField } from '../src/scene/terrainField'

type RolloutSummary = {
  steps: number
  timestep: number
  final: Record<string, number | number[]>
}

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const fixturePath = resolve(root, 'fixtures/mounted-runtime-smoke.json')
const environmentPath = resolve(root, 'public/configs/environments/adaptive-skate-terrarium.json')
const updateFixture = process.argv.includes('update')
const environment = JSON.parse(await readFile(environmentPath, 'utf8')) as EnvironmentConfig

verifyActionSanitizer()
verifyObservationAdapter(environment)
await verifyScenePolicyIntegration()

const summary = runMountedSimulatorSmokeRollout(environment, 180)

if (updateFixture) {
  await mkdir(dirname(fixturePath), { recursive: true })
  await writeFile(fixturePath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8')
  console.log(`Updated ${fixturePath}`)
} else {
  const expected = JSON.parse(await readFile(fixturePath, 'utf8')) as RolloutSummary
  assertDeepEqual(summary, expected, 'mounted simulator smoke rollout')
  console.log('Policy runtime verification passed.')
}

function verifyActionSanitizer() {
  const valid = new Float32Array(ACTION_SIZE)
  valid[0] = 1.5
  valid[1] = -1.5
  valid[2] = 0.25
  const sanitized = sanitizePolicyAction(valid)
  assert(sanitized !== null, 'Expected valid action data to sanitize.')
  assert(sanitized[0] === 1, 'Expected high action value to clamp to 1.')
  assert(sanitized[1] === -1, 'Expected low action value to clamp to -1.')
  assert(sanitized[2] === 0.25, 'Expected in-range action value to remain unchanged.')

  assert(
    sanitizePolicyAction(new Float32Array(ACTION_SIZE - 1)) === null,
    'Expected short action data to fail.',
  )

  const nonFinite = new Float32Array(ACTION_SIZE)
  nonFinite[0] = Number.NaN
  assert(sanitizePolicyAction(nonFinite) === null, 'Expected NaN action data to fail.')
}

function verifyObservationAdapter(environmentConfig: EnvironmentConfig) {
  const field = createTerrainField(environmentConfig)
  const snapshot = toSnapshot(createStuntState(field, environmentConfig))
  snapshot.time = 1.25
  snapshot.board.x = -0.5
  snapshot.board.vx = Number.NaN
  snapshot.segments[0].x = 0.25
  snapshot.previousAction[0] = 0.42

  const observation = snapshotToObservation(snapshot)
  assert(observation.length === OBSERVATION_SIZE, 'Observation adapter returned the wrong channel count.')
  assertClose(observation[0], 1.25, 'Observation time channel moved.')
  assertClose(observation[1], -0.5, 'Observation board-x channel moved.')
  assert(observation[4] === 0, 'Observation adapter did not sanitize a non-finite channel.')
  assertClose(observation[14], 0.75, 'Observation segment-relative-x channel moved.')
  assertClose(observation[14 + SEGMENT_COUNT * 8], 0.42, 'Observation previous-action channel moved.')
}

async function verifyScenePolicyIntegration() {
  const scene = await readFile(resolve(root, 'src/scene/WurmkickflipScene.tsx'), 'utf8')
  assert(
    /snapshotToObservation\s*\(/.test(scene),
    'Scene must construct policy observations with snapshotToObservation.',
  )
  assert(
    /policyRunner\s*\.\s*run\s*\(/.test(scene),
    'Scene must invoke PolicyRunner.run so the loaded mounted-stunt policy cannot be silently ignored.',
  )
  assert(
    /locomotionRunner\s*\.\s*run\s*\(/.test(scene) && /locomotionSensorsFor\s*\(/.test(scene),
    'Scene must invoke LocomotionPolicyRunner with live segment sensors during detached locomotion.',
  )
}

/**
 * Small deterministic smoke fixture for the production terrarium simulator.
 *
 * This deliberately starts from a coherent mounted pose so it can exercise the
 * mounted diagnostic policy without paying for the longer autonomous boarding
 * scenario already covered by verify-stunt-motion.ts. It mirrors the scene's
 * action smoothing and procedural-obstacle inputs.
 */
function runMountedSimulatorSmokeRollout(
  environmentConfig: EnvironmentConfig,
  steps: number,
): RolloutSummary {
  const policy = new ScriptedMusclePolicy()
  const field = createTerrainField(environmentConfig)
  const state = createMountedSmokeState(field, environmentConfig)
  const decor = makeTerrariumDecor(
    environmentConfig.seed,
    field,
    state.resources,
    environmentConfig.terrain.obstacleDensity,
  )
  const appliedAction = makeInitialAction()
  let observationChecksum = 0

  for (let step = 0; step < steps; step += 1) {
    const observation = snapshotToObservation(toSnapshot(state))
    observationChecksum += observation.reduce(
      (checksum, value, index) => checksum + value * (index + 1) * (step + 1),
      0,
    )
    const targetAction = policy.run(observation)
    smoothAction(appliedAction, targetAction, POLICY_TIMESTEP)
    advanceStunt(
      state,
      appliedAction,
      POLICY_TIMESTEP,
      Math.abs(environmentConfig.world.gravity[1]),
      'kickflip',
      field,
      environmentConfig,
      decor.obstacles,
    )
  }

  const segmentIndexes = [0, 8, 15]
  const segmentSample = segmentIndexes.flatMap(index => {
    const segment = state.segments[index]
    return [round(segment.x), round(segment.y), round(segment.z), round(segment.pitch), round(segment.yaw)]
  })

  return {
    steps,
    timestep: round(POLICY_TIMESTEP),
    final: {
      time: round(state.time),
      boardX: round(state.boardX),
      boardY: round(state.boardY),
      boardZ: round(state.boardZ),
      boardVx: round(state.boardVx),
      boardVz: round(state.boardVz),
      boardPitch: round(state.boardPitch),
      boardRoll: round(state.boardRoll),
      boardYaw: round(state.boardYaw),
      wheelSpin: round(state.wheelSpin),
      reward: round(state.reward),
      contactRatio: round(state.contactRatio),
      distance: round(state.distance),
      observationChecksum: round(observationChecksum),
      previousActionHead: Array.from(state.previousAction.slice(0, 8), round),
      segmentSample,
    },
  }
}

function createMountedSmokeState(
  field: ReturnType<typeof createTerrainField>,
  environmentConfig: EnvironmentConfig,
): StuntState {
  const state = createStuntState(field, environmentConfig)
  const centerY = state.boardY + 0.18
  const forwardX = Math.cos(state.boardHeading)
  const forwardZ = Math.sin(state.boardHeading)

  state.locomotionState = 'riding'
  state.phase = 'terrarium cruise'
  state.mountBlend = 1
  state.grounded = true
  state.boardContactRatio = 1
  state.boardContactSegmentCount = state.segments.length
  state.boardContactStableSeconds = 1
  state.boardContactHeadWeight = 1
  state.boardContactMidbodyWeight = 1
  state.boardContactTailWeight = 1
  state.boardContactRelativeSpeed = 0
  state.contactRatio = 1
  state.distanceToBoard = 0
  state.wormX = state.boardX
  state.wormY = centerY
  state.wormZ = state.boardZ
  state.wormVx = 0
  state.wormVz = 0
  state.wormHeading = state.boardHeading
  state.transitionStartX = state.boardX
  state.transitionStartZ = state.boardZ
  state.transitionStartHeading = state.boardHeading

  state.segments.forEach((segment, index) => {
    const centered = index / (SEGMENT_COUNT - 1) - 0.5
    const axial = centered * 1.42
    segment.x = state.boardX + forwardX * axial
    segment.y = centerY
    segment.z = state.boardZ + forwardZ * axial
    segment.vx = 0
    segment.vy = 0
    segment.vz = 0
    segment.pitch = 0
    segment.yaw = state.boardHeading
    state.segmentGroundContacts[index].anchorX = segment.x
    state.segmentGroundContacts[index].anchorZ = segment.z
    state.segmentGroundContacts[index].strength = 0
  })

  return state
}

function round(value: number): number {
  return Number(value.toFixed(6))
}

function assertClose(actual: number, expected: number, message: string) {
  assert(Math.abs(actual - expected) < 1e-6, `${message} Expected ${expected}, got ${actual}.`)
}

function assertDeepEqual(actual: unknown, expected: unknown, label: string) {
  const actualJson = JSON.stringify(actual, null, 2)
  const expectedJson = JSON.stringify(expected, null, 2)
  if (actualJson !== expectedJson) {
    console.error(`${label} fixture mismatch.`)
    console.error('Run `npm run verify:runtime -- update` only when the rollout change is intentional.')
    console.error('Actual:')
    console.error(actualJson)
    console.error('Expected:')
    console.error(expectedJson)
    process.exit(1)
  }
}

function assert(condition: boolean, message: string): asserts condition {
  if (!condition) {
    console.error(message)
    process.exit(1)
  }
}
