import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { advanceSimulation, createInitialState, toSnapshot } from '../src/policy/localSimulation'
import { sanitizePolicyAction } from '../src/policy/policyRunner'
import { ScriptedMusclePolicy } from '../src/policy/scriptedPolicy'
import { snapshotToObservation } from '../src/policy/simulationAdapter'
import { ACTION_SIZE, POLICY_TIMESTEP } from '../src/policy/types'

type RolloutSummary = {
  steps: number
  timestep: number
  final: Record<string, number | number[]>
}

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const fixturePath = resolve(root, 'fixtures/scripted-rollout.json')
const updateFixture = process.argv.includes('update')

verifyActionSanitizer()
await verifyScenePolicyIntegration()

const summary = runScriptedRollout(180)

if (updateFixture) {
  await mkdir(dirname(fixturePath), { recursive: true })
  await writeFile(fixturePath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8')
  console.log(`Updated ${fixturePath}`)
} else {
  const expected = JSON.parse(await readFile(fixturePath, 'utf8')) as RolloutSummary
  assertDeepEqual(summary, expected, 'scripted rollout')
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

function runScriptedRollout(steps: number): RolloutSummary {
  const policy = new ScriptedMusclePolicy()
  const state = createInitialState()

  for (let step = 0; step < steps; step += 1) {
    const observation = snapshotToObservation(toSnapshot(state))
    const action = policy.run(observation)
    advanceSimulation(state, action, POLICY_TIMESTEP)
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
      previousActionHead: Array.from(state.previousAction.slice(0, 8), round),
      segmentSample,
    },
  }
}

function round(value: number): number {
  return Number(value.toFixed(6))
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
