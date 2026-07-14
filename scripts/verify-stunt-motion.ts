import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import type { EnvironmentConfig } from '../src/creature/types'
import { NeuralStuntPolicy, parseStuntPolicy } from '../src/policy/neuralPolicy'
import { makeInitialAction, snapshotToObservation } from '../src/policy/simulationAdapter'
import { POLICY_TIMESTEP } from '../src/policy/types'
import {
  advanceStunt,
  createStuntState,
  smoothAction,
  toSnapshot,
  type ShowcaseMode,
} from '../src/scene/WurmkickflipScene'
import { createTerrainField } from '../src/scene/terrainField'

const root = resolve(import.meta.dirname, '..')
const environment = JSON.parse(
  await readFile(resolve(root, 'public/configs/environments/adaptive-skate-terrarium.json'), 'utf8'),
) as EnvironmentConfig
const artifact = parseStuntPolicy(
  JSON.parse(await readFile(resolve(root, 'public/models/wurmkickflip_stunt_policy.json'), 'utf8')),
)

const first = simulate('kickflip', 46)
const second = simulate('kickflip', 46)
const freestyle = simulate('freestyle', 18)

expect(first.hash === second.hash, 'identical seeded kickflip rollouts must be deterministic')
expect(first.flipsLanded >= 2, `expected a kickflip before and after crawling, got ${first.flipsLanded}`)
for (const state of ['riding', 'dismounting', 'crawling', 'seeking', 'mounting']) {
  expect(first.locomotionStates.includes(state), `kickflip rollout never entered ${state}`)
}
expect(first.remounted, 'kickflip rollout never returned to riding after its detached crawl')
expect(first.maximumSeparation > 0.8, `worm never clearly detached from board (${first.maximumSeparation.toFixed(3)} m)`)
expect(first.maximumRootStep < 0.08, `root motion jumped ${first.maximumRootStep.toFixed(4)} m in one tick`)
expect(
  first.maximumSegmentStep < 0.09,
  `segment motion jumped ${first.maximumSegmentStep.toFixed(4)} m in one tick (${first.maximumSegmentContext})`,
)
expect(first.bodySpeedP99 < 2.8, `body-speed p99 is too jittery (${first.bodySpeedP99.toFixed(3)} m/s)`)
expect(first.maximumBodySpeed < 4.2, `body-speed maximum is too jittery (${first.maximumBodySpeed.toFixed(3)} m/s)`)
expect(
  first.maximumYawHeadingError < 0.16,
  `board yaw drifted away from its route heading (${first.maximumYawHeadingError.toFixed(3)} rad)`,
)
expect(first.boardRangeX > 2.5 && first.boardRangeZ > 2.5, 'board did not traverse meaningful distance on both arena axes')
expect(first.inBounds, 'kickflip rollout left the configured terrarium bounds')
expect(first.observationsFinite, 'kickflip rollout produced non-finite policy observations')

expect(freestyle.locomotionStates.includes('crawling'), 'free crawl never detached onto the terrain')
expect(!freestyle.remounted, 'free crawl unexpectedly remounted the skateboard')
expect(freestyle.wormRangeX > 1.2 && freestyle.wormRangeZ > 1.2, 'free crawl did not explore both arena axes')
expect(freestyle.inBounds, 'free crawl left the configured terrarium bounds')

console.log(
  JSON.stringify(
    {
      bodySpeedMax: round(first.maximumBodySpeed),
      bodySpeedP99: round(first.bodySpeedP99),
      boardRange: [round(first.boardRangeX), round(first.boardRangeZ)],
      deterministicHash: first.hash,
      flipsLanded: first.flipsLanded,
      freestyleWormRange: [round(freestyle.wormRangeX), round(freestyle.wormRangeZ)],
      locomotionStates: first.locomotionStates,
      maximumRootStep: round(first.maximumRootStep),
      maximumSegmentStep: round(first.maximumSegmentStep),
      maximumYawHeadingError: round(first.maximumYawHeadingError),
      maximumSeparation: round(first.maximumSeparation),
      remounted: first.remounted,
    },
    null,
    2,
  ),
)
console.log('Stunt motion verification passed.')

function simulate(mode: ShowcaseMode, seconds: number) {
  const field = createTerrainField(environment)
  const state = createStuntState(field)
  const policy = new NeuralStuntPolicy(artifact)
  const appliedAction = makeInitialAction()
  const bodySpeeds: number[] = []
  const locomotionStates: string[] = []
  const boardX: number[] = [state.boardX]
  const boardZ: number[] = [state.boardZ]
  const wormX: number[] = [state.wormX]
  const wormZ: number[] = [state.wormZ]
  const trace: string[] = []
  let maximumBodySpeed = 0
  let maximumRootStep = 0
  let maximumSegmentStep = 0
  let maximumSegmentContext = ''
  let maximumSeparation = 0
  let maximumYawHeadingError = 0
  let observationsFinite = true
  let inBounds = true
  let sawDetached = false
  let remounted = false

  for (let step = 0; step < Math.round(seconds / POLICY_TIMESTEP); step += 1) {
    const observation = snapshotToObservation(toSnapshot(state))
    observationsFinite &&= observation.every(Number.isFinite)
    const action = policy.run(observation)
    smoothAction(appliedAction, action, POLICY_TIMESTEP)
    const previousBoardX = state.boardX
    const previousBoardZ = state.boardZ
    const previousWormX = state.wormX
    const previousWormZ = state.wormZ
    const previousSegments = state.segments.map((segment) => [segment.x, segment.y, segment.z] as const)

    advanceStunt(state, appliedAction, POLICY_TIMESTEP, 9.81, mode, field, environment)

    if (locomotionStates.at(-1) !== state.locomotionState) locomotionStates.push(state.locomotionState)
    if (state.locomotionState !== 'riding') sawDetached = true
    if (sawDetached && state.locomotionState === 'riding') remounted = true
    maximumRootStep = Math.max(
      maximumRootStep,
      Math.hypot(state.boardX - previousBoardX, state.boardZ - previousBoardZ),
      Math.hypot(state.wormX - previousWormX, state.wormZ - previousWormZ),
    )
    const rootVx = state.mountBlend > 0.5 ? state.boardVx : state.wormVx
    const rootVz = state.mountBlend > 0.5 ? state.boardVz : state.wormVz
    let stepBodySpeed = 0
    state.segments.forEach((segment, index) => {
      stepBodySpeed = Math.max(stepBodySpeed, Math.hypot(segment.vx - rootVx, segment.vy, segment.vz - rootVz))
      const segmentStep = Math.hypot(
        segment.x - previousSegments[index][0],
        segment.y - previousSegments[index][1],
        segment.z - previousSegments[index][2],
      )
      if (segmentStep > maximumSegmentStep) {
        maximumSegmentStep = segmentStep
        maximumSegmentContext = `t=${state.time.toFixed(3)} phase=${state.phase} locomotion=${state.locomotionState} segment=${index} mount=${state.mountBlend.toFixed(3)} range=${state.distanceToBoard.toFixed(3)}`
      }
    })
    bodySpeeds.push(stepBodySpeed)
    maximumBodySpeed = Math.max(maximumBodySpeed, stepBodySpeed)
    maximumSeparation = Math.max(maximumSeparation, state.distanceToBoard)
    maximumYawHeadingError = Math.max(maximumYawHeadingError, Math.abs(wrapAngle(state.boardYaw + state.boardHeading)))
    boardX.push(state.boardX)
    boardZ.push(state.boardZ)
    wormX.push(state.wormX)
    wormZ.push(state.wormZ)
    const xLimit = field.width * 0.5
    const zLimit = field.depth * 0.5
    inBounds &&=
      Math.abs(state.boardX) <= xLimit &&
      Math.abs(state.wormX) <= xLimit &&
      Math.abs(state.boardZ) <= zLimit &&
      Math.abs(state.wormZ) <= zLimit

    if (step % 10 === 0) {
      trace.push(
        [
          state.boardX,
          state.boardY,
          state.boardZ,
          state.wormX,
          state.wormY,
          state.wormZ,
          state.boardRoll,
          state.flipsLanded,
          state.locomotionState,
        ]
          .map((value) => (typeof value === 'number' ? value.toFixed(6) : value))
          .join(','),
      )
    }
  }

  const sortedBodySpeeds = bodySpeeds.toSorted((a, b) => a - b)
  return {
    boardRangeX: range(boardX),
    boardRangeZ: range(boardZ),
    bodySpeedP99: sortedBodySpeeds[Math.min(sortedBodySpeeds.length - 1, Math.floor(sortedBodySpeeds.length * 0.99))],
    flipsLanded: state.flipsLanded,
    hash: createHash('sha256').update(trace.join('\n')).digest('hex'),
    inBounds,
    locomotionStates,
    maximumBodySpeed,
    maximumRootStep,
    maximumSegmentStep,
    maximumSegmentContext,
    maximumSeparation,
    maximumYawHeadingError,
    observationsFinite,
    remounted,
    wormRangeX: range(wormX),
    wormRangeZ: range(wormZ),
  }
}

function range(values: number[]) {
  return Math.max(...values) - Math.min(...values)
}

function round(value: number) {
  return Number(value.toFixed(6))
}

function wrapAngle(angle: number) {
  return Math.atan2(Math.sin(angle), Math.cos(angle))
}

function expect(condition: boolean, message: string): asserts condition {
  if (!condition) {
    console.error(`Stunt motion verification failed: ${message}`)
    process.exit(1)
  }
}
