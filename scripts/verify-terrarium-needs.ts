import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import type { EnvironmentConfig, Vec3 } from '../src/creature/types'
import {
  NEEDS_DIAGNOSTIC_OBSERVATION_SIZE,
  advanceNeeds,
  createNeedsDiagnosticObservation,
  createNeedsState,
  createTerrariumResources,
  readNeedsMetrics,
  selectNeedTarget,
  syncSkateboardResource,
  type NeedsState,
  type TerrariumResource,
} from '../src/scene/terrariumNeeds'
import { createTerrainField } from '../src/scene/terrainField'

const environmentUrl = new URL('../public/configs/environments/adaptive-skate-terrarium.json', import.meta.url)
const environment = JSON.parse(await readFile(environmentUrl, 'utf8')) as EnvironmentConfig
const field = createTerrainField(environment)
const first = createTerrariumResources(field, environment.seed)
const repeated = createTerrariumResources(field, environment.seed)
const alternate = createTerrariumResources(field, environment.seed + 1)

expect(JSON.stringify(first) === JSON.stringify(repeated), 'resource placement must be deterministic for a fixed seed')
expect(JSON.stringify(first) !== JSON.stringify(alternate), 'resource placement must respond to the environment seed')
expect(first.length === 3, 'terrarium must expose food, water, and skateboard resources')
expect(new Set(first.map((resource) => resource.need)).size === 3, 'each need must have exactly one resource kind')

const food = resourceFor(first, 'hunger')
const water = resourceFor(first, 'thirst')
const board = resourceFor(first, 'wellbeing')
expect(food.presentation === 'bowl' && water.presentation === 'bowl', 'food and water must render as bowls')
expect(board.presentation === 'skateboard' && board.requiresMount, 'the skateboard must be the mounted well-being resource')
expect(distanceXZ(food.position, water.position) > Math.min(field.width, field.depth) * 0.3, 'bowls must be well separated')

for (const resource of first) {
  expect(Math.abs(resource.position[0]) <= field.width * 0.5 - 0.71, resource.id + ' must remain inside the x boundary')
  expect(Math.abs(resource.position[2]) <= field.depth * 0.5 - 0.71, resource.id + ' must remain inside the z boundary')
  expect(resource.terrainNormal.every(Number.isFinite), resource.id + ' terrain normal must be finite')
  expect(resource.terrainFriction > 0, resource.id + ' terrain friction must be positive')
  expect(resource.groundHeight === field.sample(resource.position[0], resource.position[2]).height, resource.id + ' must use terrain height')
}
expect(
  Math.abs(food.position[1] - food.groundHeight - food.appearance.height * 0.5) < 1e-12,
  'food bowl must rest on terrain',
)
expect(
  Math.abs(water.position[1] - water.groundHeight - water.appearance.height * 0.5) < 1e-12,
  'water bowl must rest on terrain',
)
expect(food.terrainNormal[1] > 0.9 && water.terrainNormal[1] > 0.9, 'bowls should prefer level terrain')

const movedBoardPosition: Vec3 = [field.width * 0.28, 1.25, -field.depth * 0.31]
const synchronized = syncSkateboardResource(first, field, movedBoardPosition)
const movedBoard = resourceFor(synchronized, 'wellbeing')
expect(movedBoard.position.every((value, index) => value === movedBoardPosition[index]), 'board resource must follow the live board pose')
expect(movedBoard.groundHeight === field.sample(movedBoardPosition[0], movedBoardPosition[2]).height, 'moving board must resample terrain')
expect(resourceFor(synchronized, 'hunger') === food, 'synchronizing the board must preserve fixed bowl objects')

const urgentHunger: NeedsState = {
  ...createNeedsState(environment.seed),
  hunger: 0.92,
  thirst: 0.2,
  wellbeing: 0.15,
  targetResourceId: null,
}
expect(selectNeedTarget(urgentHunger, first, [0, 0, 0])?.need === 'hunger', 'highest urgency must select its resource')

const unmountedNeed: NeedsState = {
  ...createNeedsState(environment.seed),
  hunger: 0.1,
  thirst: 0.1,
  wellbeing: 0.8,
  targetResourceId: board.id,
}
const unmounted = advanceNeeds(unmountedNeed, 0.1, { resources: [board], wormPosition: board.position, mounted: false })
expect(unmounted.state.wellbeing > unmountedNeed.wellbeing, 'nearby board must not restore well-being while unmounted')
const mounted = advanceNeeds(unmountedNeed, 0.1, { resources: [board], wormPosition: board.position, mounted: true })
expect(mounted.state.wellbeing < unmountedNeed.wellbeing, 'riding the board must restore well-being')
expect(mounted.interaction === 'wellbeing', 'mounted board restoration must identify the interaction')

const rolloutA = runFollowerRollout(first, environment.seed)
const rolloutB = runFollowerRollout(first, environment.seed)
expect(rolloutA.hash === rolloutB.hash, 'needs rollout must be deterministic')
expect(rolloutA.visited.size === 3, 'need selection must visit food, water, and board')
expect(rolloutA.state.fulfillment.hunger > 0.3, 'food must substantially fulfill hunger')
expect(rolloutA.state.fulfillment.thirst > 0.3, 'water must substantially fulfill thirst')
expect(rolloutA.state.fulfillment.wellbeing > 0.3, 'board riding must substantially fulfill well-being')
expect(rolloutA.state.hunger >= 0 && rolloutA.state.hunger <= 1, 'hunger must stay normalized')
expect(rolloutA.state.thirst >= 0 && rolloutA.state.thirst <= 1, 'thirst must stay normalized')
expect(rolloutA.state.wellbeing >= 0 && rolloutA.state.wellbeing <= 1, 'well-being must stay normalized')

const observation = createNeedsDiagnosticObservation(rolloutA.state, synchronized, [0.4, 0.3, -0.2], 0.73, field)
expect(
  observation.length === NEEDS_DIAGNOSTIC_OBSERVATION_SIZE,
  'diagnostic needs observation size changed unexpectedly',
)
expect(observation.every(Number.isFinite), 'neural needs observation must be finite')
expect(observation.every((value) => value >= -1 && value <= 1), 'neural needs observation must be normalized')
expect(observation.slice(-3).filter((value) => value === 1).length === 1, 'active target must be one-hot encoded')

const metrics = readNeedsMetrics(mounted.state, first, board.position)
expect(metrics.activeNeed === 'wellbeing', 'metrics must expose the active need')
expect(metrics.ridingForWellbeing, 'metrics must expose board fulfillment')
expect(metrics.targetDistance === 0, 'metrics must measure planar target distance')

console.log(
  JSON.stringify(
    {
      resources: first.map((resource) => ({
        id: resource.id,
        position: resource.position.map(round),
        surface: resource.surface,
      })),
      diagnosticObservationSize: observation.length,
      visitedNeeds: [...rolloutA.visited],
      fulfillment: mapValues(rolloutA.state.fulfillment, round),
      deterministicHash: rolloutA.hash,
    },
    null,
    2,
  ),
)
console.log('Terrarium needs verification passed.')

function runFollowerRollout(resources: readonly TerrariumResource[], seed: number) {
  let state = createNeedsState(seed)
  let wormPosition: Vec3 = [0, field.sample(0, 0).height, 0]
  const visited = new Set<string>()
  const trace: number[] = []

  for (let stepIndex = 0; stepIndex < 3600; stepIndex += 1) {
    const target = resources.find((resource) => resource.id === state.targetResourceId)
    if (target) {
      wormPosition = [...target.position]
      visited.add(target.need)
    }
    const result = advanceNeeds(state, 0.05, {
      resources,
      wormPosition,
      mounted: target?.need === 'wellbeing',
    })
    state = result.state
    if (stepIndex % 10 === 0) {
      trace.push(
        round(state.hunger),
        round(state.thirst),
        round(state.wellbeing),
        state.targetResourceId ? resources.findIndex((resource) => resource.id === state.targetResourceId) : -1,
      )
    }
  }

  const hash = createHash('sha256').update(JSON.stringify(trace)).digest('hex')
  return { state, visited, hash }
}

function resourceFor(resources: readonly TerrariumResource[], need: TerrariumResource['need']) {
  const resource = resources.find((candidate) => candidate.need === need)
  if (!resource) throw new Error('Missing resource for ' + need)
  return resource
}

function distanceXZ(first: Vec3, second: Vec3) {
  return Math.hypot(first[0] - second[0], first[2] - second[2])
}

function mapValues<T extends Record<string, number>>(value: T, mapper: (number: number) => number) {
  return Object.fromEntries(Object.entries(value).map(([key, number]) => [key, mapper(number)]))
}

function round(value: number) {
  return Math.round(value * 1_000_000) / 1_000_000
}

function expect(condition: boolean, message: string): asserts condition {
  if (!condition) throw new Error(message)
}
