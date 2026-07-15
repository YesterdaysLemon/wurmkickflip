import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import type { EnvironmentConfig } from '../src/creature/types'
import { makeInitialAction } from '../src/policy/simulationAdapter'
import { advanceStunt, createStuntState, toSnapshot } from '../src/scene/terrariumSimulation'
import { createTerrainField } from '../src/scene/terrainField'

const simulationSourceUrl = new URL('../src/scene/terrariumSimulation.ts', import.meta.url)
const viewSourceUrl = new URL('../src/scene/WurmkickflipScene.tsx', import.meta.url)
const environmentUrl = new URL(
  '../public/configs/environments/adaptive-skate-terrarium.json',
  import.meta.url,
)
const [simulationSource, viewSource, environmentSource] = await Promise.all([
  readFile(simulationSourceUrl, 'utf8'),
  readFile(viewSourceUrl, 'utf8'),
  readFile(environmentUrl, 'utf8'),
])

for (const forbiddenImport of ["from 'react'", '@react-three/fiber', '@react-three/drei']) {
  expect(
    !simulationSource.includes(forbiddenImport),
    `headless terrarium simulation imports view dependency ${forbiddenImport}`,
  )
}
expect(!simulationSource.includes('useFrame('), 'headless terrarium simulation contains a render hook')
expect(
  viewSource.includes("from './terrariumSimulation'"),
  'scene view does not delegate to the headless terrarium simulation',
)

const environment = JSON.parse(environmentSource) as EnvironmentConfig
const field = createTerrainField(environment)
const first = createStuntState(field, environment)
const second = createStuntState(field, environment)
const firstAction = makeInitialAction()
const secondAction = makeInitialAction()

for (let step = 0; step < 360; step += 1) {
  advanceStunt(first, firstAction, 1 / 60, 9.81, 'kickflip', field, environment)
  advanceStunt(second, secondAction, 1 / 60, 9.81, 'kickflip', field, environment)
}

const firstTrace = JSON.stringify({ state: first, snapshot: toSnapshot(first) })
const secondTrace = JSON.stringify({ state: second, snapshot: toSnapshot(second) })
expect(firstTrace === secondTrace, 'headless terrarium stepping is not deterministic')
expect(Math.abs(first.time - 6) < 1e-10, 'headless terrarium stepping did not advance fixed time')
expect(
  first.segments.every(segment => Object.values(segment).every(Number.isFinite)),
  'headless step produced non-finite segments',
)

const apertureLatch = verifyBowlRimApertureLatch()

console.log(
  JSON.stringify({
    apertureLatch,
    moduleLines: simulationSource.split(/\r?\n/).length,
    steps: 360,
    finalState: first.locomotionState,
    traceHash: createHash('sha256').update(firstTrace).digest('hex'),
  }),
)
console.log('Headless terrarium simulation verification passed.')

function verifyBowlRimApertureLatch() {
  const state = createStuntState(field, environment)
  const food = state.resources.find(resource => resource.id === 'food-bowl')
  expect(food !== undefined, 'food bowl is missing from aperture-latch fixture')

  const timestep = 1 / 600
  const linkLength = 0.102
  const rimRadius = food.appearance.radius * 0.82
  state.locomotionState = 'crawling'
  state.mountBlend = 0
  state.wormHeading = 0
  state.needs.hunger = 1
  state.needs.thirst = 0
  state.needs.wellbeing = 0
  state.needs.targetResourceId = 'food-bowl'
  state.feedingCooldown = 10

  for (const [index, segment] of state.segments.entries()) {
    segment.x = food.position[0] - rimRadius + (index - 11) * linkLength
    segment.y = food.groundHeight + food.appearance.height
    segment.z = food.position[2]
    segment.vx = 0
    segment.vy = 0
    segment.vz = 0
    segment.pitch = index === state.segments.length - 1 ? -Math.PI / 2 : 0
    segment.yaw = 0
  }
  state.wormX = state.segments.reduce((sum, segment) => sum + segment.x, 0) / state.segments.length
  state.wormY = state.segments.reduce((sum, segment) => sum + segment.y, 0) / state.segments.length
  state.wormZ = food.position[2]

  advanceStunt(state, makeInitialAction(), timestep, 9.81, 'kickflip', field, environment)
  expect(state.bowlRimAperture !== null, 'approaching the food bowl did not acquire a rim aperture')
  expect(state.bowlRimAperture.resourceId === 'food-bowl', 'aperture latched the wrong bowl')
  const acquiredAngle = state.bowlRimAperture.approachAngle

  // A moving controller may rotate the posterior centroid enough to cross one
  // of the twelve rim-sample cutoffs. Once acquired, that must not rotate the
  // aperture or change which solid circles exist around the embedded midbody.
  for (const segment of state.segments.slice(0, 8)) segment.z += 0.08
  advanceStunt(state, makeInitialAction(), timestep, 9.81, 'kickflip', field, environment)
  expect(state.bowlRimAperture !== null, 'posterior motion unexpectedly released the aperture')
  expect(
    state.bowlRimAperture.approachAngle === acquiredAngle,
    'posterior motion rotated an already acquired aperture',
  )

  state.needs.hunger = 0
  state.needs.thirst = 0
  state.needs.wellbeing = 1
  state.needs.targetResourceId = 'skateboard'
  const beforeHandoff = state.segments.map(segment => [segment.x, segment.y, segment.z] as const)
  advanceStunt(state, makeInitialAction(), timestep, 9.81, 'kickflip', field, environment)
  expect(
    String(state.locomotionState) === 'seeking',
    `aperture handoff fixture entered ${state.locomotionState} for target ${state.needs.targetResourceId ?? 'none'} ` +
      `after interaction ${state.needs.lastInteraction ?? 'none'}`,
  )
  expect(
    state.bowlRimAperture?.resourceId === 'food-bowl',
    'target switch moved the aperture to another resource',
  )
  expect(
    state.bowlRimAperture.approachAngle === acquiredAngle,
    'target/lifecycle switch rotated the latched aperture',
  )
  const maximumHandoffStep = Math.max(
    ...state.segments.map((segment, index) =>
      Math.hypot(
        segment.x - beforeHandoff[index][0],
        segment.y - beforeHandoff[index][1],
        segment.z - beforeHandoff[index][2],
      ),
    ),
  )
  expect(
    maximumHandoffStep < 0.09,
    `target/lifecycle switch caused a ${maximumHandoffStep.toFixed(4)} m segment correction`,
  )

  const towardCenterX = -food.position[0]
  const towardCenterZ = -food.position[2]
  const towardCenterLength = Math.max(1e-6, Math.hypot(towardCenterX, towardCenterZ))
  const clearX = food.position[0] + (towardCenterX / towardCenterLength) * 1.25
  const clearZ = food.position[2] + (towardCenterZ / towardCenterLength) * 1.25
  for (const [index, segment] of state.segments.entries()) {
    segment.x = clearX + (index - 7.5) * 0.012
    segment.y = field.sample(segment.x, clearZ).height + 0.105
    segment.z = clearZ
    segment.vx = 0
    segment.vy = 0
    segment.vz = 0
  }
  state.wormX = clearX
  state.wormZ = clearZ
  advanceStunt(state, makeInitialAction(), timestep, 9.81, 'kickflip', field, environment)
  expect(state.bowlRimAperture === null, 'aperture remained latched after the entire chain cleared the bowl')

  return {
    approachAngle: Number(acquiredAngle.toFixed(6)),
    maximumHandoffStep: Number(maximumHandoffStep.toFixed(6)),
  }
}

function expect(condition: boolean, message: string): asserts condition {
  if (!condition) throw new Error(message)
}
