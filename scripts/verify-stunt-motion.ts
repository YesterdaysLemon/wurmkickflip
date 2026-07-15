import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import type { EnvironmentConfig } from '../src/creature/types'
import { NeuralStuntPolicy, parseStuntPolicy } from '../src/policy/neuralPolicy'
import { EvolvedLocomotionPolicy, parseLocomotionPolicy } from '../src/policy/locomotionPolicy'
import { makeInitialAction, snapshotToObservation } from '../src/policy/simulationAdapter'
import { POLICY_TIMESTEP } from '../src/policy/types'
import {
  advanceStunt,
  createStuntState,
  locomotionSensorsFor,
  makeTerrariumDecor,
  smoothAction,
  toSnapshot,
  type ShowcaseMode,
} from '../src/scene/WurmkickflipScene'
import { createTerrainField } from '../src/scene/terrainField'

const root = resolve(import.meta.dirname, '..')
const environment = JSON.parse(
  await readFile(resolve(root, 'public/configs/environments/adaptive-skate-terrarium.json'), 'utf8'),
) as EnvironmentConfig
const rippleEnvironment = JSON.parse(
  await readFile(resolve(root, 'public/configs/environments/ripple-yard.json'), 'utf8'),
) as EnvironmentConfig
const tiltEnvironment = JSON.parse(
  await readFile(resolve(root, 'public/configs/environments/tilt-basin.json'), 'utf8'),
) as EnvironmentConfig
const artifact = parseStuntPolicy(
  JSON.parse(await readFile(resolve(root, 'public/models/wurmkickflip_stunt_policy.json'), 'utf8')),
)
const locomotionArtifact = parseLocomotionPolicy(
  JSON.parse(await readFile(resolve(root, 'public/models/wurmkickflip_locomotion_policy.json'), 'utf8')),
)

const first = simulate('kickflip', 150)
const second = simulate('kickflip', 150)
const freestyle = simulate('freestyle', 32)
const zeroFreestyle = simulate('freestyle', 32, 'zero')
const frozenFreestyle = simulate('freestyle', 32, 'frozen')
const shuffledFreestyle = simulate('freestyle', 32, 'shuffled')
const zeroFrictionFreestyle = simulate('freestyle', 32, 'full', 0)
const rippleFreestyle = simulate('freestyle', 48, 'full', undefined, rippleEnvironment)
const tiltFreestyle = simulate('freestyle', 48, 'full', undefined, tiltEnvironment)
const incidentalBoardContact = verifyIncidentalBoardCollision()

expect(first.hash === second.hash, 'identical seeded kickflip rollouts must be deterministic')
expect(
  first.flipsLanded >= 2,
  `expected a kickflip before and after crawling, got ${first.flipsLanded}; ` +
    `states=${first.locomotionStates.join('>')} needs=${first.foodFulfillment.toFixed(2)}/` +
    `${first.waterFulfillment.toFixed(2)}/${first.wellbeingFulfillment.toFixed(2)} ` +
    `crawl=${first.crawlDistance.toFixed(2)} separation=${first.maximumSeparation.toFixed(2)} ` +
    `board=${first.finalDistanceToBoard.toFixed(2)}`,
)
for (const state of ['riding', 'dismounting', 'crawling', 'feeding', 'seeking', 'mounting']) {
  expect(first.locomotionStates.includes(state), `kickflip rollout never entered ${state}`)
}
expect(first.remounted, 'kickflip rollout never returned to riding after its detached crawl')
expect(first.foodFulfillment > 0.2, 'needs-driven rollout never ate from the food bowl')
expect(first.waterFulfillment > 0.2, 'needs-driven rollout never drank from the water bowl')
expect(first.ateWhileNeurallyCrawling, 'food interaction never occurred during neural crawl/seek locomotion')
expect(first.drankWhileNeurallyCrawling, 'water interaction never occurred during neural crawl/seek locomotion')
expect(first.wellbeingFulfillment > 0.2, 'needs-driven rollout never restored well-being on the skateboard')
expect(
  first.postRemountWellbeingFulfillment > 0.2,
  'well-being was not restored after the worm neurally returned to and remounted the skateboard',
)
expect(first.maximumSeparation > 0.8, `worm never clearly detached from board (${first.maximumSeparation.toFixed(3)} m)`)
expect(
  first.maximumRootStep < 0.08,
  `root motion jumped ${first.maximumRootStep.toFixed(4)} m in one tick (${first.maximumRootContext})`,
)
expect(
  first.maximumSegmentStep < 0.09,
  `segment motion jumped ${first.maximumSegmentStep.toFixed(4)} m in one tick (${first.maximumSegmentContext})`,
)
expect(
  first.bodySpeedP99 < 3.2,
  `body-speed p99 is too jittery (${first.bodySpeedP99.toFixed(3)} m/s; max ${first.maximumBodyContext})`,
)
expect(
  first.maximumBodySpeed < 4.8,
  `body-speed maximum is too jittery (${first.maximumBodySpeed.toFixed(3)} m/s; ${first.maximumBodyContext})`,
)
expect(
  first.maximumYawHeadingError < 0.16,
  `board yaw drifted away from its route heading (${first.maximumYawHeadingError.toFixed(3)} rad)`,
)
expect(
  first.boardRangeX > 1.2 && first.boardRangeZ > 1.2,
  `board did not traverse meaningful distance on both arena axes (${first.boardRangeX.toFixed(2)} × ${first.boardRangeZ.toFixed(2)})`,
)
expect(first.inBounds, 'kickflip rollout left the configured terrarium bounds')
expect(first.observationsFinite, 'kickflip rollout produced non-finite policy observations')

expect(freestyle.locomotionStates.includes('crawling'), 'free crawl never detached onto the terrain')
expect(!freestyle.remounted, 'free crawl unexpectedly remounted the skateboard')
expect(freestyle.wormRangeX > 1.2 && freestyle.wormRangeZ > 1.2, 'free crawl did not explore both arena axes')
expect(freestyle.inBounds, 'free crawl left the configured terrarium bounds')
for (const [label, rollout] of [['ripple yard', rippleFreestyle], ['tilt basin', tiltFreestyle]] as const) {
  expect(
    rollout.crawlDistance > 3,
    `${label} crawl did not produce meaningful segment-driven travel ` +
      `(${rollout.crawlDistance.toFixed(2)} m; contacts=${rollout.obstacleContacts.join(',') || 'none'}; ` +
      `states=${rollout.locomotionStates.join('>')})`,
  )
  expect(
    rollout.ateWhileNeurallyCrawling && rollout.drankWhileNeurallyCrawling,
    `${label} crawl did not reach both food and water resources ` +
      `(distance=${rollout.crawlDistance.toFixed(2)} food=${rollout.ateWhileNeurallyCrawling} ` +
      `water=${rollout.drankWhileNeurallyCrawling} contact=${rollout.finalObstacleContact ?? 'none'})`,
  )
  expect(rollout.inBounds, `${label} crawl left its configured terrarium bounds`)
}
for (const [label, rollout] of [
  ['autonomous', first],
  ['freestyle', freestyle],
  ['ripple yard', rippleFreestyle],
  ['tilt basin', tiltFreestyle],
] as const) {
  expect(
    rollout.minimumStaticObstacleClearance >= -1e-6,
    `${label} rollout penetrated a rendered tree or rock by ${(-rollout.minimumStaticObstacleClearance).toFixed(5)} m ` +
      `(${rollout.minimumStaticObstacleContext})`,
  )
}
const staticContactIds = [first, freestyle, rippleFreestyle, tiltFreestyle]
  .flatMap((rollout) => rollout.obstacleContacts)
  .filter((id) => id.startsWith('tree-') || id.startsWith('rock-'))
expect(staticContactIds.some((id) => id.startsWith('tree-')), 'integrated rollouts never exercised a tree collision')
expect(staticContactIds.some((id) => id.startsWith('rock-')), 'integrated rollouts never exercised a rock collision')
expect(zeroFreestyle.neuralActionMax > 0.5, 'zero intervention did not exercise an active neural controller')
expect(zeroFreestyle.crawlDistance < 1e-10, 'zero segment commands moved the integrated worm root')
expect(zeroFreestyle.finalLateralSpan < 0.05, 'zero commands left a hidden time-authored crawl pose')
expect(zeroFrictionFreestyle.neuralActionMax > 0.5, 'zero-friction scene did not exercise active neural commands')
expect(zeroFrictionFreestyle.crawlDistance < 1e-10, 'integrated worm moved without terrain traction')
expect(
  freestyle.crawlDistance > frozenFreestyle.crawlDistance + 1,
  'full recurrent crawl did not beat a frozen-command scene intervention',
)
expect(
  freestyle.crawlDistance > shuffledFreestyle.crawlDistance + 0.5,
  'full recurrent crawl did not beat a shuffled-segment scene intervention',
)

console.log(
  JSON.stringify(
    {
      bodySpeedMax: round(first.maximumBodySpeed),
      bodySpeedP99: round(first.bodySpeedP99),
      boardRange: [round(first.boardRangeX), round(first.boardRangeZ)],
      deterministicHash: first.hash,
      flipsLanded: first.flipsLanded,
      freestyleWormRange: [round(freestyle.wormRangeX), round(freestyle.wormRangeZ)],
      integratedCrawlAblations: {
        full: round(freestyle.crawlDistance),
        zero: round(zeroFreestyle.crawlDistance),
        frozen: round(frozenFreestyle.crawlDistance),
        shuffled: round(shuffledFreestyle.crawlDistance),
        zeroFriction: round(zeroFrictionFreestyle.crawlDistance),
        zeroFinalLateralSpan: round(zeroFreestyle.finalLateralSpan),
      },
      presetCrawl: {
        ripple: {
          distance: round(rippleFreestyle.crawlDistance),
          food: rippleFreestyle.ateWhileNeurallyCrawling,
          water: rippleFreestyle.drankWhileNeurallyCrawling,
        },
        tilt: {
          distance: round(tiltFreestyle.crawlDistance),
          food: tiltFreestyle.ateWhileNeurallyCrawling,
          water: tiltFreestyle.drankWhileNeurallyCrawling,
        },
      },
      environmentContacts: {
        autonomous: first.obstacleContacts,
        freestyle: freestyle.obstacleContacts,
        ripple: rippleFreestyle.obstacleContacts,
        tilt: tiltFreestyle.obstacleContacts,
      },
      locomotionStates: first.locomotionStates,
      maximumRootStep: round(first.maximumRootStep),
      maximumSegmentStep: round(first.maximumSegmentStep),
      maximumYawHeadingError: round(first.maximumYawHeadingError),
      maximumSeparation: round(first.maximumSeparation),
      minimumStaticObstacleClearance: {
        autonomous: [round(first.minimumStaticObstacleClearance), first.minimumStaticObstacleContext],
        freestyle: [round(freestyle.minimumStaticObstacleClearance), freestyle.minimumStaticObstacleContext],
        ripple: [round(rippleFreestyle.minimumStaticObstacleClearance), rippleFreestyle.minimumStaticObstacleContext],
        tilt: [round(tiltFreestyle.minimumStaticObstacleClearance), tiltFreestyle.minimumStaticObstacleContext],
      },
      needsFulfillment: {
        food: round(first.foodFulfillment),
        water: round(first.waterFulfillment),
        wellbeing: round(first.wellbeingFulfillment),
        wellbeingAfterRemount: round(first.postRemountWellbeingFulfillment),
      },
      neuralResourceContacts: {
        food: first.ateWhileNeurallyCrawling,
        water: first.drankWhileNeurallyCrawling,
      },
      remounted: first.remounted,
      incidentalBoardContact,
    },
    null,
    2,
  ),
)
console.log('Stunt motion verification passed.')

type LocomotionIntervention = 'full' | 'zero' | 'frozen' | 'shuffled'

function verifyIncidentalBoardCollision() {
  const field = createTerrainField(environment)
  const state = createStuntState(field, environment)
  const heading = state.boardHeading
  const deckLength = Math.max(1.65, Math.min(2.1, environment.skateboard.deckSize[0]))
  const approachDistance = deckLength * 0.5 + 0.145
  state.locomotionState = 'crawling'
  state.locomotionTime = 1
  state.mountBlend = 0
  state.boardCollisionCooldown = 0
  state.needs.hunger = 0.95
  state.needs.thirst = 0.1
  state.needs.wellbeing = 0.05
  state.needs.targetResourceId = 'food-bowl'
  state.wormX = state.boardX - Math.cos(heading) * approachDistance
  state.wormZ = state.boardZ - Math.sin(heading) * approachDistance
  state.wormY = field.sample(state.wormX, state.wormZ).height + 0.105
  state.wormHeading = heading
  state.wormVx = Math.cos(heading) * 1.45
  state.wormVz = Math.sin(heading) * 1.45
  state.locomotionPlant.forwardSpeed = 1.45
  state.locomotionPlant.angularSpeed = 0
  state.segments.forEach((segment, index) => {
    segment.x = state.wormX
    segment.y = state.wormY
    segment.z = state.wormZ
    segment.vx = state.wormVx
    segment.vy = 0
    segment.vz = state.wormVz
    segment.pitch = 0
    segment.yaw = heading
    state.segmentGroundContacts[index].anchorX = state.wormX
    state.segmentGroundContacts[index].anchorZ = state.wormZ
    state.segmentGroundContacts[index].strength = 0
  })

  advanceStunt(
    state,
    makeInitialAction(),
    POLICY_TIMESTEP,
    9.81,
    'freestyle',
    field,
    environment,
    [],
  )

  const clearance =
    Math.hypot(state.wormX - state.boardX, state.wormZ - state.boardZ) - deckLength * 0.5 - 0.13
  expect(state.locomotionState === 'crawling', 'incidental board probe unexpectedly entered the mount lifecycle')
  expect(
    clearance >= -1e-6,
    `worm root penetrated the skateboard outside its mount lifecycle (${clearance.toFixed(6)} m)`,
  )
  expect(
    state.obstacleContactId?.startsWith('skateboard-') === true,
    'incidental board probe did not report visible-deck contact',
  )
  return { clearance: round(clearance), contact: state.obstacleContactId }
}

function simulate(
  mode: ShowcaseMode,
  seconds: number,
  intervention: LocomotionIntervention = 'full',
  frictionOverride?: number,
  simulationEnvironment: EnvironmentConfig = environment,
) {
  const originalField = createTerrainField(simulationEnvironment)
  const field = frictionOverride === undefined
    ? originalField
    : {
      ...originalField,
      sample: (x: number, z: number) => ({ ...originalField.sample(x, z), friction: frictionOverride }),
    }
  const state = createStuntState(field, simulationEnvironment)
  const decor = makeTerrariumDecor(
    simulationEnvironment.seed,
    field,
    state.resources,
    simulationEnvironment.terrain.obstacleDensity,
  )
  const policy = new NeuralStuntPolicy(artifact)
  const locomotionPolicy = new EvolvedLocomotionPolicy(locomotionArtifact)
  const appliedAction = makeInitialAction()
  const idleAction = makeInitialAction()
  const intervenedAction = makeInitialAction()
  const frozenAction = makeInitialAction()
  const bodySpeeds: number[] = []
  const locomotionStates: string[] = []
  const boardX: number[] = [state.boardX]
  const boardZ: number[] = [state.boardZ]
  const wormX: number[] = [state.wormX]
  const wormZ: number[] = [state.wormZ]
  const trace: string[] = []
  const obstacleContacts = new Set<string>()
  let minimumStaticObstacleClearance = Number.POSITIVE_INFINITY
  let minimumStaticObstacleContext = ''
  let maximumBodySpeed = 0
  let maximumBodyContext = ''
  let maximumRootStep = 0
  let maximumRootContext = ''
  let maximumSegmentStep = 0
  let maximumSegmentContext = ''
  let maximumSeparation = 0
  let maximumYawHeadingError = 0
  let observationsFinite = true
  let inBounds = true
  let sawDetached = false
  let remounted = false
  let wellbeingFulfillmentAtFirstDetach: number | null = null
  let postRemountWellbeingFulfillment = 0
  let ateWhileNeurallyCrawling = false
  let drankWhileNeurallyCrawling = false
  let locomotionStep = 0
  let neuralActionMax = 0

  for (let step = 0; step < Math.round(seconds / POLICY_TIMESTEP); step += 1) {
    const wasRiding = state.locomotionState === 'riding'
    const observation = snapshotToObservation(toSnapshot(state))
    observationsFinite &&= observation.every(Number.isFinite)
    const locomotionOwnsBody = state.locomotionState === 'crawling' || state.locomotionState === 'seeking'
    const rawAction = state.locomotionState === 'riding'
      ? policy.run(observation)
      : locomotionOwnsBody
        ? locomotionPolicy.run(
        locomotionSensorsFor(state, field),
        state.locomotionPlant.joints,
        state.locomotionPlant.jointVelocities,
      )
        : idleAction
    if (state.locomotionState === 'riding') smoothAction(appliedAction, rawAction, POLICY_TIMESTEP)
    else if (locomotionOwnsBody) {
      for (const activation of rawAction) neuralActionMax = Math.max(neuralActionMax, Math.abs(activation))
      appliedAction.set(interveneLocomotionAction(rawAction, intervention, locomotionStep, frozenAction, intervenedAction))
      locomotionStep += 1
    }
    else appliedAction.fill(0)
    const previousBoardX = state.boardX
    const previousBoardZ = state.boardZ
    const previousWormX = state.wormX
    const previousWormZ = state.wormZ
    const previousWormHeading = state.wormHeading
    const previousSegments = state.segments.map((segment) => [segment.x, segment.y, segment.z] as const)

    advanceStunt(
      state,
      appliedAction,
      POLICY_TIMESTEP,
      9.81,
      mode,
      field,
      simulationEnvironment,
      decor.obstacles,
    )
    if (state.obstacleContactId) obstacleContacts.add(state.obstacleContactId)
    const deckLength = Math.max(1.65, Math.min(2.1, simulationEnvironment.skateboard.deckSize[0]))
    const boardForwardX = Math.cos(state.boardHeading)
    const boardForwardZ = Math.sin(state.boardHeading)
    const boardProbes = [
      {
        x: state.boardX + boardForwardX * deckLength * 0.28,
        z: state.boardZ + boardForwardZ * deckLength * 0.28,
      },
      {
        x: state.boardX - boardForwardX * deckLength * 0.28,
        z: state.boardZ - boardForwardZ * deckLength * 0.28,
      },
    ]
    const detachedRoot =
      state.locomotionState === 'crawling' ||
      state.locomotionState === 'seeking' ||
      state.locomotionState === 'feeding'
    const recordStaticClearance = (clearance: number, context: string) => {
      if (clearance >= minimumStaticObstacleClearance) return
      minimumStaticObstacleClearance = clearance
      minimumStaticObstacleContext = `t=${state.time.toFixed(3)} ${context}`
    }
    for (const obstacle of decor.obstacles) {
      for (const [probeIndex, probe] of boardProbes.entries()) {
        recordStaticClearance(
          Math.hypot(probe.x - obstacle.center.x, probe.z - obstacle.center.z) - obstacle.radius - 0.34,
          `board-${probeIndex} obstacle=${obstacle.id}`,
        )
      }
      if (detachedRoot) {
        recordStaticClearance(
          Math.hypot(state.wormX - obstacle.center.x, state.wormZ - obstacle.center.z) - obstacle.radius - 0.13,
          `worm-root obstacle=${obstacle.id}`,
        )
      }
      for (const [segmentIndex, segment] of state.segments.entries()) {
        recordStaticClearance(
          Math.hypot(segment.x - obstacle.center.x, segment.z - obstacle.center.z) - obstacle.radius - 0.074,
          `segment=${segmentIndex} obstacle=${obstacle.id}`,
        )
      }
    }
    ateWhileNeurallyCrawling ||=
      locomotionOwnsBody &&
      state.locomotionState === 'feeding' &&
      state.feedingResourceId === 'food-bowl'
    drankWhileNeurallyCrawling ||=
      locomotionOwnsBody &&
      state.locomotionState === 'feeding' &&
      state.feedingResourceId === 'water-bowl'
    if (wasRiding && state.locomotionState !== 'riding') {
      locomotionPolicy.reset()
      appliedAction.fill(0)
      wellbeingFulfillmentAtFirstDetach ??= state.needs.fulfillment.wellbeing
      locomotionStep = 0
      frozenAction.fill(0)
    }

    if (locomotionStates.at(-1) !== state.locomotionState) locomotionStates.push(state.locomotionState)
    if (state.locomotionState !== 'riding') sawDetached = true
    if (sawDetached && state.locomotionState === 'riding') {
      remounted = true
      if (wellbeingFulfillmentAtFirstDetach !== null) {
        postRemountWellbeingFulfillment = Math.max(
          postRemountWellbeingFulfillment,
          state.needs.fulfillment.wellbeing - wellbeingFulfillmentAtFirstDetach,
        )
      }
    }
    const boardRootStep = Math.hypot(state.boardX - previousBoardX, state.boardZ - previousBoardZ)
    const wormRootStep = Math.hypot(state.wormX - previousWormX, state.wormZ - previousWormZ)
    const rootStep = Math.max(boardRootStep, wormRootStep)
    if (rootStep > maximumRootStep) {
      maximumRootStep = rootStep
      maximumRootContext =
        `t=${state.time.toFixed(3)} phase=${state.phase} locomotion=${state.locomotionState} ` +
        `board=${boardRootStep.toFixed(4)} worm=${wormRootStep.toFixed(4)} contact=${state.obstacleContactId ?? 'none'}`
    }
    const rootVx = state.mountBlend > 0.5 ? state.boardVx : state.wormVx
    const rootVz = state.mountBlend > 0.5 ? state.boardVz : state.wormVz
    let stepBodySpeed = 0
    let stepBodySegment = 0
    state.segments.forEach((segment, index) => {
      const relativeBodySpeed = Math.hypot(segment.vx - rootVx, segment.vy, segment.vz - rootVz)
      if (relativeBodySpeed > stepBodySpeed) {
        stepBodySpeed = relativeBodySpeed
        stepBodySegment = index
      }
      const segmentStep = Math.hypot(
        segment.x - previousSegments[index][0],
        segment.y - previousSegments[index][1],
        segment.z - previousSegments[index][2],
      )
      if (segmentStep > maximumSegmentStep) {
        maximumSegmentStep = segmentStep
        const previousStaticClearance = Math.min(
          ...decor.obstacles.map(
            (obstacle) =>
              Math.hypot(
                previousSegments[index][0] - obstacle.center.x,
                previousSegments[index][2] - obstacle.center.z,
              ) - obstacle.radius - 0.074,
          ),
        )
        const currentStaticClearance = Math.min(
          ...decor.obstacles.map(
            (obstacle) =>
              Math.hypot(segment.x - obstacle.center.x, segment.z - obstacle.center.z) -
              obstacle.radius - 0.074,
          ),
        )
        const previousResourceClearances = state.resources.map((resource) => ({
          id: resource.id,
          clearance:
            Math.hypot(
              previousSegments[index][0] - resource.position[0],
              previousSegments[index][2] - resource.position[2],
            ) -
            (resource.presentation === 'skateboard' ? 0.6 : resource.appearance.radius * 0.94) -
            0.074,
        }))
        const closestPreviousResource = previousResourceClearances.reduce((closest, resource) =>
          resource.clearance < closest.clearance ? resource : closest,
        )
        maximumSegmentContext =
          `t=${state.time.toFixed(3)} phase=${state.phase} locomotion=${state.locomotionState} ` +
          `segment=${index} mount=${state.mountBlend.toFixed(3)} range=${state.distanceToBoard.toFixed(3)} ` +
          `contact=${state.obstacleContactId ?? 'none'} grip=${state.segmentGroundContacts[index]?.strength.toFixed(3) ?? 'none'} ` +
          `locomotionTime=${state.locomotionTime.toFixed(3)} ` +
          `delta=${(segment.x - previousSegments[index][0]).toFixed(3)}/` +
          `${(segment.y - previousSegments[index][1]).toFixed(3)}/` +
          `${(segment.z - previousSegments[index][2]).toFixed(3)} ` +
          `anchor=${(state.segmentGroundContacts[index]?.anchorX - segment.x).toFixed(3)}/` +
          `${(state.segmentGroundContacts[index]?.anchorZ - segment.z).toFixed(3)} ` +
          `clearance=${previousStaticClearance.toFixed(4)}->${currentStaticClearance.toFixed(4)} ` +
          `resource=${closestPreviousResource.id}:${closestPreviousResource.clearance.toFixed(4)} ` +
          `headingDelta=${wrapAngle(state.wormHeading - previousWormHeading).toFixed(3)}`
      }
    })
    bodySpeeds.push(stepBodySpeed)
    if (stepBodySpeed > maximumBodySpeed) {
      maximumBodySpeed = stepBodySpeed
      maximumBodyContext =
        `t=${state.time.toFixed(3)} phase=${state.phase} locomotion=${state.locomotionState} ` +
        `segment=${stepBodySegment} mount=${state.mountBlend.toFixed(3)} contact=${state.obstacleContactId ?? 'none'}`
    }
    maximumSeparation = Math.max(maximumSeparation, state.distanceToBoard)
    maximumYawHeadingError = Math.max(maximumYawHeadingError, Math.abs(wrapAngle(state.boardYaw + state.boardHeading)))
    boardX.push(state.boardX)
    boardZ.push(state.boardZ)
    wormX.push(state.wormX)
    wormZ.push(state.wormZ)
    const xLimit = field.width * 0.5
    const zLimit = field.depth * 0.5
    const boardFootprintInBounds = boardProbes.every(
      (probe) => Math.abs(probe.x) + 0.34 <= xLimit && Math.abs(probe.z) + 0.34 <= zLimit,
    )
    const segmentFootprintsInBounds = state.segments.every(
      (segment) => Math.abs(segment.x) + 0.074 <= xLimit && Math.abs(segment.z) + 0.074 <= zLimit,
    )
    const detachedRootInBounds =
      !detachedRoot ||
      (Math.abs(state.wormX) + 0.13 <= xLimit && Math.abs(state.wormZ) + 0.13 <= zLimit)
    inBounds &&=
      boardFootprintInBounds &&
      segmentFootprintsInBounds &&
      detachedRootInBounds

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
          state.needs.targetResourceId ?? 'none',
        ]
          .map((value) => (typeof value === 'number' ? value.toFixed(6) : value))
          .join(','),
      )
    }
  }

  const sortedBodySpeeds = bodySpeeds.toSorted((a, b) => a - b)
  const rightX = -Math.sin(state.wormHeading)
  const rightZ = Math.cos(state.wormHeading)
  const finalLaterals = state.segments.map(
    (segment) => (segment.x - state.wormX) * rightX + (segment.z - state.wormZ) * rightZ,
  )
  return {
    ateWhileNeurallyCrawling,
    boardRangeX: range(boardX),
    boardRangeZ: range(boardZ),
    bodySpeedP99: sortedBodySpeeds[Math.min(sortedBodySpeeds.length - 1, Math.floor(sortedBodySpeeds.length * 0.99))],
    crawlDistance: state.wormDistance,
    flipsLanded: state.flipsLanded,
    foodFulfillment: state.needs.fulfillment.hunger,
    finalDistanceToBoard: state.distanceToBoard,
    finalObstacleContact: state.obstacleContactId,
    obstacleContacts: [...obstacleContacts].sort(),
    finalLateralSpan: range(finalLaterals),
    hash: createHash('sha256').update(trace.join('\n')).digest('hex'),
    inBounds,
    locomotionStates,
    maximumBodySpeed,
    maximumBodyContext,
    maximumRootStep,
    maximumRootContext,
    maximumSegmentStep,
    maximumSegmentContext,
    maximumSeparation,
    maximumYawHeadingError,
    minimumStaticObstacleClearance,
    minimumStaticObstacleContext,
    neuralActionMax,
    observationsFinite,
    postRemountWellbeingFulfillment,
    remounted,
    drankWhileNeurallyCrawling,
    waterFulfillment: state.needs.fulfillment.thirst,
    wellbeingFulfillment: state.needs.fulfillment.wellbeing,
    wormRangeX: range(wormX),
    wormRangeZ: range(wormZ),
  }
}

function interveneLocomotionAction(
  rawAction: Float32Array,
  intervention: LocomotionIntervention,
  locomotionStep: number,
  frozenAction: Float32Array,
  output: Float32Array,
) {
  if (intervention === 'zero') {
    output.fill(0)
    return output
  }
  if (intervention === 'frozen') {
    if (locomotionStep < 24) {
      output.set(rawAction)
      if (locomotionStep === 23) frozenAction.set(rawAction)
    } else {
      output.set(frozenAction)
    }
    return output
  }
  if (intervention === 'shuffled') {
    const shuffle = [0, 9, 2, 13, 4, 15, 6, 11, 8, 1, 10, 3, 12, 5, 14, 7]
    for (let segment = 0; segment < shuffle.length; segment += 1) {
      const source = shuffle[segment]
      output[segment * 2] = rawAction[source * 2]
      output[segment * 2 + 1] = rawAction[source * 2 + 1]
    }
    return output
  }
  output.set(rawAction)
  return output
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
