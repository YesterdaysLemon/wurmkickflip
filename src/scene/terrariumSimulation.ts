import { MathUtils } from 'three'
import { deriveWurmAnatomy, type WurmAnatomy } from '../creature/anatomy'
import type { EnvironmentConfig, Vec3 } from '../creature/types'
import type { LocomotionSensors } from '../policy/locomotionPolicy'
import { makeInitialAction } from '../policy/simulationAdapter'
import {
  SEGMENT_COUNT,
  type PolicyAction,
  type SegmentSnapshot,
  type SimulationSnapshot,
} from '../policy/types'
import {
  advanceNeeds,
  createNeedsState,
  createTerrariumResources,
  syncSkateboardResource,
  type NeedsState,
  type TerrariumResource,
} from './terrariumNeeds'
import {
  createWormLocomotionPlant,
  deriveWormLocalPose,
  resetWormLocomotionPlant,
  type WormLocomotionPlant,
} from './wormLocomotion'
import { stepArticulatedWorm } from './wormDynamics'
import {
  sampleWormInteractionAnimation,
  type WormInteractionAnimationSample,
  type WormInteractionKind,
} from './wormInteractionAnimation'
import {
  createTerrariumArenaBounds,
  prepareTerrariumCollisionWorld,
  resolveSweptTerrariumMotion,
  type TerrariumCircleObstacle,
  type TerrariumCollisionWorld,
} from './terrariumCollisions'
import type { TerrainField } from './terrainField'

export type ShowcaseMode = 'kickflip' | 'freestyle'

export type StuntPhase =
  | 'terrarium cruise'
  | 'coil'
  | 'pop'
  | 'kickflip'
  | 'landing'
  | 'victory wiggle'
  | 'dismount'
  | 'ground crawl'
  | 'seeking food'
  | 'seeking water'
  | 'eating'
  | 'drinking'
  | 'finding board'
  | 'mounting up'
  | 'free crawl'

export type LocomotionState = 'riding' | 'dismounting' | 'crawling' | 'seeking' | 'mounting' | 'feeding'

type SegmentGroundContact = {
  anchorX: number
  anchorZ: number
  strength: number
}

type BowlRimAperture = {
  resourceId: 'food-bowl' | 'water-bowl'
  approachAngle: number
}

export type StuntState = {
  time: number
  cycleTime: number
  locomotionState: LocomotionState
  locomotionTime: number
  mountBlend: number
  rideLandings: number
  boardX: number
  boardY: number
  boardZ: number
  boardVx: number
  boardVy: number
  boardVz: number
  boardPitch: number
  boardRoll: number
  boardYaw: number
  boardHeading: number
  boardSpeed: number
  rollVelocity: number
  wheelSpin: number
  distance: number
  reward: number
  grounded: boolean
  coilMemory: number
  contactRatio: number
  currentAirtime: number
  lastAirtime: number
  maxHeight: number
  flipProgress: number
  flipsLanded: number
  landingQuality: number
  attempt: number
  landingFlash: number
  poke: number
  wormX: number
  wormY: number
  wormZ: number
  wormVx: number
  wormVz: number
  wormHeading: number
  wormDistance: number
  boardWaypointIndex: number
  terrainFriction: number
  distanceToBoard: number
  feedingResourceId: 'food-bowl' | 'water-bowl' | null
  feedingReleaseResourceId: 'food-bowl' | 'water-bowl' | null
  feedingCooldown: number
  bowlRimAperture: BowlRimAperture | null
  boardCollisionCooldown: number
  boardCollisionReleasePending: boolean
  transitionStartX: number
  transitionStartZ: number
  transitionStartHeading: number
  dismountSide: -1 | 1
  segmentGroundContacts: SegmentGroundContact[]
  obstacleContactId: string | null
  collisionCount: number
  needs: NeedsState
  resources: TerrariumResource[]
  locomotionPlant: WormLocomotionPlant
  phase: StuntPhase
  segments: SegmentSnapshot[]
  previousAction: PolicyAction
}

export type DecodedAction = {
  bends: number[]
  contractions: number[]
  propulsion: number
  coil: number
  release: number
  kick: number
  energy: number
}

const CYCLE_SECONDS = 7.2
const BOARD_CLEARANCE = 0.28
const DISMOUNT_SECONDS = 1.2
const MOUNT_SECONDS = 1.8
const FEED_SECONDS = 2.65
const ARENA_MARGIN = 0.82
const SEGMENT_SMOOTHING = 9
const ROTATION_SMOOTHING = 10

export function createStuntState(
  field: TerrainField,
  environmentConfig: EnvironmentConfig | null = null,
): StuntState {
  const start = field.waypoints[0] ?? [-2.8, -1.8]
  const next = field.waypoints[1] ?? [2.4, -1.5]
  const boardHeading = Math.atan2(next[1] - start[1], next[0] - start[0])
  const boardY = boardGroundY(field, start[0], start[1])
  const resources = createTerrariumResources(field, environmentConfig?.seed ?? 1337, [
    start[0],
    boardY,
    start[1],
  ])
  return {
    time: 0,
    cycleTime: 0,
    locomotionState: 'riding',
    locomotionTime: 0,
    mountBlend: 1,
    rideLandings: 0,
    boardX: start[0],
    boardY,
    boardZ: start[1],
    boardVx: Math.cos(boardHeading) * 0.7,
    boardVy: 0,
    boardVz: Math.sin(boardHeading) * 0.7,
    boardPitch: 0,
    boardRoll: 0,
    boardYaw: -boardHeading,
    boardHeading,
    boardSpeed: 0.7,
    rollVelocity: 0,
    wheelSpin: 0,
    distance: 0,
    reward: 0,
    grounded: true,
    coilMemory: 0,
    contactRatio: 1,
    currentAirtime: 0,
    lastAirtime: 0,
    maxHeight: 0,
    flipProgress: 0,
    flipsLanded: 0,
    landingQuality: 0,
    attempt: 0,
    landingFlash: 0,
    poke: 0,
    wormX: start[0],
    wormY: boardY + 0.18,
    wormZ: start[1],
    wormVx: 0,
    wormVz: 0,
    wormHeading: boardHeading,
    wormDistance: 0,
    boardWaypointIndex: Math.min(1, Math.max(0, field.waypoints.length - 1)),
    terrainFriction: field.sample(start[0], start[1]).friction,
    distanceToBoard: 0,
    feedingResourceId: null,
    feedingReleaseResourceId: null,
    feedingCooldown: 0,
    bowlRimAperture: null,
    boardCollisionCooldown: 0,
    boardCollisionReleasePending: false,
    transitionStartX: start[0],
    transitionStartZ: start[1],
    transitionStartHeading: boardHeading,
    dismountSide: 1,
    segmentGroundContacts: Array.from({ length: SEGMENT_COUNT }, () => ({
      anchorX: start[0],
      anchorZ: start[1],
      strength: 0,
    })),
    obstacleContactId: null,
    collisionCount: 0,
    needs: createNeedsState(environmentConfig?.seed ?? 1337),
    resources,
    locomotionPlant: createWormLocomotionPlant(),
    phase: 'terrarium cruise',
    segments: Array.from({ length: SEGMENT_COUNT }, (_, index) =>
      makeSegment(index, start[0], boardY, start[1], boardHeading),
    ),
    previousAction: makeInitialAction(),
  }
}

function makeSegment(
  index: number,
  centerX: number,
  centerY: number,
  centerZ: number,
  heading: number,
): SegmentSnapshot {
  const axial = (index / (SEGMENT_COUNT - 1) - 0.5) * 1.35
  return {
    x: centerX + Math.cos(heading) * axial,
    y: centerY + 0.2,
    z: centerZ + Math.sin(heading) * axial,
    vx: 0,
    vy: 0,
    vz: 0,
    pitch: 0,
    yaw: heading,
  }
}

export function advanceStunt(
  state: StuntState,
  action: PolicyAction,
  delta: number,
  gravity: number,
  mode: ShowcaseMode,
  field: TerrainField,
  environmentConfig: EnvironmentConfig | null,
  staticObstacles: readonly TerrariumCircleObstacle[] = [],
  anatomy: WurmAnatomy = deriveWurmAnatomy(null),
) {
  state.time += delta
  state.locomotionTime += delta
  const decoded = decodeAction(action)
  state.previousAction = action
  state.poke = Math.max(0, state.poke - delta * 0.58)
  state.landingFlash = Math.max(0, state.landingFlash - delta * 1.35)
  state.feedingCooldown = Math.max(0, state.feedingCooldown - delta)
  state.boardCollisionCooldown = Math.max(0, state.boardCollisionCooldown - delta)
  if (
    state.boardCollisionReleasePending &&
    state.boardCollisionCooldown <= 0 &&
    !wormBodyOverlapsBoardCollider(state, environmentConfig)
  ) {
    state.boardCollisionReleasePending = false
  }
  if (
    state.feedingCooldown <= 0 &&
    state.feedingReleaseResourceId &&
    !wormBodyTouchesResource(state, state.feedingReleaseResourceId)
  ) {
    state.feedingReleaseResourceId = null
  }
  state.obstacleContactId = null

  state.resources = syncSkateboardResource(state.resources, field, [state.boardX, state.boardY, state.boardZ])
  const head = state.segments.at(-1)
  const needResources =
    mode === 'freestyle' ? state.resources.filter(resource => resource.id !== 'skateboard') : state.resources
  const needsStep = advanceNeeds(state.needs, delta, {
    resources: needResources,
    wormPosition: [head?.x ?? state.wormX, head?.y ?? state.wormY, head?.z ?? state.wormZ],
    mouthPosition: wormMouthContactPoint(state),
    mouthRadius: 0.052,
    mounted: state.mountBlend > 0.92,
  })
  state.needs = needsStep.state

  updateLocomotionLifecycle(state, mode, environmentConfig, staticObstacles, needsStep.interaction)

  if (state.locomotionState === 'riding') {
    state.cycleTime += delta
    if (state.cycleTime >= CYCLE_SECONDS) {
      state.cycleTime %= CYCLE_SECONDS
      state.coilMemory = 0
      state.flipProgress = 0
      if (!state.grounded) settleBoard(state, field)
    }
  } else {
    // The authored stunt clock is inactive while the evolved crawl brain owns
    // the body. Detached motion has no gait phase or animation clock.
    state.cycleTime = 0
  }

  updateBoardPlanar(state, decoded, delta, field, environmentConfig, staticObstacles)
  state.resources = syncSkateboardResource(state.resources, field, [state.boardX, state.boardY, state.boardZ])
  updateWormRoot(state, action, delta, field, environmentConfig, staticObstacles, anatomy)
  const groundY = boardGroundY(field, state.boardX, state.boardZ)
  const terrain = field.sample(state.boardX, state.boardZ)

  if (state.grounded) {
    state.boardY = groundY - (state.locomotionState === 'riding' ? decoded.coil * 0.028 : 0)
    const forwardSlope =
      terrain.normal[0] * Math.cos(state.boardHeading) + terrain.normal[2] * Math.sin(state.boardHeading)
    const terrainPitch = Math.atan2(-forwardSlope, Math.max(0.2, terrain.normal[1]))
    state.boardPitch = MathUtils.damp(
      state.boardPitch,
      terrainPitch + (state.locomotionState === 'riding' ? decoded.coil * -0.095 : 0) + state.poke * 0.025,
      7,
      delta,
    )
    state.boardRoll = MathUtils.damp(state.boardRoll, state.poke * Math.sin(state.time * 9) * 0.14, 7, delta)
    // Grounded collision probes and the rendered deck share one orientation.
    // Smoothing this value independently made the visible board lag behind its
    // footprint after an obstacle deflection.
    state.boardYaw = -state.boardHeading
    state.coilMemory = Math.max(state.coilMemory * 0.992, decoded.coil)
    const semanticContact =
      state.cycleTime < 2.65 ? 0.84 : state.cycleTime < 2.9 ? 0.24 : state.cycleTime < 3.75 ? 0.12 : 0.72
    state.contactRatio =
      state.locomotionState === 'riding'
        ? MathUtils.clamp(semanticContact - state.poke * 0.2 - decoded.coil * 0.025, 0.08, 0.9)
        : state.locomotionPlant.contactRatio

    const scriptedPop =
      mode === 'kickflip' &&
      state.locomotionState === 'riding' &&
      state.mountBlend > 0.985 &&
      state.cycleTime > 2.72 &&
      state.cycleTime < 2.82

    if (scriptedPop) launchKickflip(state, gravity)
  } else {
    state.currentAirtime += delta
    state.boardVy -= gravity * delta
    state.boardY += state.boardVy * delta
    state.boardRoll += state.rollVelocity * delta
    state.rollVelocity *= Math.pow(0.9985, delta * 60)
    state.boardPitch = MathUtils.damp(
      state.boardPitch,
      Math.sin(state.currentAirtime * Math.PI) * -0.12,
      4.5,
      delta,
    )
    state.boardYaw = -state.boardHeading + decoded.kick * 0.045
    state.maxHeight = Math.max(state.maxHeight, state.boardY - groundY)
    state.flipProgress = MathUtils.clamp(Math.abs(state.boardRoll) / (Math.PI * 2), 0, 1)
    state.contactRatio = MathUtils.clamp(0.14 + decoded.coil * 0.28, 0.08, 0.45)

    if (state.boardY <= groundY && state.boardVy < 0) landKickflip(state, field)
  }

  updateSegments(state, decoded, delta, field, environmentConfig, staticObstacles, anatomy)
  state.phase = phaseFor(state, mode)
  state.reward =
    state.distance * 0.62 +
    state.wormDistance * 0.3 +
    state.flipsLanded * 24 +
    state.landingQuality * 6 +
    state.contactRatio * 1.8 -
    decoded.energy * 0.18
}

function updateLocomotionLifecycle(
  state: StuntState,
  mode: ShowcaseMode,
  environmentConfig: EnvironmentConfig | null,
  staticObstacles: readonly TerrariumCircleObstacle[],
  substantialFeedingContact: NeedsState['lastInteraction'],
) {
  if (state.locomotionState === 'riding') {
    state.mountBlend = 1
    const clearToDismount =
      state.resources.every(
        resource =>
          resource.presentation !== 'bowl' ||
          Math.hypot(state.boardX - resource.position[0], state.boardZ - resource.position[2]) >
            resource.appearance.radius + 1.05,
      ) &&
      staticObstacles.every(
        obstacle =>
          Math.hypot(state.boardX - obstacle.center.x, state.boardZ - obstacle.center.z) >
          obstacle.radius + 1.35,
      )
    if (
      mode === 'freestyle' ||
      (clearToDismount && state.rideLandings > 0 && state.cycleTime > 5.55 && state.needs.wellbeing < 0.08)
    ) {
      transitionLocomotion(state, 'dismounting')
    }
    return
  }

  if (state.locomotionState === 'dismounting') {
    const contact = interactionSampleFor(state).contact
    state.mountBlend = (contact.headWeight + contact.midbodyWeight + contact.tailWeight) / 3
    if (state.locomotionTime >= DISMOUNT_SECONDS) {
      state.mountBlend = 0
      transitionLocomotion(state, 'crawling')
    }
    return
  }

  if (state.locomotionState === 'crawling') {
    state.mountBlend = 0
    if (
      state.feedingCooldown <= 0 &&
      state.feedingReleaseResourceId === null &&
      (substantialFeedingContact === 'hunger' || substantialFeedingContact === 'thirst')
    ) {
      state.feedingResourceId = substantialFeedingContact === 'hunger' ? 'food-bowl' : 'water-bowl'
      transitionLocomotion(state, 'feeding')
      return
    }
    if (mode === 'kickflip' && state.needs.targetResourceId === 'skateboard') {
      transitionLocomotion(state, 'seeking')
    }
    return
  }

  if (state.locomotionState === 'feeding') {
    state.mountBlend = 0
    if (state.locomotionTime >= FEED_SECONDS) {
      state.feedingReleaseResourceId = state.feedingResourceId
      state.feedingResourceId = null
      state.feedingCooldown = 1.1
      transitionLocomotion(
        state,
        mode === 'kickflip' && state.needs.targetResourceId === 'skateboard' ? 'seeking' : 'crawling',
      )
    }
    return
  }

  if (state.locomotionState === 'seeking') {
    state.mountBlend = 0
    // Once well-being wins selection, commit to the board long enough to make
    // physical contact. Faster-growing food/water urgencies may queue next,
    // but they should not make the worm oscillate between distant resources.
    state.needs.targetResourceId = 'skateboard'
    const discoveryRadius = environmentConfig?.skateboard.discoveryRadius ?? 1.35
    const closestBodyDistance = state.segments.reduce(
      (closest, segment) => Math.min(closest, Math.hypot(state.boardX - segment.x, state.boardZ - segment.z)),
      state.distanceToBoard,
    )
    if (closestBodyDistance < Math.max(0.48, discoveryRadius * 0.55)) {
      transitionLocomotion(state, 'mounting')
    }
    return
  }

  const contact = interactionSampleFor(state).contact
  state.mountBlend = (contact.headWeight + contact.midbodyWeight + contact.tailWeight) / 3
  if (state.locomotionTime >= MOUNT_SECONDS) {
    state.mountBlend = 1
    state.rideLandings = 0
    state.cycleTime = 0
    state.coilMemory = 0
    state.flipProgress = 0
    state.wormX = state.boardX
    state.wormZ = state.boardZ
    state.wormHeading = state.boardHeading
    transitionLocomotion(state, 'riding')
  }
}

function transitionLocomotion(state: StuntState, next: LocomotionState) {
  const previous = state.locomotionState
  state.transitionStartX = state.wormX
  state.transitionStartZ = state.wormZ
  state.transitionStartHeading = state.wormHeading
  state.locomotionState = next
  state.locomotionTime = 0
  if (previous === 'dismounting' && next === 'crawling') {
    state.boardCollisionCooldown = 1.2
    state.boardCollisionReleasePending = true
    resetWormLocomotionPlant(state.locomotionPlant)
    state.wormVx = 0
    state.wormVz = 0
    state.segments.forEach(segment => {
      segment.vx = 0
      segment.vy = 0
      segment.vz = 0
    })
    const overlappedBowl = state.resources.find(
      resource => resource.presentation === 'bowl' && wormBodyTouchesResource(state, resource.id),
    )
    if (overlappedBowl) {
      // An authored dismount may descend through the hollow center of a nearby
      // bowl. Defer its planar rim collider until the chain has naturally
      // cleared it; enabling a newly intersecting collider would create an
      // unphysical hand-off impulse in the free articulated plant.
      state.feedingReleaseResourceId = overlappedBowl.id as 'food-bowl' | 'water-bowl'
      state.feedingCooldown = Math.max(state.feedingCooldown, 0.8)
    }
  }
  if (next === 'dismounting') {
    state.dismountSide = state.attempt % 2 === 0 ? 1 : -1
    state.flipProgress = 0
    state.currentAirtime = 0
  }
  if (next === 'riding') {
    state.boardCollisionReleasePending = false
    resetWormLocomotionPlant(state.locomotionPlant)
    state.segmentGroundContacts.forEach(contact => {
      contact.strength = 0
    })
  }
}

export function interactionSampleFor(state: StuntState): WormInteractionAnimationSample {
  const kind: WormInteractionKind =
    state.locomotionState === 'mounting'
      ? 'mounting'
      : state.locomotionState === 'dismounting'
        ? 'dismounting'
        : state.feedingResourceId === 'water-bowl'
          ? 'drinking'
          : 'eating'
  const durationSeconds =
    kind === 'mounting' ? MOUNT_SECONDS : kind === 'dismounting' ? DISMOUNT_SECONDS : FEED_SECONDS
  return sampleWormInteractionAnimation({
    kind,
    elapsedSeconds: state.locomotionTime,
    durationSeconds,
    segmentCount: SEGMENT_COUNT,
    side: state.dismountSide,
  })
}

function contactWeightForSegment(interaction: WormInteractionAnimationSample, posteriorToAnterior: number) {
  if (posteriorToAnterior <= 0.5) {
    return MathUtils.lerp(
      interaction.contact.tailWeight,
      interaction.contact.midbodyWeight,
      smootherStep(posteriorToAnterior * 2),
    )
  }
  return MathUtils.lerp(
    interaction.contact.midbodyWeight,
    interaction.contact.headWeight,
    smootherStep((posteriorToAnterior - 0.5) * 2),
  )
}

function updateBoardPlanar(
  state: StuntState,
  decoded: DecodedAction,
  delta: number,
  field: TerrainField,
  environmentConfig: EnvironmentConfig | null,
  staticObstacles: readonly TerrariumCircleObstacle[],
) {
  const terrain = field.sample(state.boardX, state.boardZ)
  const oldBoardHeading = state.boardHeading
  const halfWidth = field.width * 0.5 - ARENA_MARGIN
  const halfDepth = field.depth * 0.5 - ARENA_MARGIN
  const wheelFriction = environmentConfig?.skateboard.wheelFriction ?? 0.82
  const boardMass = environmentConfig?.skateboard.mass ?? 1.1
  const traction = MathUtils.clamp((terrain.friction * wheelFriction) / Math.sqrt(boardMass), 0.3, 1.45)
  let desiredHeading = state.boardHeading
  let edgeBlend = 0

  if (state.locomotionState === 'riding') {
    const waypoint = field.waypoints[state.boardWaypointIndex % field.waypoints.length] ?? [0, 0]
    const waypointDistance = Math.hypot(waypoint[0] - state.boardX, waypoint[1] - state.boardZ)
    if (waypointDistance < 0.72)
      state.boardWaypointIndex = (state.boardWaypointIndex + 1) % field.waypoints.length
    const activeWaypoint = field.waypoints[state.boardWaypointIndex % field.waypoints.length] ?? [0, 0]
    desiredHeading = Math.atan2(activeWaypoint[1] - state.boardZ, activeWaypoint[0] - state.boardX)

    const edgeTurnZone = 1.68
    const edgeX = Math.max(0, Math.abs(state.boardX) - (halfWidth - edgeTurnZone))
    const edgeZ = Math.max(0, Math.abs(state.boardZ) - (halfDepth - edgeTurnZone))
    edgeBlend = MathUtils.clamp(Math.max(edgeX, edgeZ) / edgeTurnZone, 0, 1)
    if (edgeBlend > 0) {
      const inwardHeading = Math.atan2(-state.boardZ, -state.boardX)
      desiredHeading = lerpAngle(desiredHeading, inwardHeading, edgeBlend)
    }
  }

  const headingRate =
    state.locomotionState === 'riding' && state.grounded ? 0.72 + traction * 0.38 + edgeBlend * 4.2 : 0.18
  const requestedHeading = dampAngle(state.boardHeading, desiredHeading, headingRate, delta)
  const boardCollisionWorld = collisionWorldFor(
    field,
    staticObstacles,
    state.resources,
    ['skateboard'],
    environmentConfig,
    oldBoardHeading,
  )
  state.boardHeading = collisionFreeBoardHeading(
    oldBoardHeading,
    requestedHeading,
    state.boardX,
    state.boardY,
    state.boardZ,
    boardCollisionWorld,
    environmentConfig,
  )
  const terrainResistance = 1 / (0.84 + terrain.friction * 0.18)
  const targetSpeed =
    state.locomotionState === 'riding' ? (0.5 + decoded.propulsion * 0.7) * terrainResistance : 0
  state.boardSpeed = MathUtils.damp(
    state.boardSpeed,
    targetSpeed,
    state.locomotionState === 'riding' ? 2.2 + traction : 0.7 + terrain.friction,
    delta,
  )

  const targetVx = Math.cos(state.boardHeading) * state.boardSpeed
  const targetVz = Math.sin(state.boardHeading) * state.boardSpeed
  const velocityResponse = state.grounded ? 1.4 + traction * 2.2 : 0.25
  state.boardVx = MathUtils.damp(state.boardVx, targetVx, velocityResponse, delta)
  state.boardVz = MathUtils.damp(state.boardVz, targetVz, velocityResponse, delta)
  const oldBoardX = state.boardX
  const oldBoardZ = state.boardZ
  state.boardX += state.boardVx * delta
  state.boardZ += state.boardVz * delta

  const boardForwardX = Math.cos(state.boardHeading)
  const boardForwardZ = Math.sin(state.boardHeading)
  const incomingBoardSpeed = Math.hypot(state.boardVx, state.boardVz)
  const boardBody = skateboardFootprintObstacles(
    state.boardX,
    state.boardY,
    state.boardZ,
    state.boardHeading,
    environmentConfig,
  ).map(sample => ({
    id: `board-${sample.id}`,
    offset: {
      x: sample.center.x - state.boardX,
      z: sample.center.z - state.boardZ,
    },
    radius: sample.radius,
  }))
  const boardCollision = resolveSweptTerrariumMotion(boardCollisionWorld, {
    position: { x: oldBoardX, z: oldBoardZ },
    displacement: { x: state.boardX - oldBoardX, z: state.boardZ - oldBoardZ },
    velocity: { x: state.boardVx, z: state.boardVz },
    body: boardBody,
    ground: {
      grounded: state.grounded,
      friction: terrain.friction * wheelFriction,
      normalY: terrain.normal[1],
      contactRatio: state.contactRatio,
    },
  })
  state.boardX = boardCollision.position.x
  state.boardZ = boardCollision.position.z
  state.boardVx = boardCollision.velocity.x
  state.boardVz = boardCollision.velocity.z
  if (boardCollision.contacts.length > 0) {
    state.obstacleContactId = boardCollision.contacts[0]?.id ?? null
    state.collisionCount += boardCollision.contacts.length
    const contact = boardCollision.contacts[0]
    const collisionSpeed = Math.hypot(state.boardVx, state.boardVz)
    if (collisionSpeed > 0.025) {
      state.boardSpeed = collisionSpeed
      const deflectedHeading = dampAngle(
        state.boardHeading,
        Math.atan2(state.boardVz, state.boardVx),
        6,
        delta,
      )
      state.boardHeading = collisionFreeBoardHeading(
        state.boardHeading,
        deflectedHeading,
        state.boardX,
        state.boardY,
        state.boardZ,
        boardCollisionWorld,
        environmentConfig,
      )
    } else if (contact) {
      const tangentA = { x: -contact.normal.z, z: contact.normal.x }
      const tangentB = { x: -tangentA.x, z: -tangentA.z }
      const tangent = tangentA.x * boardForwardX + tangentA.z * boardForwardZ >= 0 ? tangentA : tangentB
      const deflectionSpeed = Math.max(0.08, incomingBoardSpeed * 0.42)
      state.boardVx = tangent.x * deflectionSpeed
      state.boardVz = tangent.z * deflectionSpeed
      state.boardSpeed = deflectionSpeed
      const deflectedHeading = dampAngle(state.boardHeading, Math.atan2(tangent.z, tangent.x), 6, delta)
      state.boardHeading = collisionFreeBoardHeading(
        state.boardHeading,
        deflectedHeading,
        state.boardX,
        state.boardY,
        state.boardZ,
        boardCollisionWorld,
        environmentConfig,
      )
      state.boardWaypointIndex = (state.boardWaypointIndex + 1) % Math.max(1, field.waypoints.length)
    }
  }

  const planarSpeed = Math.hypot(state.boardVx, state.boardVz)
  if (state.locomotionState === 'riding') state.distance += planarSpeed * delta
  state.wheelSpin -= planarSpeed * delta * 8.4
}

function updateWormRoot(
  state: StuntState,
  action: PolicyAction,
  delta: number,
  field: TerrainField,
  environmentConfig: EnvironmentConfig | null,
  staticObstacles: readonly TerrariumCircleObstacle[],
  anatomy: WurmAnatomy,
) {
  if (state.locomotionState === 'riding') {
    state.wormX = MathUtils.damp(state.wormX, state.boardX, 18, delta)
    state.wormZ = MathUtils.damp(state.wormZ, state.boardZ, 18, delta)
    state.wormHeading = dampAngle(state.wormHeading, state.boardHeading, 18, delta)
    state.wormVx = state.boardVx
    state.wormVz = state.boardVz
  } else if (state.locomotionState === 'dismounting') {
    const interaction = interactionSampleFor(state)
    const progress = smoothStep(interaction.progress)
    const forwardX = Math.cos(state.boardHeading)
    const forwardZ = Math.sin(state.boardHeading)
    const rightX = -forwardZ
    const rightZ = forwardX
    const sideOffset = interaction.side * (0.2 + progress * 0.78) + interaction.root.lateral
    const targetX =
      MathUtils.lerp(state.transitionStartX, state.boardX, progress) +
      forwardX * interaction.root.forward +
      rightX * sideOffset
    const targetZ =
      MathUtils.lerp(state.transitionStartZ, state.boardZ, progress) +
      forwardZ * interaction.root.forward +
      rightZ * sideOffset
    const oldX = state.wormX
    const oldZ = state.wormZ
    state.wormX = MathUtils.damp(state.wormX, targetX, 8.5, delta)
    state.wormZ = MathUtils.damp(state.wormZ, targetZ, 8.5, delta)
    state.wormHeading = dampAngle(
      state.wormHeading,
      lerpAngle(state.transitionStartHeading, state.boardHeading + interaction.side * 0.62, progress) +
        interaction.root.yaw,
      7,
      delta,
    )
    state.wormVx = (state.wormX - oldX) / delta
    state.wormVz = (state.wormZ - oldZ) / delta
  } else if (state.locomotionState === 'mounting') {
    const interaction = interactionSampleFor(state)
    const progress = smoothStep(interaction.progress)
    const forwardX = Math.cos(state.boardHeading)
    const forwardZ = Math.sin(state.boardHeading)
    const rightX = -forwardZ
    const rightZ = forwardX
    const targetX =
      MathUtils.lerp(state.transitionStartX, state.boardX, progress) +
      forwardX * interaction.root.forward +
      rightX * interaction.root.lateral
    const targetZ =
      MathUtils.lerp(state.transitionStartZ, state.boardZ, progress) +
      forwardZ * interaction.root.forward +
      rightZ * interaction.root.lateral
    const oldX = state.wormX
    const oldZ = state.wormZ
    state.wormX = MathUtils.damp(state.wormX, targetX, 7.2, delta)
    state.wormZ = MathUtils.damp(state.wormZ, targetZ, 7.2, delta)
    state.wormHeading = dampAngle(
      state.wormHeading,
      lerpAngle(state.transitionStartHeading, state.boardHeading, progress) + interaction.root.yaw,
      6.5,
      delta,
    )
    state.wormVx = (state.wormX - oldX) / delta
    state.wormVz = (state.wormZ - oldZ) / delta
  } else if (state.locomotionState === 'feeding') {
    const resource = state.feedingResourceId
      ? state.resources.find(candidate => candidate.id === state.feedingResourceId)
      : null
    if (resource) {
      state.wormHeading = dampAngle(
        state.wormHeading,
        Math.atan2(resource.position[2] - state.wormZ, resource.position[0] - state.wormX),
        7.5,
        delta,
      )
    }
    state.locomotionPlant.forwardSpeed = MathUtils.damp(state.locomotionPlant.forwardSpeed, 0, 14, delta)
    state.locomotionPlant.angularSpeed = MathUtils.damp(state.locomotionPlant.angularSpeed, 0, 14, delta)
    state.wormVx = MathUtils.damp(state.wormVx, 0, 18, delta)
    state.wormVz = MathUtils.damp(state.wormVz, 0, 18, delta)
  } else {
    const omittedResources: TerrariumResource['id'][] = []
    if (
      state.boardCollisionCooldown > 0 ||
      state.boardCollisionReleasePending ||
      state.locomotionState === 'seeking'
    ) {
      omittedResources.push('skateboard')
    }
    if (
      state.feedingReleaseResourceId &&
      (state.feedingCooldown > 0 || wormBodyTouchesResource(state, state.feedingReleaseResourceId))
    ) {
      omittedResources.push(state.feedingReleaseResourceId)
    }
    const bowlRimAperture = activeBowlRimAperture(state)
    const movement = stepArticulatedWorm(
      state.locomotionPlant,
      state.segments,
      action,
      delta,
      field,
      collisionWorldFor(
        field,
        staticObstacles,
        state.resources,
        omittedResources,
        environmentConfig,
        state.boardHeading,
        bowlRimAperture,
      ),
      anatomy,
    )
    state.wormX = movement.root.x
    state.wormY = movement.root.y
    state.wormZ = movement.root.z
    state.wormVx = movement.root.vx
    state.wormVz = movement.root.vz
    state.wormHeading = movement.root.heading
    state.wormDistance += movement.distance
    state.contactRatio = movement.contactRatio
    state.collisionCount += movement.collisionCount
    state.obstacleContactId = movement.contacts[0]?.obstacleId ?? null
    state.terrainFriction = field.sample(state.wormX, state.wormZ).friction
  }

  const wormTerrain = field.sample(state.wormX, state.wormZ)
  state.wormY = wormTerrain.height + 0.115
  state.distanceToBoard = Math.hypot(state.boardX - state.wormX, state.boardZ - state.wormZ)
}

function launchKickflip(state: StuntState, gravity: number) {
  state.grounded = false
  state.attempt += 1
  state.boardY += 0.015
  state.boardVy = 4.18
  const predictedFlight = (state.boardVy * 2) / Math.max(1, gravity)
  // The aerial stunt is intentionally authored; detached travel is the evolved,
  // causal behavior under test. Finish one turn at the sampled substrate.
  state.rollVelocity = (Math.PI * 2 * 1.035) / predictedFlight
  state.currentAirtime = 0
  state.lastAirtime = 0
  state.maxHeight = 0
  state.flipProgress = 0
  state.landingQuality = 0
  state.coilMemory = 0
}

function landKickflip(state: StuntState, field: TerrainField) {
  const turnCount = Math.abs(state.boardRoll) / (Math.PI * 2)
  const turnError = Math.abs(turnCount - 1)
  const wrappedRoll = Math.abs(wrapAngle(state.boardRoll))
  const verticalPenalty = MathUtils.clamp(Math.abs(state.boardVy + 4.4) / 9, 0, 0.25)
  const quality = MathUtils.clamp(1 - turnError * 2.6 - wrappedRoll * 0.22 - verticalPenalty, 0, 1)
  const landed = turnCount > 0.76 && turnCount < 1.24 && wrappedRoll < 0.78

  state.boardY = boardGroundY(field, state.boardX, state.boardZ)
  state.boardVy = 0
  state.boardRoll = 0
  state.rollVelocity = 0
  state.grounded = true
  state.lastAirtime = state.currentAirtime
  state.currentAirtime = 0
  state.landingQuality = quality
  state.flipProgress = landed ? 1 : MathUtils.clamp(turnCount, 0, 1)
  state.contactRatio = landed ? 0.94 : 0.58
  state.landingFlash = landed ? 1 : 0.35
  if (landed) {
    state.flipsLanded += 1
    state.rideLandings += 1
  }
}

function settleBoard(state: StuntState, field: TerrainField) {
  state.grounded = true
  state.boardY = boardGroundY(field, state.boardX, state.boardZ)
  state.boardVy = 0
  state.boardRoll = 0
  state.rollVelocity = 0
  state.currentAirtime = 0
  state.contactRatio = 1
}

function updateSegments(
  state: StuntState,
  decoded: DecodedAction,
  delta: number,
  field: TerrainField,
  environmentConfig: EnvironmentConfig | null,
  staticObstacles: readonly TerrariumCircleObstacle[],
  anatomy: WurmAnatomy,
) {
  if ((['crawling', 'seeking'] as readonly LocomotionState[]).includes(state.locomotionState)) {
    state.contactRatio = state.locomotionPlant.contactRatio
    return
  }
  const airborneTuck = state.grounded
    ? 0
    : MathUtils.clamp(Math.sin((state.currentAirtime / 0.92) * Math.PI), 0, 1)
  const victoryProgress = MathUtils.clamp((state.cycleTime - 4.1) / (5.25 - 4.1), 0, 1)
  const victory = state.cycleTime > 4.1 && state.cycleTime < 5.25 ? Math.sin(victoryProgress * Math.PI) : 0
  const poke = state.poke
  const lengthScale = 1 - airborneTuck * 0.3 - decoded.coil * 0.1
  const centerY = state.boardY + 0.18 + airborneTuck * 0.18 + victory * 0.07
  const mount = smoothStep(state.mountBlend)
  const interaction =
    state.locomotionState === 'mounting' ||
    state.locomotionState === 'dismounting' ||
    state.locomotionState === 'feeding'
      ? interactionSampleFor(state)
      : null
  const feedingResource = state.feedingResourceId
    ? state.resources.find(resource => resource.id === state.feedingResourceId)
    : null
  const boardForwardX = Math.cos(state.boardHeading)
  const boardForwardZ = Math.sin(state.boardHeading)
  const boardRightX = -boardForwardZ
  const boardRightZ = boardForwardX
  const wormForwardX = Math.cos(state.wormHeading)
  const wormForwardZ = Math.sin(state.wormHeading)
  const wormRightX = -wormForwardZ
  const wormRightZ = wormForwardX
  const poseSmoothing = interaction
    ? 4.5
    : !state.grounded
      ? SEGMENT_SMOOTHING
      : state.obstacleContactId
        ? 5.2
        : SEGMENT_SMOOTHING
  const locomotionPose = deriveWormLocalPose(state.locomotionPlant, state.previousAction)
  const detached =
    state.locomotionState === 'crawling' ||
    state.locomotionState === 'seeking' ||
    state.locomotionState === 'feeding'
  const baseOmittedResources: TerrariumResource['id'][] = []
  if (
    state.boardCollisionCooldown > 0 ||
    state.boardCollisionReleasePending ||
    state.locomotionState === 'seeking' ||
    state.locomotionState === 'mounting' ||
    state.locomotionState === 'dismounting' ||
    state.locomotionState === 'riding'
  ) {
    baseOmittedResources.push('skateboard')
  }
  if (
    state.feedingReleaseResourceId &&
    (state.feedingCooldown > 0 || wormBodyTouchesResource(state, state.feedingReleaseResourceId))
  ) {
    baseOmittedResources.push(state.feedingReleaseResourceId)
  }
  const collisionWorld = collisionWorldFor(
    field,
    staticObstacles,
    state.resources,
    baseOmittedResources,
    environmentConfig,
    state.boardHeading,
  )
  const feedingCollisionWorld =
    state.locomotionState === 'feeding' && state.feedingResourceId
      ? collisionWorldFor(
          field,
          staticObstacles,
          state.resources,
          [...baseOmittedResources, state.feedingResourceId],
          environmentConfig,
          state.boardHeading,
        )
      : collisionWorld
  let gripTotal = 0

  for (let index = 0; index < SEGMENT_COUNT; index += 1) {
    const segment = state.segments[index]
    const oldX = segment.x
    const oldY = segment.y
    const oldZ = segment.z
    const u = index / (SEGMENT_COUNT - 1)
    const centered = u - 0.5
    const bend = decoded.bends[index] ?? 0
    const pokeCurve = centered * poke
    const mountedAxial = centered * 1.42 * lengthScale
    const mountedLateral = bend * 0.15 + pokeCurve * 0.1 + victory * Math.sin(index * 0.82) * 0.048
    const mountedX = state.boardX + boardForwardX * mountedAxial + boardRightX * mountedLateral
    const mountedZ = state.boardZ + boardForwardZ * mountedAxial + boardRightZ * mountedLateral
    const mountedY =
      centerY +
      Math.abs(bend) * 0.035 +
      airborneTuck * Math.cos(centered * Math.PI) * 0.08 +
      Math.abs(pokeCurve) * 0.025

    const localPose = locomotionPose[index]
    const locomotionWeight = interaction?.locomotionWeight ?? 1
    const neutralForward = centered * 0.102 * (SEGMENT_COUNT - 1)
    const ownedForward = MathUtils.lerp(neutralForward, localPose.forward, locomotionWeight)
    const ownedLateral = localPose.lateral * locomotionWeight
    const ownedLift = localPose.lift * locomotionWeight
    const ownedYaw = localPose.yaw * locomotionWeight
    const crawlX = state.wormX + wormForwardX * ownedForward + wormRightX * ownedLateral
    const crawlZ = state.wormZ + wormForwardZ * ownedForward + wormRightZ * ownedLateral
    const crawlTerrain = field.sample(crawlX, crawlZ)
    const crawlY = crawlTerrain.height + 0.105 + ownedLift
    const interactionOffset = interaction?.segments[index]
    const interactionHeading = lerpAngle(state.wormHeading, state.boardHeading, mount)
    const interactionForwardX = Math.cos(interactionHeading)
    const interactionForwardZ = Math.sin(interactionHeading)
    const interactionRightX = -interactionForwardZ
    const interactionRightZ = interactionForwardX
    const offsetX = interactionOffset
      ? interactionForwardX * interactionOffset.forward + interactionRightX * interactionOffset.lateral
      : 0
    const offsetZ = interactionOffset
      ? interactionForwardZ * interactionOffset.forward + interactionRightZ * interactionOffset.lateral
      : 0
    const segmentMount =
      interaction && (interaction.kind === 'mounting' || interaction.kind === 'dismounting')
        ? contactWeightForSegment(interaction, u)
        : mount
    let targetX = MathUtils.lerp(crawlX, mountedX, segmentMount) + offsetX
    let targetY = MathUtils.lerp(crawlY, mountedY, segmentMount) + (interactionOffset?.vertical ?? 0)
    let targetZ = MathUtils.lerp(crawlZ, mountedZ, segmentMount) + offsetZ
    let segmentSmoothing = state.obstacleContactId ? Math.min(poseSmoothing, 3.4) : poseSmoothing

    if (feedingResource && interaction) {
      const bowlDx = state.wormX - feedingResource.position[0]
      const bowlDz = state.wormZ - feedingResource.position[2]
      const bowlDistance = Math.hypot(bowlDx, bowlDz)
      const outwardX = bowlDistance > 1e-6 ? bowlDx / bowlDistance : -wormForwardX
      const outwardZ = bowlDistance > 1e-6 ? bowlDz / bowlDistance : -wormForwardZ
      const anterior = smootherStep(MathUtils.clamp((u - 0.48) / 0.52, 0, 1))
      const headContact = anterior * interaction.contact.headWeight
      const neckOffset = (1 - u) * 0.58
      const contentsY =
        feedingResource.groundHeight +
        feedingResource.appearance.height * 0.84 +
        (feedingResource.id === 'water-bowl' ? 0.012 : 0.05)
      targetX = MathUtils.lerp(targetX, feedingResource.position[0] + outwardX * neckOffset, headContact)
      targetY = MathUtils.lerp(targetY, contentsY + (interactionOffset?.vertical ?? 0) * 0.22, headContact)
      targetZ = MathUtils.lerp(targetZ, feedingResource.position[2] + outwardZ * neckOffset, headContact)
    }

    if (state.locomotionState === 'feeding') {
      const posteriorGrip = 1 - smootherStep(MathUtils.clamp((u - 0.48) / 0.32, 0, 1))
      const contact = state.segmentGroundContacts[index]
      contact.anchorX = MathUtils.damp(contact.anchorX, targetX, 3.5, delta)
      contact.anchorZ = MathUtils.damp(contact.anchorZ, targetZ, 3.5, delta)
      contact.strength = MathUtils.damp(contact.strength, posteriorGrip * 0.86, 9, delta)
      targetX = MathUtils.lerp(targetX, contact.anchorX, contact.strength * 0.42)
      targetZ = MathUtils.lerp(targetZ, contact.anchorZ, contact.strength * 0.42)
      gripTotal += contact.strength
    } else {
      state.segmentGroundContacts[index].strength = MathUtils.damp(
        state.segmentGroundContacts[index].strength,
        0,
        12,
        delta,
      )
    }

    const proposedX = MathUtils.damp(segment.x, targetX, segmentSmoothing, delta)
    const proposedY = MathUtils.damp(segment.y, targetY, segmentSmoothing, delta)
    const proposedZ = MathUtils.damp(segment.z, targetZ, segmentSmoothing, delta)
    const proposedDx = proposedX - segment.x
    const proposedDy = proposedY - segment.y
    const proposedDz = proposedZ - segment.z
    const proposedDistance = Math.hypot(proposedDx, proposedDy, proposedDz)
    const connectiveStepLimit = state.obstacleContactId ? 0.045 : state.grounded ? 0.052 : 0.066
    const connectiveScale =
      proposedDistance > connectiveStepLimit ? connectiveStepLimit / proposedDistance : 1
    let nextX = segment.x + proposedDx * connectiveScale
    const nextY = segment.y + proposedDy * connectiveScale
    let nextZ = segment.z + proposedDz * connectiveScale

    if (collisionWorld) {
      const allowBowlContact = state.locomotionState === 'feeding' && u > 0.46
      const world = allowBowlContact ? (feedingCollisionWorld ?? collisionWorld) : collisionWorld
      const collision = resolveSweptTerrariumMotion(world, {
        position: { x: oldX, z: oldZ },
        displacement: { x: nextX - oldX, z: nextZ - oldZ },
        velocity: { x: (nextX - oldX) / delta, z: (nextZ - oldZ) / delta },
        body: [
          {
            id: `segment-${index}`,
            offset: { x: 0, z: 0 },
            radius: 0.074 * anatomy.thicknessScale,
            vertical: {
              minY: oldY - 0.074 * anatomy.verticalScale,
              maxY: oldY + 0.074 * anatomy.verticalScale,
            },
          },
        ],
        ground: {
          grounded: true,
          friction: crawlTerrain.friction,
          normalY: crawlTerrain.normal[1],
          contactRatio: state.segmentGroundContacts[index].strength,
        },
      })
      nextX = collision.position.x
      nextZ = collision.position.z
      if (collision.contacts.length > 0) {
        state.obstacleContactId = collision.contacts[0]?.id ?? state.obstacleContactId
        state.collisionCount += collision.contacts.length
      }
    }

    segment.x = nextX
    segment.y = nextY
    segment.z = nextZ
    segment.vx = (segment.x - oldX) / delta
    segment.vy = (segment.y - oldY) / delta
    segment.vz = (segment.z - oldZ) / delta
    const plantSegment = SEGMENT_COUNT - 1 - index
    const crawlPitch = -(state.locomotionPlant.jointVelocities[plantSegment] ?? 0) * 0.028 * locomotionWeight
    const mountedPitch = bend * 0.19 + airborneTuck * centered * 0.78
    const targetPitch =
      MathUtils.lerp(crawlPitch, mountedPitch, segmentMount) + (interactionOffset?.pitch ?? 0)
    const targetYaw =
      lerpAngle(
        state.wormHeading + ownedYaw,
        state.boardHeading + bend * 0.28 + pokeCurve * 0.12,
        segmentMount,
      ) + (interactionOffset?.yaw ?? 0)
    segment.pitch = MathUtils.damp(segment.pitch, targetPitch, ROTATION_SMOOTHING, delta)
    segment.yaw = dampAngle(segment.yaw, targetYaw, ROTATION_SMOOTHING, delta)
  }

  if (detached) {
    state.contactRatio = MathUtils.clamp(gripTotal / SEGMENT_COUNT, 0.12, 0.96)
  }
  if (state.locomotionState === 'feeding' || state.locomotionState === 'dismounting') {
    // Close-contact and dismount poses are authored, but the free crawler
    // measures its root from the articulated body's center of mass. Keep those
    // representations synchronized every frame so handing control back to the
    // neural plant cannot introduce a one-tick root correction.
    let centerX = 0
    let centerY = 0
    let centerZ = 0
    let velocityX = 0
    let velocityZ = 0
    for (const segment of state.segments) {
      centerX += segment.x
      centerY += segment.y
      centerZ += segment.z
      velocityX += segment.vx
      velocityZ += segment.vz
    }
    const inverseCount = 1 / Math.max(1, state.segments.length)
    state.wormX = centerX * inverseCount
    state.wormY = centerY * inverseCount
    state.wormZ = centerZ * inverseCount
    state.wormVx = velocityX * inverseCount
    state.wormVz = velocityZ * inverseCount
  }
}

function collisionFreeBoardHeading(
  startHeading: number,
  requestedHeading: number,
  centerX: number,
  centerY: number,
  centerZ: number,
  collisionWorld: TerrariumCollisionWorld,
  environmentConfig: EnvironmentConfig | null,
) {
  const headingIsClear = (heading: number) => {
    const body = skateboardFootprintObstacles(centerX, centerY, centerZ, heading, environmentConfig).map(
      sample => ({
        id: `rotation-${sample.id}`,
        offset: { x: sample.center.x - centerX, z: sample.center.z - centerZ },
        radius: sample.radius,
      }),
    )
    const probe = resolveSweptTerrariumMotion(collisionWorld, {
      position: { x: centerX, z: centerZ },
      displacement: { x: 0, z: 0 },
      velocity: { x: 0, z: 0 },
      body,
    })
    return Math.hypot(probe.actualDisplacement.x, probe.actualDisplacement.z) < 5e-4
  }

  if (headingIsClear(requestedHeading)) return requestedHeading
  let clearFraction = 0
  let blockedFraction = 1
  for (let iteration = 0; iteration < 9; iteration += 1) {
    const fraction = (clearFraction + blockedFraction) * 0.5
    const candidate = lerpAngle(startHeading, requestedHeading, fraction)
    if (headingIsClear(candidate)) clearFraction = fraction
    else blockedFraction = fraction
  }
  return lerpAngle(startHeading, requestedHeading, clearFraction)
}

export function decodeAction(action: PolicyAction): DecodedAction {
  const bends = new Array<number>(SEGMENT_COUNT)
  const contractions = new Array<number>(SEGMENT_COUNT)
  let propulsion = 0
  let coil = 0
  let release = 0
  let kick = 0
  let energy = 0

  for (let index = 0; index < SEGMENT_COUNT; index += 1) {
    const dorsal = action[index * 2] ?? 0
    const ventral = action[index * 2 + 1] ?? 0
    const bend = MathUtils.clamp((dorsal - ventral) * 0.5, -1, 1)
    const contraction = MathUtils.clamp((dorsal + ventral) * 0.5, -1, 1)
    const sideWeight = index < SEGMENT_COUNT / 2 ? -1 : 1
    bends[index] = bend
    contractions[index] = contraction
    propulsion += Math.abs(bend)
    coil += Math.max(0, contraction)
    release += Math.max(0, -contraction)
    kick += bend * sideWeight
    energy += (Math.abs(dorsal) + Math.abs(ventral)) * 0.5
  }

  return {
    bends,
    contractions,
    propulsion: propulsion / SEGMENT_COUNT,
    coil: coil / SEGMENT_COUNT,
    release: release / SEGMENT_COUNT,
    kick: kick / SEGMENT_COUNT,
    energy: energy / SEGMENT_COUNT,
  }
}

export function toSnapshot(state: StuntState): SimulationSnapshot {
  // Navigation happens in a large world, but the distilled controller learned
  // local stunt semantics. Present a canonical controller frame so arena turns,
  // hills, and board/worm separation cannot excite unsupported kinematic inputs.
  const controllerHeading = state.mountBlend > 0.5 ? state.boardHeading : state.wormHeading
  const canonicalBoardY = 0.42
  return {
    time: state.cycleTime,
    board: {
      x: 0,
      y: canonicalBoardY,
      z: 0,
      vx: 0.8,
      vy: 0,
      vz: 0,
      pitch: -0.04,
      roll: wrapAngle(state.boardRoll),
      yaw: 0,
    },
    segments: state.segments.map((segment, index) => ({
      x: (index / (SEGMENT_COUNT - 1) - 0.5) * 1.52,
      y: canonicalBoardY + 0.19,
      z: 0,
      vx: 0,
      vy: 0,
      vz: 0,
      pitch: segment.pitch,
      yaw: wrapAngle(segment.yaw - controllerHeading),
    })),
    contactRatio: state.contactRatio,
    targetDirection: [1, 0, 0],
    previousAction: state.previousAction,
  }
}

export function locomotionSensorsFor(state: StuntState, field: TerrainField): LocomotionSensors {
  const targetId = state.locomotionState === 'seeking' ? 'skateboard' : state.needs.targetResourceId
  const target = state.resources.find(resource => resource.id === targetId)
  if (!target) {
    return {
      targetForward: 1,
      targetRight: 0,
      targetDistance: 0,
      forwardSpeed: state.locomotionPlant.forwardSpeed,
      angularSpeed: state.locomotionPlant.angularSpeed,
      terrainFriction: field.sample(state.wormX, state.wormZ).friction,
      urgency: 0,
      contactLoads: state.locomotionPlant.contactLoads,
      slipSpeeds: state.locomotionPlant.slipSpeeds,
      obstacleForward: state.locomotionPlant.obstacleForward,
      obstacleRight: state.locomotionPlant.obstacleRight,
    }
  }

  const dx = target.position[0] - state.wormX
  const dz = target.position[2] - state.wormZ
  const distance = Math.hypot(dx, dz)
  const inverseDistance = distance > 1e-9 ? 1 / distance : 0
  const forwardX = Math.cos(state.wormHeading)
  const forwardZ = Math.sin(state.wormHeading)
  const rightX = -forwardZ
  const rightZ = forwardX
  return {
    targetForward: (dx * forwardX + dz * forwardZ) * inverseDistance,
    targetRight: (dx * rightX + dz * rightZ) * inverseDistance,
    targetDistance: MathUtils.clamp(distance / 5, 0, 1.5),
    forwardSpeed: state.locomotionPlant.forwardSpeed,
    angularSpeed: state.locomotionPlant.angularSpeed,
    terrainFriction: field.sample(state.wormX, state.wormZ).friction,
    urgency: state.needs[target.need],
    contactLoads: state.locomotionPlant.contactLoads,
    slipSpeeds: state.locomotionPlant.slipSpeeds,
    obstacleForward: state.locomotionPlant.obstacleForward,
    obstacleRight: state.locomotionPlant.obstacleRight,
  }
}

export function phaseFor(state: StuntState, mode: ShowcaseMode): StuntPhase {
  if (state.locomotionState === 'dismounting') return 'dismount'
  if (state.locomotionState === 'feeding') {
    return state.feedingResourceId === 'water-bowl' ? 'drinking' : 'eating'
  }
  if (state.locomotionState === 'crawling') {
    const target = state.resources.find(resource => resource.id === state.needs.targetResourceId)
    if (target?.need === 'hunger') return 'seeking food'
    if (target?.need === 'thirst') return 'seeking water'
    if (target?.need === 'wellbeing') return 'finding board'
    return mode === 'freestyle' ? 'free crawl' : 'ground crawl'
  }
  if (state.locomotionState === 'seeking') return 'finding board'
  if (state.locomotionState === 'mounting') return 'mounting up'
  if (!state.grounded) return 'kickflip'
  if (state.landingFlash > 0.05) return state.landingFlash > 0.45 ? 'landing' : 'victory wiggle'
  if (state.cycleTime < 2.05) return 'terrarium cruise'
  if (state.cycleTime < 2.62) return 'coil'
  if (state.cycleTime < 3.05) return 'pop'
  if (state.cycleTime < 4.15) return 'landing'
  if (state.cycleTime < 5.35) return 'victory wiggle'
  return 'terrarium cruise'
}

export function stuntNameFor(state: StuntState, mode: ShowcaseMode) {
  if (state.locomotionState === 'dismounting') return 'Board dismount'
  if (state.locomotionState === 'feeding') {
    return state.feedingResourceId === 'water-bowl' ? 'Drinking at the water bowl' : 'Eating at the food bowl'
  }
  if (state.locomotionState === 'crawling') {
    const target = state.resources.find(resource => resource.id === state.needs.targetResourceId)
    if (target?.need === 'hunger') return 'Hunting for food'
    if (target?.need === 'thirst') return 'Looking for water'
    return mode === 'freestyle' ? 'Evolved free crawl' : 'Needs-driven crawl'
  }
  if (state.locomotionState === 'seeking') return 'Finding the board'
  if (state.locomotionState === 'mounting') return 'Mounting up'
  return 'Scripted kickflip'
}

export type TerrariumDecor = {
  stones: Array<{ id: string; position: Vec3; scale: Vec3; color: string; collisionRadius: number }>
  sprouts: Array<{ id: string; position: Vec3; rotation: number; scale: number; collisionRadius: number }>
  obstacles: TerrariumCircleObstacle[]
}

export function makeTerrariumDecor(
  seed: number,
  field: TerrainField,
  resources: readonly TerrariumResource[],
  obstacleDensity: number,
): TerrariumDecor {
  const stones: TerrariumDecor['stones'] = []
  const sprouts: TerrariumDecor['sprouts'] = []
  const obstacles: TerrariumCircleObstacle[] = []
  const occupied: Array<{ x: number; z: number; radius: number }> = []
  const stoneColors = ['#8a9b78', '#a77f5f', '#6f8c71', '#b39b75']
  const density = MathUtils.clamp(obstacleDensity, 0, 0.3)
  const rockCount = Math.round(10 + density * 55)
  const treeCount = Math.round(5 + density * 28)

  for (let index = 0; index < rockCount; index += 1) {
    const scale = 0.72 + seededNoise(seed + 12, index) * 0.92
    const collisionRadius = scale * 0.2
    const position = chooseDecorPosition(seed + 3, index, field, resources, occupied, collisionRadius, 0.26)
    if (!position) continue
    const [x, z] = position
    const id = `rock-${index}`
    stones.push({
      id,
      position: [x, field.sample(x, z).height + 0.1 * scale, z],
      scale: [scale * 1.25, scale * 0.75, scale],
      color: stoneColors[index % stoneColors.length],
      collisionRadius,
    })
    occupied.push({ x, z, radius: collisionRadius })
    obstacles.push({
      id,
      kind: 'rock',
      center: { x, z },
      radius: collisionRadius,
      vertical: {
        minY: field.sample(x, z).height - 0.025,
        maxY: field.sample(x, z).height + scale * 0.24,
      },
      friction: 0.74,
      restitution: 0.035,
    })
  }

  for (let index = 0; index < treeCount; index += 1) {
    const scale = 0.82 + seededNoise(seed + 32, index * 2) * 0.62
    const collisionRadius = 0.065 + scale * 0.055
    const position = chooseDecorPosition(seed + 21, index, field, resources, occupied, collisionRadius, 0.42)
    if (!position) continue
    const [x, z] = position
    const id = `tree-${index}`
    sprouts.push({
      id,
      position: [x, field.sample(x, z).height, z],
      rotation: seededNoise(seed + 30, index * 4) * Math.PI,
      scale,
      collisionRadius,
    })
    occupied.push({ x, z, radius: collisionRadius })
    obstacles.push({
      id,
      kind: 'tree',
      center: { x, z },
      radius: collisionRadius,
      vertical: {
        minY: field.sample(x, z).height,
        maxY: field.sample(x, z).height + scale * 1.02,
      },
      friction: 0.86,
      restitution: 0.01,
    })
  }
  return { stones, sprouts, obstacles }
}

function chooseDecorPosition(
  seed: number,
  index: number,
  field: TerrainField,
  resources: readonly TerrariumResource[],
  occupied: readonly { x: number; z: number; radius: number }[],
  radius: number,
  spacing: number,
): [number, number] | null {
  for (let attempt = 0; attempt < 36; attempt += 1) {
    const sample = index * 41 + attempt
    const x = (seededNoise(seed + 5, sample * 7) - 0.5) * (field.width - 1.65)
    const z = (seededNoise(seed + 11, sample * 5) - 0.5) * (field.depth - 1.65)
    const clearOfResources = resources.every(resource => {
      const clearance = resource.interactionRadius + radius + (resource.id === 'skateboard' ? 0.48 : 0.26)
      return Math.hypot(x - resource.position[0], z - resource.position[2]) >= clearance
    })
    if (!clearOfResources) continue
    const clearOfRoute = field.waypoints.every(
      waypoint => Math.hypot(x - waypoint[0], z - waypoint[1]) >= radius + 0.34,
    )
    if (!clearOfRoute) continue
    const clearOfDecor = occupied.every(
      candidate => Math.hypot(x - candidate.x, z - candidate.z) >= radius + candidate.radius + spacing,
    )
    if (clearOfDecor) return [x, z]
  }
  return null
}

export function smoothAction(applied: PolicyAction, target: PolicyAction, delta: number) {
  const alpha = 1 - Math.exp(-18 * delta)
  for (let index = 0; index < applied.length; index += 1) {
    applied[index] = MathUtils.lerp(applied[index] ?? 0, target[index] ?? 0, alpha)
  }
}

export function boardGroundY(field: TerrainField, x: number, z: number) {
  return field.sample(x, z).height + BOARD_CLEARANCE
}

export function smoothStep(value: number) {
  const t = MathUtils.clamp(value, 0, 1)
  return t * t * (3 - 2 * t)
}

function collisionWorldFor(
  field: TerrainField,
  staticObstacles: readonly TerrariumCircleObstacle[],
  resources: readonly TerrariumResource[],
  omittedResourceIds: readonly TerrariumResource['id'][],
  environmentConfig: EnvironmentConfig | null,
  skateboardHeading: number,
  bowlRimAperture: BowlRimAperture | null = null,
): TerrariumCollisionWorld {
  const omitted = new Set(omittedResourceIds)
  const resourceObstacles: TerrariumCircleObstacle[] = resources
    .filter(resource => !omitted.has(resource.id))
    .flatMap(resource => {
      if (resource.presentation === 'skateboard') {
        return skateboardFootprintObstacles(
          resource.position[0],
          resource.position[1],
          resource.position[2],
          skateboardHeading,
          environmentConfig,
        )
      }
      return bowlRimFootprintObstacles(
        resource,
        bowlRimAperture?.resourceId === resource.id ? bowlRimAperture.approachAngle : null,
      )
    })
  return prepareTerrariumCollisionWorld({
    bounds: createTerrariumArenaBounds(field.width, field.depth, ARENA_MARGIN),
    obstacles: [...staticObstacles, ...resourceObstacles],
    boundaryFriction: Math.max(0.25, (environmentConfig?.terrain.baseFriction ?? 0.9) * 0.55),
    boundaryRestitution: environmentConfig?.terrain.restitution ?? 0.04,
  })
}

function bowlRimFootprintObstacles(
  resource: TerrariumResource,
  apertureAngle: number | null = null,
): TerrariumCircleObstacle[] {
  // The mesh is a hollow vessel, so its collider must not be a solid disk.
  // Small circles around the lip let an anterior segment enter the contents
  // while the rest of the articulated body still contacts and slides over the
  // visible rim.
  const sampleCount = 12
  const rimRadius = resource.appearance.radius * 0.82
  const rimThickness = Math.max(0.035, resource.appearance.radius * 0.12)
  const obstacles: TerrariumCircleObstacle[] = []
  for (let index = 0; index < sampleCount; index += 1) {
    const angle = (index / sampleCount) * Math.PI * 2
    if (apertureAngle !== null && Math.abs(wrapAngle(angle - apertureAngle)) < 0.42) continue
    obstacles.push({
      id: `${resource.id}-rim-${index}`,
      kind: 'bowl' as const,
      center: {
        x: resource.position[0] + Math.cos(angle) * rimRadius,
        z: resource.position[2] + Math.sin(angle) * rimRadius,
      },
      radius: rimThickness,
      vertical: {
        minY: resource.groundHeight - 0.015,
        maxY: resource.groundHeight + resource.appearance.height * 1.08,
      },
      friction: 0.72,
      restitution: 0.015,
    })
  }
  return obstacles
}

function activeBowlRimAperture(state: StuntState): BowlRimAperture | null {
  const latched = state.bowlRimAperture
  if (latched) {
    const latchedResource = state.resources.find(candidate => candidate.id === latched.resourceId)
    const releaseDistance = latchedResource ? latchedResource.interactionRadius + 0.18 : 0
    const bodyStillNearLatchedBowl =
      latchedResource !== undefined &&
      state.segments.some(
        segment =>
          Math.hypot(segment.x - latchedResource.position[0], segment.z - latchedResource.position[2]) <=
          releaseDistance,
      )
    if (bodyStillNearLatchedBowl) return latched
    state.bowlRimAperture = null
  }

  if (state.locomotionState !== 'crawling') return null
  const resource = state.resources.find(
    candidate => candidate.id === state.needs.targetResourceId && candidate.presentation === 'bowl',
  )
  if (!resource || (resource.id !== 'food-bowl' && resource.id !== 'water-bowl')) return null

  // Acquire early from the anterior quarter, then freeze this resource and
  // angle until the entire chain has cleared a conservative release radius.
  // Recomputing it from the current need target or centroid would make discrete
  // rim samples reappear around a segment during target and lifecycle changes.
  const anterior = state.segments.slice(-Math.max(1, Math.ceil(state.segments.length * 0.25)))
  const closestAnteriorDistance = anterior.reduce(
    (closest, segment) =>
      Math.min(closest, Math.hypot(segment.x - resource.position[0], segment.z - resource.position[2])),
    Number.POSITIVE_INFINITY,
  )
  if (closestAnteriorDistance > resource.interactionRadius + 0.18) return null

  const posterior = state.segments.slice(0, Math.max(1, Math.floor(state.segments.length * 0.5)))
  const approachPoint = posterior.reduce(
    (sum, segment) => ({ x: sum.x + segment.x / posterior.length, z: sum.z + segment.z / posterior.length }),
    { x: 0, z: 0 },
  )
  const dx = approachPoint.x - resource.position[0]
  const dz = approachPoint.z - resource.position[2]
  const aperture = {
    resourceId: resource.id,
    approachAngle: Math.hypot(dx, dz) > 1e-6 ? Math.atan2(dz, dx) : wrapAngle(state.wormHeading + Math.PI),
  }
  state.bowlRimAperture = aperture
  return aperture
}

function wormBodyTouchesResource(state: StuntState, resourceId: TerrariumResource['id']) {
  const resource = state.resources.find(candidate => candidate.id === resourceId)
  if (!resource) return false
  const clearance = resource.presentation === 'skateboard' ? 0.82 : resource.appearance.radius + 0.09
  return state.segments.some(
    segment => Math.hypot(segment.x - resource.position[0], segment.z - resource.position[2]) < clearance,
  )
}

/** World-space center of the visible anterior mouth, derived from the live head pose. */
function wormMouthContactPoint(state: StuntState): Vec3 {
  const head = state.segments.at(-1)
  if (!head) return [state.wormX, state.wormY, state.wormZ]
  const reach = 0.055
  const horizontalReach = Math.cos(head.pitch) * reach
  return [
    head.x + Math.cos(head.yaw) * horizontalReach,
    head.y - Math.sin(head.pitch) * reach,
    head.z + Math.sin(head.yaw) * horizontalReach,
  ]
}

function wormBodyOverlapsBoardCollider(state: StuntState, environmentConfig: EnvironmentConfig | null) {
  const footprint = skateboardFootprintObstacles(
    state.boardX,
    state.boardY,
    state.boardZ,
    state.boardHeading,
    environmentConfig,
  )
  if (
    footprint.some(
      circle =>
        Math.hypot(state.wormX - circle.center.x, state.wormZ - circle.center.z) < circle.radius + 0.13,
    )
  )
    return true
  return state.segments.some(segment =>
    footprint.some(
      circle => Math.hypot(segment.x - circle.center.x, segment.z - circle.center.z) < circle.radius + 0.074,
    ),
  )
}

export function skateboardFootprintObstacles(
  centerX: number,
  centerY: number,
  centerZ: number,
  heading: number,
  environmentConfig: EnvironmentConfig | null,
): TerrariumCircleObstacle[] {
  const configured = environmentConfig?.skateboard.deckSize ?? [1.9, 0.12, 0.58]
  const deckLength = MathUtils.clamp(configured[0], 1.65, 2.1)
  const deckWidth = MathUtils.clamp(configured[2], 0.52, 0.68)
  const axialCount = 7
  const lateralCount = 2
  const cellLength = deckLength / axialCount
  const cellWidth = deckWidth / lateralCount
  const radius = Math.hypot(cellLength * 0.5, cellWidth * 0.5)
  const forwardX = Math.cos(heading)
  const forwardZ = Math.sin(heading)
  const rightX = -forwardZ
  const rightZ = forwardX
  const circles: TerrariumCircleObstacle[] = []
  for (let axialIndex = 0; axialIndex < axialCount; axialIndex += 1) {
    const axial = -deckLength * 0.5 + cellLength * (axialIndex + 0.5)
    for (let lateralIndex = 0; lateralIndex < lateralCount; lateralIndex += 1) {
      const lateral = -deckWidth * 0.5 + cellWidth * (lateralIndex + 0.5)
      circles.push({
        id: `skateboard-${axialIndex}-${lateralIndex}`,
        kind: 'skateboard',
        center: {
          x: centerX + forwardX * axial + rightX * lateral,
          z: centerZ + forwardZ * axial + rightZ * lateral,
        },
        radius,
        vertical: {
          minY: centerY - Math.max(0.1, configured[1] * 0.75),
          maxY: centerY + Math.max(0.08, configured[1] * 0.75),
        },
        friction: 0.48,
        restitution: 0.015,
      })
    }
  }
  return circles
}

function smootherStep(value: number) {
  const t = MathUtils.clamp(value, 0, 1)
  return t * t * t * (t * (t * 6 - 15) + 10)
}

export function dampAngle(current: number, target: number, lambda: number, delta: number) {
  return current + wrapAngle(target - current) * (1 - Math.exp(-lambda * delta))
}

export function lerpAngle(start: number, end: number, alpha: number) {
  return start + wrapAngle(end - start) * MathUtils.clamp(alpha, 0, 1)
}

function seededNoise(seed: number, value: number) {
  const x = Math.sin(seed * 12.9898 + value * 78.233) * 43758.5453
  return x - Math.floor(x)
}

function wrapAngle(angle: number) {
  return Math.atan2(Math.sin(angle), Math.cos(angle))
}
