import { OrbitControls, PerspectiveCamera, RoundedBox } from '@react-three/drei'
import { Canvas, useFrame } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import type { Group, Mesh, MeshStandardMaterial } from 'three'
import { BufferAttribute, BufferGeometry, Color, MathUtils, Quaternion, Vector3 } from 'three'
import { deriveWurmAnatomy, type GenomeAppendage, type WurmAnatomy } from '../creature/anatomy'
import type { CreatureGenome, EnvironmentConfig, Vec3 } from '../creature/types'
import { PolicyRunner } from '../policy/policyRunner'
import { LocomotionPolicyRunner } from '../policy/locomotionRunner'
import type { LocomotionSensors } from '../policy/locomotionPolicy'
import { makeInitialAction, snapshotToObservation } from '../policy/simulationAdapter'
import { createTerrainField, TERRAIN_GRID_RESOLUTION, type TerrainField } from './terrainField'
import {
  advanceNeeds,
  createNeedsState,
  createTerrariumResources,
  readNeedsMetrics,
  syncSkateboardResource,
  type NeedsState,
  type TerrariumResource,
} from './terrariumNeeds'
import {
  createWormLocomotionPlant,
  deriveWormLocalPose,
  resetWormLocomotionPlant,
  stepWormLocomotion,
  type WormLocomotionPlant,
} from './wormLocomotion'
import {
  sampleWormInteractionAnimation,
  type WormInteractionAnimationSample,
  type WormInteractionKind,
} from './wormInteractionAnimation'
import {
  createTerrariumArenaBounds,
  resolveSweptTerrariumMotion,
  type TerrariumCircleObstacle,
  type TerrariumCollisionWorld,
} from './terrariumCollisions'
import {
  POLICY_TIMESTEP,
  SEGMENT_COUNT,
  type PolicyAction,
  type PolicyStatus,
  type SegmentSnapshot,
  type SimulationSnapshot,
  type ViewerMetrics,
} from '../policy/types'

export type ShowcaseMode = 'kickflip' | 'freestyle'

type SceneProps = {
  policyRunner: PolicyRunner
  running: boolean
  resetNonce: number
  interactionNonce?: number
  showcaseMode?: ShowcaseMode
  creature: CreatureGenome | null
  environmentConfig: EnvironmentConfig | null
  onMetrics: (metrics: ViewerMetrics) => void
  onPolicyStatus: (status: PolicyStatus) => void
}

type StuntPhase =
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

type LocomotionState = 'riding' | 'dismounting' | 'crawling' | 'seeking' | 'mounting' | 'feeding'

type SegmentGroundContact = {
  anchorX: number
  anchorZ: number
  strength: number
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

type DecodedAction = {
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
const up = new Vector3(0, 1, 0)
const connectorQuaternion = new Quaternion()
const connectorDelta = new Vector3()
const connectorStart = new Vector3()
const connectorEnd = new Vector3()
const renderTarget = new Vector3()

export function WurmkickflipScene({
  policyRunner,
  running,
  resetNonce,
  interactionNonce = 0,
  showcaseMode = 'kickflip',
  creature,
  environmentConfig,
  onMetrics,
  onPolicyStatus,
}: SceneProps) {
  const locomotionRunner = useMemo(() => new LocomotionPolicyRunner(), [])
  useEffect(() => {
    let mounted = true
    void Promise.all([policyRunner.load(), locomotionRunner.load()]).then(([stuntStatus, locomotionStatus]) => {
      if (!mounted) return
      onPolicyStatus({
        backend: locomotionStatus.loaded ? 'neural-js' : 'unavailable',
        message:
          `${locomotionStatus.message} Detached segment motion is the evolved controller driving the causal plant. ` +
          `Obstacle response, mounting, feeding, and the aerial kickflip are authored environment choreography. ` +
          `Mounted stunt rig: ${stuntStatus.modelVersion}.`,
        modelVersion: locomotionStatus.loaded
          ? `${locomotionStatus.modelVersion} + scripted-kickflip`
          : `${locomotionStatus.modelVersion} + mounted-${stuntStatus.modelVersion}`,
      })
    })
    return () => {
      mounted = false
    }
  }, [locomotionRunner, onPolicyStatus, policyRunner])

  const sceneKey = `${creature?.id ?? 'wurm'}-${environmentConfig?.id ?? 'terrarium'}-${resetNonce}`
  const arenaSpan = Math.max(environmentConfig?.world.size[0] ?? 11.5, environmentConfig?.world.size[2] ?? 11.5)
  const cameraDistance = arenaSpan * 0.69

  return (
    <Canvas dpr={[1, 1.5]} shadows="percentage">
      <PerspectiveCamera makeDefault position={[cameraDistance, arenaSpan * 0.52, cameraDistance]} fov={45} />
      <color attach="background" args={['#dce9df']} />
      <fog attach="fog" args={['#dce9df', arenaSpan * 0.85, arenaSpan * 1.75]} />
      <hemisphereLight color="#fffaf0" groundColor="#4e7660" intensity={1.25} />
      <directionalLight
        castShadow
        color="#fff4d6"
        intensity={2.55}
        position={[-arenaSpan * 0.38, arenaSpan * 0.78, arenaSpan * 0.42]}
        shadow-camera-far={arenaSpan * 1.7}
        shadow-camera-left={-arenaSpan * 0.62}
        shadow-camera-right={arenaSpan * 0.62}
        shadow-camera-top={arenaSpan * 0.62}
        shadow-camera-bottom={-arenaSpan * 0.18}
        shadow-mapSize={[1536, 1536]}
      />
      <TerrariumWorld
        creature={creature}
        environmentConfig={environmentConfig}
        interactionNonce={interactionNonce}
        key={sceneKey}
        onMetrics={onMetrics}
        policyRunner={policyRunner}
        locomotionRunner={locomotionRunner}
        running={running}
        showcaseMode={showcaseMode}
      />
      <OrbitControls
        enableDamping
        enablePan={false}
        maxDistance={arenaSpan * 1.55}
        maxPolarAngle={Math.PI / 2.08}
        minDistance={4.2}
        minPolarAngle={0.42}
        target={[0, 0.6, 0]}
      />
    </Canvas>
  )
}


type TerrariumWorldProps = {
  policyRunner: PolicyRunner
  locomotionRunner: LocomotionPolicyRunner
  running: boolean
  interactionNonce: number
  showcaseMode: ShowcaseMode
  creature: CreatureGenome | null
  environmentConfig: EnvironmentConfig | null
  onMetrics: (metrics: ViewerMetrics) => void
}

function TerrariumWorld({
  policyRunner,
  locomotionRunner,
  running,
  interactionNonce,
  showcaseMode,
  creature,
  environmentConfig,
  onMetrics,
}: TerrariumWorldProps) {
  const terrainField = useMemo(() => createTerrainField(environmentConfig), [environmentConfig])
  const state = useRef(createStuntState(terrainField, environmentConfig))
  const latestAction = useRef<PolicyAction>(makeInitialAction())
  const appliedAction = useRef<PolicyAction>(makeInitialAction())
  const inferencePending = useRef(false)
  const physicsAccumulator = useRef(0)
  const inferenceAccumulator = useRef(POLICY_TIMESTEP)
  const metricsAccumulator = useRef(0)
  const lastInteractionNonce = useRef(interactionNonce)
  const boardRef = useRef<Group>(null)
  const wheelRefs = useRef<Array<Mesh | null>>([])
  const segmentRefs = useRef<Array<Group | null>>([])
  const segmentMaterialRefs = useRef<Array<MeshStandardMaterial | null>>([])
  const connectorRefs = useRef<Array<Mesh | null>>([])
  const burstRefs = useRef<Array<Mesh | null>>([])
  const mouthRef = useRef<Mesh>(null)

  useEffect(() => {
    locomotionRunner.reset()
  }, [locomotionRunner])

  const gravity = Math.abs(environmentConfig?.world.gravity[1] ?? -9.81)
  const palette = useMemo(() => makeWurmPalette(creature), [creature])
  const anatomy = useMemo(() => deriveWurmAnatomy(creature), [creature])
  const terrain = useMemo(
    () => makeTerrariumDecor(
      environmentConfig?.seed ?? 1337,
      terrainField,
      state.current.resources,
      environmentConfig?.terrain.obstacleDensity ?? 0.08,
    ),
    [environmentConfig?.seed, environmentConfig?.terrain.obstacleDensity, terrainField],
  )

  useEffect(() => {
    if (lastInteractionNonce.current !== interactionNonce) {
      state.current.poke = 1
      lastInteractionNonce.current = interactionNonce
    }
  }, [interactionNonce])

  useFrame((_, rawDelta) => {
    const frameDelta = Math.min(rawDelta, 0.08)
    if (running) {
      physicsAccumulator.current += frameDelta
      inferenceAccumulator.current += frameDelta
      metricsAccumulator.current += frameDelta

      let steps = 0
      while (physicsAccumulator.current >= POLICY_TIMESTEP && steps < 5) {
        const wasRiding = state.current.locomotionState === 'riding'
        const locomotionOwnsBody =
          state.current.locomotionState === 'crawling' || state.current.locomotionState === 'seeking'
        if (state.current.locomotionState === 'riding') {
          smoothAction(appliedAction.current, latestAction.current, POLICY_TIMESTEP)
        } else if (locomotionOwnsBody) {
          appliedAction.current.set(locomotionRunner.run(
            locomotionSensorsFor(state.current, terrainField),
            state.current.locomotionPlant.joints,
            state.current.locomotionPlant.jointVelocities,
          ))
        } else {
          appliedAction.current.fill(0)
        }
        advanceStunt(
          state.current,
          appliedAction.current,
          POLICY_TIMESTEP,
          gravity,
          showcaseMode,
          terrainField,
          environmentConfig,
          terrain.obstacles,
        )
        if (wasRiding && state.current.locomotionState !== 'riding') {
          locomotionRunner.reset()
          appliedAction.current.fill(0)
        }
        physicsAccumulator.current -= POLICY_TIMESTEP
        steps += 1
      }

      if (
        state.current.locomotionState === 'riding' &&
        inferenceAccumulator.current >= POLICY_TIMESTEP &&
        !inferencePending.current
      ) {
        inferenceAccumulator.current %= POLICY_TIMESTEP
        inferencePending.current = true
        const observation = snapshotToObservation(toSnapshot(state.current))
        void policyRunner.run(observation)
          .then((action) => {
            latestAction.current = action
          })
          .catch(() => {
            // Keep the last known-safe action; PolicyRunner reports backend failures in its status.
          })
          .finally(() => {
            inferencePending.current = false
          })
      }

      if (metricsAccumulator.current >= 0.1) {
        metricsAccumulator.current %= 0.1
        const status = policyRunner.getStatus()
        const current = state.current
        const needsMetrics = readNeedsMetrics(
          current.needs,
          current.resources,
          [current.wormX, current.wormY, current.wormZ],
        )
        const interactionResource = current.feedingResourceId
          ? current.resources.find((resource) => resource.id === current.feedingResourceId)
          : null
        const displayedNeedTarget = interactionResource?.id ?? needsMetrics.targetResourceId
        const displayedNeedDistance = interactionResource
          ? Math.hypot(
              interactionResource.position[0] - current.wormX,
              interactionResource.position[2] - current.wormZ,
            )
          : needsMetrics.targetDistance
        const rootVx = current.mountBlend > 0.5 ? current.boardVx : current.wormVx
        const rootVz = current.mountBlend > 0.5 ? current.boardVz : current.wormVz
        const bodySpeed = current.segments.reduce(
          (maximum, segment) =>
            Math.max(maximum, Math.hypot(segment.vx - rootVx, segment.vy, segment.vz - rootVz)),
          0,
        )
        onMetrics({
          time: current.time,
          reward: current.reward,
          distance: current.distance,
          contactRatio: current.contactRatio,
          backend: status.backend,
          message: status.message,
          running,
          muscleActivity: decodeAction(current.previousAction).bends,
          phase: current.phase,
          flipProgress: current.flipProgress,
          flipsLanded: current.flipsLanded,
          airtime: current.grounded ? current.lastAirtime : current.currentAirtime,
          height:
            current.locomotionState === 'riding'
              ? Math.max(0, current.boardY - boardGroundY(terrainField, current.boardX, current.boardZ))
              : Math.max(0, current.wormY - terrainField.sample(current.wormX, current.wormZ).height),
          speed:
            current.locomotionState === 'riding'
              ? Math.hypot(current.boardVx, current.boardVz)
              : Math.hypot(current.wormVx, current.wormVz),
          landingQuality: current.landingQuality,
          attempt: current.attempt,
          stuntName: stuntNameFor(current, showcaseMode),
          bodySpeed,
          mounted: current.mountBlend > 0.92,
          distanceToBoard: current.distanceToBoard,
          terrainFriction: current.terrainFriction,
          crawlDistance: current.wormDistance,
          hunger: needsMetrics.hunger,
          thirst: needsMetrics.thirst,
          wellbeing: needsMetrics.wellbeing,
          activeNeed: interactionResource?.need ?? needsMetrics.activeNeed,
          needTarget: displayedNeedTarget,
          needTargetDistance: displayedNeedDistance,
        })
      }
    }

    renderStunt(
      state.current,
      state.current.previousAction,
      frameDelta,
      showcaseMode,
      anatomy,
      boardRef.current,
      wheelRefs.current,
      segmentRefs.current,
      segmentMaterialRefs.current,
      connectorRefs.current,
      burstRefs.current,
      mouthRef.current,
    )
  })

  return (
    <group>
      <Terrarium terrain={terrain} environmentConfig={environmentConfig} field={terrainField} />
      <TerrariumResources resources={state.current.resources} stateRef={state} />
      <BoardVisual boardRef={boardRef} environmentConfig={environmentConfig} wheelRefs={wheelRefs} />
      <WurmVisual
        anatomy={anatomy}
        connectorRefs={connectorRefs}
        materialRefs={segmentMaterialRefs}
        palette={palette}
        segmentRefs={segmentRefs}
        mouthRef={mouthRef}
      />
      <LandingBurst burstRefs={burstRefs} palette={palette} />
    </group>
  )
}

export function createStuntState(field: TerrainField, environmentConfig: EnvironmentConfig | null = null): StuntState {
  const start = field.waypoints[0] ?? [-2.8, -1.8]
  const next = field.waypoints[1] ?? [2.4, -1.5]
  const boardHeading = Math.atan2(next[1] - start[1], next[0] - start[0])
  const boardY = boardGroundY(field, start[0], start[1])
  const resources = createTerrariumResources(field, environmentConfig?.seed ?? 1337, [start[0], boardY, start[1]])
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
  const feedingResource = state.feedingResourceId
    ? state.resources.find((resource) => resource.id === state.feedingResourceId)
    : null
  const needResources = mode === 'freestyle'
    ? state.resources.filter((resource) => resource.id !== 'skateboard')
    : state.resources
  const needsStep = advanceNeeds(state.needs, delta, {
    resources: needResources,
    wormPosition: feedingResource?.position ?? [head?.x ?? state.wormX, head?.y ?? state.wormY, head?.z ?? state.wormZ],
    mounted: state.mountBlend > 0.92,
  })
  state.needs = needsStep.state

  updateLocomotionLifecycle(
    state,
    mode,
    environmentConfig,
    needsStep.restored >= 0.004 ? needsStep.interaction : null,
  )

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
  updateWormRoot(state, action, delta, field, environmentConfig, staticObstacles)
  const groundY = boardGroundY(field, state.boardX, state.boardZ)
  const terrain = field.sample(state.boardX, state.boardZ)

  if (state.grounded) {
    state.boardY = groundY - (state.locomotionState === 'riding' ? decoded.coil * 0.028 : 0)
    const forwardSlope = terrain.normal[0] * Math.cos(state.boardHeading) + terrain.normal[2] * Math.sin(state.boardHeading)
    const terrainPitch = Math.atan2(-forwardSlope, Math.max(0.2, terrain.normal[1]))
    state.boardPitch = MathUtils.damp(
      state.boardPitch,
      terrainPitch + (state.locomotionState === 'riding' ? decoded.coil * -0.095 : 0) + state.poke * 0.025,
      7,
      delta,
    )
    state.boardRoll = MathUtils.damp(state.boardRoll, state.poke * Math.sin(state.time * 9) * 0.14, 7, delta)
    state.boardYaw = dampAngle(state.boardYaw, -state.boardHeading, 8, delta)
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
    state.boardPitch = MathUtils.damp(state.boardPitch, Math.sin(state.currentAirtime * Math.PI) * -0.12, 4.5, delta)
    state.boardYaw = dampAngle(state.boardYaw, -state.boardHeading + decoded.kick * 0.045, 4.5, delta)
    state.maxHeight = Math.max(state.maxHeight, state.boardY - groundY)
    state.flipProgress = MathUtils.clamp(Math.abs(state.boardRoll) / (Math.PI * 2), 0, 1)
    state.contactRatio = MathUtils.clamp(0.14 + decoded.coil * 0.28, 0.08, 0.45)

    if (state.boardY <= groundY && state.boardVy < 0) landKickflip(state, field)
  }

  updateSegments(state, decoded, delta, field, environmentConfig, staticObstacles)
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
  substantialFeedingContact: NeedsState['lastInteraction'],
) {
  if (state.locomotionState === 'riding') {
    state.mountBlend = 1
    if (mode === 'freestyle' || (state.rideLandings > 0 && state.cycleTime > 5.55 && state.needs.wellbeing < 0.08)) {
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
  }
  if (next === 'dismounting') {
    state.dismountSide = state.attempt % 2 === 0 ? 1 : -1
    state.flipProgress = 0
    state.currentAirtime = 0
  }
  if (next === 'riding') {
    state.boardCollisionReleasePending = false
    resetWormLocomotionPlant(state.locomotionPlant)
    state.segmentGroundContacts.forEach((contact) => {
      contact.strength = 0
    })
  }
}

function interactionSampleFor(state: StuntState): WormInteractionAnimationSample {
  const kind: WormInteractionKind =
    state.locomotionState === 'mounting'
      ? 'mounting'
      : state.locomotionState === 'dismounting'
        ? 'dismounting'
        : state.feedingResourceId === 'water-bowl'
          ? 'drinking'
          : 'eating'
  const durationSeconds =
    kind === 'mounting'
      ? MOUNT_SECONDS
      : kind === 'dismounting'
        ? DISMOUNT_SECONDS
        : FEED_SECONDS
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

function updateSegmentGroundGrip(
  contact: SegmentGroundContact,
  targetX: number,
  targetZ: number,
  jointSpeed: number,
  terrainFriction: number,
  delta: number,
) {
  if (contact.strength < 0.035) {
    contact.anchorX = targetX
    contact.anchorZ = targetZ
  }
  const strain = Math.hypot(targetX - contact.anchorX, targetZ - contact.anchorZ)
  if (strain > 0.22) {
    // Skin contact cannot store meters of invisible spring energy. Once a
    // planted segment is overextended it releases and re-contacts at the
    // kinematic pose, producing stick-slip instead of an elastic teleport.
    contact.anchorX = targetX
    contact.anchorZ = targetZ
    contact.strength = Math.min(contact.strength, 0.06)
    return { x: contact.anchorX, z: contact.anchorZ, strength: contact.strength }
  }
  const quietJoint = 1 - MathUtils.clamp(jointSpeed / 2.35, 0, 1)
  const substrateGrip = MathUtils.clamp((terrainFriction - 0.08) / 1.02, 0.08, 1)
  const overloadRelease = smootherStep(MathUtils.clamp((strain - 0.085) / 0.12, 0, 1))
  const desiredStrength = quietJoint * substrateGrip * (1 - overloadRelease)
  contact.strength = MathUtils.damp(contact.strength, desiredStrength, desiredStrength > contact.strength ? 12 : 18, delta)
  if (contact.strength < 0.12 || overloadRelease > 0.82) {
    const anchorResponse = 3 + overloadRelease * 18
    contact.anchorX = MathUtils.damp(contact.anchorX, targetX, anchorResponse, delta)
    contact.anchorZ = MathUtils.damp(contact.anchorZ, targetZ, anchorResponse, delta)
  }
  return { x: contact.anchorX, z: contact.anchorZ, strength: contact.strength }
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
  const halfWidth = field.width * 0.5 - ARENA_MARGIN
  const halfDepth = field.depth * 0.5 - ARENA_MARGIN
  const wheelFriction = environmentConfig?.skateboard.wheelFriction ?? 0.82
  const boardMass = environmentConfig?.skateboard.mass ?? 1.1
  const traction = MathUtils.clamp(terrain.friction * wheelFriction / Math.sqrt(boardMass), 0.3, 1.45)
  let desiredHeading = state.boardHeading

  if (state.locomotionState === 'riding') {
    const waypoint = field.waypoints[state.boardWaypointIndex % field.waypoints.length] ?? [0, 0]
    const waypointDistance = Math.hypot(waypoint[0] - state.boardX, waypoint[1] - state.boardZ)
    if (waypointDistance < 0.72) state.boardWaypointIndex = (state.boardWaypointIndex + 1) % field.waypoints.length
    const activeWaypoint = field.waypoints[state.boardWaypointIndex % field.waypoints.length] ?? [0, 0]
    desiredHeading = Math.atan2(activeWaypoint[1] - state.boardZ, activeWaypoint[0] - state.boardX)

    const edgeX = Math.max(0, Math.abs(state.boardX) - (halfWidth - 1.05))
    const edgeZ = Math.max(0, Math.abs(state.boardZ) - (halfDepth - 1.05))
    const edgeBlend = MathUtils.clamp(Math.max(edgeX, edgeZ) / 1.05, 0, 1)
    if (edgeBlend > 0) {
      const inwardHeading = Math.atan2(-state.boardZ, -state.boardX)
      desiredHeading = lerpAngle(desiredHeading, inwardHeading, edgeBlend)
    }
  }

  const headingRate = state.locomotionState === 'riding' && state.grounded ? 0.72 + traction * 0.38 : 0.18
  state.boardHeading = dampAngle(state.boardHeading, desiredHeading, headingRate, delta)
  const terrainResistance = 1 / (0.84 + terrain.friction * 0.18)
  const targetSpeed =
    state.locomotionState === 'riding'
      ? (0.5 + decoded.propulsion * 0.7) * terrainResistance
      : 0
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
  const deckLength = MathUtils.clamp(environmentConfig?.skateboard.deckSize[0] ?? 1.9, 1.65, 2.1)
  const boardCollision = resolveSweptTerrariumMotion(
    collisionWorldFor(
      field,
      staticObstacles,
      state.resources,
      ['skateboard'],
      environmentConfig,
      state.boardHeading,
    ),
    {
      position: { x: oldBoardX, z: oldBoardZ },
      displacement: { x: state.boardX - oldBoardX, z: state.boardZ - oldBoardZ },
      velocity: { x: state.boardVx, z: state.boardVz },
      body: [
        {
          id: 'board-front',
          offset: { x: boardForwardX * deckLength * 0.28, z: boardForwardZ * deckLength * 0.28 },
          radius: 0.34,
        },
        {
          id: 'board-rear',
          offset: { x: -boardForwardX * deckLength * 0.28, z: -boardForwardZ * deckLength * 0.28 },
          radius: 0.34,
        },
      ],
      ground: {
        grounded: state.grounded,
        friction: terrain.friction * wheelFriction,
        normalY: terrain.normal[1],
        contactRatio: state.contactRatio,
      },
    },
  )
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
      state.boardHeading = dampAngle(state.boardHeading, Math.atan2(state.boardVz, state.boardVx), 6, delta)
    } else if (contact) {
      const tangentA = { x: -contact.normal.z, z: contact.normal.x }
      const tangentB = { x: -tangentA.x, z: -tangentA.z }
      const tangent = tangentA.x * boardForwardX + tangentA.z * boardForwardZ >= 0 ? tangentA : tangentB
      const deflectionSpeed = Math.max(0.08, incomingBoardSpeed * 0.42)
      state.boardVx = tangent.x * deflectionSpeed
      state.boardVz = tangent.z * deflectionSpeed
      state.boardSpeed = deflectionSpeed
      state.boardHeading = dampAngle(state.boardHeading, Math.atan2(tangent.z, tangent.x), 6, delta)
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
      lerpAngle(state.transitionStartHeading, state.boardHeading + interaction.side * 0.62, progress) + interaction.root.yaw,
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
      ? state.resources.find((candidate) => candidate.id === state.feedingResourceId)
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
    const oldRootX = state.wormX
    const oldRootZ = state.wormZ
    const root = {
      x: state.wormX,
      z: state.wormZ,
      heading: state.wormHeading,
      vx: state.wormVx,
      vz: state.wormVz,
    }
    const movement = stepWormLocomotion(state.locomotionPlant, root, action, delta, field, ARENA_MARGIN)
    const incomingForwardSpeed = state.locomotionPlant.forwardSpeed
    const terrain = field.sample(oldRootX, oldRootZ)
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
    const collision = resolveSweptTerrariumMotion(
      collisionWorldFor(
        field,
        staticObstacles,
        state.resources,
        omittedResources,
        environmentConfig,
        state.boardHeading,
      ),
      {
        position: { x: oldRootX, z: oldRootZ },
        displacement: { x: root.x - oldRootX, z: root.z - oldRootZ },
        velocity: { x: root.vx, z: root.vz },
        body: [{ id: 'worm-root', offset: { x: 0, z: 0 }, radius: 0.13 }],
        ground: {
          grounded: true,
          friction: terrain.friction,
          normalY: terrain.normal[1],
          contactRatio: state.locomotionPlant.contactRatio,
        },
      },
    )
    state.wormX = collision.position.x
    state.wormZ = collision.position.z
    state.wormVx = collision.velocity.x
    state.wormVz = collision.velocity.z
    const collisionSpeed = Math.hypot(state.wormVx, state.wormVz)
    const priorForwardSign = Math.sign(state.locomotionPlant.forwardSpeed) || 1
    if (collision.hitObstacle || collision.hitBoundary) {
      const contact = collision.contacts[0]
      state.obstacleContactId = contact?.id ?? null
      state.collisionCount += collision.contacts.length
      if (collisionSpeed > 0.018) {
        root.heading = dampAngle(root.heading, Math.atan2(state.wormVz, state.wormVx), 5.5, delta)
      } else if (contact) {
        root.heading = dampAngle(root.heading, contactTangentHeading(state, contact.normal.x, contact.normal.z), 8.5, delta)
      }
    }
    state.wormHeading = root.heading
    state.locomotionPlant.forwardSpeed = collision.hitObstacle || collision.hitBoundary
      ? priorForwardSign * (
        collisionSpeed > 0.018
          ? collisionSpeed
          : Math.min(Math.abs(incomingForwardSpeed) * 0.42, 0.11)
      )
      : incomingForwardSpeed
    state.wormDistance += Math.hypot(collision.actualDisplacement.x, collision.actualDisplacement.z)
    state.terrainFriction = movement.friction
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
) {
  const airborneTuck = state.grounded ? 0 : MathUtils.clamp(Math.sin((state.currentAirtime / 0.92) * Math.PI), 0, 1)
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
    ? state.resources.find((resource) => resource.id === state.feedingResourceId)
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
      : (state.locomotionState === 'crawling' || state.locomotionState === 'seeking') && state.locomotionTime < 0.45
        ? 3.2
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
  const feedingCollisionWorld = state.locomotionState === 'feeding' && state.feedingResourceId
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
    const mountedLateral =
      bend * 0.15 +
      pokeCurve * 0.1 +
      victory * Math.sin(index * 0.82) * 0.048
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
    const segmentMount = interaction && (interaction.kind === 'mounting' || interaction.kind === 'dismounting')
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

    if (state.locomotionState === 'crawling' || state.locomotionState === 'seeking') {
      const plantIndex = SEGMENT_COUNT - 1 - index
      const grip = updateSegmentGroundGrip(
        state.segmentGroundContacts[index],
        targetX,
        targetZ,
        Math.abs(state.locomotionPlant.jointVelocities[plantIndex] ?? 0),
        crawlTerrain.friction,
        delta,
      )
      targetX = MathUtils.lerp(targetX, grip.x, grip.strength * 0.46)
      targetZ = MathUtils.lerp(targetZ, grip.z, grip.strength * 0.46)
      gripTotal += grip.strength
    } else if (state.locomotionState === 'feeding') {
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
    const connectiveScale = proposedDistance > connectiveStepLimit ? connectiveStepLimit / proposedDistance : 1
    let nextX = segment.x + proposedDx * connectiveScale
    const nextY = segment.y + proposedDy * connectiveScale
    let nextZ = segment.z + proposedDz * connectiveScale

    if (collisionWorld) {
      const allowBowlContact = state.locomotionState === 'feeding' && u > 0.46
      const world = allowBowlContact ? feedingCollisionWorld ?? collisionWorld : collisionWorld
      const collision = resolveSweptTerrariumMotion(world, {
        position: { x: oldX, z: oldZ },
        displacement: { x: nextX - oldX, z: nextZ - oldZ },
        velocity: { x: (nextX - oldX) / delta, z: (nextZ - oldZ) / delta },
        body: [{ id: `segment-${index}`, offset: { x: 0, z: 0 }, radius: 0.074 }],
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
    const crawlPitch =
      -(state.locomotionPlant.jointVelocities[plantSegment] ?? 0) * 0.028 * locomotionWeight
    const mountedPitch = bend * 0.19 + airborneTuck * centered * 0.78
    const targetPitch =
      MathUtils.lerp(crawlPitch, mountedPitch, segmentMount) + (interactionOffset?.pitch ?? 0)
    const targetYaw = lerpAngle(
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
}

function decodeAction(action: PolicyAction): DecodedAction {
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
  const target = state.resources.find((resource) => resource.id === targetId)
  if (!target) {
    return {
      targetForward: 1,
      targetRight: 0,
      targetDistance: 0,
      forwardSpeed: state.locomotionPlant.forwardSpeed,
      angularSpeed: state.locomotionPlant.angularSpeed,
      terrainFriction: field.sample(state.wormX, state.wormZ).friction,
      urgency: 0,
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
  }
}

function phaseFor(state: StuntState, mode: ShowcaseMode): StuntPhase {
  if (state.locomotionState === 'dismounting') return 'dismount'
  if (state.locomotionState === 'feeding') {
    return state.feedingResourceId === 'water-bowl' ? 'drinking' : 'eating'
  }
  if (state.locomotionState === 'crawling') {
    const target = state.resources.find((resource) => resource.id === state.needs.targetResourceId)
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

function stuntNameFor(state: StuntState, mode: ShowcaseMode) {
  if (state.locomotionState === 'dismounting') return 'Board dismount'
  if (state.locomotionState === 'feeding') {
    return state.feedingResourceId === 'water-bowl' ? 'Drinking at the water bowl' : 'Eating at the food bowl'
  }
  if (state.locomotionState === 'crawling') {
    const target = state.resources.find((resource) => resource.id === state.needs.targetResourceId)
    if (target?.need === 'hunger') return 'Hunting for food'
    if (target?.need === 'thirst') return 'Looking for water'
    return mode === 'freestyle' ? 'Evolved free crawl' : 'Needs-driven crawl'
  }
  if (state.locomotionState === 'seeking') return 'Finding the board'
  if (state.locomotionState === 'mounting') return 'Mounting up'
  return 'Scripted kickflip'
}

function renderStunt(
  state: StuntState,
  action: PolicyAction,
  delta: number,
  mode: ShowcaseMode,
  anatomy: WurmAnatomy,
  board: Group | null,
  wheels: Array<Mesh | null>,
  segments: Array<Group | null>,
  materials: Array<MeshStandardMaterial | null>,
  connectors: Array<Mesh | null>,
  bursts: Array<Mesh | null>,
  mouth: Mesh | null,
) {
  const renderAlpha = 1 - Math.exp(-22 * delta)
  if (board) {
    renderTarget.set(state.boardX, state.boardY, state.boardZ)
    board.position.lerp(renderTarget, renderAlpha)
    board.rotation.x = MathUtils.damp(board.rotation.x, state.boardRoll, 24, delta)
    board.rotation.y = dampAngle(board.rotation.y, state.boardYaw, 20, delta)
    board.rotation.z = MathUtils.damp(board.rotation.z, state.boardPitch, 20, delta)
  }
  for (const wheel of wheels) {
    if (wheel) wheel.rotation.y = state.wheelSpin
  }

  const decoded = decodeAction(action)
  const mount = smoothStep(state.mountBlend)
  const anchorX = MathUtils.lerp(state.wormX, state.boardX, mount)
  const anchorZ = MathUtils.lerp(state.wormZ, state.boardZ, mount)
  const heading = lerpAngle(state.wormHeading, state.boardHeading, mount)
  const forwardX = Math.cos(heading)
  const forwardZ = Math.sin(heading)
  const rightX = -forwardZ
  const rightZ = forwardX
  const plantOwnsBody = state.locomotionState === 'crawling' || state.locomotionState === 'seeking'
  const interaction =
    state.locomotionState === 'mounting' ||
    state.locomotionState === 'dismounting' ||
    state.locomotionState === 'feeding'
      ? interactionSampleFor(state)
      : null
  segments.forEach((segmentGroup, index) => {
    if (!segmentGroup) return
    const pose = state.segments[index]
    const dx = pose.x - anchorX
    const dz = pose.z - anchorZ
    const axial = dx * forwardX + dz * forwardZ
    const lateral = dx * rightX + dz * rightZ
    renderTarget.set(
      anchorX + forwardX * axial * anatomy.visualLengthScale + rightX * lateral * anatomy.motionWidthScale,
      pose.y + (anatomy.verticalScale - 1) * 0.025,
      anchorZ + forwardZ * axial * anatomy.visualLengthScale + rightZ * lateral * anatomy.motionWidthScale,
    )
    segmentGroup.position.lerp(renderTarget, renderAlpha)
    segmentGroup.rotation.x = MathUtils.damp(segmentGroup.rotation.x, 0, 20, delta)
    segmentGroup.rotation.y = dampAngle(segmentGroup.rotation.y, -pose.yaw, 18, delta)
    segmentGroup.rotation.z = MathUtils.damp(segmentGroup.rotation.z, Math.PI / 2 + pose.pitch, 18, delta)
    const muscleIndex = plantOwnsBody ? SEGMENT_COUNT - 1 - index : index
    const squeeze = 1 - Math.abs(decoded.contractions[muscleIndex] ?? 0) * 0.12
    const taper = 0.74 + Math.sin((index / (SEGMENT_COUNT - 1)) * Math.PI) * 0.28
    const interactionScale = interaction?.segments[index]
    const radialScale = interactionScale?.radialScale ?? 1
    const axialScale = interactionScale?.axialScale ?? 1
    segmentGroup.scale.x = MathUtils.damp(
      segmentGroup.scale.x,
      taper * squeeze * radialScale * anatomy.thicknessScale,
      16,
      delta,
    )
    segmentGroup.scale.y = MathUtils.damp(
      segmentGroup.scale.y,
      taper * (2 - squeeze) * axialScale * anatomy.axialScale,
      16,
      delta,
    )
    segmentGroup.scale.z = MathUtils.damp(
      segmentGroup.scale.z,
      taper * radialScale * anatomy.thicknessScale,
      16,
      delta,
    )
    const material = materials[index]
    if (material) {
      material.emissiveIntensity = 0.06 + Math.abs(decoded.bends[muscleIndex] ?? 0) * 0.28
    }
  })

  if (mouth) {
    const mouthOpen = interaction?.contact.mouthOpen ?? 0
    mouth.scale.x = MathUtils.damp(mouth.scale.x, 0.52, 18, delta)
    mouth.scale.y = MathUtils.damp(mouth.scale.y, 0.16 + mouthOpen * 0.84, 18, delta)
    mouth.scale.z = MathUtils.damp(mouth.scale.z, 0.72 + mouthOpen * 0.18, 18, delta)
  }

  connectors.forEach((connector, index) => {
    if (!connector) return
    const startGroup = segments[index]
    const endGroup = segments[index + 1]
    if (!startGroup || !endGroup) return
    connectorStart.copy(startGroup.position)
    connectorEnd.copy(endGroup.position)
    connectorDelta.copy(connectorEnd).sub(connectorStart)
    const length = connectorDelta.length()
    renderTarget.copy(connectorStart).add(connectorEnd).multiplyScalar(0.5)
    connector.position.lerp(renderTarget, renderAlpha)
    connector.scale.x = MathUtils.damp(connector.scale.x, anatomy.connectorScale, 18, delta)
    connector.scale.y = MathUtils.damp(connector.scale.y, length, 18, delta)
    connector.scale.z = MathUtils.damp(connector.scale.z, anatomy.connectorScale, 18, delta)
    connectorQuaternion.setFromUnitVectors(up, connectorDelta.normalize())
    connector.quaternion.slerp(connectorQuaternion, renderAlpha)
  })

  bursts.forEach((burst, index) => {
    if (!burst) return
    const strength = state.landingFlash
    const angle = (index / bursts.length) * Math.PI * 2
    const radius = (1 - strength) * (0.28 + (index % 3) * 0.16)
    burst.visible = strength > 0.02 && mode === 'kickflip'
    burst.position.set(
      state.boardX + Math.cos(angle) * radius,
      state.boardY + 0.08 + Math.sin(angle * 2) * 0.08 + strength * 0.18,
      state.boardZ + Math.sin(angle) * radius * 0.65,
    )
    const scale = Math.max(0.001, strength * (0.5 + (index % 2) * 0.35))
    burst.scale.setScalar(scale)
    burst.rotation.x += delta * 4
  })
}

function TerrariumResources({
  resources,
  stateRef,
}: {
  resources: readonly TerrariumResource[]
  stateRef: React.RefObject<StuntState>
}) {
  return (
    <group>
      {resources
        .filter((resource) => resource.presentation === 'bowl')
        .map((resource) => <ResourceBowl key={resource.id} resource={resource} stateRef={stateRef} />)}
    </group>
  )
}

function ResourceBowl({
  resource,
  stateRef,
}: {
  resource: TerrariumResource
  stateRef: React.RefObject<StuntState>
}) {
  const { radius, height, vesselColor, contentsColor } = resource.appearance
  const food = resource.id === 'food-bowl'
  const contentsRef = useRef<Group>(null)
  const rippleRefs = useRef<Array<Mesh | null>>([])
  useFrame((_, delta) => {
    const state = stateRef.current
    const active = state?.locomotionState === 'feeding' && state.feedingResourceId === resource.id
    const interaction = active && state ? interactionSampleFor(state) : null
    const headContact = interaction?.contact.headWeight ?? 0
    const mouth = interaction?.contact.mouthOpen ?? 0
    const swallow = interaction?.contact.swallow ?? 0
    if (contentsRef.current) {
      contentsRef.current.position.y = MathUtils.damp(contentsRef.current.position.y, mouth * 0.024, 14, delta)
      contentsRef.current.rotation.y = dampAngle(
        contentsRef.current.rotation.y,
        food ? interaction?.root.yaw ?? 0 : 0,
        12,
        delta,
      )
      const contentsScale = 1 + headContact * (food ? 0.035 : 0.018)
      contentsRef.current.scale.setScalar(MathUtils.damp(contentsRef.current.scale.x, contentsScale, 12, delta))
    }
    rippleRefs.current.forEach((ripple, index) => {
      if (!ripple) return
      ripple.visible = !food && headContact > 0.025
      const rippleScale = 0.25 + headContact * (0.85 + index * 0.4) + swallow * 0.28
      ripple.scale.setScalar(MathUtils.damp(ripple.scale.x, rippleScale, 16, delta))
      ripple.position.y = height * 0.91 + index * 0.008
    })
  })
  return (
    <group position={[resource.position[0], resource.groundHeight, resource.position[2]]}>
      <mesh position={[0, 0.018, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[radius * 1.38, 0.025, 7, 32]} />
        <meshStandardMaterial
          color={contentsColor}
          emissive={contentsColor}
          emissiveIntensity={0.42}
          opacity={0.62}
          transparent
        />
      </mesh>
      <mesh castShadow position={[0, height * 0.48, 0]}>
        <cylinderGeometry args={[radius, radius * 0.78, height, 28]} />
        <meshStandardMaterial color={vesselColor} roughness={0.68} />
      </mesh>
      <mesh position={[0, height * 0.84, 0]}>
        <cylinderGeometry args={[radius * 0.76, radius * 0.76, 0.035, 28]} />
        <meshStandardMaterial
          color={contentsColor}
          emissive={contentsColor}
          emissiveIntensity={food ? 0.05 : 0.14}
          metalness={food ? 0 : 0.12}
          opacity={food ? 1 : 0.82}
          roughness={food ? 0.82 : 0.18}
          transparent={!food}
        />
      </mesh>
      <mesh position={[0, height * 0.92, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[radius * 0.83, radius * 0.075, 8, 28]} />
        <meshStandardMaterial color={vesselColor} roughness={0.58} />
      </mesh>
      <mesh castShadow position={[radius * 0.92, 0.24, 0]}>
        <cylinderGeometry args={[0.018, 0.025, 0.38, 8]} />
        <meshStandardMaterial color={vesselColor} roughness={0.72} />
      </mesh>
      <mesh castShadow position={[radius * 0.92, 0.46, 0]} rotation={[0, 0, Math.PI / 4]}>
        <boxGeometry args={[0.14, 0.14, 0.055]} />
        <meshStandardMaterial color={contentsColor} emissive={contentsColor} emissiveIntensity={0.25} roughness={0.4} />
      </mesh>
      <group ref={contentsRef}>
        {food
          ? Array.from({ length: 9 }, (_, index) => {
          const angle = index * 2.399
          const pelletRadius = radius * (0.18 + (index % 3) * 0.13)
          return (
            <mesh
              castShadow
              key={index}
              position={[Math.cos(angle) * pelletRadius, height * 1.02, Math.sin(angle) * pelletRadius]}
              scale={0.72 + (index % 2) * 0.2}
            >
              <dodecahedronGeometry args={[0.055, 0]} />
              <meshStandardMaterial color={index % 2 === 0 ? '#d59a43' : '#b87531'} roughness={0.9} />
            </mesh>
          )
          })
          : null}
      </group>
      {!food
        ? [0, 1].map((index) => (
          <mesh
            key={`ripple-${index}`}
            position={[0, height * 0.91, 0]}
            ref={(value) => {
              rippleRefs.current[index] = value
            }}
            rotation={[Math.PI / 2, 0, 0]}
            visible={false}
          >
            <torusGeometry args={[radius * 0.46, 0.012, 6, 30]} />
            <meshStandardMaterial color="#b9edff" emissive="#59b9dc" emissiveIntensity={0.55} transparent opacity={0.68} />
          </mesh>
        ))
        : null}
    </group>
  )
}

function BoardVisual({
  boardRef,
  wheelRefs,
  environmentConfig,
}: {
  boardRef: React.RefObject<Group | null>
  wheelRefs: React.MutableRefObject<Array<Mesh | null>>
  environmentConfig: EnvironmentConfig | null
}) {
  const configured = environmentConfig?.skateboard.deckSize ?? [1.9, 0.12, 0.58]
  const deck: Vec3 = [
    MathUtils.clamp(configured[0], 1.65, 2.1),
    MathUtils.clamp(configured[1], 0.09, 0.15),
    MathUtils.clamp(configured[2], 0.52, 0.68),
  ]
  const wheelRadius = MathUtils.clamp(environmentConfig?.skateboard.wheelRadius ?? 0.12, 0.1, 0.15)
  const wheelX = deck[0] * 0.34
  const wheelZ = deck[2] * 0.68
  const wheelPositions = [
    [-wheelX, -0.15, -wheelZ],
    [-wheelX, -0.15, wheelZ],
    [wheelX, -0.15, -wheelZ],
    [wheelX, -0.15, wheelZ],
  ] as Vec3[]

  return (
    <group ref={boardRef}>
      <RoundedBox args={deck} castShadow radius={0.08} smoothness={4}>
        <meshStandardMaterial color="#d8523f" metalness={0.08} roughness={0.38} />
      </RoundedBox>
      <RoundedBox args={[deck[0] * 0.86, deck[1] + 0.012, deck[2] * 0.92]} position={[0, 0.014, 0]} radius={0.055} smoothness={3}>
        <meshStandardMaterial color="#28352f" roughness={0.82} />
      </RoundedBox>
      {[-wheelX, wheelX].map((x) => (
        <group key={`truck-${x}`} position={[x, -0.085, 0]}>
          <mesh castShadow>
            <boxGeometry args={[0.18, 0.06, deck[2] * 1.12]} />
            <meshStandardMaterial color="#d8d8ce" metalness={0.72} roughness={0.24} />
          </mesh>
          <mesh position={[0, -0.035, 0]} rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.055, 0.07, 0.12, 12]} />
            <meshStandardMaterial color="#867c6f" metalness={0.6} roughness={0.32} />
          </mesh>
        </group>
      ))}
      {wheelPositions.map((position, index) => (
        <mesh
          castShadow
          key={`wheel-${index}`}
          position={position}
          ref={(value) => {
            wheelRefs.current[index] = value
          }}
          rotation={[Math.PI / 2, 0, 0]}
        >
          <cylinderGeometry args={[wheelRadius, wheelRadius, 0.1, 20]} />
          <meshStandardMaterial color={index % 2 === 0 ? '#f2c964' : '#f19d67'} roughness={0.58} />
        </mesh>
      ))}
    </group>
  )
}

function WurmVisual({
  anatomy,
  palette,
  segmentRefs,
  materialRefs,
  connectorRefs,
  mouthRef,
}: {
  anatomy: WurmAnatomy
  palette: Color[]
  segmentRefs: React.MutableRefObject<Array<Group | null>>
  materialRefs: React.MutableRefObject<Array<MeshStandardMaterial | null>>
  connectorRefs: React.MutableRefObject<Array<Mesh | null>>
  mouthRef: React.RefObject<Mesh | null>
}) {
  return (
    <group>
      {Array.from({ length: SEGMENT_COUNT }, (_, index) => (
        <group
          key={`wurm-segment-${index}`}
          ref={(value) => {
            segmentRefs.current[index] = value
          }}
        >
          <mesh castShadow receiveShadow>
            <WurmSegmentGeometry shape={anatomy.segmentShape} />
            <meshStandardMaterial
              color={palette[index]}
              emissive={palette[index]}
              emissiveIntensity={0.08}
              ref={(value) => {
                materialRefs.current[index] = value
              }}
              roughness={0.62}
            />
          </mesh>
          {anatomy.appendages
            .filter((appendage) => appendage.anchorIndex === index)
            .map((appendage) => <GenomeAppendageVisual appendage={appendage} key={appendage.id} />)}
          {index === SEGMENT_COUNT - 1 ? <WurmFace mouthRef={mouthRef} /> : null}
        </group>
      ))}
      {Array.from({ length: SEGMENT_COUNT - 1 }, (_, index) => (
        <mesh
          castShadow
          key={`wurm-connector-${index}`}
          ref={(value) => {
            connectorRefs.current[index] = value
          }}
        >
          <cylinderGeometry args={[0.062, 0.062, 1, 12]} />
          <meshStandardMaterial color={palette[Math.min(index + 1, palette.length - 1)]} roughness={0.7} />
        </mesh>
      ))}
    </group>
  )
}

function WurmSegmentGeometry({ shape }: { shape: WurmAnatomy['segmentShape'] }) {
  if (shape === 'sphere') return <sphereGeometry args={[0.082, 16, 12]} />
  if (shape === 'box') return <boxGeometry args={[0.15, 0.11, 0.14]} />
  if (shape === 'cylinder') return <cylinderGeometry args={[0.085, 0.085, 0.13, 14]} />
  return <capsuleGeometry args={[0.085, 0.085, 5, 14]} />
}

function GenomeAppendageVisual({ appendage }: { appendage: GenomeAppendage }) {
  const lateral = appendage.side !== 0
  const side = appendage.side === 0 ? 1 : appendage.side
  return (
    <group
      position={lateral ? [0, 0, side * 0.13] : [0.12, 0, 0]}
      rotation={lateral ? [0, 0, side * 0.08] : [0, Math.PI / 2, 0]}
      scale={[appendage.thicknessScale, appendage.thicknessScale, appendage.lengthScale]}
    >
      <mesh
        castShadow
        rotation={appendage.shape === 'cylinder' || appendage.shape === 'capsule' ? [Math.PI / 2, 0, 0] : [0, 0, 0]}
      >
        <GenomeAppendageGeometry shape={appendage.shape} />
        <meshStandardMaterial color={appendage.color} roughness={0.68} />
      </mesh>
      <mesh position={[0, 0, side * 0.16]} scale={[1.12, 0.72, 0.42]}>
        <sphereGeometry args={[0.075, 10, 8]} />
        <meshStandardMaterial color={appendage.color} roughness={0.78} />
      </mesh>
    </group>
  )
}

function GenomeAppendageGeometry({ shape }: { shape: GenomeAppendage['shape'] }) {
  if (shape === 'box') return <boxGeometry args={[0.11, 0.1, 0.34]} />
  if (shape === 'sphere') return <sphereGeometry args={[0.11, 12, 10]} />
  if (shape === 'cylinder') return <cylinderGeometry args={[0.055, 0.07, 0.34, 12]} />
  return <capsuleGeometry args={[0.055, 0.24, 4, 10]} />
}

function WurmFace({ mouthRef }: { mouthRef: React.RefObject<Mesh | null> }) {
  return (
    <group rotation={[0, 0, -Math.PI / 2]}>
      {[-0.055, 0.055].map((z) => (
        <group key={z} position={[0.078, 0.055, z]}>
          <mesh>
            <sphereGeometry args={[0.032, 12, 12]} />
            <meshStandardMaterial color="#fffaf0" roughness={0.5} />
          </mesh>
          <mesh position={[0.026, 0.004, 0]}>
            <sphereGeometry args={[0.014, 10, 10]} />
            <meshStandardMaterial color="#17231e" roughness={0.35} />
          </mesh>
        </group>
      ))}
      <mesh position={[0.092, -0.018, 0]} ref={mouthRef} scale={[0.52, 0.16, 0.72]}>
        <sphereGeometry args={[0.035, 12, 9]} />
        <meshStandardMaterial color="#6f302b" roughness={0.64} />
      </mesh>
    </group>
  )
}

type TerrariumDecor = {
  stones: Array<{ id: string; position: Vec3; scale: Vec3; color: string; collisionRadius: number }>
  sprouts: Array<{ id: string; position: Vec3; rotation: number; scale: number; collisionRadius: number }>
  obstacles: TerrariumCircleObstacle[]
}

function Terrarium({
  terrain,
  environmentConfig,
  field,
}: {
  terrain: TerrariumDecor
  environmentConfig: EnvironmentConfig | null
  field: TerrainField
}) {
  const wallHeight = Math.max(2.35, environmentConfig?.world.size[1] ?? 2.8)
  const supportDepth = 0.42
  const supportTop = field.minimumHeight - 0.05
  const wallBottom = supportTop - 0.03
  const wallSpan = wallHeight - wallBottom
  const wallCenter = wallBottom + wallSpan * 0.5
  return (
    <group>
      <mesh receiveShadow position={[0, supportTop - supportDepth * 0.5, 0]}>
        <boxGeometry args={[field.width + 0.42, supportDepth, field.depth + 0.42]} />
        <meshStandardMaterial color="#7da17e" roughness={0.96} />
      </mesh>
      <TerrainGround field={field} roughness={environmentConfig?.terrain.roughness ?? 0.18} />
      {terrain.stones.map((stone, index) => (
        <mesh castShadow key={stone.id} position={stone.position} rotation={[0.2, index * 0.7, 0.1]} scale={stone.scale}>
          <dodecahedronGeometry args={[0.16, 0]} />
          <meshStandardMaterial color={stone.color} roughness={0.92} />
        </mesh>
      ))}
      {terrain.sprouts.map((sprout) => (
        <group key={sprout.id} position={sprout.position} rotation={[0, sprout.rotation, 0]} scale={sprout.scale}>
          <mesh castShadow position={[0, 0.34, 0]}>
            <cylinderGeometry args={[0.05, 0.075, 0.68, 9]} />
            <meshStandardMaterial color="#5f4a32" roughness={0.94} />
          </mesh>
          <mesh castShadow position={[0.09, 0.64, 0]} rotation={[0, 0, -0.48]}>
            <sphereGeometry args={[0.19, 14, 10]} />
            <meshStandardMaterial color="#4f865b" roughness={0.82} />
          </mesh>
          <mesh castShadow position={[-0.11, 0.55, 0.035]} rotation={[0, 0, 0.58]}>
            <sphereGeometry args={[0.16, 13, 9]} />
            <meshStandardMaterial color="#6d9f65" roughness={0.84} />
          </mesh>
          <mesh castShadow position={[0, 0.8, -0.03]}>
            <sphereGeometry args={[0.17, 13, 9]} />
            <meshStandardMaterial color="#5b965f" roughness={0.8} />
          </mesh>
          <mesh position={[0, 0.035, 0]} rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[0.095, 0.022, 6, 16]} />
            <meshStandardMaterial color="#58442f" roughness={0.98} />
          </mesh>
        </group>
      ))}
      <GlassWall position={[0, wallCenter, -field.depth * 0.5 - 0.08]} scale={[field.width + 0.42, wallSpan, 0.045]} />
      <GlassWall position={[0, wallCenter, field.depth * 0.5 + 0.08]} scale={[field.width + 0.42, wallSpan, 0.045]} />
      <GlassWall position={[-field.width * 0.5 - 0.08, wallCenter, 0]} scale={[0.045, wallSpan, field.depth + 0.42]} />
      <GlassWall position={[field.width * 0.5 + 0.08, wallCenter, 0]} scale={[0.045, wallSpan, field.depth + 0.42]} />
      <TerrariumRim bottom={supportTop} depth={field.depth} height={wallHeight} width={field.width} />
    </group>
  )
}

function TerrainGround({ field, roughness }: { field: TerrainField; roughness: number }) {
  const geometry = useMemo(() => makeTerrainGeometry(field), [field])
  useEffect(() => () => geometry.dispose(), [geometry])
  return (
    <mesh geometry={geometry} receiveShadow>
      <meshStandardMaterial roughness={MathUtils.clamp(0.78 + roughness * 0.28, 0.78, 0.98)} vertexColors />
    </mesh>
  )
}

function makeTerrainGeometry(field: TerrainField) {
  const resolution = TERRAIN_GRID_RESOLUTION
  const row = resolution + 1
  const positions: number[] = []
  const colors: number[] = []
  const indices: number[] = []
  const color = new Color()
  const sand = new Color('#c8ad78')
  const moss = new Color('#6f9c69')
  const clay = new Color('#b87555')

  for (let zIndex = 0; zIndex <= resolution; zIndex += 1) {
    const z = (zIndex / resolution - 0.5) * field.depth
    for (let xIndex = 0; xIndex <= resolution; xIndex += 1) {
      const x = (xIndex / resolution - 0.5) * field.width
      const sample = field.sample(x, z)
      positions.push(x, sample.height, z)
      color.copy(sample.surface === 'moss' ? moss : sample.surface === 'clay' ? clay : sand)
      const heightTint = MathUtils.clamp((sample.height - 0.04) * 0.16, -0.05, 0.08)
      color.offsetHSL(0, 0, heightTint)
      colors.push(color.r, color.g, color.b)
    }
  }

  for (let zIndex = 0; zIndex < resolution; zIndex += 1) {
    for (let xIndex = 0; xIndex < resolution; xIndex += 1) {
      const topLeft = zIndex * row + xIndex
      const topRight = topLeft + 1
      const bottomLeft = (zIndex + 1) * row + xIndex
      const bottomRight = bottomLeft + 1
      indices.push(topLeft, bottomLeft, topRight, topRight, bottomLeft, bottomRight)
    }
  }

  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3))
  geometry.setAttribute('color', new BufferAttribute(new Float32Array(colors), 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  return geometry
}

function GlassWall({ position, scale }: { position: Vec3; scale: Vec3 }) {
  return (
    <mesh position={position} scale={scale}>
      <boxGeometry args={[1, 1, 1]} />
      <meshPhysicalMaterial color="#c9edf0" opacity={0.11} roughness={0.05} thickness={0.12} transparent />
    </mesh>
  )
}

function TerrariumRim({ width, depth, height, bottom }: { width: number; depth: number; height: number; bottom: number }) {
  const x = width * 0.5 + 0.1
  const z = depth * 0.5 + 0.1
  const longBars: Array<{ position: Vec3; scale: Vec3 }> = [
    { position: [0, bottom, -z], scale: [width + 0.42, 0.09, 0.09] },
    { position: [0, bottom, z], scale: [width + 0.42, 0.09, 0.09] },
    { position: [0, height, -z], scale: [width + 0.42, 0.075, 0.075] },
    { position: [0, height, z], scale: [width + 0.42, 0.075, 0.075] },
    { position: [-x, bottom, 0], scale: [0.09, 0.09, depth + 0.42] },
    { position: [x, bottom, 0], scale: [0.09, 0.09, depth + 0.42] },
  ]
  return (
    <group>
      {longBars.map((bar, index) => (
        <mesh key={index} position={bar.position}>
          <boxGeometry args={bar.scale} />
          <meshStandardMaterial color="#315947" metalness={0.18} roughness={0.52} />
        </mesh>
      ))}
    </group>
  )
}

function LandingBurst({
  burstRefs,
  palette,
}: {
  burstRefs: React.MutableRefObject<Array<Mesh | null>>
  palette: Color[]
}) {
  return (
    <group>
      {Array.from({ length: 14 }, (_, index) => (
        <mesh
          key={`burst-${index}`}
          ref={(value) => {
            burstRefs.current[index] = value
          }}
          visible={false}
        >
          <tetrahedronGeometry args={[0.065, 0]} />
          <meshStandardMaterial color={palette[(index * 3) % palette.length]} emissive={palette[(index * 3) % palette.length]} emissiveIntensity={0.45} />
        </mesh>
      ))}
    </group>
  )
}

function makeWurmPalette(creature: CreatureGenome | null): Color[] {
  const configured = creature?.morphology.bodyParts.map((part) => part.visual.color) ?? []
  const start = new Color(configured[0] ?? '#ef765f')
  const middle = new Color(configured[Math.floor(configured.length / 2)] ?? '#f6c882')
  const end = new Color(configured.at(-1) ?? '#df5948')
  return Array.from({ length: SEGMENT_COUNT }, (_, index) => {
    const t = index / (SEGMENT_COUNT - 1)
    return t < 0.5 ? start.clone().lerp(middle, t * 2) : middle.clone().lerp(end, (t - 0.5) * 2)
  })
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
    const position = chooseDecorPosition(
      seed + 3,
      index,
      field,
      resources,
      occupied,
      collisionRadius,
      0.26,
    )
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
      friction: 0.74,
      restitution: 0.035,
    })
  }

  for (let index = 0; index < treeCount; index += 1) {
    const scale = 0.82 + seededNoise(seed + 32, index * 2) * 0.62
    const collisionRadius = 0.065 + scale * 0.055
    const position = chooseDecorPosition(
      seed + 21,
      index,
      field,
      resources,
      occupied,
      collisionRadius,
      0.42,
    )
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
    const clearOfResources = resources.every((resource) => {
      const clearance = resource.interactionRadius + radius + (resource.id === 'skateboard' ? 0.48 : 0.26)
      return Math.hypot(x - resource.position[0], z - resource.position[2]) >= clearance
    })
    if (!clearOfResources) continue
    const clearOfRoute = field.waypoints.every(
      (waypoint) => Math.hypot(x - waypoint[0], z - waypoint[1]) >= radius + 0.34,
    )
    if (!clearOfRoute) continue
    const clearOfDecor = occupied.every(
      (candidate) => Math.hypot(x - candidate.x, z - candidate.z) >= radius + candidate.radius + spacing,
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

function boardGroundY(field: TerrainField, x: number, z: number) {
  return field.sample(x, z).height + BOARD_CLEARANCE
}

function smoothStep(value: number) {
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
): TerrariumCollisionWorld {
  const omitted = new Set(omittedResourceIds)
  const resourceObstacles: TerrariumCircleObstacle[] = resources
    .filter((resource) => !omitted.has(resource.id))
    .flatMap((resource) => {
      if (resource.presentation === 'skateboard') {
        return skateboardFootprintObstacles(
          resource.position[0],
          resource.position[2],
          skateboardHeading,
          environmentConfig,
        )
      }
      return [{
        id: resource.id,
        kind: 'bowl' as const,
        center: { x: resource.position[0], z: resource.position[2] },
        radius: resource.appearance.radius * 0.94,
        friction: 0.72,
        restitution: 0.015,
      }]
    })
  return {
    bounds: createTerrariumArenaBounds(field.width, field.depth, ARENA_MARGIN),
    obstacles: [...staticObstacles, ...resourceObstacles],
    boundaryFriction: Math.max(0.25, (environmentConfig?.terrain.baseFriction ?? 0.9) * 0.55),
    boundaryRestitution: environmentConfig?.terrain.restitution ?? 0.04,
  }
}

function wormBodyTouchesResource(state: StuntState, resourceId: TerrariumResource['id']) {
  const resource = state.resources.find((candidate) => candidate.id === resourceId)
  if (!resource) return false
  const clearance = resource.presentation === 'skateboard' ? 0.82 : resource.appearance.radius + 0.09
  return state.segments.some(
    (segment) => Math.hypot(segment.x - resource.position[0], segment.z - resource.position[2]) < clearance,
  )
}

function wormBodyOverlapsBoardCollider(
  state: StuntState,
  environmentConfig: EnvironmentConfig | null,
) {
  const footprint = skateboardFootprintObstacles(
    state.boardX,
    state.boardZ,
    state.boardHeading,
    environmentConfig,
  )
  if (footprint.some(
    (circle) => Math.hypot(state.wormX - circle.center.x, state.wormZ - circle.center.z) < circle.radius + 0.13,
  )) return true
  return state.segments.some((segment) => footprint.some(
    (circle) => Math.hypot(segment.x - circle.center.x, segment.z - circle.center.z) < circle.radius + 0.074,
  ))
}

function skateboardFootprintObstacles(
  centerX: number,
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
        friction: 0.48,
        restitution: 0.015,
      })
    }
  }
  return circles
}

function contactTangentHeading(state: StuntState, normalX: number, normalZ: number) {
  const first = { x: -normalZ, z: normalX }
  const second = { x: normalZ, z: -normalX }
  const targetId = state.locomotionState === 'seeking' ? 'skateboard' : state.needs.targetResourceId
  const target = state.resources.find((resource) => resource.id === targetId)
  const desiredX = target ? target.position[0] - state.wormX : Math.cos(state.wormHeading)
  const desiredZ = target ? target.position[2] - state.wormZ : Math.sin(state.wormHeading)
  const tangent = first.x * desiredX + first.z * desiredZ >= second.x * desiredX + second.z * desiredZ
    ? first
    : second
  return Math.atan2(tangent.z, tangent.x)
}

function smootherStep(value: number) {
  const t = MathUtils.clamp(value, 0, 1)
  return t * t * t * (t * (t * 6 - 15) + 10)
}

function dampAngle(current: number, target: number, lambda: number, delta: number) {
  return current + wrapAngle(target - current) * (1 - Math.exp(-lambda * delta))
}

function lerpAngle(start: number, end: number, alpha: number) {
  return start + wrapAngle(end - start) * MathUtils.clamp(alpha, 0, 1)
}

function seededNoise(seed: number, value: number) {
  const x = Math.sin(seed * 12.9898 + value * 78.233) * 43758.5453
  return x - Math.floor(x)
}

function wrapAngle(angle: number) {
  return Math.atan2(Math.sin(angle), Math.cos(angle))
}
