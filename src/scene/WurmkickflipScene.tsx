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

type LocomotionState = 'riding' | 'dismounting' | 'crawling' | 'seeking' | 'mounting'

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
        message: `${locomotionStatus.message} Mounted stunt rig: ${stuntStatus.modelVersion}.`,
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

  useEffect(() => {
    locomotionRunner.reset()
  }, [locomotionRunner])

  const gravity = Math.abs(environmentConfig?.world.gravity[1] ?? -9.81)
  const palette = useMemo(() => makeWurmPalette(creature), [creature])
  const anatomy = useMemo(() => deriveWurmAnatomy(creature), [creature])
  const terrain = useMemo(
    () => makeTerrariumDecor(environmentConfig?.seed ?? 1337, terrainField),
    [environmentConfig?.seed, terrainField],
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
          activeNeed: needsMetrics.activeNeed,
          needTarget: needsMetrics.targetResourceId,
          needTargetDistance: needsMetrics.targetDistance,
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
    )
  })

  return (
    <group>
      <Terrarium terrain={terrain} environmentConfig={environmentConfig} field={terrainField} />
      <TerrariumResources resources={state.current.resources} />
      <BoardVisual boardRef={boardRef} environmentConfig={environmentConfig} wheelRefs={wheelRefs} />
      <WurmVisual
        anatomy={anatomy}
        connectorRefs={connectorRefs}
        materialRefs={segmentMaterialRefs}
        palette={palette}
        segmentRefs={segmentRefs}
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
) {
  state.time += delta
  state.locomotionTime += delta
  const decoded = decodeAction(action)
  state.previousAction = action
  state.poke = Math.max(0, state.poke - delta * 0.58)
  state.landingFlash = Math.max(0, state.landingFlash - delta * 1.35)

  state.resources = syncSkateboardResource(state.resources, field, [state.boardX, state.boardY, state.boardZ])
  const head = state.segments.at(-1)
  const needResources = mode === 'freestyle'
    ? state.resources.filter((resource) => resource.id !== 'skateboard')
    : state.resources
  const needsStep = advanceNeeds(state.needs, delta, {
    resources: needResources,
    wormPosition: [head?.x ?? state.wormX, head?.y ?? state.wormY, head?.z ?? state.wormZ],
    mounted: state.mountBlend > 0.92,
  })
  state.needs = needsStep.state

  updateLocomotionLifecycle(state, mode, environmentConfig)

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

  updateBoardPlanar(state, decoded, delta, field, environmentConfig)
  updateWormRoot(state, action, delta, field)
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

  updateSegments(state, decoded, delta, field)
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
) {
  if (state.locomotionState === 'riding') {
    state.mountBlend = 1
    if (mode === 'freestyle' || (state.rideLandings > 0 && state.cycleTime > 5.55 && state.needs.wellbeing < 0.08)) {
      transitionLocomotion(state, 'dismounting')
    }
    return
  }

  if (state.locomotionState === 'dismounting') {
    state.mountBlend = 1 - smoothStep(state.locomotionTime / DISMOUNT_SECONDS)
    if (state.locomotionTime >= DISMOUNT_SECONDS) {
      state.mountBlend = 0
      transitionLocomotion(state, 'crawling')
    }
    return
  }

  if (state.locomotionState === 'crawling') {
    state.mountBlend = 0
    if (mode === 'kickflip' && state.needs.targetResourceId === 'skateboard') {
      transitionLocomotion(state, 'seeking')
    }
    return
  }

  if (state.locomotionState === 'seeking') {
    state.mountBlend = 0
    if (state.needs.targetResourceId !== 'skateboard') {
      transitionLocomotion(state, 'crawling')
      return
    }
    const discoveryRadius = environmentConfig?.skateboard.discoveryRadius ?? 1.35
    if (state.distanceToBoard < Math.max(0.48, discoveryRadius * 0.55)) {
      transitionLocomotion(state, 'mounting')
    }
    return
  }

  state.mountBlend = smoothStep(state.locomotionTime / MOUNT_SECONDS)
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
  state.locomotionState = next
  state.locomotionTime = 0
  if (next === 'dismounting') {
    state.flipProgress = 0
    state.currentAirtime = 0
  }
  if (next === 'riding') resetWormLocomotionPlant(state.locomotionPlant)
}

function updateBoardPlanar(
  state: StuntState,
  decoded: DecodedAction,
  delta: number,
  field: TerrainField,
  environmentConfig: EnvironmentConfig | null,
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
  state.boardX += state.boardVx * delta
  state.boardZ += state.boardVz * delta

  if (Math.abs(state.boardX) > halfWidth) {
    state.boardX = MathUtils.clamp(state.boardX, -halfWidth, halfWidth)
    state.boardVx *= -(environmentConfig?.terrain.restitution ?? 0.04)
    state.boardHeading = Math.atan2(-state.boardZ, -state.boardX)
  }
  if (Math.abs(state.boardZ) > halfDepth) {
    state.boardZ = MathUtils.clamp(state.boardZ, -halfDepth, halfDepth)
    state.boardVz *= -(environmentConfig?.terrain.restitution ?? 0.04)
    state.boardHeading = Math.atan2(-state.boardZ, -state.boardX)
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
) {
  if (state.locomotionState === 'riding') {
    state.wormX = MathUtils.damp(state.wormX, state.boardX, 18, delta)
    state.wormZ = MathUtils.damp(state.wormZ, state.boardZ, 18, delta)
    state.wormHeading = dampAngle(state.wormHeading, state.boardHeading, 18, delta)
    state.wormVx = state.boardVx
    state.wormVz = state.boardVz
  } else if (state.locomotionState === 'dismounting') {
    const side = state.attempt % 2 === 0 ? 1 : -1
    const progress = 1 - state.mountBlend
    const targetX = state.boardX - Math.sin(state.boardHeading) * side * (0.2 + progress * 0.78)
    const targetZ = state.boardZ + Math.cos(state.boardHeading) * side * (0.2 + progress * 0.78)
    const oldX = state.wormX
    const oldZ = state.wormZ
    state.wormX = MathUtils.damp(state.wormX, targetX, 6.2, delta)
    state.wormZ = MathUtils.damp(state.wormZ, targetZ, 6.2, delta)
    state.wormHeading = dampAngle(state.wormHeading, state.boardHeading + side * 0.62, 4.5, delta)
    state.wormVx = (state.wormX - oldX) / delta
    state.wormVz = (state.wormZ - oldZ) / delta
  } else if (state.locomotionState === 'mounting') {
    const oldX = state.wormX
    const oldZ = state.wormZ
    state.wormX = MathUtils.damp(state.wormX, state.boardX, 3.2, delta)
    state.wormZ = MathUtils.damp(state.wormZ, state.boardZ, 3.2, delta)
    state.wormHeading = dampAngle(state.wormHeading, state.boardHeading, 3.4, delta)
    state.wormVx = (state.wormX - oldX) / delta
    state.wormVz = (state.wormZ - oldZ) / delta
  } else {
    const root = {
      x: state.wormX,
      z: state.wormZ,
      heading: state.wormHeading,
      vx: state.wormVx,
      vz: state.wormVz,
    }
    const movement = stepWormLocomotion(state.locomotionPlant, root, action, delta, field, ARENA_MARGIN)
    state.wormX = root.x
    state.wormZ = root.z
    state.wormHeading = root.heading
    state.wormVx = root.vx
    state.wormVz = root.vz
    state.wormDistance += movement.distance
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
) {
  const airborneTuck = state.grounded ? 0 : MathUtils.clamp(Math.sin((state.currentAirtime / 0.92) * Math.PI), 0, 1)
  const victoryProgress = MathUtils.clamp((state.cycleTime - 4.1) / (5.25 - 4.1), 0, 1)
  const victory = state.cycleTime > 4.1 && state.cycleTime < 5.25 ? Math.sin(victoryProgress * Math.PI) : 0
  const poke = state.poke
  const lengthScale = 1 - airborneTuck * 0.3 - decoded.coil * 0.1
  const centerY = state.boardY + 0.18 + airborneTuck * 0.18 + victory * 0.07
  const mount = smoothStep(state.mountBlend)
  const boardForwardX = Math.cos(state.boardHeading)
  const boardForwardZ = Math.sin(state.boardHeading)
  const boardRightX = -boardForwardZ
  const boardRightZ = boardForwardX
  const wormForwardX = Math.cos(state.wormHeading)
  const wormForwardZ = Math.sin(state.wormHeading)
  const wormRightX = -wormForwardZ
  const wormRightZ = wormForwardX
  const poseSmoothing =
    state.locomotionState === 'mounting' || state.locomotionState === 'dismounting' ? 4.5 : SEGMENT_SMOOTHING
  const locomotionPose = deriveWormLocalPose(state.locomotionPlant, state.previousAction)

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
    const crawlX = state.wormX + wormForwardX * localPose.forward + wormRightX * localPose.lateral
    const crawlZ = state.wormZ + wormForwardZ * localPose.forward + wormRightZ * localPose.lateral
    const crawlTerrain = field.sample(crawlX, crawlZ)
    const crawlY = crawlTerrain.height + 0.105 + localPose.lift

    const targetX = MathUtils.lerp(crawlX, mountedX, mount)
    const targetY = MathUtils.lerp(crawlY, mountedY, mount)
    const targetZ = MathUtils.lerp(crawlZ, mountedZ, mount)
    segment.x = MathUtils.damp(segment.x, targetX, poseSmoothing, delta)
    segment.y = MathUtils.damp(segment.y, targetY, poseSmoothing, delta)
    segment.z = MathUtils.damp(segment.z, targetZ, poseSmoothing, delta)
    segment.vx = (segment.x - oldX) / delta
    segment.vy = (segment.y - oldY) / delta
    segment.vz = (segment.z - oldZ) / delta
    const plantSegment = SEGMENT_COUNT - 1 - index
    const crawlPitch = -(state.locomotionPlant.jointVelocities[plantSegment] ?? 0) * 0.028
    const mountedPitch = bend * 0.19 + airborneTuck * centered * 0.78
    const targetPitch = MathUtils.lerp(crawlPitch, mountedPitch, mount)
    const targetYaw = lerpAngle(
      state.wormHeading + localPose.yaw,
      state.boardHeading + bend * 0.28 + pokeCurve * 0.12,
      mount,
    )
    segment.pitch = MathUtils.damp(segment.pitch, targetPitch, ROTATION_SMOOTHING, delta)
    segment.yaw = dampAngle(segment.yaw, targetYaw, ROTATION_SMOOTHING, delta)
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
  const target = state.resources.find((resource) => resource.id === state.needs.targetResourceId)
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
  if (state.locomotionState === 'crawling') {
    if (state.needs.lastInteraction === 'hunger') return 'eating'
    if (state.needs.lastInteraction === 'thirst') return 'drinking'
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
  if (state.locomotionState === 'crawling') {
    if (state.needs.lastInteraction === 'hunger') return 'Eating at the food bowl'
    if (state.needs.lastInteraction === 'thirst') return 'Drinking at the water bowl'
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
    segmentGroup.scale.x = MathUtils.damp(
      segmentGroup.scale.x,
      taper * squeeze * anatomy.thicknessScale,
      16,
      delta,
    )
    segmentGroup.scale.y = MathUtils.damp(
      segmentGroup.scale.y,
      taper * (2 - squeeze) * anatomy.axialScale,
      16,
      delta,
    )
    segmentGroup.scale.z = MathUtils.damp(segmentGroup.scale.z, taper * anatomy.thicknessScale, 16, delta)
    const material = materials[index]
    if (material) {
      material.emissiveIntensity = 0.06 + Math.abs(decoded.bends[muscleIndex] ?? 0) * 0.28
    }
  })

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

function TerrariumResources({ resources }: { resources: readonly TerrariumResource[] }) {
  return (
    <group>
      {resources
        .filter((resource) => resource.presentation === 'bowl')
        .map((resource) => <ResourceBowl key={resource.id} resource={resource} />)}
    </group>
  )
}

function ResourceBowl({ resource }: { resource: TerrariumResource }) {
  const { radius, height, vesselColor, contentsColor } = resource.appearance
  const food = resource.id === 'food-bowl'
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
}: {
  anatomy: WurmAnatomy
  palette: Color[]
  segmentRefs: React.MutableRefObject<Array<Group | null>>
  materialRefs: React.MutableRefObject<Array<MeshStandardMaterial | null>>
  connectorRefs: React.MutableRefObject<Array<Mesh | null>>
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
          {index === SEGMENT_COUNT - 1 ? <WurmFace /> : null}
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

function WurmFace() {
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
    </group>
  )
}

type TerrariumDecor = {
  stones: Array<{ position: Vec3; scale: Vec3; color: string }>
  sprouts: Array<{ position: Vec3; rotation: number; scale: number }>
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
        <mesh castShadow key={`stone-${index}`} position={stone.position} rotation={[0.2, index * 0.7, 0.1]} scale={stone.scale}>
          <dodecahedronGeometry args={[0.12, 0]} />
          <meshStandardMaterial color={stone.color} roughness={0.92} />
        </mesh>
      ))}
      {terrain.sprouts.map((sprout, index) => (
        <group key={`sprout-${index}`} position={sprout.position} rotation={[0, sprout.rotation, 0]} scale={sprout.scale}>
          <mesh castShadow position={[0, 0.16, 0]}>
            <cylinderGeometry args={[0.018, 0.028, 0.32, 8]} />
            <meshStandardMaterial color="#376c4e" roughness={0.84} />
          </mesh>
          <mesh castShadow position={[0.07, 0.27, 0]} rotation={[0, 0, -0.72]}>
            <sphereGeometry args={[0.09, 12, 8]} />
            <meshStandardMaterial color="#579162" roughness={0.78} />
          </mesh>
          <mesh castShadow position={[-0.065, 0.19, 0.02]} rotation={[0, 0, 0.72]}>
            <sphereGeometry args={[0.075, 12, 8]} />
            <meshStandardMaterial color="#6fa06d" roughness={0.78} />
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

function makeTerrariumDecor(seed: number, field: TerrainField): TerrariumDecor {
  const stones: TerrariumDecor['stones'] = []
  const sprouts: TerrariumDecor['sprouts'] = []
  const stoneColors = ['#8a9b78', '#a77f5f', '#6f8c71', '#b39b75']
  for (let index = 0; index < 30; index += 1) {
    const x = (seededNoise(seed + 3, index * 7) - 0.5) * (field.width - 1.1)
    const z = (seededNoise(seed + 8, index * 5) - 0.5) * (field.depth - 1.1)
    const scale = 0.48 + seededNoise(seed + 12, index) * 0.72
    stones.push({
      position: [x, field.sample(x, z).height + 0.1 * scale, z],
      scale: [scale * 1.25, scale * 0.75, scale],
      color: stoneColors[index % stoneColors.length],
    })
  }
  for (let index = 0; index < 14; index += 1) {
    const x = (seededNoise(seed + 21, index * 6) - 0.5) * (field.width - 1.25)
    const z = (seededNoise(seed + 29, index * 4) - 0.5) * (field.depth - 1.25)
    sprouts.push({
      position: [x, field.sample(x, z).height, z],
      rotation: seededNoise(seed + 30, index * 4) * Math.PI,
      scale: 0.72 + seededNoise(seed + 32, index * 2) * 0.55,
    })
  }
  return { stones, sprouts }
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
