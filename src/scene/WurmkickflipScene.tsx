import { Grid, OrbitControls, PerspectiveCamera, RoundedBox } from '@react-three/drei'
import { Canvas, useFrame } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import type { Group, Mesh, MeshStandardMaterial } from 'three'
import { Color, MathUtils, Quaternion, Vector3 } from 'three'
import { deriveWurmAnatomy, type GenomeAppendage, type WurmAnatomy } from '../creature/anatomy'
import type { CreatureGenome, EnvironmentConfig, Vec3 } from '../creature/types'
import { PolicyRunner } from '../policy/policyRunner'
import { makeInitialAction, snapshotToObservation } from '../policy/simulationAdapter'
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

type StuntPhase = 'cruise' | 'coil' | 'pop' | 'kickflip' | 'landing' | 'victory wiggle' | 'free flop'

type StuntState = {
  time: number
  cycleTime: number
  boardX: number
  boardY: number
  boardZ: number
  boardVx: number
  boardVy: number
  boardVz: number
  boardPitch: number
  boardRoll: number
  boardYaw: number
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
const BOARD_GROUND_Y = 0.32
const TRACK_MIN_X = -3.7
const TRACK_MAX_X = 3.7
const up = new Vector3(0, 1, 0)
const connectorQuaternion = new Quaternion()
const connectorDelta = new Vector3()
const connectorStart = new Vector3()
const connectorEnd = new Vector3()

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
  useEffect(() => {
    let mounted = true
    void policyRunner.load().then((status) => {
      if (mounted) onPolicyStatus(status)
    })
    return () => {
      mounted = false
    }
  }, [onPolicyStatus, policyRunner])

  const sceneKey = `${creature?.id ?? 'wurm'}-${environmentConfig?.id ?? 'terrarium'}-${resetNonce}`

  return (
    <Canvas dpr={[1, 1.5]} shadows="percentage">
      <PerspectiveCamera makeDefault position={[4.9, 3.05, 5.25]} fov={43} />
      <color attach="background" args={['#dce9df']} />
      <fog attach="fog" args={['#dce9df', 7.5, 15]} />
      <hemisphereLight color="#fffaf0" groundColor="#4e7660" intensity={1.25} />
      <directionalLight
        castShadow
        color="#fff4d6"
        intensity={2.55}
        position={[-3.5, 6.5, 4.5]}
        shadow-camera-far={14}
        shadow-camera-left={-6}
        shadow-camera-right={6}
        shadow-camera-top={5}
        shadow-camera-bottom={-2}
        shadow-mapSize={[1536, 1536]}
      />
      <TerrariumWorld
        creature={creature}
        environmentConfig={environmentConfig}
        interactionNonce={interactionNonce}
        key={sceneKey}
        onMetrics={onMetrics}
        policyRunner={policyRunner}
        running={running}
        showcaseMode={showcaseMode}
      />
      <OrbitControls
        enableDamping
        enablePan={false}
        maxDistance={9.4}
        maxPolarAngle={Math.PI / 2.08}
        minDistance={3.5}
        minPolarAngle={0.42}
        target={[0, 0.6, 0]}
      />
    </Canvas>
  )
}


type TerrariumWorldProps = {
  policyRunner: PolicyRunner
  running: boolean
  interactionNonce: number
  showcaseMode: ShowcaseMode
  creature: CreatureGenome | null
  environmentConfig: EnvironmentConfig | null
  onMetrics: (metrics: ViewerMetrics) => void
}

function TerrariumWorld({
  policyRunner,
  running,
  interactionNonce,
  showcaseMode,
  creature,
  environmentConfig,
  onMetrics,
}: TerrariumWorldProps) {
  const state = useRef(createStuntState())
  const latestAction = useRef<PolicyAction>(makeInitialAction())
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

  const gravity = Math.abs(environmentConfig?.world.gravity[1] ?? -9.81)
  const palette = useMemo(() => makeWurmPalette(creature), [creature])
  const anatomy = useMemo(() => deriveWurmAnatomy(creature), [creature])
  const terrain = useMemo(() => makeTerrariumDecor(environmentConfig?.seed ?? 1337), [environmentConfig?.seed])

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
        advanceStunt(state.current, latestAction.current, POLICY_TIMESTEP, gravity, showcaseMode)
        physicsAccumulator.current -= POLICY_TIMESTEP
        steps += 1
      }

      if (inferenceAccumulator.current >= POLICY_TIMESTEP && !inferencePending.current) {
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
          height: Math.max(0, current.boardY - BOARD_GROUND_Y),
          speed: current.boardVx,
          landingQuality: current.landingQuality,
          attempt: current.attempt,
          stuntName: showcaseMode === 'kickflip' ? 'Neural kickflip' : 'Free flop',
        })
      }
    }

    renderStunt(
      state.current,
      latestAction.current,
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
      <Terrarium terrain={terrain} environmentConfig={environmentConfig} />
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

function createStuntState(): StuntState {
  return {
    time: 0,
    cycleTime: 0,
    boardX: -2.85,
    boardY: BOARD_GROUND_Y,
    boardZ: 0,
    boardVx: 0.7,
    boardVy: 0,
    boardVz: 0,
    boardPitch: 0,
    boardRoll: 0,
    boardYaw: 0,
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
    phase: 'cruise',
    segments: Array.from({ length: SEGMENT_COUNT }, (_, index) => makeSegment(index)),
    previousAction: makeInitialAction(),
  }
}

function makeSegment(index: number): SegmentSnapshot {
  const x = -2.85 + (index / (SEGMENT_COUNT - 1) - 0.5) * 1.35
  return { x, y: BOARD_GROUND_Y + 0.2, z: 0, vx: 0, vy: 0, vz: 0, pitch: 0, yaw: 0 }
}

function advanceStunt(
  state: StuntState,
  action: PolicyAction,
  delta: number,
  gravity: number,
  mode: ShowcaseMode,
) {
  const priorCycleTime = state.cycleTime
  state.time += delta
  state.cycleTime = state.time % CYCLE_SECONDS
  if (state.cycleTime < priorCycleTime) {
    state.coilMemory = 0
    state.flipProgress = 0
    if (!state.grounded) settleBoard(state)
  }

  const decoded = decodeAction(action)
  state.previousAction = action
  state.phase = phaseFor(state, mode)
  state.poke = Math.max(0, state.poke - delta * 0.58)
  state.landingFlash = Math.max(0, state.landingFlash - delta * 1.35)

  const terrainDrag = 0.018 + (decoded.energy * 0.006)
  const targetSpeed = 0.52 + decoded.propulsion * 0.72
  state.boardVx = MathUtils.damp(state.boardVx, targetSpeed, 2.8 + terrainDrag, delta)
  state.boardVz = MathUtils.damp(
    state.boardVz,
    mode === 'freestyle' ? Math.sin(state.time * 0.75) * 0.06 + state.poke * 0.08 : decoded.kick * 0.025,
    3.1,
    delta,
  )
  state.boardX += state.boardVx * delta
  state.boardZ = MathUtils.clamp(state.boardZ + state.boardVz * delta, -0.72, 0.72)
  state.distance += Math.max(0, state.boardVx * delta)
  state.wheelSpin -= state.boardVx * delta * 8.4
  if (state.boardX > TRACK_MAX_X) state.boardX = TRACK_MIN_X

  if (state.grounded) {
    state.boardY = BOARD_GROUND_Y - decoded.coil * 0.028
    state.boardPitch = MathUtils.damp(state.boardPitch, decoded.coil * -0.095 + state.poke * 0.05, 8, delta)
    state.boardRoll = MathUtils.damp(state.boardRoll, state.poke * Math.sin(state.time * 9) * 0.14, 7, delta)
    state.boardYaw = MathUtils.damp(state.boardYaw, state.boardVz * 0.22, 5, delta)
    state.coilMemory = Math.max(state.coilMemory * 0.992, decoded.coil)
    state.contactRatio = MathUtils.clamp(0.98 - state.poke * 0.38 - decoded.coil * 0.06, 0.45, 1)

    const learnedPop =
      mode === 'kickflip' &&
      state.cycleTime > 2.45 &&
      state.cycleTime < 3.2 &&
      state.coilMemory > 0.28 &&
      decoded.release > 0.22 &&
      Math.abs(decoded.kick) > 0.055

    if (learnedPop) launchKickflip(state, decoded, gravity)
  } else {
    state.currentAirtime += delta
    state.boardVy -= gravity * delta
    state.boardY += state.boardVy * delta
    state.boardRoll += state.rollVelocity * delta
    state.rollVelocity *= Math.pow(0.9985, delta * 60)
    state.boardPitch = MathUtils.damp(state.boardPitch, Math.sin(state.currentAirtime * Math.PI) * -0.12, 4.5, delta)
    state.boardYaw = MathUtils.damp(state.boardYaw, decoded.kick * 0.045, 4.5, delta)
    state.maxHeight = Math.max(state.maxHeight, state.boardY - BOARD_GROUND_Y)
    state.flipProgress = MathUtils.clamp(Math.abs(state.boardRoll) / (Math.PI * 2), 0, 1)
    state.contactRatio = MathUtils.clamp(0.14 + decoded.coil * 0.28, 0.08, 0.45)

    if (state.boardY <= BOARD_GROUND_Y && state.boardVy < 0) landKickflip(state)
  }

  updateSegments(state, decoded, delta, mode)
  state.reward =
    state.distance * 0.62 +
    state.flipsLanded * 24 +
    state.landingQuality * 6 +
    state.contactRatio * 1.8 -
    decoded.energy * 0.18
}

function launchKickflip(state: StuntState, decoded: DecodedAction, gravity: number) {
  const popStrength = MathUtils.clamp((state.coilMemory + decoded.release) * 0.72, 0.72, 1.12)
  state.grounded = false
  state.attempt += 1
  state.boardY = BOARD_GROUND_Y + 0.015
  state.boardVy = 4.48 * popStrength
  const predictedFlight = (state.boardVy * 2) / Math.max(1, gravity)
  // Once the learned kick clears its semantic threshold, scale angular speed to
  // the learned pop height so one turn finishes at the substrate, not mid-air.
  state.rollVelocity = (decoded.kick >= 0 ? 1 : -1) * ((Math.PI * 2 * 1.035) / predictedFlight)
  state.currentAirtime = 0
  state.lastAirtime = 0
  state.maxHeight = 0
  state.flipProgress = 0
  state.landingQuality = 0
  state.coilMemory = 0
}

function landKickflip(state: StuntState) {
  const turnCount = Math.abs(state.boardRoll) / (Math.PI * 2)
  const turnError = Math.abs(turnCount - 1)
  const wrappedRoll = Math.abs(wrapAngle(state.boardRoll))
  const verticalPenalty = MathUtils.clamp(Math.abs(state.boardVy + 4.4) / 9, 0, 0.25)
  const quality = MathUtils.clamp(1 - turnError * 2.6 - wrappedRoll * 0.22 - verticalPenalty, 0, 1)
  const landed = turnCount > 0.76 && turnCount < 1.24 && wrappedRoll < 0.78

  state.boardY = BOARD_GROUND_Y
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
  if (landed) state.flipsLanded += 1
}

function settleBoard(state: StuntState) {
  state.grounded = true
  state.boardY = BOARD_GROUND_Y
  state.boardVy = 0
  state.boardRoll = 0
  state.rollVelocity = 0
  state.currentAirtime = 0
  state.contactRatio = 1
}

function updateSegments(state: StuntState, decoded: DecodedAction, delta: number, mode: ShowcaseMode) {
  const airborneTuck = state.grounded ? 0 : MathUtils.clamp(Math.sin((state.currentAirtime / 0.92) * Math.PI), 0, 1)
  const victory = state.cycleTime > 4.1 && state.cycleTime < 5.25 ? Math.sin((state.cycleTime - 4.1) * Math.PI) : 0
  const freeFlop = mode === 'freestyle' ? 0.62 + Math.sin(state.time * 1.7) * 0.18 : 0
  const poke = state.poke
  const lengthScale = 1 - airborneTuck * 0.3 - decoded.coil * 0.1
  const centerY = state.boardY + 0.18 + airborneTuck * 0.18 + victory * 0.07

  for (let index = 0; index < SEGMENT_COUNT; index += 1) {
    const segment = state.segments[index]
    const oldX = segment.x
    const oldY = segment.y
    const oldZ = segment.z
    const u = index / (SEGMENT_COUNT - 1)
    const centered = u - 0.5
    const bend = decoded.bends[index] ?? 0
    const flopWave = Math.sin(state.time * (4.2 + freeFlop * 2) - index * 0.7)
    const pokeWave = Math.sin(state.time * 10.5 - index * 0.9) * poke

    segment.x = state.boardX + centered * 1.42 * lengthScale
    segment.y =
      centerY +
      Math.abs(bend) * 0.035 +
      airborneTuck * Math.cos(centered * Math.PI) * 0.08 +
      (freeFlop * flopWave + pokeWave) * 0.045 * Math.cos(centered * Math.PI * 0.65)
    segment.z =
      state.boardZ +
      bend * (0.15 + freeFlop * 0.12) +
      flopWave * (0.018 + freeFlop * 0.04) +
      pokeWave * 0.085 +
      victory * Math.sin(index * 0.9) * 0.055
    segment.vx = (segment.x - oldX) / delta
    segment.vy = (segment.y - oldY) / delta
    segment.vz = (segment.z - oldZ) / delta
    segment.pitch = bend * 0.28 + airborneTuck * centered * 0.9
    segment.yaw = bend * 0.42 + pokeWave * 0.2
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

function toSnapshot(state: StuntState): SimulationSnapshot {
  // The learned curriculum uses the original policy contract's 0.42 m deck datum.
  // Rendering sits 0.10 m lower so the simplified wheels meet the substrate.
  const observationDatum = 0.1
  return {
    time: state.cycleTime,
    board: {
      x: state.boardX,
      y: state.boardY + observationDatum,
      z: state.boardZ,
      vx: state.boardVx,
      vy: state.boardVy,
      vz: state.boardVz,
      pitch: state.boardPitch,
      roll: wrapAngle(state.boardRoll),
      yaw: state.boardYaw,
    },
    segments: state.segments.map((segment) => ({ ...segment, y: segment.y + observationDatum })),
    contactRatio: state.contactRatio,
    targetDirection: [1, 0, 0],
    previousAction: state.previousAction,
  }
}

function phaseFor(state: StuntState, mode: ShowcaseMode): StuntPhase {
  if (mode === 'freestyle') return 'free flop'
  if (!state.grounded) return 'kickflip'
  if (state.landingFlash > 0.05) return state.landingFlash > 0.45 ? 'landing' : 'victory wiggle'
  if (state.cycleTime < 2.05) return 'cruise'
  if (state.cycleTime < 2.62) return 'coil'
  if (state.cycleTime < 3.05) return 'pop'
  if (state.cycleTime < 4.15) return 'landing'
  if (state.cycleTime < 5.35) return 'victory wiggle'
  return 'cruise'
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
  if (board) {
    board.position.set(state.boardX, state.boardY, state.boardZ)
    board.rotation.set(state.boardRoll, state.boardYaw, state.boardPitch)
  }
  for (const wheel of wheels) {
    if (wheel) wheel.rotation.y = state.wheelSpin
  }

  const decoded = decodeAction(action)
  segments.forEach((segmentGroup, index) => {
    if (!segmentGroup) return
    const pose = state.segments[index]
    segmentGroup.position.set(
      state.boardX + (pose.x - state.boardX) * anatomy.visualLengthScale,
      pose.y + (anatomy.verticalScale - 1) * 0.025,
      state.boardZ + (pose.z - state.boardZ) * anatomy.motionWidthScale,
    )
    segmentGroup.rotation.set(0, -pose.yaw, Math.PI / 2 + pose.pitch)
    const squeeze = 1 - Math.abs(decoded.contractions[index] ?? 0) * 0.12
    const taper = 0.74 + Math.sin((index / (SEGMENT_COUNT - 1)) * Math.PI) * 0.28
    segmentGroup.scale.set(
      taper * squeeze * anatomy.thicknessScale,
      taper * (2 - squeeze) * anatomy.axialScale,
      taper * anatomy.thicknessScale,
    )
    const material = materials[index]
    if (material) {
      material.emissiveIntensity = 0.06 + Math.abs(decoded.bends[index] ?? 0) * 0.28
    }
  })

  connectors.forEach((connector, index) => {
    if (!connector) return
    const startPose = state.segments[index]
    const endPose = state.segments[index + 1]
    connectorStart.set(
      state.boardX + (startPose.x - state.boardX) * anatomy.visualLengthScale,
      startPose.y + (anatomy.verticalScale - 1) * 0.025,
      state.boardZ + (startPose.z - state.boardZ) * anatomy.motionWidthScale,
    )
    connectorEnd.set(
      state.boardX + (endPose.x - state.boardX) * anatomy.visualLengthScale,
      endPose.y + (anatomy.verticalScale - 1) * 0.025,
      state.boardZ + (endPose.z - state.boardZ) * anatomy.motionWidthScale,
    )
    connectorDelta.copy(connectorEnd).sub(connectorStart)
    const length = connectorDelta.length()
    connector.position.copy(connectorStart).add(connectorEnd).multiplyScalar(0.5)
    connector.scale.set(anatomy.connectorScale, length, anatomy.connectorScale)
    connectorQuaternion.setFromUnitVectors(up, connectorDelta.normalize())
    connector.quaternion.copy(connectorQuaternion)
  })

  bursts.forEach((burst, index) => {
    if (!burst) return
    const strength = state.landingFlash
    const angle = (index / bursts.length) * Math.PI * 2
    const radius = (1 - strength) * (0.28 + (index % 3) * 0.16)
    burst.visible = strength > 0.02 && mode === 'kickflip'
    burst.position.set(
      state.boardX + Math.cos(angle) * radius,
      BOARD_GROUND_Y + 0.08 + Math.sin(angle * 2) * 0.08 + strength * 0.18,
      state.boardZ + Math.sin(angle) * radius * 0.65,
    )
    const scale = Math.max(0.001, strength * (0.5 + (index % 2) * 0.35))
    burst.scale.setScalar(scale)
    burst.rotation.x += delta * 4
  })
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
}: {
  terrain: TerrariumDecor
  environmentConfig: EnvironmentConfig | null
}) {
  const terrainKind = environmentConfig?.terrain.kind ?? 'bumps'
  return (
    <group>
      <mesh receiveShadow position={[0, -0.17, 0]}>
        <boxGeometry args={[9.2, 0.34, 4.3]} />
        <meshStandardMaterial color="#7da17e" roughness={0.96} />
      </mesh>
      <mesh receiveShadow position={[0, 0.012, 0]}>
        <boxGeometry args={[8.78, 0.055, 3.88]} />
        <meshStandardMaterial color={terrainKind === 'slope' ? '#c69c66' : '#c8ad78'} roughness={0.92} />
      </mesh>
      <Grid
        args={[8.5, 3.6]}
        cellColor="#8c7355"
        cellSize={0.42}
        fadeDistance={10}
        fadeStrength={1}
        position={[0, 0.045, 0]}
        sectionColor="#aa8b60"
        sectionSize={1.68}
      />
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
      <GlassWall position={[0, 1.08, -2.12]} scale={[9.2, 2.15, 0.045]} />
      <GlassWall position={[0, 1.08, 2.12]} scale={[9.2, 2.15, 0.045]} />
      <GlassWall position={[-4.58, 1.08, 0]} scale={[0.045, 2.15, 4.25]} />
      <GlassWall position={[4.58, 1.08, 0]} scale={[0.045, 2.15, 4.25]} />
      <TerrariumRim />
    </group>
  )
}

function GlassWall({ position, scale }: { position: Vec3; scale: Vec3 }) {
  return (
    <mesh position={position} scale={scale}>
      <boxGeometry args={[1, 1, 1]} />
      <meshPhysicalMaterial color="#c9edf0" opacity={0.11} roughness={0.05} thickness={0.12} transparent />
    </mesh>
  )
}

function TerrariumRim() {
  const longBars: Array<{ position: Vec3; scale: Vec3 }> = [
    { position: [0, 0.06, -2.14], scale: [9.25, 0.09, 0.09] },
    { position: [0, 0.06, 2.14], scale: [9.25, 0.09, 0.09] },
    { position: [0, 2.15, -2.14], scale: [9.25, 0.075, 0.075] },
    { position: [0, 2.15, 2.14], scale: [9.25, 0.075, 0.075] },
    { position: [-4.6, 0.06, 0], scale: [0.09, 0.09, 4.35] },
    { position: [4.6, 0.06, 0], scale: [0.09, 0.09, 4.35] },
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

function makeTerrariumDecor(seed: number): TerrariumDecor {
  const stones: TerrariumDecor['stones'] = []
  const sprouts: TerrariumDecor['sprouts'] = []
  const stoneColors = ['#8a9b78', '#a77f5f', '#6f8c71', '#b39b75']
  for (let index = 0; index < 22; index += 1) {
    const edge = seededNoise(seed, index * 11) > 0.5 ? 1 : -1
    const x = -4.05 + seededNoise(seed + 3, index * 7) * 8.1
    const z = edge * (1.48 + seededNoise(seed + 8, index * 5) * 0.34)
    const scale = 0.48 + seededNoise(seed + 12, index) * 0.72
    stones.push({
      position: [x, 0.1 * scale, z],
      scale: [scale * 1.25, scale * 0.75, scale],
      color: stoneColors[index % stoneColors.length],
    })
  }
  for (let index = 0; index < 8; index += 1) {
    const edge = index % 2 === 0 ? 1 : -1
    sprouts.push({
      position: [-3.8 + index * 1.08, 0.04, edge * (1.62 + seededNoise(seed + 21, index) * 0.16)],
      rotation: seededNoise(seed + 30, index * 4) * Math.PI,
      scale: 0.72 + seededNoise(seed + 32, index * 2) * 0.55,
    })
  }
  return { stones, sprouts }
}

function seededNoise(seed: number, value: number) {
  const x = Math.sin(seed * 12.9898 + value * 78.233) * 43758.5453
  return x - Math.floor(x)
}

function wrapAngle(angle: number) {
  return Math.atan2(Math.sin(angle), Math.cos(angle))
}
