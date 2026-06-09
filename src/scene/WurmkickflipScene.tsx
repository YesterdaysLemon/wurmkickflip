import { Environment, Grid, OrbitControls, PerspectiveCamera, Sparkles } from '@react-three/drei'
import { Canvas, useFrame } from '@react-three/fiber'
import { CuboidCollider, Physics, RigidBody } from '@react-three/rapier'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { Group } from 'three'
import { Color, MathUtils, PCFShadowMap } from 'three'
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

type SceneProps = {
  policyRunner: PolicyRunner
  running: boolean
  resetNonce: number
  onMetrics: (metrics: ViewerMetrics) => void
  onPolicyStatus: (status: PolicyStatus) => void
}

type SegmentPose = SegmentSnapshot & {
  scale: number
}

type SimState = {
  time: number
  boardX: number
  boardY: number
  boardZ: number
  boardVx: number
  boardVy: number
  boardVz: number
  boardPitch: number
  boardRoll: number
  boardYaw: number
  wheelSpin: number
  reward: number
  contactRatio: number
  distance: number
  segments: SegmentPose[]
  previousAction: PolicyAction
}

const deckColor = new Color('#6f3f2f')
const muscleHot = new Color('#d9574a')
const muscleCool = new Color('#f3c08c')

export function WurmkickflipScene({
  policyRunner,
  running,
  resetNonce,
  onMetrics,
  onPolicyStatus,
}: SceneProps) {
  useEffect(() => {
    let mounted = true
    policyRunner.load().then((status) => {
      if (mounted) {
        onPolicyStatus(status)
      }
    })
    return () => {
      mounted = false
    }
  }, [onPolicyStatus, policyRunner])

  return (
    <Canvas shadows={{ type: PCFShadowMap }} dpr={[1, 2]}>
      <PerspectiveCamera makeDefault position={[4.7, 3.1, 5.2]} fov={42} />
      <color attach="background" args={['#e9f2ee']} />
      <fog attach="fog" args={['#e9f2ee', 6, 13]} />
      <ambientLight intensity={0.58} />
      <directionalLight castShadow intensity={2.1} position={[4, 6, 3]} shadow-mapSize={[2048, 2048]} />
      <Environment preset="park" />
      <Physics gravity={[0, -9.81, 0]} timeStep={POLICY_TIMESTEP}>
        <Terrarium />
        <SimulationLoop
          onMetrics={onMetrics}
          policyRunner={policyRunner}
          resetNonce={resetNonce}
          running={running}
        />
      </Physics>
      <OrbitControls
        enableDamping
        maxDistance={10}
        maxPolarAngle={Math.PI / 2.08}
        minDistance={3.5}
        target={[0.6, 0.7, 0]}
      />
    </Canvas>
  )
}

type SimulationLoopProps = {
  policyRunner: PolicyRunner
  running: boolean
  resetNonce: number
  onMetrics: (metrics: ViewerMetrics) => void
}

function SimulationLoop({ policyRunner, running, resetNonce, onMetrics }: SimulationLoopProps) {
  const boardRef = useRef<Group>(null)
  const wormRef = useRef<Group>(null)
  const pendingPolicy = useRef(false)
  const latestAction = useRef<PolicyAction>(makeInitialAction())
  const sim = useRef<SimState>(createInitialState())
  const [renderState, setRenderState] = useState<SimState>(() => createInitialState())

  useEffect(() => {
    sim.current = createInitialState()
    latestAction.current = makeInitialAction()
    policyRunner.reset()
    setRenderState(sim.current)
  }, [policyRunner, resetNonce])

  useFrame((_, rawDelta) => {
    const delta = Math.min(rawDelta, 1 / 30)
    if (!running) {
      return
    }

    const current = sim.current
    const snapshot = toSnapshot(current)

    if (!pendingPolicy.current) {
      pendingPolicy.current = true
      policyRunner
        .run(snapshotToObservation(snapshot))
        .then((action) => {
          latestAction.current = action
        })
        .finally(() => {
          pendingPolicy.current = false
        })
    }

    advanceSimulation(current, latestAction.current, delta)
    current.reward = scoreState(current)
    setRenderState({ ...current, segments: current.segments.map((segment) => ({ ...segment })) })

    if (boardRef.current) {
      boardRef.current.position.set(current.boardX, current.boardY, current.boardZ)
      boardRef.current.rotation.set(current.boardPitch, current.boardYaw, current.boardRoll)
    }
    if (wormRef.current) {
      wormRef.current.position.set(0, 0, 0)
    }

    onMetrics({
      time: current.time,
      reward: current.reward,
      distance: current.distance,
      contactRatio: current.contactRatio,
      backend: policyRunner.getStatus().backend,
      message: policyRunner.getStatus().message,
      running,
      muscleActivity: Array.from({ length: SEGMENT_COUNT }, (_, index) => {
        const dorsal = latestAction.current[index * 2] ?? 0
        const ventral = latestAction.current[index * 2 + 1] ?? 0
        return MathUtils.clamp((dorsal - ventral) * 0.5, -1, 1)
      }),
    })
  })

  return (
    <>
      <group ref={boardRef}>
        <Skateboard wheelSpin={renderState.wheelSpin} />
      </group>
      <group ref={wormRef}>
        <Worm segments={renderState.segments} action={latestAction.current} />
      </group>
    </>
  )
}

function Terrarium() {
  return (
    <group>
      <RigidBody type="fixed" colliders={false}>
        <CuboidCollider args={[4.6, 0.08, 2.15]} position={[0, -0.08, 0]} />
        <CuboidCollider args={[4.6, 0.7, 0.06]} position={[0, 0.62, -2.15]} />
        <CuboidCollider args={[4.6, 0.7, 0.06]} position={[0, 0.62, 2.15]} />
        <CuboidCollider args={[0.06, 0.7, 2.15]} position={[-4.6, 0.62, 0]} />
        <CuboidCollider args={[0.06, 0.7, 2.15]} position={[4.6, 0.62, 0]} />
      </RigidBody>

      <mesh receiveShadow position={[0, -0.09, 0]}>
        <boxGeometry args={[9.2, 0.12, 4.3]} />
        <meshStandardMaterial color="#5e8f70" roughness={0.95} />
      </mesh>
      <mesh receiveShadow position={[0, -0.02, 0]}>
        <boxGeometry args={[8.65, 0.05, 3.75]} />
        <meshStandardMaterial color="#b78d61" roughness={0.9} />
      </mesh>
      <Grid
        args={[8.4, 3.5]}
        cellColor="#7b6248"
        cellSize={0.35}
        fadeDistance={8}
        fadeStrength={1}
        position={[0, 0.012, 0]}
        sectionColor="#557c60"
      />

      <GlassPanel position={[0, 0.72, -2.15]} scale={[9.2, 1.6, 0.04]} />
      <GlassPanel position={[0, 0.72, 2.15]} scale={[9.2, 1.6, 0.04]} />
      <GlassPanel position={[-4.6, 0.72, 0]} scale={[0.04, 1.6, 4.3]} />
      <GlassPanel position={[4.6, 0.72, 0]} scale={[0.04, 1.6, 4.3]} />
      <Sparkles count={32} opacity={0.35} position={[0, 1.2, 0]} scale={[8, 1.8, 3.7]} size={1.8} />
    </group>
  )
}

type GlassPanelProps = {
  position: [number, number, number]
  scale: [number, number, number]
}

function GlassPanel({ position, scale }: GlassPanelProps) {
  return (
    <mesh position={position} scale={scale}>
      <boxGeometry args={[1, 1, 1]} />
      <meshPhysicalMaterial color="#d7f4ff" metalness={0} opacity={0.2} roughness={0.05} transparent />
    </mesh>
  )
}

function Skateboard({ wheelSpin }: { wheelSpin: number }) {
  return (
    <group>
      <mesh castShadow receiveShadow position={[0, 0, 0]}>
        <boxGeometry args={[1.9, 0.12, 0.58]} />
        <meshStandardMaterial color={deckColor} roughness={0.42} />
      </mesh>
      <mesh castShadow position={[-0.78, 0.07, 0]}>
        <boxGeometry args={[0.16, 0.07, 0.66]} />
        <meshStandardMaterial color="#aeb5b0" metalness={0.45} roughness={0.26} />
      </mesh>
      <mesh castShadow position={[0.78, 0.07, 0]}>
        <boxGeometry args={[0.16, 0.07, 0.66]} />
        <meshStandardMaterial color="#aeb5b0" metalness={0.45} roughness={0.26} />
      </mesh>
      {[-0.78, 0.78].map((x) =>
        [-0.42, 0.42].map((z) => (
          <mesh castShadow key={`${x}-${z}`} position={[x, -0.13, z]} rotation={[Math.PI / 2, wheelSpin, 0]}>
            <cylinderGeometry args={[0.14, 0.14, 0.12, 24]} />
            <meshStandardMaterial color="#202725" roughness={0.7} />
          </mesh>
        )),
      )}
    </group>
  )
}

function Worm({ segments, action }: { segments: SegmentPose[]; action: PolicyAction }) {
  const materials = useMemo(
    () =>
      segments.map((_, index) => {
        const color = muscleCool.clone().lerp(muscleHot, Math.abs(action[index * 2] ?? 0))
        return color
      }),
    [action, segments],
  )

  return (
    <group>
      {segments.map((segment, index) => (
        <mesh
          castShadow
          key={index}
          position={[segment.x, segment.y, segment.z]}
          rotation={[0, segment.yaw, Math.PI / 2 + segment.pitch]}
          scale={[segment.scale, segment.scale, segment.scale]}
        >
          <capsuleGeometry args={[0.085, 0.13, 6, 12]} />
          <meshStandardMaterial color={materials[index]} roughness={0.7} />
        </mesh>
      ))}
      <mesh castShadow position={[segments[0]?.x ?? 0, (segments[0]?.y ?? 0) + 0.035, segments[0]?.z ?? 0.08]}>
        <sphereGeometry args={[0.026, 12, 12]} />
        <meshStandardMaterial color="#202725" />
      </mesh>
      <mesh castShadow position={[segments[0]?.x ?? 0, (segments[0]?.y ?? 0) + 0.035, segments[0]?.z ?? -0.08]}>
        <sphereGeometry args={[0.026, 12, 12]} />
        <meshStandardMaterial color="#202725" />
      </mesh>
    </group>
  )
}

function createInitialState(): SimState {
  return {
    time: 0,
    boardX: -1,
    boardY: 0.42,
    boardZ: 0,
    boardVx: 0,
    boardVy: 0,
    boardVz: 0,
    boardPitch: -0.03,
    boardRoll: 0.02,
    boardYaw: 0,
    wheelSpin: 0,
    reward: 0,
    contactRatio: 1,
    distance: 0,
    previousAction: makeInitialAction(),
    segments: Array.from({ length: SEGMENT_COUNT }, (_, index) => ({
      x: -1.62 + index * 0.08,
      y: 0.63,
      z: 0,
      vx: 0,
      vy: 0,
      vz: 0,
      pitch: 0,
      yaw: 0,
      scale: 1 - index * 0.012,
    })),
  }
}

function advanceSimulation(state: SimState, action: PolicyAction, delta: number) {
  const previousZ = state.boardZ
  state.time += delta
  state.previousAction = action

  let propulsion = 0
  let lean = 0
  for (let i = 0; i < SEGMENT_COUNT; i += 1) {
    const dorsal = action[i * 2] ?? 0
    const ventral = action[i * 2 + 1] ?? 0
    const bend = MathUtils.clamp((dorsal - ventral) * 0.5, -1, 1)
    propulsion += Math.abs(bend)
    lean += bend * (i / SEGMENT_COUNT - 0.5)
  }
  propulsion /= SEGMENT_COUNT

  const targetVelocity = 0.35 + propulsion * 0.95
  state.boardVx = MathUtils.damp(state.boardVx, targetVelocity, 1.8, delta)
  state.boardVz = MathUtils.damp(state.boardVz, Math.sin(state.time * 0.9) * 0.05 + lean * 0.06, 2.4, delta)
  state.boardX += state.boardVx * delta
  state.distance += Math.max(0, state.boardVx * delta)
  state.boardZ = MathUtils.clamp(state.boardZ + state.boardVz * delta, -1.15, 1.15)
  state.boardY = 0.42 + Math.sin(state.time * 1.8) * 0.012
  state.boardRoll = MathUtils.damp(state.boardRoll, lean * 0.18 + Math.sin(state.time * 2.1) * 0.025, 3.2, delta)
  state.boardPitch = MathUtils.damp(state.boardPitch, -0.04 + propulsion * 0.045, 2.5, delta)
  state.boardYaw = MathUtils.damp(state.boardYaw, state.boardVz * 0.18, 2.2, delta)
  state.wheelSpin -= state.boardVx * delta * 7

  let contacts = 0
  const spacing = 0.105
  const centerOffset = ((SEGMENT_COUNT - 1) * spacing) / 2
  for (let i = 0; i < SEGMENT_COUNT; i += 1) {
    const dorsal = action[i * 2] ?? 0
    const ventral = action[i * 2 + 1] ?? 0
    const bend = MathUtils.clamp((dorsal - ventral) * 0.5, -1, 1)
    const prior = state.segments[i]
    const x = state.boardX - centerOffset + i * spacing
    const z = state.boardZ + bend * 0.18 + Math.sin(state.time * 3.2 - i * 0.35) * 0.025
    const y = state.boardY + 0.18 + Math.abs(bend) * 0.035
    prior.vx = (x - prior.x) / Math.max(delta, 0.001)
    prior.vy = (y - prior.y) / Math.max(delta, 0.001)
    prior.vz = (z - prior.z) / Math.max(delta, 0.001)
    prior.x = x
    prior.y = y
    prior.z = z
    prior.pitch = bend * 0.36
    prior.yaw = state.boardYaw + bend * 0.18

    const onDeckX = Math.abs(x - state.boardX) < 0.98
    const onDeckZ = Math.abs(z - state.boardZ) < 0.42
    if (onDeckX && onDeckZ) {
      contacts += 1
    }
  }

  state.contactRatio = contacts / SEGMENT_COUNT
  if (state.boardX > 3.35) {
    state.boardX = -3.2
  }

  state.boardVy = 0
  state.boardVz = (state.boardZ - previousZ) / Math.max(delta, 0.001)
}

function scoreState(state: SimState): number {
  const balancePenalty = Math.abs(state.boardRoll) * 1.5 + Math.abs(state.boardPitch) * 0.8
  const energyPenalty = state.previousAction.reduce((total, value) => total + Math.abs(value), 0) * 0.003
  return state.distance * 0.7 + state.contactRatio * 4 - balancePenalty - energyPenalty
}

function toSnapshot(state: SimState): SimulationSnapshot {
  return {
    time: state.time,
    board: {
      x: state.boardX,
      y: state.boardY,
      z: state.boardZ,
      vx: state.boardVx,
      vy: state.boardVy,
      vz: state.boardVz,
      pitch: state.boardPitch,
      roll: state.boardRoll,
      yaw: state.boardYaw,
    },
    segments: state.segments,
    contactRatio: state.contactRatio,
    targetDirection: [1, 0, 0],
    previousAction: state.previousAction,
  }
}
