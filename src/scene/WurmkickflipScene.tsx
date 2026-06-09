import { Environment, Grid, OrbitControls, PerspectiveCamera, Sparkles } from '@react-three/drei'
import { Canvas, useFrame } from '@react-three/fiber'
import { CuboidCollider, Physics, RigidBody } from '@react-three/rapier'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { Group } from 'three'
import { Color, MathUtils, PCFShadowMap } from 'three'
import { CreaturePreview } from '../creature/CreaturePreview'
import type { CreatureGenome, EnvironmentConfig } from '../creature/types'
import {
  advanceSimulation,
  createInitialState,
  type SegmentPose,
  type SimState,
  toSnapshot,
} from '../policy/localSimulation'
import { PolicyRunner } from '../policy/policyRunner'
import { makeInitialAction, snapshotToObservation } from '../policy/simulationAdapter'
import {
  POLICY_TIMESTEP,
  SEGMENT_COUNT,
  type PolicyAction,
  type PolicyStatus,
  type ViewerMetrics,
} from '../policy/types'

type SceneProps = {
  policyRunner: PolicyRunner
  running: boolean
  resetNonce: number
  creature: CreatureGenome | null
  environmentConfig: EnvironmentConfig | null
  onMetrics: (metrics: ViewerMetrics) => void
  onPolicyStatus: (status: PolicyStatus) => void
}

const deckColor = new Color('#6f3f2f')
const muscleHot = new Color('#d9574a')
const muscleCool = new Color('#f3c08c')

export function WurmkickflipScene({
  policyRunner,
  running,
  resetNonce,
  creature,
  environmentConfig,
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
        <Terrarium environmentConfig={environmentConfig} />
        <SimulationLoop
          onMetrics={onMetrics}
          policyRunner={policyRunner}
          resetNonce={resetNonce}
          running={running}
        />
        <CreaturePreview genome={creature} />
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

function Terrarium({ environmentConfig }: { environmentConfig: EnvironmentConfig | null }) {
  const slope = environmentConfig?.terrain.slopeDegrees ?? 0
  const roughness = environmentConfig?.terrain.roughness ?? 0
  const obstacleDensity = environmentConfig?.terrain.obstacleDensity ?? 0

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
      <EnvironmentMarkers obstacleDensity={obstacleDensity} roughness={roughness} slope={slope} />
    </group>
  )
}

function EnvironmentMarkers({
  obstacleDensity,
  roughness,
  slope,
}: {
  obstacleDensity: number
  roughness: number
  slope: number
}) {
  const count = Math.max(2, Math.round(obstacleDensity * 32 + roughness * 10))
  return (
    <group rotation={[0, 0, MathUtils.degToRad(slope) * 0.05]}>
      {Array.from({ length: count }, (_, index) => {
        const x = -3.4 + (index % 8) * 0.92
        const z = -1.35 + Math.floor(index / 8) * 0.58
        const height = 0.025 + ((index * 17) % 7) * 0.009 * (1 + roughness)
        return (
          <mesh castShadow key={index} position={[x, 0.04 + height * 0.5, z]}>
            <boxGeometry args={[0.18, height, 0.18]} />
            <meshStandardMaterial color="#7d9b76" roughness={0.92} />
          </mesh>
        )
      })}
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
