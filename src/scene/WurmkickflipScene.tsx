import { Environment, Grid, OrbitControls, PerspectiveCamera } from '@react-three/drei'
import { Canvas, useFrame } from '@react-three/fiber'
import {
  BallCollider,
  CapsuleCollider,
  CuboidCollider,
  CylinderCollider,
  Physics,
  type RapierRigidBody,
  RigidBody,
} from '@react-three/rapier'
import { createRef, useEffect, useMemo, useRef } from 'react'
import type { RefObject } from 'react'
import { Color, MathUtils, PCFShadowMap } from 'three'
import type { Mesh } from 'three'
import { Vector3 } from 'three'
import type { CreatureBodyPart, CreatureGenome, EnvironmentConfig, Vec3 } from '../creature/types'
import { PolicyRunner } from '../policy/policyRunner'
import { makeInitialAction } from '../policy/simulationAdapter'
import {
  ACTION_SIZE,
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

type TerrainTile = {
  key: string
  position: Vec3
  size: Vec3
  color: string
}

type Obstacle = {
  key: string
  position: Vec3
  size: Vec3
  color: string
}

const deckColor = new Color('#70432f')
const truckColor = new Color('#aeb5b0')
const wheelColor = new Color('#202725')

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

  const gravity = environmentConfig?.world.gravity ?? [0, -9.81, 0]
  const sceneKey = `${creature?.id ?? 'creature'}-${environmentConfig?.id ?? 'environment'}-${resetNonce}`

  return (
    <Canvas shadows={{ type: PCFShadowMap }} dpr={[1, 2]}>
      <PerspectiveCamera makeDefault position={[5.2, 3.4, 5.4]} fov={42} />
      <color attach="background" args={['#e8f0ee']} />
      <fog attach="fog" args={['#e8f0ee', 6, 14]} />
      <ambientLight intensity={0.58} />
      <directionalLight castShadow intensity={2.15} position={[4, 7, 3]} shadow-mapSize={[2048, 2048]} />
      <Environment preset="park" />
      <Physics key={sceneKey} gravity={gravity} paused={!running} timeStep={POLICY_TIMESTEP}>
        <PhysicsLab
          creature={creature}
          environmentConfig={environmentConfig}
          onMetrics={onMetrics}
          policyRunner={policyRunner}
          running={running}
        />
      </Physics>
      <OrbitControls
        enableDamping
        maxDistance={11}
        maxPolarAngle={Math.PI / 2.05}
        minDistance={3.2}
        target={[0.4, 0.55, 0]}
      />
    </Canvas>
  )
}

type PhysicsLabProps = {
  creature: CreatureGenome | null
  environmentConfig: EnvironmentConfig | null
  onMetrics: (metrics: ViewerMetrics) => void
  policyRunner: PolicyRunner
  running: boolean
}

function PhysicsLab({ creature, environmentConfig, onMetrics, policyRunner, running }: PhysicsLabProps) {
  const boardRef = useRef<RapierRigidBody>(null)
  const latestAction = useRef<PolicyAction>(makeInitialAction())
  const time = useRef(0)
  const distance = useRef(0)
  const reward = useRef(0)
  const lastMetricsAt = useRef(0)
  const previousBoardX = useRef(environmentConfig?.skateboard.spawnPosition[0] ?? 1.1)
  const partRefs = usePartRefs(creature)

  useEffect(() => {
    time.current = 0
    distance.current = 0
    reward.current = 0
    lastMetricsAt.current = 0
    previousBoardX.current = environmentConfig?.skateboard.spawnPosition[0] ?? 1.1
    latestAction.current = makeInitialAction()
    policyRunner.reset()
  }, [creature?.id, environmentConfig?.id, environmentConfig?.skateboard.spawnPosition, policyRunner])

  useFrame((_, rawDelta) => {
    if (!running || !creature || !environmentConfig) {
      return
    }

    const delta = Math.min(rawDelta, 1 / 30)
    time.current += delta
    latestAction.current = makeCreatureAction(creature, time.current)

    const contactRatio = computeContactRatio(boardRef.current, partRefs, creature, environmentConfig)
    applyJointSprings(partRefs, creature)
    applyCreatureDrive(partRefs, creature, latestAction.current, time.current, contactRatio)
    applyBoardAssist(boardRef.current, latestAction.current, contactRatio, environmentConfig)

    const boardPosition = boardRef.current?.translation()
    const boardVelocity = boardRef.current?.linvel()
    const boardX = boardPosition?.x ?? previousBoardX.current
    distance.current += Math.max(0, boardX - previousBoardX.current)
    previousBoardX.current = boardX

    const energy = latestAction.current.reduce((total, value) => total + Math.abs(value), 0) / ACTION_SIZE
    const boardSpeed = Math.max(0, boardVelocity?.x ?? 0)
    reward.current = distance.current * 0.9 + contactRatio * 3 + boardSpeed * 0.6 - energy * 0.35

    if (time.current - lastMetricsAt.current < 0.1) {
      return
    }
    lastMetricsAt.current = time.current

    onMetrics({
      time: time.current,
      reward: reward.current,
      distance: distance.current,
      contactRatio,
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

  if (!creature || !environmentConfig) {
    return <ProceduralTerrain environmentConfig={environmentConfig} />
  }

  return (
    <>
      <ProceduralTerrain environmentConfig={environmentConfig} />
      <PhysicsSkateboard boardRef={boardRef} environmentConfig={environmentConfig} />
      <PhysicsCreature creature={creature} partRefs={partRefs} spawnOffset={[0, 0, 0]} />
    </>
  )
}

function usePartRefs(creature: CreatureGenome | null) {
  return useMemo(() => {
    const refs: Record<string, RefObject<RapierRigidBody>> = {}
    for (const part of creature?.morphology.bodyParts ?? []) {
      refs[part.id] = createRef<RapierRigidBody>() as RefObject<RapierRigidBody>
    }
    return refs
  }, [creature?.id, creature?.morphology.bodyParts])
}

function ProceduralTerrain({ environmentConfig }: { environmentConfig: EnvironmentConfig | null }) {
  const terrain = useMemo(() => generateTerrain(environmentConfig), [environmentConfig])
  const slope = MathUtils.degToRad(environmentConfig?.terrain.slopeDegrees ?? 0)

  return (
    <group>
      <RigidBody colliders={false} rotation={[0, 0, slope * 0.08]} type="fixed">
        <CuboidCollider args={[4.6, 0.08, 2.15]} friction={environmentConfig?.terrain.baseFriction ?? 0.9} position={[0, -0.12, 0]} />
        {terrain.obstacles.map((obstacle) => (
          <CuboidCollider
            args={[obstacle.size[0] / 2, obstacle.size[1] / 2, obstacle.size[2] / 2]}
            key={`oc-${obstacle.key}`}
            position={obstacle.position}
          />
        ))}
        <CuboidCollider args={[4.6, 1.25, 0.12]} position={[0, 1.05, -2.15]} />
        <CuboidCollider args={[4.6, 1.25, 0.12]} position={[0, 1.05, 2.15]} />
        <CuboidCollider args={[0.12, 1.25, 2.15]} position={[-4.6, 1.05, 0]} />
        <CuboidCollider args={[0.12, 1.25, 2.15]} position={[4.6, 1.05, 0]} />
        <CuboidCollider args={[4.6, 0.08, 2.15]} position={[0, 2.32, 0]} />
      </RigidBody>

      <group rotation={[0, 0, slope * 0.08]}>
        <mesh receiveShadow position={[0, -0.15, 0]}>
          <boxGeometry args={[9.2, 0.14, 4.3]} />
          <meshStandardMaterial color="#5e8f70" roughness={0.96} />
        </mesh>
        {terrain.tiles.map((tile) => (
          <mesh castShadow receiveShadow key={tile.key} position={tile.position}>
            <boxGeometry args={tile.size} />
            <meshStandardMaterial color={tile.color} roughness={0.92} />
          </mesh>
        ))}
        {terrain.obstacles.map((obstacle) => (
          <mesh castShadow receiveShadow key={obstacle.key} position={obstacle.position}>
            <boxGeometry args={obstacle.size} />
            <meshStandardMaterial color={obstacle.color} roughness={0.88} />
          </mesh>
        ))}
        <Grid
          args={[8.4, 3.5]}
          cellColor="#735f4a"
          cellSize={0.35}
          fadeDistance={8}
          fadeStrength={1}
          position={[0, 0.04, 0]}
          sectionColor="#557c60"
        />
      </group>
      <GlassPanel position={[0, 0.72, -2.15]} scale={[9.2, 1.6, 0.04]} />
      <GlassPanel position={[0, 0.72, 2.15]} scale={[9.2, 1.6, 0.04]} />
      <GlassPanel position={[-4.6, 0.72, 0]} scale={[0.04, 1.6, 4.3]} />
      <GlassPanel position={[4.6, 0.72, 0]} scale={[0.04, 1.6, 4.3]} />
      <GlassPanel position={[0, 2.32, 0]} scale={[9.2, 0.04, 4.3]} />
    </group>
  )
}

function generateTerrain(environmentConfig: EnvironmentConfig | null): { tiles: TerrainTile[]; obstacles: Obstacle[] } {
  const seed = environmentConfig?.seed ?? 1
  const roughness = environmentConfig?.terrain.roughness ?? 0.15
  const obstacleDensity = environmentConfig?.terrain.obstacleDensity ?? 0.06
  const tiles: TerrainTile[] = []
  const tileSize: Vec3 = [0.96, 0.055, 0.68]
  for (let xIndex = 0; xIndex < 8; xIndex += 1) {
    for (let zIndex = 0; zIndex < 5; zIndex += 1) {
      const noise = seededNoise(seed, xIndex * 31 + zIndex * 17)
      const height = 0.018 + noise * roughness * 0.18
      const x = -3.35 + xIndex * 0.96
      const z = -1.35 + zIndex * 0.68
      tiles.push({
        key: `tile-${xIndex}-${zIndex}`,
        position: [x, -0.04 + height / 2, z],
        size: [tileSize[0], height, tileSize[2]],
        color: noise > 0.58 ? '#c09a66' : '#b98758',
      })
    }
  }

  const obstacles: Obstacle[] = []
  const count = Math.round(3 + obstacleDensity * 18)
  for (let index = 0; index < count; index += 1) {
    const noise = seededNoise(seed + 19, index * 23)
    const x = -3.3 + seededNoise(seed + 3, index) * 6.7
    const z = -1.35 + seededNoise(seed + 7, index * 5) * 2.7
    const height = 0.08 + noise * (0.22 + roughness * 0.25)
    obstacles.push({
      key: `obstacle-${index}`,
      position: [x, height / 2 + 0.02, z],
      size: [0.16 + noise * 0.22, height, 0.16 + seededNoise(seed + 11, index) * 0.22],
      color: '#6f8f67',
    })
  }

  return { tiles, obstacles }
}

function PhysicsSkateboard({
  boardRef,
  environmentConfig,
}: {
  boardRef: RefObject<RapierRigidBody | null>
  environmentConfig: EnvironmentConfig
}) {
  const board = environmentConfig.skateboard
  const deck = board.deckSize
  const wheelRadius = board.wheelRadius
  const wheelX = deck[0] * 0.38
  const wheelZ = deck[2] * 0.72

  return (
    <RigidBody
      angularDamping={0.35}
      colliders={false}
      friction={board.wheelFriction}
      linearDamping={0.12}
      mass={board.mass}
      position={board.spawnPosition}
      ref={boardRef}
      restitution={0.05}
    >
      <CuboidCollider args={[deck[0] / 2, deck[1] / 2, deck[2] / 2]} friction={0.82} restitution={0.04} />
      {[-wheelX, wheelX].map((x) =>
        [-wheelZ, wheelZ].map((z) => (
          <BallCollider args={[wheelRadius]} friction={board.wheelFriction} key={`${x}-${z}`} position={[x, -0.17, z]} />
        )),
      )}
      <mesh castShadow receiveShadow>
        <boxGeometry args={deck} />
        <meshStandardMaterial color={deckColor} roughness={0.42} />
      </mesh>
      {[-wheelX, wheelX].map((x) => (
        <mesh castShadow key={`truck-${x}`} position={[x, 0.07, 0]}>
          <boxGeometry args={[0.16, 0.07, deck[2] * 1.16]} />
          <meshStandardMaterial color={truckColor} metalness={0.45} roughness={0.26} />
        </mesh>
      ))}
      {[-wheelX, wheelX].map((x) =>
        [-wheelZ, wheelZ].map((z) => (
          <mesh castShadow key={`wheel-${x}-${z}`} position={[x, -0.17, z]} rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[wheelRadius, wheelRadius, 0.12, 24]} />
            <meshStandardMaterial color={wheelColor} roughness={0.7} />
          </mesh>
        )),
      )}
    </RigidBody>
  )
}

function PhysicsCreature({
  creature,
  partRefs,
  spawnOffset,
}: {
  creature: CreatureGenome
  partRefs: Record<string, RefObject<RapierRigidBody>>
  spawnOffset: Vec3
}) {
  return (
    <group>
      {creature.morphology.bodyParts.map((part) => (
        <RigidBody
          angularDamping={part.shape === 'sphere' ? 0.5 : 0.82}
          colliders={false}
          key={part.id}
          linearDamping={0.28}
          mass={part.mass}
          position={addVec3(part.position, spawnOffset)}
          ref={partRefs[part.id]}
          rotation={part.rotation}
        >
          <PartCollider part={part} />
          <mesh castShadow receiveShadow scale={part.size}>
            <CreatureGeometry shape={part.shape} />
            <meshStandardMaterial color={part.visual.color} roughness={0.72} />
          </mesh>
        </RigidBody>
      ))}
      {creature.morphology.joints.map((joint) => (
        <CreatureConnector
          childRef={partRefs[joint.childId]}
          key={`connector-${joint.id}`}
          parentRef={partRefs[joint.parentId]}
        />
      ))}
    </group>
  )
}

function CreatureConnector({
  childRef,
  parentRef,
}: {
  childRef: RefObject<RapierRigidBody>
  parentRef: RefObject<RapierRigidBody>
}) {
  const meshRef = useRef<Mesh>(null)

  useFrame(() => {
    const mesh = meshRef.current
    const parent = parentRef.current?.translation()
    const child = childRef.current?.translation()
    if (!mesh || !parent || !child) return

    const start = new Vector3(parent.x, parent.y, parent.z)
    const end = new Vector3(child.x, child.y, child.z)
    const delta = end.clone().sub(start)
    const length = delta.length()
    if (length < 0.01) {
      mesh.visible = false
      return
    }
    mesh.visible = true
    mesh.position.copy(start.add(end).multiplyScalar(0.5))
    mesh.scale.set(1, length, 1)
    mesh.quaternion.setFromUnitVectors(new Vector3(0, 1, 0), delta.normalize())
  })

  return (
    <mesh castShadow ref={meshRef}>
      <cylinderGeometry args={[0.035, 0.035, 1, 10]} />
      <meshStandardMaterial color="#26342e" roughness={0.58} />
    </mesh>
  )
}

function PartCollider({ part }: { part: CreatureBodyPart }) {
  const radius = Math.max(part.size[0], part.size[2]) * 0.5
  if (part.shape === 'sphere') {
    return <BallCollider args={[Math.max(...part.size) * 0.5]} friction={part.material.friction} restitution={part.material.restitution} />
  }
  if (part.shape === 'cylinder') {
    return (
      <CylinderCollider
        args={[part.size[1] * 0.5, radius]}
        friction={part.material.friction}
        restitution={part.material.restitution}
      />
    )
  }
  if (part.shape === 'capsule') {
    return (
      <CapsuleCollider
        args={[part.size[1] * 0.42, radius]}
        friction={part.material.friction}
        restitution={part.material.restitution}
      />
    )
  }
  return (
    <CuboidCollider
      args={[part.size[0] / 2, part.size[1] / 2, part.size[2] / 2]}
      friction={part.material.friction}
      restitution={part.material.restitution}
    />
  )
}

function CreatureGeometry({ shape }: { shape: CreatureBodyPart['shape'] }) {
  if (shape === 'box') {
    return <boxGeometry args={[1, 1, 1]} />
  }
  if (shape === 'sphere') {
    return <sphereGeometry args={[0.5, 20, 20]} />
  }
  if (shape === 'cylinder') {
    return <cylinderGeometry args={[0.5, 0.5, 1, 18]} />
  }
  return <capsuleGeometry args={[0.5, 1, 6, 16]} />
}

function applyCreatureDrive(
  partRefs: Record<string, RefObject<RapierRigidBody>>,
  creature: CreatureGenome,
  action: PolicyAction,
  time: number,
  contactRatio: number,
) {
  const amplitude = creature.controller.parameters.waveAmplitude ?? 0.55
  const frequency = creature.controller.parameters.waveFrequency ?? 1.6
  const phaseOffset = creature.controller.parameters.phaseOffset ?? 0.75
  const forwardBias = creature.controller.parameters.forwardBias ?? 0.1

  creature.morphology.bodyParts.forEach((part, index) => {
    const body = partRefs[part.id]?.current
    if (!body) return

    const dorsal = action[(index * 2) % action.length] ?? 0
    const ventral = action[(index * 2 + 1) % action.length] ?? 0
    const policyDrive = MathUtils.clamp((dorsal - ventral) * 0.5, -1, 1)
    const cpgDrive = Math.sin(time * frequency * Math.PI * 2 - index * phaseOffset) * amplitude + forwardBias
    const drive = MathUtils.clamp(policyDrive * 0.45 + cpgDrive * 0.55, -1, 1)
    const massScale = Math.max(0.12, part.mass)
    const contactBoost = 1 + contactRatio * 0.6

    body.applyImpulse(
      {
        x: (0.010 + Math.abs(drive) * 0.018) * contactBoost * massScale,
        y: 0.004 * Math.sin(time * 3 + index),
        z: drive * 0.018 * massScale,
      },
      true,
    )
    body.applyTorqueImpulse(
      {
        x: drive * 0.006 * massScale,
        y: Math.sin(time * 2 + index) * 0.003 * massScale,
        z: drive * 0.012 * massScale,
      },
      true,
    )
  })
}

function makeCreatureAction(creature: CreatureGenome, time: number): PolicyAction {
  const action = makeInitialAction()
  const amplitude = creature.controller.parameters.waveAmplitude ?? 0.55
  const frequency = creature.controller.parameters.waveFrequency ?? 1.6
  const phaseOffset = creature.controller.parameters.phaseOffset ?? 0.75
  const forwardBias = creature.controller.parameters.forwardBias ?? 0.1

  for (let index = 0; index < ACTION_SIZE / 2; index += 1) {
    const phase = time * frequency * Math.PI * 2 - index * phaseOffset
    const drive = MathUtils.clamp(Math.sin(phase) * amplitude + forwardBias, -1, 1)
    action[index * 2] = drive
    action[index * 2 + 1] = -drive
  }

  return action
}

function applyJointSprings(partRefs: Record<string, RefObject<RapierRigidBody>>, creature: CreatureGenome) {
  for (const joint of creature.morphology.joints) {
    const parentPart = creature.morphology.bodyParts.find((part) => part.id === joint.parentId)
    const childPart = creature.morphology.bodyParts.find((part) => part.id === joint.childId)
    const parent = partRefs[joint.parentId]?.current
    const child = partRefs[joint.childId]?.current
    if (!parentPart || !childPart || !parent || !child) continue

    const parentPosition = parent.translation()
    const childPosition = child.translation()
    const current = new Vector3(
      childPosition.x - parentPosition.x,
      childPosition.y - parentPosition.y,
      childPosition.z - parentPosition.z,
    )
    const currentLength = current.length()
    if (currentLength < 0.001) continue

    const rest = new Vector3(
      childPart.position[0] - parentPart.position[0],
      childPart.position[1] - parentPart.position[1],
      childPart.position[2] - parentPart.position[2],
    ).length()
    const error = MathUtils.clamp(currentLength - rest, -0.35, 0.35)
    const direction = current.normalize()
    const strength = 0.018 + joint.stiffness * 0.0009
    const impulse = direction.multiplyScalar(error * strength)

    child.applyImpulse({ x: -impulse.x, y: -impulse.y, z: -impulse.z }, true)
    parent.applyImpulse({ x: impulse.x, y: impulse.y, z: impulse.z }, true)
  }
}

function applyBoardAssist(
  board: RapierRigidBody | null,
  action: PolicyAction,
  contactRatio: number,
  environmentConfig: EnvironmentConfig,
) {
  if (!board || contactRatio <= 0.04) return
  const energy = action.reduce((total, value) => total + Math.abs(value), 0) / ACTION_SIZE
  const weight = environmentConfig.rewardWeights.skateboardVelocity
  board.applyImpulse({ x: contactRatio * energy * 0.018 * weight, y: 0, z: 0 }, true)
}

function computeContactRatio(
  board: RapierRigidBody | null,
  partRefs: Record<string, RefObject<RapierRigidBody>>,
  creature: CreatureGenome,
  environmentConfig: EnvironmentConfig,
) {
  const boardPosition = board?.translation()
  if (!boardPosition) return 0
  const deck = environmentConfig.skateboard.deckSize
  let contacts = 0
  for (const part of creature.morphology.bodyParts) {
    const body = partRefs[part.id]?.current
    const position = body?.translation()
    if (!position) continue
    const onDeckX = Math.abs(position.x - boardPosition.x) < deck[0] * 0.58
    const onDeckZ = Math.abs(position.z - boardPosition.z) < deck[2] * 0.76
    const nearDeckY = Math.abs(position.y - boardPosition.y) < environmentConfig.skateboard.mountHeightTolerance
    if (onDeckX && onDeckZ && nearDeckY) {
      contacts += 1
    }
  }
  return contacts / Math.max(1, creature.morphology.bodyParts.length)
}

function GlassPanel({ position, scale }: { position: Vec3; scale: Vec3 }) {
  return (
    <mesh position={position} scale={scale}>
      <boxGeometry args={[1, 1, 1]} />
      <meshPhysicalMaterial color="#d7f4ff" metalness={0} opacity={0.2} roughness={0.05} transparent />
    </mesh>
  )
}

function addVec3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

function seededNoise(seed: number, value: number): number {
  const x = Math.sin(seed * 12.9898 + value * 78.233) * 43758.5453
  return x - Math.floor(x)
}
