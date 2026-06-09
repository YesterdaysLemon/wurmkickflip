export type Vec3 = [number, number, number]

export type PrimitiveShape = 'capsule' | 'box' | 'sphere' | 'cylinder'

export type CreatureBodyPart = {
  id: string
  shape: PrimitiveShape
  parentId: string | null
  position: Vec3
  rotation: Vec3
  size: Vec3
  mass: number
  material: {
    friction: number
    restitution: number
  }
  visual: {
    color: string
  }
}

export type CreatureJoint = {
  id: string
  parentId: string
  childId: string
  type: 'hinge' | 'ball' | 'fixed'
  axis: Vec3
  anchor: Vec3
  limits: [number, number]
  stiffness: number
  damping: number
  motorStrength: number
}

export type CreatureSensor =
  | 'body_pose'
  | 'body_velocity'
  | 'joint_angles'
  | 'joint_velocities'
  | 'contacts'
  | 'target_direction'
  | 'environment_probe'

export type CreatureGenome = {
  schemaVersion: 1
  id: string
  name: string
  description: string
  morphology: {
    symmetry: 'none' | 'bilateral' | 'radial'
    bodyParts: CreatureBodyPart[]
    joints: CreatureJoint[]
  }
  controller: {
    kind: 'cpg' | 'onnx_policy' | 'hybrid'
    actionSize: number
    parameters: Record<string, number>
  }
  sensors: CreatureSensor[]
  mutation: {
    enabled: boolean
    partScaleRange: [number, number]
    massScaleRange: [number, number]
    jointLimitJitter: number
    controllerJitter: number
  }
}

export type Range = [number, number]

export type EnvironmentConfig = {
  schemaVersion: 1
  id: string
  name: string
  description: string
  seed: number
  world: {
    size: Vec3
    gravity: Vec3
    airDrag: number
    solverIterations: number
  }
  terrain: {
    kind: 'flat' | 'slope' | 'bumps' | 'obstacle_field'
    baseFriction: number
    restitution: number
    slopeDegrees: number
    roughness: number
    obstacleDensity: number
  }
  task: {
    kind: 'forward_locomotion' | 'target_reach' | 'skateboard_balance' | 'find_and_ride_skateboard'
    targetDirection: Vec3
    durationSeconds: number
    failureHeight: number
  }
  skateboard: {
    enabled: boolean
    spawnPosition: Vec3
    deckSize: Vec3
    mass: number
    wheelRadius: number
    wheelFriction: number
    discoveryRadius: number
    mountHeightTolerance: number
  }
  randomization: {
    gravityScale: Range
    frictionScale: Range
    dragScale: Range
    slopeDegrees: Range
    roughness: Range
    obstacleDensity: Range
    actuatorStrength: Range
    actuatorLatencyMs: Range
    sensorNoise: Range
    spawnYawDegrees: Range
    skateboardSpawnX: Range
    skateboardSpawnZ: Range
    skateboardMass: Range
    wheelFriction: Range
  }
  rewardWeights: {
    forwardProgress: number
    stability: number
    energy: number
    survival: number
    targetAlignment: number
    skateboardDiscovery: number
    skateboardContact: number
    skateboardVelocity: number
  }
}
