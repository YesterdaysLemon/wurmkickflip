export const REPLAY_SCHEMA_VERSION = 1
export const REPLAY_KIND = 'wurmkickflip.replay'

export type ReplayPolicyBackend = 'scripted' | 'onnx-webgpu' | 'onnx-wasm' | 'training'

export type ReplayFallReason = 'none' | 'lost_contact' | 'fell_below_height' | 'out_of_bounds' | 'timeout'

export type ReplayVec3 = [number, number, number]

export type ReplayPose = {
  position: ReplayVec3
  velocity: ReplayVec3
  rotation: ReplayVec3
}

export type ReplayFrame = {
  time: number
  board: ReplayPose
  creatureRoot: ReplayPose
  contactRatio: number
  skateboardDiscovered: boolean
  rollingVelocity: number
  reward: number
}

export type ReplayTaskMetrics = {
  skateboardDiscoveredAt: number | null
  firstContactAt: number | null
  mountTime: number
  rollingDistance: number
  maxRollingVelocity: number
  averageContactRatio: number
  energy: number
  survivalTime: number
  fallReason: ReplayFallReason
}

export type ReplayEnvironmentSample = {
  seed: number
  gravityScale: number
  frictionScale: number
  dragScale: number
  slopeDegrees: number
  roughness: number
  obstacleDensity: number
  skateboardSpawn: [number, number]
  skateboardMass: number
  wheelFriction: number
}

export type ReplayArtifact = {
  schemaVersion: typeof REPLAY_SCHEMA_VERSION
  kind: typeof REPLAY_KIND
  replayId: string
  createdAt: string
  source: {
    creatureId: string
    environmentId: string
    policyBackend: ReplayPolicyBackend
    modelVersion: string
  }
  timestep: number
  durationSeconds: number
  frameCount: number
  environmentSample: ReplayEnvironmentSample
  taskMetrics: ReplayTaskMetrics
  frames: ReplayFrame[]
}
