export const REPLAY_SCHEMA_VERSION = 1
export const REPLAY_KIND = 'wurmkickflip.replay'
export const REPLAY_CORE_VERSION = 1
export const REPLAY_INTERPOLATION = 'linear-v1'
export const REPLAY_INTEGRITY_ALGORITHM = 'fnv1a-64-v1'

// The current controller lattice has 16 anatomical segments. Each segment owns
// an antagonistic dorsal/ventral actuator pair, producing 32 muscle channels.
export const REPLAY_SEGMENT_COUNT = 16
export const REPLAY_CHANNELS_PER_SEGMENT = 2
export const REPLAY_MUSCLE_CHANNEL_COUNT = REPLAY_SEGMENT_COUNT * REPLAY_CHANNELS_PER_SEGMENT

export type ReplayPolicyBackend = 'scripted' | 'neural-js' | 'onnx-webgpu' | 'onnx-wasm' | 'training'

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
  /**
   * Optional for legacy schema-v1 artifacts. Recorder-core artifacts always
   * include all 32 channels in segment order: dorsal, then ventral.
   */
  muscleActivations?: number[]
}

export type ReplayPlaybackContract = {
  coreVersion: typeof REPLAY_CORE_VERSION
  interpolation: typeof REPLAY_INTERPOLATION
  actuatorLayout: {
    segmentCount: typeof REPLAY_SEGMENT_COUNT
    channelsPerSegment: typeof REPLAY_CHANNELS_PER_SEGMENT
    muscleChannelCount: typeof REPLAY_MUSCLE_CHANNEL_COUNT
    channelOrder: ['dorsal', 'ventral']
  }
}

export type ReplayIntegrity = {
  algorithm: typeof REPLAY_INTEGRITY_ALGORITHM
  digest: string
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
  /**
   * These extensions are absent on early schema-v1 artifacts. When either is
   * present, validation requires both so a player never mistakes unchecked
   * legacy data for a recorder-core artifact.
   */
  playback?: ReplayPlaybackContract
  integrity?: ReplayIntegrity
}

export type RecordedReplayFrame = ReplayFrame & {
  muscleActivations: number[]
}

export type RecordedReplayArtifact = Omit<ReplayArtifact, 'frames' | 'playback' | 'integrity'> & {
  frames: RecordedReplayFrame[]
  playback: ReplayPlaybackContract
  integrity: ReplayIntegrity
}
