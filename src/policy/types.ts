export const SEGMENT_COUNT = 16
export const MUSCLE_COUNT = SEGMENT_COUNT * 2
export const OBSERVATION_SIZE = 174
export const ACTION_SIZE = MUSCLE_COUNT
export const POLICY_TIMESTEP = 1 / 60

export type PolicyBackend = 'loading' | 'neural-js' | 'onnx-webgpu' | 'onnx-wasm' | 'scripted'

export type PolicyStatus = {
  backend: PolicyBackend
  message: string
  modelVersion: string
}

export type PolicyMeta = {
  modelVersion: string
  modelPath: string
  observationSize: number
  actionSize: number
  timestep: number
  trainingReward: number | null
  observationMean: number[]
  observationStd: number[]
}

export type PolicyObservation = Float32Array
export type PolicyAction = Float32Array

export type SegmentSnapshot = {
  x: number
  y: number
  z: number
  vx: number
  vy: number
  vz: number
  pitch: number
  yaw: number
}

export type BoardSnapshot = {
  x: number
  y: number
  z: number
  vx: number
  vy: number
  vz: number
  pitch: number
  roll: number
  yaw: number
}

export type SimulationSnapshot = {
  time: number
  board: BoardSnapshot
  segments: SegmentSnapshot[]
  contactRatio: number
  targetDirection: [number, number, number]
  previousAction: PolicyAction
}

export type ViewerMetrics = {
  time: number
  reward: number
  distance: number
  contactRatio: number
  backend: PolicyBackend
  message: string
  running: boolean
  muscleActivity: number[]
  phase: string
  flipProgress: number
  flipsLanded: number
  airtime: number
  height: number
  speed: number
  landingQuality: number
  attempt: number
  stuntName: string
  bodySpeed: number
  mounted: boolean
  distanceToBoard: number
  terrainFriction: number
  crawlDistance: number
}
