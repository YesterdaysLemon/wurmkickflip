import { LOCOMOTION_CONTRACT } from './locomotionContract'

export const SEGMENT_COUNT = LOCOMOTION_CONTRACT.segmentCount
export const MUSCLE_COUNT = LOCOMOTION_CONTRACT.muscleChannelCount
export const OBSERVATION_SIZE = 174
export const ACTION_SIZE = MUSCLE_COUNT
export const POLICY_TIMESTEP = LOCOMOTION_CONTRACT.timestep

export type PolicyBackend = 'loading' | 'neural-js' | 'scripted' | 'unavailable'

export type PolicyStatus = {
  backend: PolicyBackend
  message: string
  modelVersion: string
}

export type PolicyObservation = Float32Array
export type PolicyAction = Float32Array

export type NeuralGaitPerturbationKind = 'numb-neuron' | 'reverse-sensors'

export type GaitExperimentCommand =
  | {
      sequence: number
      kind: 'numb-neuron'
      /** Anterior-to-posterior neural segment index. */
      segment: number
      durationSeconds: number
    }
  | {
      sequence: number
      kind: 'reverse-sensors'
      durationSeconds: number
    }
  | {
      sequence: number
      kind: 'lateral-shove'
      durationSeconds: number
      impulse: number
    }
  | {
      sequence: number
      kind: 'clear'
    }

export type ActiveGaitExperiment = {
  kind: NeuralGaitPerturbationKind | 'lateral-shove'
  segment: number | null
  remainingSeconds: number
}

export type GaitSegmentTelemetry = {
  /** Anterior-to-posterior neural segment index. */
  segment: number
  neuralActivation: number
  neuralDrive: number
  muscleCommand: number
  requestedMuscleCommand: number
  jointBend: number
  jointVelocity: number
  afferentJointBend: number
  afferentJointVelocity: number
  afferentContactLoad: number
  afferentSlipSpeed: number
  afferentObstacleForward: number
  afferentObstacleRight: number
  contactLoad: number
  slipSpeed: number
  obstacleForward: number
  obstacleRight: number
}

export type GaitTelemetry = {
  controllerActive: boolean
  source:
    | 'live-neural'
    | 'scripted-stunt'
    | 'lifecycle-handoff'
    | 'replay-muscles'
    | 'awaiting-sample'
    | 'unavailable'
  /** Legacy replay v1 muscle channels do not record which lifecycle ordering produced them. */
  segmentOrder: 'anterior-to-posterior' | 'unavailable'
  tractionScale: number
  tractionAvailable: boolean
  bodyForwardSpeed: number
  bodyLateralSpeed: number
  targetAlignment: number
  targetAlignmentAvailable: boolean
  activeExperiment: ActiveGaitExperiment | null
  /** Persistent acknowledgement when an experiment command could not be applied. */
  experimentNotice: string | null
  segments: GaitSegmentTelemetry[]
}

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
  hunger: number
  thirst: number
  wellbeing: number
  activeNeed: 'hunger' | 'thirst' | 'wellbeing' | null
  needTarget: 'food-bowl' | 'water-bowl' | 'skateboard' | null
  needTargetDistance: number
  gait: GaitTelemetry
}
