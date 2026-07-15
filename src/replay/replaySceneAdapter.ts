import type { PolicyAction, SegmentSnapshot } from '../policy/types'
import type { ReplayPlaybackSample } from './replayPlayer'
import type { ReplayRecorderFrame } from './replayRecorder'
import { REPLAY_MUSCLE_CHANNEL_COUNT, REPLAY_SEGMENT_COUNT } from './types'

export type ReplayCaptureSceneSnapshot = {
  boardX: number
  boardY: number
  boardZ: number
  boardVx: number
  boardVy: number
  boardVz: number
  boardRoll: number
  boardYaw: number
  boardPitch: number
  wormX: number
  wormY: number
  wormZ: number
  wormVx: number
  wormVz: number
  wormHeading: number
  mountBlend: number
  distanceToBoard: number
  contactRatio: number
  reward: number
  previousAction: ArrayLike<number>
}

export type ReplaySceneProjection = {
  time: number
  board: {
    x: number
    y: number
    z: number
    vx: number
    vy: number
    vz: number
    roll: number
    yaw: number
    pitch: number
    heading: number
    speed: number
    wheelSpin: number
  }
  creatureRoot: {
    x: number
    y: number
    z: number
    vx: number
    vz: number
    heading: number
  }
  contactRatio: number
  reward: number
  action: PolicyAction
  segments: SegmentSnapshot[]
}

/** Copies the exact live fixed-step state into recorder-core frame fields. */
export function replayRecorderFrameForScene(
  state: ReplayCaptureSceneSnapshot,
  skateboardDiscoveryRadius: number,
): ReplayRecorderFrame {
  const mount = Math.max(0, Math.min(1, state.mountBlend))
  return {
    board: {
      position: [state.boardX, state.boardY, state.boardZ],
      velocity: [state.boardVx, state.boardVy, state.boardVz],
      rotation: [state.boardRoll, state.boardYaw, state.boardPitch],
    },
    creatureRoot: {
      position: [
        lerp(state.wormX, state.boardX, mount),
        lerp(state.wormY, state.boardY + 0.18, mount),
        lerp(state.wormZ, state.boardZ, mount),
      ],
      velocity: [
        lerp(state.wormVx, state.boardVx, mount),
        state.boardVy * mount,
        lerp(state.wormVz, state.boardVz, mount),
      ],
      rotation: [0, lerpAngle(state.wormHeading, -state.boardYaw, mount), 0],
    },
    contactRatio: state.contactRatio,
    skateboardDiscovered: mount > 0.01 || state.distanceToBoard <= skateboardDiscoveryRadius,
    rollingVelocity: Math.hypot(state.boardVx, state.boardVz),
    reward: state.reward,
    muscleActivations: state.previousAction,
  }
}

/**
 * Projects recorder-core samples into the current visual rig. Schema v1 does
 * not store every segment pose, so the body is deterministically reconstructed
 * from the recorded root, contact ratio, and all 32 muscle channels.
 */
export function projectReplaySample(sample: ReplayPlaybackSample): ReplaySceneProjection {
  const action = new Float32Array(REPLAY_MUSCLE_CHANNEL_COUNT)
  if (sample.muscleActivations) action.set(sample.muscleActivations)

  const heading = sample.creatureRoot.rotation[1]
  const segments = reconstructReplaySegments(sample, action, heading)
  return {
    time: sample.time,
    board: {
      x: sample.board.position[0],
      y: sample.board.position[1],
      z: sample.board.position[2],
      vx: sample.board.velocity[0],
      vy: sample.board.velocity[1],
      vz: sample.board.velocity[2],
      roll: sample.board.rotation[0],
      yaw: sample.board.rotation[1],
      pitch: sample.board.rotation[2],
      heading: -sample.board.rotation[1],
      speed: Math.abs(sample.rollingVelocity),
      wheelSpin: (sample.time * sample.rollingVelocity) / 0.14,
    },
    creatureRoot: {
      x: sample.creatureRoot.position[0],
      y: sample.creatureRoot.position[1],
      z: sample.creatureRoot.position[2],
      vx: sample.creatureRoot.velocity[0],
      vz: sample.creatureRoot.velocity[2],
      heading,
    },
    contactRatio: sample.contactRatio,
    reward: sample.reward,
    action,
    segments,
  }
}

/** Converts 32 dorsal/ventral channels into the 16 pair-drive UI values. */
export function replayPairDrive(sample: ReplayPlaybackSample): number[] {
  const action = sample.muscleActivations ?? []
  return Array.from({ length: REPLAY_SEGMENT_COUNT }, (_, index) => {
    const dorsal = action[index * 2] ?? 0
    const ventral = action[index * 2 + 1] ?? 0
    return (dorsal - ventral) * 0.5
  })
}

function reconstructReplaySegments(
  sample: ReplayPlaybackSample,
  action: PolicyAction,
  rootHeading: number,
): SegmentSnapshot[] {
  const forwardX = Math.cos(rootHeading)
  const forwardZ = Math.sin(rootHeading)
  const rightX = -forwardZ
  const rightZ = forwardX
  const contactLift = (1 - sample.contactRatio) * 0.09

  return Array.from({ length: REPLAY_SEGMENT_COUNT }, (_, index) => {
    const u = index / (REPLAY_SEGMENT_COUNT - 1) - 0.5
    const dorsal = action[index * 2] ?? 0
    const ventral = action[index * 2 + 1] ?? 0
    const bend = (dorsal - ventral) * 0.5
    const contraction = (dorsal + ventral) * 0.5
    const neighborIndex = Math.min(REPLAY_SEGMENT_COUNT - 1, index + 1)
    const neighborBend = ((action[neighborIndex * 2] ?? 0) - (action[neighborIndex * 2 + 1] ?? 0)) * 0.5
    const lateral = (bend * 0.72 + neighborBend * 0.28) * 0.16
    const axial = u * (1.35 - Math.abs(contraction) * 0.08)
    return {
      x: sample.creatureRoot.position[0] + forwardX * axial + rightX * lateral,
      y: sample.creatureRoot.position[1] + contactLift + Math.max(0, contraction) * 0.035,
      z: sample.creatureRoot.position[2] + forwardZ * axial + rightZ * lateral,
      vx: sample.creatureRoot.velocity[0],
      vy: sample.creatureRoot.velocity[1],
      vz: sample.creatureRoot.velocity[2],
      pitch: contraction * 0.08,
      yaw: rootHeading + bend * 0.22,
    }
  })
}

function lerp(lower: number, upper: number, alpha: number) {
  return lower + (upper - lower) * alpha
}

function lerpAngle(lower: number, upper: number, alpha: number) {
  const delta = Math.atan2(Math.sin(upper - lower), Math.cos(upper - lower))
  return lower + delta * alpha
}
