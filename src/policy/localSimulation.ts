import { MathUtils } from 'three'
import {
  POLICY_TIMESTEP,
  SEGMENT_COUNT,
  type PolicyAction,
  type SegmentSnapshot,
  type SimulationSnapshot,
} from './types'
import { makeInitialAction } from './simulationAdapter'

export type SegmentPose = SegmentSnapshot & {
  scale: number
}

export type SimState = {
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

export function createInitialState(): SimState {
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

export function advanceSimulation(state: SimState, action: PolicyAction, delta = POLICY_TIMESTEP) {
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
  state.reward = scoreState(state)
}

export function scoreState(state: SimState): number {
  const balancePenalty = Math.abs(state.boardRoll) * 1.5 + Math.abs(state.boardPitch) * 0.8
  const energyPenalty = state.previousAction.reduce((total, value) => total + Math.abs(value), 0) * 0.003
  return state.distance * 0.7 + state.contactRatio * 4 - balancePenalty - energyPenalty
}

export function toSnapshot(state: SimState): SimulationSnapshot {
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
