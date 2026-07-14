import {
  ACTION_SIZE,
  OBSERVATION_SIZE,
  SEGMENT_COUNT,
  type PolicyAction,
  type PolicyObservation,
  type SimulationSnapshot,
} from './types'

export function makeInitialAction(): PolicyAction {
  return new Float32Array(ACTION_SIZE)
}

export function snapshotToObservation(snapshot: SimulationSnapshot): PolicyObservation {
  const observation = new Float32Array(OBSERVATION_SIZE)
  let index = 0

  const push = (value: number) => {
    if (index < observation.length) {
      observation[index] = Number.isFinite(value) ? value : 0
      index += 1
    }
  }

  push(snapshot.time)
  push(snapshot.board.x)
  push(snapshot.board.y)
  push(snapshot.board.z)
  push(snapshot.board.vx)
  push(snapshot.board.vy)
  push(snapshot.board.vz)
  push(snapshot.board.pitch)
  push(snapshot.board.roll)
  push(snapshot.board.yaw)
  push(snapshot.contactRatio)
  push(snapshot.targetDirection[0])
  push(snapshot.targetDirection[1])
  push(snapshot.targetDirection[2])

  for (let i = 0; i < SEGMENT_COUNT; i += 1) {
    const segment = snapshot.segments[i]
    if (!segment) {
      for (let field = 0; field < 8; field += 1) {
        push(0)
      }
      continue
    }
    push(segment.x - snapshot.board.x)
    push(segment.y - snapshot.board.y)
    push(segment.z - snapshot.board.z)
    push(segment.vx)
    push(segment.vy)
    push(segment.vz)
    push(segment.pitch)
    push(segment.yaw)
  }

  for (let i = 0; i < ACTION_SIZE; i += 1) {
    push(snapshot.previousAction[i] ?? 0)
  }

  return observation
}

export function scoreSnapshot(snapshot: SimulationSnapshot): number {
  const forwardProgress = snapshot.board.x * 0.7
  const contact = snapshot.contactRatio * 4
  const balancePenalty = Math.abs(snapshot.board.roll) * 1.5 + Math.abs(snapshot.board.pitch) * 0.8
  const energyPenalty = snapshot.previousAction.reduce((total, value) => total + Math.abs(value), 0) * 0.003

  return forwardProgress + contact - balancePenalty - energyPenalty
}
