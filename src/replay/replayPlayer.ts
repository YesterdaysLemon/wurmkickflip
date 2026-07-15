import { cloneReplayArtifact, cloneReplayFrame, cloneReplayJson } from './replayClone'
import type { ReplayArtifact, ReplayFrame, ReplayPose, ReplayVec3 } from './types'
import { validateReplayArtifact } from './replayValidation'

export type ReplayPlayerOptions = {
  /** Defaults to true. Set false only to inspect a legacy schema-v1 artifact. */
  requireIntegrity?: boolean
}

export type ReplayPlaybackSample = ReplayFrame & {
  lowerFrameIndex: number
  upperFrameIndex: number
  interpolationAlpha: number
}

/** Validates once, owns an immutable copy, and samples replay time deterministically. */
export class ReplayPlayer {
  private readonly replay: ReplayArtifact
  private readonly sourceArtifact: ReplayArtifact
  private cursorSeconds = 0

  constructor(value: unknown, options: ReplayPlayerOptions = {}) {
    const validation = validateReplayArtifact(value)
    if (!validation.ok) throw new Error(`Invalid replay artifact: ${validation.errors.join(' ')}`)
    if ((options.requireIntegrity ?? true) && (!validation.value.playback || !validation.value.integrity)) {
      throw new Error('Replay player requires a recorder-core artifact with verified integrity.')
    }
    if (validation.value.frames.length === 0) throw new Error('Replay player requires at least one frame.')
    this.replay = cloneReplayArtifact(validation.value)
    this.sourceArtifact = cloneReplayJson(value as ReplayArtifact)
  }

  get durationSeconds() {
    return this.replay.durationSeconds
  }

  get currentTime() {
    return this.cursorSeconds
  }

  get ended() {
    return this.cursorSeconds >= this.replay.durationSeconds
  }

  artifact(): ReplayArtifact {
    return cloneReplayJson(this.sourceArtifact)
  }

  frame(index: number): ReplayFrame {
    if (!Number.isInteger(index) || index < 0 || index >= this.replay.frames.length) {
      throw new Error(`Replay frame index must be an integer between 0 and ${this.replay.frames.length - 1}.`)
    }
    return cloneReplayFrame(this.replay.frames[index])
  }

  sample(timeSeconds: number): ReplayPlaybackSample {
    return sampleReplayAtTime(this.replay, timeSeconds)
  }

  seek(timeSeconds: number): ReplayPlaybackSample {
    assertFinite(timeSeconds, 'Replay seek time')
    this.cursorSeconds = clamp(timeSeconds, 0, this.replay.durationSeconds)
    return this.sample(this.cursorSeconds)
  }

  advance(deltaSeconds: number): ReplayPlaybackSample {
    assertFinite(deltaSeconds, 'Replay advance delta')
    return this.seek(this.cursorSeconds + deltaSeconds)
  }

  reset(): ReplayPlaybackSample {
    this.cursorSeconds = 0
    return this.sample(0)
  }
}

/** Low-level sampler for artifacts that have already passed replay validation. */
export function sampleReplayAtTime(replay: ReplayArtifact, timeSeconds: number): ReplayPlaybackSample {
  assertFinite(timeSeconds, 'Replay sample time')
  if (replay.frames.length === 0) throw new Error('Cannot sample a replay without frames.')

  const frames = replay.frames
  const first = frames[0]
  const lastIndex = frames.length - 1
  const last = frames[lastIndex]
  const sampleTime = clamp(timeSeconds, first.time, last.time)

  if (sampleTime <= first.time) return exactSample(first, 0)
  if (sampleTime >= last.time) return exactSample(last, lastIndex)

  let low = 0
  let high = lastIndex
  while (low <= high) {
    const middle = Math.floor((low + high) / 2)
    const frameTime = frames[middle].time
    if (frameTime === sampleTime) return exactSample(frames[middle], middle)
    if (frameTime < sampleTime) low = middle + 1
    else high = middle - 1
  }

  const upperFrameIndex = low
  const lowerFrameIndex = upperFrameIndex - 1
  const lower = frames[lowerFrameIndex]
  const upper = frames[upperFrameIndex]
  const interpolationAlpha = (sampleTime - lower.time) / (upper.time - lower.time)
  return {
    time: sampleTime,
    board: interpolatePose(lower.board, upper.board, interpolationAlpha),
    creatureRoot: interpolatePose(lower.creatureRoot, upper.creatureRoot, interpolationAlpha),
    contactRatio: lerp(lower.contactRatio, upper.contactRatio, interpolationAlpha),
    // Discrete events are left-continuous: they change only at a recorded frame.
    skateboardDiscovered: lower.skateboardDiscovered,
    rollingVelocity: lerp(lower.rollingVelocity, upper.rollingVelocity, interpolationAlpha),
    reward: lerp(lower.reward, upper.reward, interpolationAlpha),
    ...(lower.muscleActivations && upper.muscleActivations
      ? {
          muscleActivations: lower.muscleActivations.map((value, index) =>
            lerp(value, upper.muscleActivations![index], interpolationAlpha),
          ),
        }
      : {}),
    lowerFrameIndex,
    upperFrameIndex,
    interpolationAlpha,
  }
}

function exactSample(frame: ReplayFrame, index: number): ReplayPlaybackSample {
  return {
    ...cloneReplayFrame(frame),
    lowerFrameIndex: index,
    upperFrameIndex: index,
    interpolationAlpha: 0,
  }
}

function interpolatePose(lower: ReplayPose, upper: ReplayPose, alpha: number): ReplayPose {
  return {
    position: interpolateVec3(lower.position, upper.position, alpha),
    velocity: interpolateVec3(lower.velocity, upper.velocity, alpha),
    rotation: [
      interpolateAngle(lower.rotation[0], upper.rotation[0], alpha),
      interpolateAngle(lower.rotation[1], upper.rotation[1], alpha),
      interpolateAngle(lower.rotation[2], upper.rotation[2], alpha),
    ],
  }
}

function interpolateVec3(lower: ReplayVec3, upper: ReplayVec3, alpha: number): ReplayVec3 {
  return [lerp(lower[0], upper[0], alpha), lerp(lower[1], upper[1], alpha), lerp(lower[2], upper[2], alpha)]
}

function interpolateAngle(lower: number, upper: number, alpha: number) {
  const delta = Math.atan2(Math.sin(upper - lower), Math.cos(upper - lower))
  return lower + delta * alpha
}

function lerp(lower: number, upper: number, alpha: number) {
  return lower + (upper - lower) * alpha
}

function assertFinite(value: number, label: string) {
  if (!Number.isFinite(value)) throw new Error(`${label} must be finite.`)
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value))
}
