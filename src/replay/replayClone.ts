import type {
  RecordedReplayArtifact,
  ReplayArtifact,
  ReplayEnvironmentSample,
  ReplayFrame,
  ReplayPose,
  ReplayTaskMetrics,
} from './types'

export function cloneReplayPose(pose: ReplayPose): ReplayPose {
  return {
    position: [...pose.position],
    velocity: [...pose.velocity],
    rotation: [...pose.rotation],
  }
}

export function cloneReplayFrame(frame: ReplayFrame): ReplayFrame {
  return {
    time: frame.time,
    board: cloneReplayPose(frame.board),
    creatureRoot: cloneReplayPose(frame.creatureRoot),
    contactRatio: frame.contactRatio,
    skateboardDiscovered: frame.skateboardDiscovered,
    rollingVelocity: frame.rollingVelocity,
    reward: frame.reward,
    ...(frame.muscleActivations ? { muscleActivations: [...frame.muscleActivations] } : {}),
  }
}

export function cloneReplayEnvironmentSample(sample: ReplayEnvironmentSample): ReplayEnvironmentSample {
  return {
    ...sample,
    skateboardSpawn: [...sample.skateboardSpawn],
  }
}

export function cloneReplayTaskMetrics(metrics: ReplayTaskMetrics): ReplayTaskMetrics {
  return { ...metrics }
}

export function cloneReplayArtifact(artifact: ReplayArtifact): ReplayArtifact {
  return {
    schemaVersion: artifact.schemaVersion,
    kind: artifact.kind,
    replayId: artifact.replayId,
    createdAt: artifact.createdAt,
    source: { ...artifact.source },
    timestep: artifact.timestep,
    durationSeconds: artifact.durationSeconds,
    frameCount: artifact.frameCount,
    environmentSample: cloneReplayEnvironmentSample(artifact.environmentSample),
    taskMetrics: cloneReplayTaskMetrics(artifact.taskMetrics),
    frames: artifact.frames.map(cloneReplayFrame),
    ...(artifact.playback
      ? {
          playback: {
            coreVersion: artifact.playback.coreVersion,
            interpolation: artifact.playback.interpolation,
            actuatorLayout: {
              ...artifact.playback.actuatorLayout,
              channelOrder: [...artifact.playback.actuatorLayout.channelOrder],
            },
          },
        }
      : {}),
    ...(artifact.integrity ? { integrity: { ...artifact.integrity } } : {}),
  }
}

export function cloneRecordedReplayArtifact(artifact: RecordedReplayArtifact): RecordedReplayArtifact {
  return cloneReplayArtifact(artifact) as RecordedReplayArtifact
}

/** Preserves forward-compatible JSON extension fields that typed clones ignore. */
export function cloneReplayJson<T>(value: T): T {
  if (
    value === null ||
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean'
  ) {
    return value
  }
  if (Array.isArray(value)) return value.map(item => cloneReplayJson(item)) as T
  if (typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, item]) => [key, cloneReplayJson(item)]),
    ) as T
  }
  throw new Error(`Replay JSON cannot contain ${typeof value} values.`)
}
