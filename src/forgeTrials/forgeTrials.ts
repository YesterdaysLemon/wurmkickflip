import { deriveWurmAnatomy } from '../creature/anatomy'
import type { CreatureGenome, EnvironmentConfig } from '../creature/types'
import {
  materializeDomainEnvironment,
  nextDomainSeed,
  normalizeDomainSeed,
  sampleDomain,
  type DomainSample,
} from '../environment/seedForge'
import {
  EvolvedLocomotionPolicy,
  LOCOMOTION_POLICY_PATH,
  parseLocomotionPolicy,
  type LocomotionPolicyArtifact,
  type LocomotionSensors,
} from '../policy/locomotionPolicy'
import {
  NeuralStuntPolicy,
  parseStuntPolicy,
  STUNT_POLICY_PATH,
  type StuntPolicyArtifact,
} from '../policy/neuralPolicy'
import { POLICY_TIMESTEP, SEGMENT_COUNT, type PolicyAction } from '../policy/types'
import {
  LiveReplayCapture,
  replayEnvironmentSampleFor,
  replayIntegrityFor,
  type RecordedReplayArtifact,
  type ReplayTaskMetrics,
} from '../replay'
import { TerrariumEpisode, type EpisodeLocomotionController } from '../scene/terrariumEpisode'
import type { LocomotionState } from '../scene/terrariumSimulation'

export const FORGE_TRIAL_DURATION_SECONDS = 8
export const FORGE_TRIAL_DEFAULT_COUNT = 16
export const FORGE_TRIAL_MAX_COUNT = 32

export type ForgeTrialIntervention = 'full' | 'zero' | 'frozen'
export type ForgeTrialVerdict = 'causal-win' | 'contested' | 'miss'

export type ForgeTrialLaneResult = {
  intervention: ForgeTrialIntervention
  mounted: boolean
  rideTime: number | null
  minimumDistanceToBoard: number
  maximumContactRatio: number
  maximumStableContactSeconds: number
  neuralActionMaximum: number
  finalLocomotionState: LocomotionState
  metrics: ReplayTaskMetrics
  trajectoryDigest: string
  replay: RecordedReplayArtifact | null
}

export type ForgeTrialHeat = {
  index: number
  seed: number
  fingerprint: string
  sample: DomainSample
  verdict: ForgeTrialVerdict
  full: ForgeTrialLaneResult
  zero: ForgeTrialLaneResult
  frozen: ForgeTrialLaneResult
}

export type ForgeTrialSummary = {
  heatCount: number
  neuralMounts: number
  zeroMounts: number
  frozenMounts: number
  causalWins: number
  contested: number
  misses: number
  medianRideTime: number | null
}

export type ForgeTrialsReport = {
  kind: 'wurmkickflip.forgeTrials'
  version: 1
  environmentId: string
  creatureId: string
  modelVersion: string
  baseSeed: number
  durationSeconds: number
  startedAt: string
  summary: ForgeTrialSummary
  heats: ForgeTrialHeat[]
}

export type ForgeTrialModels = {
  locomotion: LocomotionPolicyArtifact
  stunt: StuntPolicyArtifact
}

export type ForgeTrialsProgress = {
  completedLanes: number
  totalLanes: number
  heatIndex: number
  seed: number
  intervention: ForgeTrialIntervention
}

export type RunForgeTrialsOptions = {
  environment: EnvironmentConfig
  creature: CreatureGenome | null
  models: ForgeTrialModels
  baseSeed: number
  trialCount?: number
  durationSeconds?: number
  startedAt?: string
  signal?: AbortSignal
  onProgress?: (progress: ForgeTrialsProgress) => void
  yieldBetweenLanes?: boolean
}

export type RunForgeTrialLaneOptions = {
  environment: EnvironmentConfig
  creature: CreatureGenome | null
  models: ForgeTrialModels
  sample: DomainSample
  intervention: ForgeTrialIntervention
  durationSeconds?: number
  createdAt?: string
  retainReplay?: boolean
  signal?: AbortSignal
}

const FROZEN_WARMUP_STEPS = 24

export async function loadForgeTrialModels(signal?: AbortSignal): Promise<ForgeTrialModels> {
  const [locomotionResponse, stuntResponse] = await Promise.all([
    fetch(LOCOMOTION_POLICY_PATH, { signal }),
    fetch(STUNT_POLICY_PATH, { signal }),
  ])
  if (!locomotionResponse.ok) {
    throw new Error(`Locomotion model request returned ${locomotionResponse.status}.`)
  }
  if (!stuntResponse.ok) {
    throw new Error(`Mounted model request returned ${stuntResponse.status}.`)
  }
  return {
    locomotion: parseLocomotionPolicy(await locomotionResponse.json()),
    stunt: parseStuntPolicy(await stuntResponse.json()),
  }
}

export function forgeTrialSeeds(baseSeed: number, count: number) {
  const normalizedBase = normalizeDomainSeed(baseSeed)
  const normalizedCount = positiveInteger(count, 'Forge Trial count')
  if (normalizedCount > FORGE_TRIAL_MAX_COUNT) {
    throw new RangeError(`Forge Trial count cannot exceed ${FORGE_TRIAL_MAX_COUNT}.`)
  }

  const seeds: number[] = []
  let seed = normalizedBase
  for (let index = 0; index < normalizedCount; index += 1) {
    seed = nextDomainSeed(seed)
    seeds.push(seed)
  }
  return seeds
}

export async function runForgeTrials(options: RunForgeTrialsOptions): Promise<ForgeTrialsReport> {
  const trialCount = options.trialCount ?? FORGE_TRIAL_DEFAULT_COUNT
  const durationSeconds = positiveDuration(options.durationSeconds ?? FORGE_TRIAL_DURATION_SECONDS)
  const baseSeed = normalizeDomainSeed(options.baseSeed)
  const startedAt = options.startedAt ?? new Date().toISOString()
  const seeds = forgeTrialSeeds(baseSeed, trialCount)
  const totalLanes = seeds.length * 3
  let completedLanes = 0
  const heats: ForgeTrialHeat[] = []

  for (let index = 0; index < seeds.length; index += 1) {
    assertNotAborted(options.signal)
    const seed = seeds[index]
    const sample = sampleDomain(options.environment, seed)
    const lanes = {} as Record<ForgeTrialIntervention, ForgeTrialLaneResult>

    for (const intervention of ['full', 'zero', 'frozen'] as const) {
      lanes[intervention] = await runForgeTrialLane({
        environment: options.environment,
        creature: options.creature,
        models: options.models,
        sample,
        intervention,
        durationSeconds,
        createdAt: startedAt,
        retainReplay: intervention === 'full',
        signal: options.signal,
      })
      completedLanes += 1
      options.onProgress?.({
        completedLanes,
        totalLanes,
        heatIndex: index,
        seed,
        intervention,
      })
      if (options.yieldBetweenLanes ?? true) await yieldToBrowser()
    }

    heats.push({
      index,
      seed,
      fingerprint: domainFingerprint(sample),
      sample,
      verdict: verdictFor(lanes.full, lanes.zero, lanes.frozen),
      full: lanes.full,
      zero: lanes.zero,
      frozen: lanes.frozen,
    })
  }

  return {
    kind: 'wurmkickflip.forgeTrials',
    version: 1,
    environmentId: options.environment.id,
    creatureId: options.creature?.id ?? 'canonical-wurm',
    modelVersion: options.models.locomotion.modelVersion,
    baseSeed,
    durationSeconds,
    startedAt,
    summary: summarizeHeats(heats),
    heats,
  }
}

export async function runForgeTrialLane(options: RunForgeTrialLaneOptions): Promise<ForgeTrialLaneResult> {
  const durationSeconds = positiveDuration(options.durationSeconds ?? FORGE_TRIAL_DURATION_SECONDS)
  const stepCount = Math.round(durationSeconds / POLICY_TIMESTEP)
  const anatomy = deriveWurmAnatomy(options.creature)
  const environment = materializeDomainEnvironment(options.environment, options.sample)
  const locomotionController = new ControlledLocomotionPolicy(options.models.locomotion, options.intervention)
  const episode = new TerrariumEpisode({
    anatomy,
    domainSample: options.sample,
    environmentConfig: environment,
    locomotionController,
    mountedController: new NeuralStuntPolicy(options.models.stunt),
  })
  const capture = new LiveReplayCapture({
    replayId: `forge-${options.environment.id}-${options.sample.seed}-${options.intervention}`,
    createdAt: options.createdAt ?? new Date().toISOString(),
    source: {
      creatureId: options.creature?.id ?? 'canonical-wurm',
      environmentId: options.environment.id,
      policyBackend: options.intervention === 'full' ? 'neural-js' : 'training',
      modelVersion: `${options.models.locomotion.modelVersion}+forge-${options.intervention}-v1`,
    },
    timestep: POLICY_TIMESTEP,
    environmentSample: replayEnvironmentSampleFor(options.environment, options.sample),
  })

  let rideTime: number | null = null
  let minimumDistanceToBoard = episode.state.distanceToBoard
  let maximumContactRatio = 0
  let maximumStableContactSeconds = 0

  for (let step = 0; step < stepCount; step += 1) {
    if (step % 60 === 0) assertNotAborted(options.signal)
    const result = episode.step({ showcaseMode: 'kickflip' })
    capture.record(result.replayFrame)
    minimumDistanceToBoard = Math.min(minimumDistanceToBoard, episode.state.distanceToBoard)
    maximumContactRatio = Math.max(maximumContactRatio, episode.state.boardContactRatio)
    maximumStableContactSeconds = Math.max(
      maximumStableContactSeconds,
      episode.state.boardContactStableSeconds,
    )
    if (
      rideTime === null &&
      result.previousLocomotionState === 'mounting' &&
      result.locomotionState === 'riding'
    ) {
      rideTime = episode.state.time
    }
  }

  const artifact = capture.finalize()
  return {
    intervention: options.intervention,
    mounted: rideTime !== null,
    rideTime,
    minimumDistanceToBoard,
    maximumContactRatio,
    maximumStableContactSeconds,
    neuralActionMaximum: locomotionController.neuralActionMaximum,
    finalLocomotionState: episode.state.locomotionState,
    metrics: artifact.taskMetrics,
    trajectoryDigest: replayIntegrityFor({
      environmentSample: artifact.environmentSample,
      frames: artifact.frames,
    }).digest,
    replay: options.retainReplay ? artifact : null,
  }
}

class ControlledLocomotionPolicy implements EpisodeLocomotionController {
  readonly policy: EvolvedLocomotionPolicy
  readonly intervention: ForgeTrialIntervention
  readonly frozenAction = new Float32Array(SEGMENT_COUNT * 2)
  readonly frozenFeedback = new Float64Array(SEGMENT_COUNT)
  readonly outputAction = new Float32Array(SEGMENT_COUNT * 2)
  readonly outputFeedback = new Float64Array(SEGMENT_COUNT)
  locomotionStep = 0
  neuralActionMaximum = 0

  constructor(artifact: LocomotionPolicyArtifact, intervention: ForgeTrialIntervention) {
    this.policy = new EvolvedLocomotionPolicy(artifact)
    this.intervention = intervention
  }

  run(
    sensors: LocomotionSensors,
    segmentBends: ArrayLike<number>,
    segmentBendVelocities: ArrayLike<number>,
  ): PolicyAction {
    const rawAction = this.policy.run(sensors, segmentBends, segmentBendVelocities)
    for (const activation of rawAction) {
      this.neuralActionMaximum = Math.max(this.neuralActionMaximum, Math.abs(activation))
    }
    if (this.intervention === 'full') {
      this.locomotionStep += 1
      return rawAction
    }

    const rawFeedback = this.policy.getTelemetry().commands
    if (this.intervention === 'zero') {
      this.outputAction.fill(0)
      this.outputFeedback.fill(0)
    } else {
      if (this.locomotionStep === FROZEN_WARMUP_STEPS) {
        this.frozenAction.set(rawAction)
        this.frozenFeedback.set(rawFeedback)
      }
      if (this.locomotionStep < FROZEN_WARMUP_STEPS) {
        this.outputAction.set(rawAction)
        this.outputFeedback.set(rawFeedback)
      } else {
        this.outputAction.set(this.frozenAction)
        this.outputFeedback.set(this.frozenFeedback)
      }
    }
    this.policy.commitCommandFeedback(this.outputFeedback)
    this.locomotionStep += 1
    return this.outputAction
  }
}

function verdictFor(
  full: ForgeTrialLaneResult,
  zero: ForgeTrialLaneResult,
  frozen: ForgeTrialLaneResult,
): ForgeTrialVerdict {
  if (zero.mounted || frozen.mounted) return 'contested'
  return full.mounted ? 'causal-win' : 'miss'
}

function summarizeHeats(heats: readonly ForgeTrialHeat[]): ForgeTrialSummary {
  const rideTimes = heats
    .map(heat => heat.full.rideTime)
    .filter((time): time is number => time !== null)
    .sort((left, right) => left - right)
  return {
    heatCount: heats.length,
    neuralMounts: heats.filter(heat => heat.full.mounted).length,
    zeroMounts: heats.filter(heat => heat.zero.mounted).length,
    frozenMounts: heats.filter(heat => heat.frozen.mounted).length,
    causalWins: heats.filter(heat => heat.verdict === 'causal-win').length,
    contested: heats.filter(heat => heat.verdict === 'contested').length,
    misses: heats.filter(heat => heat.verdict === 'miss').length,
    medianRideTime: median(rideTimes),
  }
}

function median(values: readonly number[]) {
  if (values.length === 0) return null
  const middle = Math.floor(values.length / 2)
  return values.length % 2 === 0 ? (values[middle - 1] + values[middle]) / 2 : values[middle]
}

function domainFingerprint(sample: DomainSample) {
  const values = [
    sample.gravityScale,
    sample.frictionScale,
    sample.actuatorStrength,
    sample.actuatorLatencyMs,
    sample.sensorNoise,
    sample.skateboardMass,
    sample.wheelFriction,
  ]
  let hash = 0x811c9dc5
  for (const value of values) {
    for (const character of value.toFixed(6)) {
      hash ^= character.charCodeAt(0)
      hash = Math.imul(hash, 0x01000193)
    }
  }
  return (hash >>> 0).toString(16).padStart(8, '0')
}

function positiveInteger(value: number, label: string) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive integer.`)
  }
  return value
}

function positiveDuration(value: number) {
  if (!Number.isFinite(value) || value <= 0 || value > 60) {
    throw new RangeError('Forge Trial duration must be finite and between 0 and 60 seconds.')
  }
  return value
}

function assertNotAborted(signal?: AbortSignal) {
  if (signal?.aborted) throw new ForgeTrialsAbortedError()
}

function yieldToBrowser() {
  return new Promise<void>(resolve => setTimeout(resolve, 0))
}

export class ForgeTrialsAbortedError extends Error {
  constructor() {
    super('Forge Trials were cancelled.')
    this.name = 'ForgeTrialsAbortedError'
  }
}
