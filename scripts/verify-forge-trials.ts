import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import type { CreatureGenome, EnvironmentConfig } from '../src/creature/types'
import { createNominalDomainSample, validateDomainSample } from '../src/environment/seedForge'
import { parseLocomotionPolicy } from '../src/policy/locomotionPolicy'
import { parseStuntPolicy } from '../src/policy/neuralPolicy'
import { POLICY_TIMESTEP } from '../src/policy/types'
import { ReplayPlayer } from '../src/replay'
import {
  forgeTrialSeeds,
  runForgeTrialLane,
  runForgeTrials,
  type ForgeTrialLaneResult,
  type ForgeTrialModels,
  type ForgeTrialsReport,
} from '../src/forgeTrials/forgeTrials'

const root = resolve(import.meta.dirname, '..')
const [environment, creature, locomotionSource, stuntSource] = await Promise.all([
  readJson<EnvironmentConfig>('public/configs/environments/adaptive-skate-terrarium.json'),
  readJson<CreatureGenome>('public/configs/creatures/segmented-starter.json'),
  readJson<unknown>('public/models/wurmkickflip_locomotion_policy.json'),
  readJson<unknown>('public/models/wurmkickflip_stunt_policy.json'),
])
const models: ForgeTrialModels = {
  locomotion: parseLocomotionPolicy(locomotionSource),
  stunt: parseStuntPolicy(stuntSource),
}
const createdAt = '2026-07-26T00:00:00.000Z'

verifySeedSchedule()
await verifyCanonicalControlLanes()
await verifyDeterministicMeet()
await verifySceneUsesCanonicalEpisode()
console.log('Forge Trials verification passed.')

function verifySeedSchedule() {
  const first = forgeTrialSeeds(0, 4)
  const second = forgeTrialSeeds(0, 4)
  assertDeepEqual(first, second, 'held-out seed schedule')
  assert(new Set(first).size === first.length, 'held-out seed schedule repeated a seed')
  assert(!first.includes(0), 'held-out seed schedule included its live base seed')
  assert(
    forgeTrialSeeds(0xffffffff, 1)[0] === 0x9e3779b8,
    'held-out seed schedule lost uint32 wrap semantics',
  )
}

async function verifyCanonicalControlLanes() {
  const sample = createNominalDomainSample(environment, environment.seed)
  const [full, zero, frozen] = await Promise.all(
    (['full', 'zero', 'frozen'] as const).map(intervention =>
      runForgeTrialLane({
        environment,
        creature,
        models,
        sample,
        intervention,
        durationSeconds: 8,
        createdAt,
        retainReplay: intervention === 'full',
      }),
    ),
  )

  assert(full.mounted, `canonical neural lane did not mount (${laneOutcome(full)})`)
  assert(!zero.mounted, `canonical zero lane mounted (${laneOutcome(zero)})`)
  assert(!frozen.mounted, `canonical frozen lane mounted (${laneOutcome(frozen)})`)
  assert(full.neuralActionMaximum > 0.5, 'canonical neural lane never exercised the active controller')
  assert(zero.metrics.energy === 0, 'zero lane leaked nonzero applied muscle energy')
  assert(full.replay !== null, 'canonical neural lane did not retain its representative replay')
  assert(full.replay!.frameCount === Math.round(8 / POLICY_TIMESTEP), 'canonical replay frame count moved')
  assert(validateDomainSample(sample, environment).length === 0, 'canonical replay sample became invalid')
  const player = new ReplayPlayer(full.replay)
  assert(player.durationSeconds === 8, 'representative replay duration moved')
  assert(player.frame(0).muscleActivations?.length === 32, 'representative replay lost muscle channels')
}

async function verifyDeterministicMeet() {
  const options = {
    environment,
    creature,
    models,
    baseSeed: 0x46524745,
    trialCount: 3,
    durationSeconds: 8,
    startedAt: createdAt,
    yieldBetweenLanes: false,
  } as const
  const first = await runForgeTrials(options)
  const second = await runForgeTrials({
    ...options,
    startedAt: '2026-07-27T00:00:00.000Z',
  })
  assertDeepEqual(reportDigest(first), reportDigest(second), 'repeated Forge Trials meet')
  assert(first.heats.length === 3, 'Forge Trials meet returned the wrong heat count')
  assert(
    first.summary.causalWins + first.summary.contested + first.summary.misses === first.heats.length,
    'Forge Trials verdict counts do not partition the heats',
  )
  for (const heat of first.heats) {
    assert(validateDomainSample(heat.sample, environment).length === 0, `seed ${heat.seed} sample is invalid`)
    assert(heat.full.replay !== null, `seed ${heat.seed} neural replay was not retained`)
    assert(
      heat.full.replay!.environmentSample.seed === heat.seed,
      `seed ${heat.seed} replay provenance drifted`,
    )
    new ReplayPlayer(heat.full.replay)
  }
}

async function verifySceneUsesCanonicalEpisode() {
  const scene = await readFile(resolve(root, 'src/scene/WurmkickflipScene.tsx'), 'utf8')
  assert(
    /new\s+TerrariumEpisode\s*\(/.test(scene),
    'live scene does not construct the canonical episode engine',
  )
  assert(
    /episodeRuntime\s*\.\s*step\s*\(/.test(scene),
    'live scene does not step the canonical episode engine',
  )
  assert(!/\badvanceStunt\s*\(/.test(scene), 'live scene still owns a second direct simulation-advance loop')
}

function reportDigest(report: ForgeTrialsReport) {
  return {
    summary: report.summary,
    heats: report.heats.map(heat => ({
      seed: heat.seed,
      fingerprint: heat.fingerprint,
      verdict: heat.verdict,
      full: laneDigest(heat.full),
      zero: laneDigest(heat.zero),
      frozen: laneDigest(heat.frozen),
    })),
  }
}

function laneDigest(lane: ForgeTrialLaneResult) {
  return {
    mounted: lane.mounted,
    rideTime: lane.rideTime,
    minimumDistanceToBoard: lane.minimumDistanceToBoard,
    maximumContactRatio: lane.maximumContactRatio,
    maximumStableContactSeconds: lane.maximumStableContactSeconds,
    neuralActionMaximum: lane.neuralActionMaximum,
    finalLocomotionState: lane.finalLocomotionState,
    metrics: lane.metrics,
    trajectoryDigest: lane.trajectoryDigest,
  }
}

function laneOutcome(lane: ForgeTrialLaneResult) {
  return JSON.stringify({
    mounted: lane.mounted,
    rideTime: lane.rideTime,
    distance: lane.minimumDistanceToBoard,
    contact: lane.maximumContactRatio,
    stable: lane.maximumStableContactSeconds,
    final: lane.finalLocomotionState,
  })
}

async function readJson<T>(path: string): Promise<T> {
  return JSON.parse(await readFile(resolve(root, path), 'utf8')) as T
}

function assertDeepEqual(actual: unknown, expected: unknown, label: string) {
  const actualJson = JSON.stringify(actual)
  const expectedJson = JSON.stringify(expected)
  assert(actualJson === expectedJson, `${label} was not deterministic`)
}

function assert(condition: boolean, message: string): asserts condition {
  if (!condition) throw new Error(`Forge Trials verification failed: ${message}`)
}
