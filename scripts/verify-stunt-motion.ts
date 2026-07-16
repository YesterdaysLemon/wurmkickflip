import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import type { CreatureGenome, EnvironmentConfig } from '../src/creature/types'
import { deriveWurmAnatomy, type WurmAnatomy } from '../src/creature/anatomy'
import { NeuralStuntPolicy, parseStuntPolicy } from '../src/policy/neuralPolicy'
import { EvolvedLocomotionPolicy, parseLocomotionPolicy } from '../src/policy/locomotionPolicy'
import { makeInitialAction, snapshotToObservation } from '../src/policy/simulationAdapter'
import { POLICY_TIMESTEP, SEGMENT_COUNT } from '../src/policy/types'
import {
  advanceStunt,
  BOARDING_STABLE_SECONDS,
  boardContactQualifiesForMount,
  chooseDismountSide,
  createStuntState,
  gaitControllerOwnsBody,
  locomotionSensorsFor,
  makeTerrariumDecor,
  skateboardFootprintObstacles,
  smoothAction,
  toSnapshot,
  type ShowcaseMode,
} from '../src/scene/terrariumSimulation'
import { createTerrainField } from '../src/scene/terrainField'
import { skateboardDeckSupportWeight } from '../src/scene/skateboardContact'

const root = resolve(import.meta.dirname, '..')
const environment = JSON.parse(
  await readFile(resolve(root, 'public/configs/environments/adaptive-skate-terrarium.json'), 'utf8'),
) as EnvironmentConfig
const rippleEnvironment = JSON.parse(
  await readFile(resolve(root, 'public/configs/environments/ripple-yard.json'), 'utf8'),
) as EnvironmentConfig
const tiltEnvironment = JSON.parse(
  await readFile(resolve(root, 'public/configs/environments/tilt-basin.json'), 'utf8'),
) as EnvironmentConfig
const selectableAnatomies = Object.fromEntries(
  await Promise.all(
    ['segmented-starter', 'tripod-pusher', 'boxfish-scrambler'].map(async id => {
      const creature = JSON.parse(
        await readFile(resolve(root, `public/configs/creatures/${id}.json`), 'utf8'),
      ) as CreatureGenome
      return [id, deriveWurmAnatomy(creature)] as const
    }),
  ),
) as Record<string, WurmAnatomy>
const defaultAnatomy = selectableAnatomies['segmented-starter']
const canonicalAnatomy = deriveWurmAnatomy(null)
const artifact = parseStuntPolicy(
  JSON.parse(await readFile(resolve(root, 'public/models/wurmkickflip_stunt_policy.json'), 'utf8')),
)
const candidateScreen = process.env.WURMKICKFLIP_CANDIDATE_SCREEN === '1'
const locomotionArtifactSource = process.env.WURMKICKFLIP_LOCOMOTION_ARTIFACT
  ? resolve(process.env.WURMKICKFLIP_LOCOMOTION_ARTIFACT)
  : resolve(root, 'public/models/wurmkickflip_locomotion_policy.json')
const rawLocomotionArtifact = JSON.parse(await readFile(locomotionArtifactSource, 'utf8'))
// Retained evolutionary checkpoints are intentionally not runtime-loadable V3
// artifacts. This explicit CLI-only screen removes the publication discriminator
// from a clone so integrated behavior can be measured without weakening the parser.
const candidateInput = candidateScreen ? structuredClone(rawLocomotionArtifact) : rawLocomotionArtifact
if (
  candidateScreen &&
  candidateInput.modelVersion !== 'locomotion-articulated-head-leading-es-v3' &&
  candidateInput.training
) {
  delete candidateInput.training.objectiveVersion
}
const locomotionArtifact = parseLocomotionPolicy(candidateInput)
const SEGMENT_SHUFFLE = [0, 9, 2, 13, 4, 15, 6, 11, 8, 1, 10, 3, 12, 5, 14, 7] as const
const candidateFailures: string[] = []
const FOOD_RESTORATION_FLOOR = 0.5
const FOOD_TARGET_PROGRESS_GAP = 0.4
const FOOD_SHUFFLED_RESTORATION_GAP = 0.1
const FOOD_HANDOFF_LEAD_SECONDS = 3
const FOOD_HANDOFF_DEADLINE_SECONDS = 16
const BOARDING_CHALLENGE_SECONDS = 8
const BOARDING_GENERALIZATION_SECONDS = 8

const fullAudit: RolloutAudit = {
  geometry: true,
  motion: true,
  observations: true,
  trace: true,
}
const geometryAudit: RolloutAudit = { geometry: true }
const traceAudit: RolloutAudit = { trace: true }
const minimalAudit: RolloutAudit = {}

if (candidateScreen && process.env.WURMKICKFLIP_CANDIDATE_FOCUS === 'food') {
  const full = simulate(
    'freestyle',
    32,
    'full',
    undefined,
    environment,
    minimalAudit,
    'food-bowl',
    canonicalAnatomy,
  )
  const frozen = simulate(
    'freestyle',
    32,
    'frozen',
    undefined,
    environment,
    minimalAudit,
    'food-bowl',
    canonicalAnatomy,
  )
  const shuffled = simulate(
    'freestyle',
    32,
    'shuffled',
    undefined,
    environment,
    minimalAudit,
    'food-bowl',
    canonicalAnatomy,
  )
  const fullOverFrozen = full.foodFulfillment - frozen.foodFulfillment
  const fullOverShuffled = full.foodFulfillment - shuffled.foodFulfillment
  const fullFeeds =
    full.firstNeuralFoodFeedingTime !== null &&
    full.firstNeuralFoodFeedingTime <= FOOD_HANDOFF_DEADLINE_SECONDS &&
    full.foodFulfillment >= FOOD_RESTORATION_FLOOR
  const fullBeatsFrozen =
    full.needTargetProgress >= frozen.needTargetProgress + FOOD_TARGET_PROGRESS_GAP &&
    foodHandoffLeads(full, frozen)
  const fullBeatsShuffled =
    full.needTargetProgress >= shuffled.needTargetProgress + FOOD_TARGET_PROGRESS_GAP &&
    full.foodFulfillment >= shuffled.foodFulfillment + FOOD_SHUFFLED_RESTORATION_GAP
  const passed = fullFeeds && fullBeatsFrozen && fullBeatsShuffled
  console.log(
    JSON.stringify(
      {
        candidateScreen: true,
        focus: 'food',
        modelVersion: locomotionArtifact.modelVersion,
        passed,
        criteria: { fullFeeds, fullBeatsFrozen, fullBeatsShuffled },
        thresholds: {
          fullRestorationAtLeast: FOOD_RESTORATION_FLOOR,
          fullFirstFoodHandoffAtMostSeconds: FOOD_HANDOFF_DEADLINE_SECONDS,
          targetProgressGapAtLeast: FOOD_TARGET_PROGRESS_GAP,
          firstFoodHandoffLeadSecondsAtLeast: FOOD_HANDOFF_LEAD_SECONDS,
          shuffledRestorationGapAtLeast: FOOD_SHUFFLED_RESTORATION_GAP,
        },
        crawlDistance: {
          full: round(full.crawlDistance),
          frozen: round(frozen.crawlDistance),
          shuffled: round(shuffled.crawlDistance),
        },
        targetProgress: {
          full: round(full.needTargetProgress),
          frozen: round(frozen.needTargetProgress),
          shuffled: round(shuffled.needTargetProgress),
        },
        foodRestoration: {
          full: round(full.foodFulfillment),
          frozen: round(frozen.foodFulfillment),
          shuffled: round(shuffled.foodFulfillment),
        },
        firstNeuralFoodFeeding: {
          full: firstFoodHandoffJson(full),
          frozen: firstFoodHandoffJson(frozen),
          shuffled: firstFoodHandoffJson(shuffled),
        },
        foodRestorationGaps: {
          fullOverFrozen: round(fullOverFrozen),
          fullOverShuffled: round(fullOverShuffled),
        },
      },
      null,
      2,
    ),
  )
  process.exit(passed ? 0 : 1)
}

if (process.env.WURMKICKFLIP_BOARD_DIAGNOSTIC === '1') {
  const boardingRollouts = {
    full: simulate('kickflip', 40, 'full', undefined, environment, minimalAudit),
    zero: simulate('kickflip', 40, 'zero', undefined, environment, minimalAudit),
    frozen: simulate('kickflip', 40, 'frozen', undefined, environment, minimalAudit),
    shuffled: simulate('kickflip', 40, 'shuffled', undefined, environment, minimalAudit),
  }
  console.log(
    JSON.stringify(
      {
        modelVersion: locomotionArtifact.modelVersion,
        rollouts: Object.fromEntries(
          Object.entries(boardingRollouts).map(([name, boarding]) => [
            name,
            {
              locomotionStates: boarding.locomotionStates,
              flipsLanded: boarding.flipsLanded,
              remounted: boarding.remounted,
              crawlDistance: round(boarding.crawlDistance),
              finalDistanceToBoard: round(boarding.finalDistanceToBoard),
              minimumDistanceToBoard: round(boarding.minimumDistanceToBoard),
              minimumSegmentDistanceToBoard: round(boarding.minimumSegmentDistanceToBoard),
              maximumPlanarBoardSupportWeight: round(boarding.maximumPlanarBoardSupportWeight),
              maximumBoardContactRatio: round(boarding.maximumBoardContactRatio),
              maximumBoardContactSegments: boarding.maximumBoardContactSegments,
              maximumBoardContactStableSeconds: round(boarding.maximumBoardContactStableSeconds),
              maximumBoardContactRegions: {
                head: round(boarding.maximumBoardContactHeadWeight),
                midbody: round(boarding.maximumBoardContactMidbodyWeight),
                tail: round(boarding.maximumBoardContactTailWeight),
              },
              maximumDismountClearance: round(boarding.maximumDismountClearance),
              firstBoardContactTime: boarding.firstBoardContactTime,
              firstNeuralBoardingTime: boarding.firstNeuralBoardingTime,
              firstAutonomousRideTime: boarding.firstAutonomousRideTime,
              firstAutonomousRideApplication: boarding.firstAutonomousRideApplication,
              autonomousRideContact: boarding.autonomousRideContact,
              mountingTicks: {
                neural: boarding.neuralMountingTicks,
                nonNeural: boarding.nonNeuralMountingTicks,
              },
            },
          ]),
        ),
      },
      null,
      2,
    ),
  )
  process.exit(0)
}

const boardingChallenge = {
  full: simulate('kickflip', BOARDING_CHALLENGE_SECONDS, 'full', undefined, environment, minimalAudit),
  zero: simulate('kickflip', BOARDING_CHALLENGE_SECONDS, 'zero', undefined, environment, minimalAudit),
  frozen: simulate('kickflip', BOARDING_CHALLENGE_SECONDS, 'frozen', undefined, environment, minimalAudit),
  shuffled: simulate(
    'kickflip',
    BOARDING_CHALLENGE_SECONDS,
    'shuffled',
    undefined,
    environment,
    minimalAudit,
  ),
}
const anatomyBoarding = Object.fromEntries(
  Object.entries(selectableAnatomies).map(([name, anatomy]) => [
    name,
    simulate(
      'kickflip',
      BOARDING_GENERALIZATION_SECONDS,
      'full',
      undefined,
      environment,
      minimalAudit,
      undefined,
      anatomy,
    ),
  ]),
)
const environmentBoarding = Object.fromEntries(
  [
    ['adaptive', environment],
    ['ripple', rippleEnvironment],
    ['tilt', tiltEnvironment],
  ].map(([name, boardEnvironment]) => [
    name as string,
    simulate(
      'kickflip',
      BOARDING_GENERALIZATION_SECONDS,
      'full',
      undefined,
      boardEnvironment as EnvironmentConfig,
      minimalAudit,
    ),
  ]),
)

expect(
  boardingChallenge.full.firstAutonomousRideTime !== null &&
    boardingChallenge.full.firstAutonomousRideTime <= BOARDING_CHALLENGE_SECONDS,
  `intact recurrent controller missed the ${BOARDING_CHALLENGE_SECONDS.toFixed(2)} s board challenge ` +
    `(${boardingOutcome(boardingChallenge.full)})`,
)
for (const [label, ablated] of Object.entries(boardingChallenge).filter(([name]) =>
  ['zero', 'frozen'].includes(name),
)) {
  expect(
    ablated.firstAutonomousRideTime === null,
    `${label} controller incorrectly earned a ride inside the causal deadline ` +
      `(${boardingOutcome(ablated)})`,
  )
}
expect(
  boardingChallenge.shuffled.firstAutonomousRideTime !== null,
  'fixed segment-channel shuffle no longer exercises the documented robustness stressor',
)
expect(
  boardingChallenge.full.firstNeuralBoardingTime !== null &&
    boardingChallenge.full.neuralMountingTicks > 0 &&
    boardingChallenge.full.nonNeuralMountingTicks === 0,
  'live mounting did not remain under neural action ownership',
)
expect(
  boardingChallenge.full.firstAutonomousRideApplication === 'lifecycle-handoff',
  'stable-contact mounting did not explicitly hand ownership to scripted riding',
)
expect(
  boardingChallenge.full.autonomousRideContact !== null &&
    boardContactQualifiesForMount({
      boardContactRatio: boardingChallenge.full.autonomousRideContact.ratio,
      boardContactSegmentCount: boardingChallenge.full.autonomousRideContact.segments,
      boardContactHeadWeight: boardingChallenge.full.autonomousRideContact.head,
      boardContactMidbodyWeight: boardingChallenge.full.autonomousRideContact.midbody,
      boardContactTailWeight: boardingChallenge.full.autonomousRideContact.tail,
      boardContactRelativeSpeed: boardingChallenge.full.autonomousRideContact.relativeSpeed,
    }) &&
    boardingChallenge.full.autonomousRideContact.stableSeconds >= BOARDING_STABLE_SECONDS,
  `ride transition lacked qualified full-body contact (${boardingOutcome(boardingChallenge.full)})`,
)
for (const [kind, rollouts] of [
  ['anatomy', anatomyBoarding],
  ['environment', environmentBoarding],
] as const) {
  for (const [name, rollout] of Object.entries(rollouts)) {
    expect(
      rollout.firstAutonomousRideTime !== null &&
        rollout.firstAutonomousRideTime <= BOARDING_GENERALIZATION_SECONDS,
      `${kind} boarding probe ${name} did not mount (${boardingOutcome(rollout)})`,
    )
  }
}

const first = simulate('kickflip', 150, 'full', undefined, environment, fullAudit)
// The duplicate pass only proves deterministic state evolution. Repeating the
// first pass's clearance and motion bookkeeping would not strengthen that assertion.
const second = simulate('kickflip', 150, 'full', undefined, environment, traceAudit)
const freestyle = simulate('freestyle', 32, 'full', undefined, environment, geometryAudit)
const zeroFreestyle = simulate(
  'freestyle',
  32,
  'zero',
  undefined,
  environment,
  minimalAudit,
  undefined,
  canonicalAnatomy,
)
const goalFreestyle = simulate(
  'freestyle',
  32,
  'full',
  undefined,
  environment,
  minimalAudit,
  'food-bowl',
  canonicalAnatomy,
)
const frozenFreestyle = simulate(
  'freestyle',
  32,
  'frozen',
  undefined,
  environment,
  minimalAudit,
  'food-bowl',
  canonicalAnatomy,
)
const shuffledFreestyle = simulate(
  'freestyle',
  32,
  'shuffled',
  undefined,
  environment,
  minimalAudit,
  'food-bowl',
  canonicalAnatomy,
)
const zeroFrictionFreestyle = simulate('freestyle', 32, 'full', 0, environment, minimalAudit)
const rippleFreestyle = simulate('freestyle', 64, 'full', undefined, rippleEnvironment, geometryAudit)
const tiltFreestyle = simulate('freestyle', 64, 'full', undefined, tiltEnvironment, geometryAudit)
const incidentalBoardContact = verifyIncidentalBoardCollision()
const boardContactLifecycleGate = verifyBoardContactLifecycleGate()
const dismountSideSelection = verifyDismountSideSelection()

expect(first.hash === second.hash, 'identical seeded kickflip rollouts must be deterministic')
expect(
  first.flipsLanded >= 2,
  `expected a kickflip before and after crawling, got ${first.flipsLanded}; ` +
    `states=${first.locomotionStates.join('>')} needs=${first.foodFulfillment.toFixed(2)}/` +
    `${first.waterFulfillment.toFixed(2)}/${first.wellbeingFulfillment.toFixed(2)} ` +
    `crawl=${first.crawlDistance.toFixed(2)} separation=${first.maximumSeparation.toFixed(2)} ` +
    `board=${first.finalDistanceToBoard.toFixed(2)} contact=${first.maximumBoardContactRatio.toFixed(3)}/` +
    `${first.maximumBoardContactSegments} stable=${first.maximumBoardContactStableSeconds.toFixed(3)} ` +
    `minBoard=${first.minimumDistanceToBoard.toFixed(3)}/${first.minimumSegmentDistanceToBoard.toFixed(3)}`,
)
for (const state of ['riding', 'dismounting', 'crawling', 'feeding', 'seeking', 'mounting']) {
  expect(first.locomotionStates.includes(state), `kickflip rollout never entered ${state}`)
}
expect(first.remounted, 'kickflip rollout never returned to riding after its detached crawl')
expect(first.foodFulfillment > 0.2, 'needs-driven rollout never ate from the food bowl')
expect(first.waterFulfillment > 0.2, 'needs-driven rollout never drank from the water bowl')
expect(first.ateWhileNeurallyCrawling, 'food interaction never occurred during neural crawl/seek locomotion')
expect(
  first.drankWhileNeurallyCrawling,
  'water interaction never occurred during neural crawl/seek locomotion',
)
expect(first.wellbeingFulfillment > 0.2, 'needs-driven rollout never restored well-being on the skateboard')
expect(
  first.postRemountWellbeingFulfillment > 0.2,
  'well-being was not restored after the worm neurally returned to and remounted the skateboard',
)
expect(
  first.maximumSeparation > 0.8,
  `worm never clearly detached from board (${first.maximumSeparation.toFixed(3)} m)`,
)
expect(
  first.maximumRootStep < 0.08,
  `root motion jumped ${first.maximumRootStep.toFixed(4)} m in one tick (${first.maximumRootContext}); ` +
    `largest segment: ${first.maximumSegmentContext}`,
)
expect(
  first.maximumSegmentStep < 0.09,
  `segment motion jumped ${first.maximumSegmentStep.toFixed(4)} m in one tick (${first.maximumSegmentContext})`,
)
expect(
  first.bodySpeedP99 < 3.5,
  `body-speed p99 is too jittery (${first.bodySpeedP99.toFixed(3)} m/s; max ${first.maximumBodyContext})`,
)
expect(
  first.maximumBodySpeed < 4.8,
  `body-speed maximum is too jittery (${first.maximumBodySpeed.toFixed(3)} m/s; ${first.maximumBodyContext})`,
)
expect(
  first.maximumYawHeadingError < 0.16,
  `board yaw drifted away from its route heading (${first.maximumYawHeadingError.toFixed(3)} rad)`,
)
expect(
  first.boardRangeX > 1.2 && first.boardRangeZ > 1.2,
  `board did not traverse meaningful distance on both arena axes (${first.boardRangeX.toFixed(2)} × ${first.boardRangeZ.toFixed(2)})`,
)
expect(first.inBounds, 'kickflip rollout left the configured terrarium bounds')
expect(first.observationsFinite, 'kickflip rollout produced non-finite policy observations')

expect(freestyle.locomotionStates.includes('crawling'), 'free crawl never detached onto the terrain')
expect(!freestyle.remounted, 'free crawl unexpectedly remounted the skateboard')
expect(freestyle.wormRangeX > 1.2 && freestyle.wormRangeZ > 1.2, 'free crawl did not explore both arena axes')
expect(freestyle.inBounds, 'free crawl left the configured terrarium bounds')
for (const [label, rollout] of [
  ['ripple yard', rippleFreestyle],
  ['tilt basin', tiltFreestyle],
] as const) {
  expect(
    rollout.crawlDistance > 3,
    `${label} crawl did not produce meaningful segment-driven travel ` +
      `(${rollout.crawlDistance.toFixed(2)} m; contacts=${rollout.obstacleContacts.join(',') || 'none'}; ` +
      `states=${rollout.locomotionStates.join('>')})`,
  )
  expect(
    rollout.ateWhileNeurallyCrawling && rollout.drankWhileNeurallyCrawling,
    `${label} crawl did not reach both food and water resources ` +
      `(distance=${rollout.crawlDistance.toFixed(2)} food=${rollout.ateWhileNeurallyCrawling} ` +
      `water=${rollout.drankWhileNeurallyCrawling} contact=${rollout.finalObstacleContact ?? 'none'})`,
  )
  expect(rollout.inBounds, `${label} crawl left its configured terrarium bounds`)
}
for (const [label, rollout] of [
  ['autonomous', first],
  ['freestyle', freestyle],
  ['ripple yard', rippleFreestyle],
  ['tilt basin', tiltFreestyle],
] as const) {
  expect(
    rollout.minimumStaticObstacleClearance >= -1e-3,
    `${label} rollout penetrated a rendered tree or rock by ${(-rollout.minimumStaticObstacleClearance).toFixed(5)} m ` +
      `(${rollout.minimumStaticObstacleContext})`,
  )
}
const staticContactIds = [first, freestyle, rippleFreestyle, tiltFreestyle]
  .flatMap(rollout => rollout.obstacleContacts)
  .filter(id => id.startsWith('tree-') || id.startsWith('rock-'))
expect(
  staticContactIds.some(id => id.startsWith('tree-')),
  'integrated rollouts never exercised a tree collision',
)
expect(
  staticContactIds.some(id => id.startsWith('rock-')),
  'integrated rollouts never exercised a rock collision',
)
expect(zeroFreestyle.neuralActionMax > 0.5, 'zero intervention did not exercise an active neural controller')
expect(
  zeroFreestyle.crawlDistance < 0.03,
  `zero segment commands moved the integrated worm root (${zeroFreestyle.crawlDistance.toFixed(6)} m)`,
)
expect(zeroFreestyle.finalLateralSpan < 0.05, 'zero commands left a hidden time-authored crawl pose')
expect(
  zeroFrictionFreestyle.neuralActionMax > 0.5,
  'zero-friction scene did not exercise active neural commands',
)
expect(
  zeroFrictionFreestyle.needTargetProgress < freestyle.needTargetProgress - 1,
  `removing terrain traction did not materially reduce purposeful target progress ` +
    `(${zeroFrictionFreestyle.crawlDistance.toFixed(3)} m vs ${freestyle.crawlDistance.toFixed(3)} m; ` +
    `target progress ${zeroFrictionFreestyle.needTargetProgress.toFixed(3)} vs ` +
    `${freestyle.needTargetProgress.toFixed(3)}); ` +
    `the exact obstacle-free COM invariant is checked by verify-worm-dynamics`,
)
expect(
  goalFreestyle.firstNeuralFoodFeedingTime !== null &&
    goalFreestyle.firstNeuralFoodFeedingTime <= FOOD_HANDOFF_DEADLINE_SECONDS &&
    goalFreestyle.foodFulfillment >= FOOD_RESTORATION_FLOOR,
  `full recurrent crawl did not reach and feed from the food bowl ` +
    `by the ${FOOD_HANDOFF_DEADLINE_SECONDS.toFixed(1)} s challenge deadline ` +
    `(handoff ${formatFoodHandoff(goalFreestyle)}, restoration ${goalFreestyle.foodFulfillment.toFixed(3)})`,
)
for (const [label, interventionRollout] of [
  ['frozen-command', frozenFreestyle],
  ['shuffled-segment', shuffledFreestyle],
] as const) {
  expect(
    goalFreestyle.needTargetProgress >= interventionRollout.needTargetProgress + FOOD_TARGET_PROGRESS_GAP,
    `full recurrent crawl did not beat the ${label} cumulative need-target-progress challenge ` +
      `(${goalFreestyle.needTargetProgress.toFixed(3)} m vs ` +
      `${interventionRollout.needTargetProgress.toFixed(3)} m; required gap ` +
      `${FOOD_TARGET_PROGRESS_GAP.toFixed(1)} m)`,
  )
}
expect(
  foodHandoffLeads(goalFreestyle, frozenFreestyle),
  `full recurrent crawl did not reach authored feeding at least ` +
    `${FOOD_HANDOFF_LEAD_SECONDS.toFixed(1)} s before the frozen-command intervention ` +
    `(full ${formatFoodHandoff(goalFreestyle)}, intervention ${formatFoodHandoff(frozenFreestyle)})`,
)
expect(
  goalFreestyle.foodFulfillment >= shuffledFreestyle.foodFulfillment + FOOD_SHUFFLED_RESTORATION_GAP,
  `full recurrent crawl did not retain a restoration advantage over fixed segment shuffling ` +
    `(${goalFreestyle.foodFulfillment.toFixed(3)} vs ${shuffledFreestyle.foodFulfillment.toFixed(3)}; ` +
    `required gap ${FOOD_SHUFFLED_RESTORATION_GAP.toFixed(1)})`,
)
console.log(
  JSON.stringify(
    {
      ...(candidateScreen
        ? {
            candidateScreen: true,
            passed: candidateFailures.length === 0,
            failures: candidateFailures,
            candidateDiagnostics: {
              traction: {
                fullCrawlDistance: round(freestyle.crawlDistance),
                zeroFrictionCrawlDistance: round(zeroFrictionFreestyle.crawlDistance),
                reduction: round(freestyle.crawlDistance - zeroFrictionFreestyle.crawlDistance),
                neuralActionMax: {
                  zeroIntervention: round(zeroFreestyle.neuralActionMax),
                  zeroFriction: round(zeroFrictionFreestyle.neuralActionMax),
                },
              },
              foodRestorationGaps: {
                fullOverFrozen: round(goalFreestyle.foodFulfillment - frozenFreestyle.foodFulfillment),
                fullOverShuffled: round(goalFreestyle.foodFulfillment - shuffledFreestyle.foodFulfillment),
              },
            },
          }
        : {}),
      autonomousBoarding: {
        causalDeadlineSeconds: BOARDING_CHALLENGE_SECONDS,
        generalizationDeadlineSeconds: BOARDING_GENERALIZATION_SECONDS,
        challenge: Object.fromEntries(
          Object.entries(boardingChallenge).map(([name, rollout]) => [name, boardingJson(rollout)]),
        ),
        anatomies: Object.fromEntries(
          Object.entries(anatomyBoarding).map(([name, rollout]) => [name, boardingJson(rollout)]),
        ),
        environments: Object.fromEntries(
          Object.entries(environmentBoarding).map(([name, rollout]) => [name, boardingJson(rollout)]),
        ),
      },
      bodySpeedMax: round(first.maximumBodySpeed),
      bodySpeedP99: round(first.bodySpeedP99),
      boardRange: [round(first.boardRangeX), round(first.boardRangeZ)],
      deterministicHash: first.hash,
      flipsLanded: first.flipsLanded,
      freestyleWormRange: [round(freestyle.wormRangeX), round(freestyle.wormRangeZ)],
      integratedCrawlAblations: {
        full: round(goalFreestyle.crawlDistance),
        zero: round(zeroFreestyle.crawlDistance),
        frozen: round(frozenFreestyle.crawlDistance),
        shuffled: round(shuffledFreestyle.crawlDistance),
        targetProgress: {
          full: round(goalFreestyle.needTargetProgress),
          frozen: round(frozenFreestyle.needTargetProgress),
          shuffled: round(shuffledFreestyle.needTargetProgress),
        },
        foodRestoration: {
          full: round(goalFreestyle.foodFulfillment),
          frozen: round(frozenFreestyle.foodFulfillment),
          shuffled: round(shuffledFreestyle.foodFulfillment),
        },
        firstNeuralFoodFeeding: {
          full: firstFoodHandoffJson(goalFreestyle),
          frozen: firstFoodHandoffJson(frozenFreestyle),
          shuffled: firstFoodHandoffJson(shuffledFreestyle),
        },
        zeroFriction: round(zeroFrictionFreestyle.crawlDistance),
        zeroFinalLateralSpan: round(zeroFreestyle.finalLateralSpan),
      },
      presetCrawl: {
        ripple: {
          distance: round(rippleFreestyle.crawlDistance),
          food: rippleFreestyle.ateWhileNeurallyCrawling,
          water: rippleFreestyle.drankWhileNeurallyCrawling,
        },
        tilt: {
          distance: round(tiltFreestyle.crawlDistance),
          food: tiltFreestyle.ateWhileNeurallyCrawling,
          water: tiltFreestyle.drankWhileNeurallyCrawling,
        },
      },
      environmentContacts: {
        autonomous: first.obstacleContacts,
        freestyle: freestyle.obstacleContacts,
        ripple: rippleFreestyle.obstacleContacts,
        tilt: tiltFreestyle.obstacleContacts,
      },
      locomotionStates: first.locomotionStates,
      maximumRootStep: round(first.maximumRootStep),
      maximumSegmentStep: round(first.maximumSegmentStep),
      maximumYawHeadingError: round(first.maximumYawHeadingError),
      maximumSeparation: round(first.maximumSeparation),
      minimumStaticObstacleClearance: {
        autonomous: [round(first.minimumStaticObstacleClearance), first.minimumStaticObstacleContext],
        freestyle: [round(freestyle.minimumStaticObstacleClearance), freestyle.minimumStaticObstacleContext],
        ripple: [
          round(rippleFreestyle.minimumStaticObstacleClearance),
          rippleFreestyle.minimumStaticObstacleContext,
        ],
        tilt: [
          round(tiltFreestyle.minimumStaticObstacleClearance),
          tiltFreestyle.minimumStaticObstacleContext,
        ],
      },
      needsFulfillment: {
        food: round(first.foodFulfillment),
        water: round(first.waterFulfillment),
        wellbeing: round(first.wellbeingFulfillment),
        wellbeingAfterRemount: round(first.postRemountWellbeingFulfillment),
      },
      neuralResourceContacts: {
        food: first.ateWhileNeurallyCrawling,
        water: first.drankWhileNeurallyCrawling,
      },
      remounted: first.remounted,
      incidentalBoardContact,
      boardContactLifecycleGate,
      dismountSideSelection,
    },
    null,
    2,
  ),
)
console.log(candidateScreen ? 'Stunt motion candidate screen complete.' : 'Stunt motion verification passed.')
if (candidateScreen && candidateFailures.length > 0) process.exitCode = 1

type LocomotionIntervention = 'full' | 'zero' | 'frozen' | 'shuffled'

interface RolloutAudit {
  /** Check all rendered footprints against arena bounds and static obstacles. */
  geometry?: boolean
  /** Collect per-tick root/segment/body speed diagnostics. */
  motion?: boolean
  /** Validate that every policy observation is finite. */
  observations?: boolean
  /** Hash the sampled trajectory for the deterministic duplicate assertion. */
  trace?: boolean
}

function verifyDismountSideSelection() {
  const field = createTerrainField(environment)
  const makeState = () => {
    const state = createStuntState(field, environment, defaultAnatomy)
    state.locomotionState = 'riding'
    state.mountBlend = 1
    state.boardX = 0
    state.boardZ = 0
    state.boardY = field.sample(0, 0).height + 0.28
    state.boardVx = 0
    state.boardVz = 0
    state.boardSpeed = 0
    state.boardHeading = 0
    state.boardYaw = 0
    state.attempt = 0
    state.rideLandings = 1
    state.cycleTime = 6
    state.needs.wellbeing = 0
    state.resources = state.resources.filter(resource => resource.presentation !== 'bowl')
    return state
  }
  const preferredBlocker = {
    id: 'preferred-side-blocker',
    kind: 'rock' as const,
    center: { x: 0, z: 0.65 },
    radius: 0.25,
  }
  const alternateBlocker = {
    id: 'alternate-side-blocker',
    kind: 'rock' as const,
    center: { x: 0, z: -0.65 },
    radius: 0.25,
  }

  const alternateState = makeState()
  expect(alternateState.attempt % 2 === 0, 'dismount fixture did not prefer the positive side')
  const alternate = chooseDismountSide(alternateState, [preferredBlocker], environment)
  expect(alternate === -1, `blocked preferred side selected ${String(alternate)} instead of clear alternate`)

  const retryState = makeState()
  const blocked = chooseDismountSide(retryState, [preferredBlocker, alternateBlocker], environment)
  expect(blocked === null, `two blocked dismount corridors selected side ${String(blocked)}`)
  advanceStunt(
    retryState,
    makeInitialAction(),
    POLICY_TIMESTEP,
    9.81,
    'kickflip',
    field,
    environment,
    [preferredBlocker, alternateBlocker],
    defaultAnatomy,
  )
  expect(retryState.locomotionState === 'riding', 'blocked dismount did not remain mounted for a retry')

  advanceStunt(
    retryState,
    makeInitialAction(),
    POLICY_TIMESTEP,
    9.81,
    'kickflip',
    field,
    environment,
    [preferredBlocker],
    defaultAnatomy,
  )
  expect(retryState.locomotionState === 'dismounting', 'cleared alternate corridor was not retried')
  expect(retryState.dismountSide === -1, 'retry did not commit to the newly clear alternate side')

  return {
    preferred: 1,
    selectedWhenPreferredBlocked: alternate,
    bothBlocked: blocked,
    blockedLifecycleState: 'riding',
    retryLifecycleState: retryState.locomotionState,
    retrySide: retryState.dismountSide,
  }
}

function verifyBoardContactLifecycleGate() {
  const field = createTerrainField(environment)
  const proximityOnly = createStuntState(field, environment, defaultAnatomy)
  proximityOnly.locomotionState = 'mounting'
  proximityOnly.locomotionTime = 99
  proximityOnly.distanceToBoard = 0
  proximityOnly.needs.targetResourceId = 'skateboard'
  const idle = makeInitialAction()
  for (let step = 0; step < Math.ceil(0.75 / POLICY_TIMESTEP); step += 1) {
    advanceStunt(
      proximityOnly,
      idle,
      POLICY_TIMESTEP,
      9.81,
      'kickflip',
      field,
      environment,
      [],
      defaultAnatomy,
    )
  }
  expect(
    String(proximityOnly.locomotionState) !== 'riding',
    'proximity and elapsed time mounted the worm without measured deck contact',
  )

  const tooFast = {
    boardContactRatio: 0.72,
    boardContactSegmentCount: 13,
    boardContactHeadWeight: 0.84,
    boardContactMidbodyWeight: 0.8,
    boardContactTailWeight: 0.76,
    boardContactRelativeSpeed: 1.61,
  }
  const stable = { ...tooFast, boardContactRelativeSpeed: 0.18 }
  expect(!boardContactQualifiesForMount(tooFast), 'an unstable fly-by qualified as a board mount')
  expect(boardContactQualifiesForMount(stable), 'stable full-body deck contact did not qualify')
  return {
    dwellSeconds: BOARDING_STABLE_SECONDS,
    proximityOnlyState: proximityOnly.locomotionState,
    speedGate: [tooFast.boardContactRelativeSpeed, stable.boardContactRelativeSpeed],
  }
}

function verifyIncidentalBoardCollision() {
  const field = createTerrainField(environment)
  const state = createStuntState(field, environment, defaultAnatomy)
  const heading = state.boardHeading
  state.boardX = 0
  state.boardZ = 0
  state.boardY = field.sample(0, 0).height + 0.28
  state.boardVx = 0
  state.boardVz = 0
  const deckLength = Math.max(1.65, Math.min(2.1, environment.skateboard.deckSize[0]))
  const bodySpacing = 0.102 * defaultAnatomy.visualLengthScale
  const bodyHalfLength = bodySpacing * (SEGMENT_COUNT - 1) * 0.5
  const approachDistance = deckLength * 0.5 + bodyHalfLength + 0.11
  state.locomotionState = 'crawling'
  state.locomotionTime = 1
  state.mountBlend = 0
  state.boardCollisionCooldown = 0
  state.needs.hunger = 0.95
  state.needs.thirst = 0.1
  state.needs.wellbeing = 0.05
  state.needs.targetResourceId = 'food-bowl'
  state.wormX = state.boardX - Math.cos(heading) * approachDistance
  state.wormZ = state.boardZ - Math.sin(heading) * approachDistance
  const groundClearance = 0.105 * defaultAnatomy.verticalScale
  state.wormY = field.sample(state.wormX, state.wormZ).height + groundClearance
  state.wormHeading = heading
  state.wormVx = Math.cos(heading) * 1.45
  state.wormVz = Math.sin(heading) * 1.45
  state.locomotionPlant.forwardSpeed = 1.45
  state.locomotionPlant.angularSpeed = 0
  state.segments.forEach((segment, index) => {
    const axial = (index - (SEGMENT_COUNT - 1) * 0.5) * bodySpacing
    segment.x = state.wormX + Math.cos(heading) * axial
    segment.y = state.wormY
    segment.z = state.wormZ + Math.sin(heading) * axial
    segment.vx = state.wormVx
    segment.vy = 0
    segment.vz = state.wormVz
    segment.pitch = 0
    segment.yaw = heading
    state.segmentGroundContacts[index].anchorX = state.wormX
    state.segmentGroundContacts[index].anchorZ = state.wormZ
    state.segmentGroundContacts[index].strength = 0
  })

  let maximumDeckLift = 0
  let maximumSegmentStep = 0
  let previousSegments = state.segments.map(segment => [segment.x, segment.y, segment.z] as const)
  for (let step = 0; step < 42; step += 1) {
    advanceStunt(
      state,
      makeInitialAction(),
      POLICY_TIMESTEP,
      9.81,
      'freestyle',
      field,
      environment,
      [],
      defaultAnatomy,
    )
    state.segments.forEach((segment, index) => {
      maximumDeckLift = Math.max(
        maximumDeckLift,
        segment.y - field.sample(segment.x, segment.z).height - groundClearance,
      )
      maximumSegmentStep = Math.max(
        maximumSegmentStep,
        Math.hypot(
          segment.x - previousSegments[index][0],
          segment.y - previousSegments[index][1],
          segment.z - previousSegments[index][2],
        ),
      )
    })
    previousSegments = state.segments.map(segment => [segment.x, segment.y, segment.z] as const)
  }

  expect(
    state.locomotionState === 'crawling',
    'incidental board probe unexpectedly entered the mount lifecycle',
  )
  expect(
    maximumDeckLift > 0.08,
    `incidental board probe did not climb the low deck support (${maximumDeckLift.toFixed(6)} m)`,
  )
  expect(
    maximumSegmentStep < 0.09,
    `incidental board crossing teleported a segment ${maximumSegmentStep.toFixed(6)} m`,
  )
  expect(
    !state.obstacleContactId?.startsWith('skateboard-'),
    'low deck support was also exposed as a conflicting vertical wall collider',
  )
  return { maximumDeckLift: round(maximumDeckLift), maximumSegmentStep: round(maximumSegmentStep) }
}

function simulate(
  mode: ShowcaseMode,
  seconds: number,
  intervention: LocomotionIntervention = 'full',
  frictionOverride?: number,
  simulationEnvironment: EnvironmentConfig = environment,
  audit: RolloutAudit = minimalAudit,
  initialNeedTarget?: 'food-bowl',
  simulationAnatomy: WurmAnatomy = defaultAnatomy,
) {
  const originalField = createTerrainField(simulationEnvironment)
  const field =
    frictionOverride === undefined
      ? originalField
      : {
          ...originalField,
          sample: (x: number, z: number) => ({ ...originalField.sample(x, z), friction: frictionOverride }),
        }
  const state = createStuntState(field, simulationEnvironment, simulationAnatomy)
  if (initialNeedTarget === 'food-bowl') {
    state.needs.hunger = 1
    state.needs.thirst = 0
    state.needs.wellbeing = 0
    state.needs.targetResourceId = 'food-bowl'
  }
  const decor = makeTerrariumDecor(
    simulationEnvironment.seed,
    field,
    state.resources,
    simulationEnvironment.terrain.obstacleDensity,
  )
  const policy = new NeuralStuntPolicy(artifact)
  const locomotionPolicy = new EvolvedLocomotionPolicy(locomotionArtifact)
  const appliedAction = makeInitialAction()
  const idleAction = makeInitialAction()
  const intervenedAction = makeInitialAction()
  const frozenAction = makeInitialAction()
  const intervenedFeedback = new Float64Array(SEGMENT_COUNT)
  const frozenFeedback = new Float64Array(SEGMENT_COUNT)
  const bodySpeeds: number[] = []
  const locomotionStates: string[] = []
  const boardX: number[] = [state.boardX]
  const boardZ: number[] = [state.boardZ]
  const wormX: number[] = [state.wormX]
  const wormZ: number[] = [state.wormZ]
  const trace: string[] = []
  const obstacleContacts = new Set<string>()
  let minimumStaticObstacleClearance = Number.POSITIVE_INFINITY
  let minimumStaticObstacleContext = ''
  let maximumBodySpeed = 0
  let maximumBodyContext = ''
  let maximumRootStep = 0
  let maximumRootContext = ''
  let maximumSegmentStep = 0
  let maximumSegmentContext = ''
  let maximumSeparation = 0
  let maximumYawHeadingError = 0
  let observationsFinite = true
  let inBounds = true
  let hasRidden = false
  let sawDetachedAfterRide = false
  let remounted = false
  let wellbeingFulfillmentAtFirstDetach: number | null = null
  let postRemountWellbeingFulfillment = 0
  let ateWhileNeurallyCrawling = false
  let drankWhileNeurallyCrawling = false
  let locomotionStep = 0
  let neuralActionMax = 0
  let needTargetProgress = 0
  let firstNeuralFoodFeedingTime: number | null = null
  let firstNeuralFoodFeedingTick: number | null = null
  let maximumBoardContactRatio = 0
  let maximumBoardContactSegments = 0
  let maximumBoardContactStableSeconds = 0
  let maximumBoardContactHeadWeight = 0
  let maximumBoardContactMidbodyWeight = 0
  let maximumBoardContactTailWeight = 0
  let firstBoardContactTime: number | null = null
  let firstNeuralBoardingTime: number | null = null
  let firstAutonomousRideTime: number | null = null
  let firstAutonomousRideApplication: string | null = null
  let autonomousRideContact: {
    ratio: number
    segments: number
    stableSeconds: number
    head: number
    midbody: number
    tail: number
    relativeSpeed: number
  } | null = null
  let neuralMountingTicks = 0
  let nonNeuralMountingTicks = 0
  let minimumDistanceToBoard = state.distanceToBoard
  let minimumSegmentDistanceToBoard = Number.POSITIVE_INFINITY
  let maximumPlanarBoardSupportWeight = 0
  let maximumDismountClearance = Number.NEGATIVE_INFINITY
  const anatomy = simulationAnatomy

  for (let step = 0; step < Math.round(seconds / POLICY_TIMESTEP); step += 1) {
    const wasRiding = state.locomotionState === 'riding'
    const previousLocomotionState = state.locomotionState
    const observation = snapshotToObservation(toSnapshot(state))
    if (audit.observations) observationsFinite &&= observation.every(Number.isFinite)
    const locomotionOwnsBody = gaitControllerOwnsBody(state)
    const previousNeedTarget = locomotionOwnsBody
      ? state.resources.find(resource => resource.id === state.needs.targetResourceId)
      : null
    const previousNeedTargetDistance = previousNeedTarget
      ? Math.hypot(previousNeedTarget.position[0] - state.wormX, previousNeedTarget.position[2] - state.wormZ)
      : null
    const rawAction =
      state.locomotionState === 'riding'
        ? policy.run(observation)
        : locomotionOwnsBody
          ? locomotionPolicy.run(
              locomotionSensorsFor(state, field),
              state.locomotionPlant.joints,
              state.locomotionPlant.jointVelocities,
            )
          : idleAction
    if (state.locomotionState === 'riding') smoothAction(appliedAction, rawAction, POLICY_TIMESTEP)
    else if (locomotionOwnsBody) {
      for (const activation of rawAction) neuralActionMax = Math.max(neuralActionMax, Math.abs(activation))
      interveneLocomotion(
        rawAction,
        locomotionPolicy.getTelemetry().commands,
        intervention,
        locomotionStep,
        frozenAction,
        frozenFeedback,
        intervenedAction,
        intervenedFeedback,
      )
      appliedAction.set(intervenedAction)
      if (intervention !== 'full') locomotionPolicy.commitCommandFeedback(intervenedFeedback)
      locomotionStep += 1
    } else appliedAction.fill(0)
    const previousBoardX = audit.motion ? state.boardX : 0
    const previousBoardZ = audit.motion ? state.boardZ : 0
    const previousWormX = audit.motion ? state.wormX : 0
    const previousWormZ = audit.motion ? state.wormZ : 0
    const previousWormHeading = audit.motion ? state.wormHeading : 0
    const previousSegments = audit.motion
      ? state.segments.map(segment => [segment.x, segment.y, segment.z] as const)
      : []

    advanceStunt(
      state,
      appliedAction,
      POLICY_TIMESTEP,
      9.81,
      mode,
      field,
      simulationEnvironment,
      decor.obstacles,
      anatomy,
    )
    maximumBoardContactRatio = Math.max(maximumBoardContactRatio, state.boardContactRatio)
    maximumBoardContactSegments = Math.max(maximumBoardContactSegments, state.boardContactSegmentCount)
    maximumBoardContactStableSeconds = Math.max(
      maximumBoardContactStableSeconds,
      state.boardContactStableSeconds,
    )
    maximumBoardContactHeadWeight = Math.max(maximumBoardContactHeadWeight, state.boardContactHeadWeight)
    maximumBoardContactMidbodyWeight = Math.max(
      maximumBoardContactMidbodyWeight,
      state.boardContactMidbodyWeight,
    )
    maximumBoardContactTailWeight = Math.max(maximumBoardContactTailWeight, state.boardContactTailWeight)
    if (state.locomotionState === 'mounting') {
      if (state.previousActionApplication === 'neural') neuralMountingTicks += 1
      else nonNeuralMountingTicks += 1
    }
    if (
      firstAutonomousRideTime === null &&
      previousLocomotionState === 'mounting' &&
      state.locomotionState === 'riding'
    ) {
      firstAutonomousRideTime = state.time
      firstAutonomousRideApplication = state.previousActionApplication
      autonomousRideContact = {
        ratio: state.boardContactRatio,
        segments: state.boardContactSegmentCount,
        stableSeconds: state.boardContactStableSeconds,
        head: state.boardContactHeadWeight,
        midbody: state.boardContactMidbodyWeight,
        tail: state.boardContactTailWeight,
        relativeSpeed: state.boardContactRelativeSpeed,
      }
    }
    minimumDistanceToBoard = Math.min(minimumDistanceToBoard, state.distanceToBoard)
    if (firstAutonomousRideTime === null) {
      for (const segment of state.segments) {
        minimumSegmentDistanceToBoard = Math.min(
          minimumSegmentDistanceToBoard,
          Math.hypot(segment.x - state.boardX, segment.z - state.boardZ),
        )
        maximumPlanarBoardSupportWeight = Math.max(
          maximumPlanarBoardSupportWeight,
          skateboardDeckSupportWeight(
            segment.x,
            segment.z,
            { x: state.boardX, z: state.boardZ, heading: state.boardHeading },
            simulationEnvironment,
            anatomy,
          ),
        )
      }
    }
    if (state.locomotionState === 'riding') {
      let dismountClearance = Number.POSITIVE_INFINITY
      for (const resource of state.resources) {
        if (resource.presentation !== 'bowl') continue
        dismountClearance = Math.min(
          dismountClearance,
          Math.hypot(state.boardX - resource.position[0], state.boardZ - resource.position[2]) -
            resource.appearance.radius -
            1.05,
        )
      }
      for (const obstacle of decor.obstacles) {
        dismountClearance = Math.min(
          dismountClearance,
          Math.hypot(state.boardX - obstacle.center.x, state.boardZ - obstacle.center.z) -
            obstacle.radius -
            1.35,
        )
      }
      maximumDismountClearance = Math.max(maximumDismountClearance, dismountClearance)
    }
    if (firstBoardContactTime === null && state.boardContactRatio > 0) firstBoardContactTime = state.time
    if (
      firstNeuralBoardingTime === null &&
      state.locomotionState === 'mounting' &&
      state.previousActionApplication === 'neural'
    ) {
      firstNeuralBoardingTime = state.time
    }
    if (
      previousNeedTarget &&
      previousNeedTargetDistance !== null &&
      state.needs.targetResourceId === previousNeedTarget.id &&
      state.previousActionApplication === 'neural'
    ) {
      needTargetProgress +=
        previousNeedTargetDistance -
        Math.hypot(previousNeedTarget.position[0] - state.wormX, previousNeedTarget.position[2] - state.wormZ)
    }
    if (audit.geometry && state.obstacleContactId) obstacleContacts.add(state.obstacleContactId)
    const boardProbes = audit.geometry
      ? skateboardFootprintObstacles(
          state.boardX,
          state.boardY,
          state.boardZ,
          state.boardHeading,
          simulationEnvironment,
        ).map(sample => ({ ...sample.center, radius: sample.radius }))
      : []
    const detachedRoot = gaitControllerOwnsBody(state) || state.locomotionState === 'feeding'
    const recordStaticClearance = (clearance: number, context: string) => {
      if (clearance >= minimumStaticObstacleClearance) return
      minimumStaticObstacleClearance = clearance
      minimumStaticObstacleContext = `t=${state.time.toFixed(3)} ${context}`
    }
    if (audit.geometry) {
      for (const obstacle of decor.obstacles) {
        for (const [probeIndex, probe] of boardProbes.entries()) {
          recordStaticClearance(
            Math.hypot(probe.x - obstacle.center.x, probe.z - obstacle.center.z) -
              obstacle.radius -
              probe.radius,
            `board-${probeIndex} obstacle=${obstacle.id}`,
          )
        }
        for (const [segmentIndex, segment] of state.segments.entries()) {
          const segmentRadius = renderedSegmentRadius(segmentIndex, anatomy)
          recordStaticClearance(
            Math.hypot(segment.x - obstacle.center.x, segment.z - obstacle.center.z) -
              obstacle.radius -
              segmentRadius,
            `segment=${segmentIndex} obstacle=${obstacle.id}`,
          )
        }
      }
    }
    ateWhileNeurallyCrawling ||=
      locomotionOwnsBody && state.locomotionState === 'feeding' && state.feedingResourceId === 'food-bowl'
    if (
      firstNeuralFoodFeedingTime === null &&
      locomotionOwnsBody &&
      state.locomotionState === 'feeding' &&
      state.feedingResourceId === 'food-bowl'
    ) {
      firstNeuralFoodFeedingTime = state.time
      firstNeuralFoodFeedingTick = locomotionStep
    }
    drankWhileNeurallyCrawling ||=
      locomotionOwnsBody && state.locomotionState === 'feeding' && state.feedingResourceId === 'water-bowl'
    if (wasRiding && state.locomotionState !== 'riding') {
      locomotionPolicy.reset()
      appliedAction.fill(0)
      wellbeingFulfillmentAtFirstDetach ??= state.needs.fulfillment.wellbeing
      locomotionStep = 0
      frozenAction.fill(0)
      frozenFeedback.fill(0)
      intervenedFeedback.fill(0)
    }

    if (locomotionStates.at(-1) !== state.locomotionState) locomotionStates.push(state.locomotionState)
    if (state.locomotionState === 'riding') {
      if (hasRidden && sawDetachedAfterRide) remounted = true
      hasRidden = true
      if (wellbeingFulfillmentAtFirstDetach !== null) {
        postRemountWellbeingFulfillment = Math.max(
          postRemountWellbeingFulfillment,
          state.needs.fulfillment.wellbeing - wellbeingFulfillmentAtFirstDetach,
        )
      }
    } else if (hasRidden) sawDetachedAfterRide = true
    if (audit.motion) {
      const boardRootStep = Math.hypot(state.boardX - previousBoardX, state.boardZ - previousBoardZ)
      const wormRootStep = Math.hypot(state.wormX - previousWormX, state.wormZ - previousWormZ)
      const rootStep = Math.max(boardRootStep, wormRootStep)
      if (rootStep > maximumRootStep) {
        maximumRootStep = rootStep
        const water = state.resources.find(resource => resource.id === 'water-bowl')
        const closestWaterBefore = water
          ? previousSegments.reduce(
              (best, segment, index) => {
                const distance = Math.hypot(segment[0] - water.position[0], segment[2] - water.position[2])
                return distance < best.distance ? { index, distance } : best
              },
              { index: -1, distance: Number.POSITIVE_INFINITY },
            )
          : null
        const closestWaterAfter = water
          ? state.segments.reduce(
              (best, segment, index) => {
                const distance = Math.hypot(segment.x - water.position[0], segment.z - water.position[2])
                return distance < best.distance ? { index, distance } : best
              },
              { index: -1, distance: Number.POSITIVE_INFINITY },
            )
          : null
        maximumRootContext =
          `t=${state.time.toFixed(3)} phase=${state.phase} locomotion=${previousLocomotionState}->${state.locomotionState} ` +
          `locomotionTime=${state.locomotionTime.toFixed(3)} ` +
          `board=${boardRootStep.toFixed(4)} worm=${wormRootStep.toFixed(4)} contact=${state.obstacleContactId ?? 'none'} ` +
          `wormFrom=${previousWormX.toFixed(3)},${previousWormZ.toFixed(3)} wormTo=${state.wormX.toFixed(3)},${state.wormZ.toFixed(3)} ` +
          `waterClosest=${closestWaterBefore?.index ?? -1}:${closestWaterBefore?.distance.toFixed(3) ?? 'n/a'}->` +
          `${closestWaterAfter?.index ?? -1}:${closestWaterAfter?.distance.toFixed(3) ?? 'n/a'} ` +
          `boardFrom=${previousBoardX.toFixed(3)},${previousBoardZ.toFixed(3)} boardTo=${state.boardX.toFixed(3)},${state.boardZ.toFixed(3)} ` +
          `boardVelocity=${state.boardVx.toFixed(3)},${state.boardVz.toFixed(3)}`
      }
      const rootVx = state.mountBlend > 0.5 ? state.boardVx : state.wormVx
      const rootVz = state.mountBlend > 0.5 ? state.boardVz : state.wormVz
      let stepBodySpeed = 0
      let stepBodySegment = 0
      state.segments.forEach((segment, index) => {
        const relativeBodySpeed = Math.hypot(segment.vx - rootVx, segment.vy, segment.vz - rootVz)
        if (relativeBodySpeed > stepBodySpeed) {
          stepBodySpeed = relativeBodySpeed
          stepBodySegment = index
        }
        const segmentStep = Math.hypot(
          segment.x - previousSegments[index][0],
          segment.y - previousSegments[index][1],
          segment.z - previousSegments[index][2],
        )
        if (segmentStep > maximumSegmentStep) {
          maximumSegmentStep = segmentStep
          const segmentRadius = renderedSegmentRadius(index, anatomy)
          const previousStaticClearance = Math.min(
            ...decor.obstacles.map(
              obstacle =>
                Math.hypot(
                  previousSegments[index][0] - obstacle.center.x,
                  previousSegments[index][2] - obstacle.center.z,
                ) -
                obstacle.radius -
                segmentRadius,
            ),
          )
          const currentStaticClearance = Math.min(
            ...decor.obstacles.map(
              obstacle =>
                Math.hypot(segment.x - obstacle.center.x, segment.z - obstacle.center.z) -
                obstacle.radius -
                segmentRadius,
            ),
          )
          const previousResourceClearances = state.resources.map(resource => ({
            id: resource.id,
            clearance:
              Math.hypot(
                previousSegments[index][0] - resource.position[0],
                previousSegments[index][2] - resource.position[2],
              ) -
              (resource.presentation === 'skateboard' ? 0.6 : resource.appearance.radius * 0.94) -
              segmentRadius,
          }))
          const closestPreviousResource = previousResourceClearances.reduce((closest, resource) =>
            resource.clearance < closest.clearance ? resource : closest,
          )
          const waterResource = state.resources.find(resource => resource.id === 'water-bowl')
          const waterTrace = waterResource
            ? ` water=${Math.hypot(previousSegments[index][0] - waterResource.position[0], previousSegments[index][2] - waterResource.position[2]).toFixed(4)}->` +
              `${Math.hypot(segment.x - waterResource.position[0], segment.z - waterResource.position[2]).toFixed(4)} ` +
              `y=${previousSegments[index][1].toFixed(3)}->${segment.y.toFixed(3)} top=${(waterResource.groundHeight + waterResource.appearance.height * 1.08).toFixed(3)}`
            : ''
          maximumSegmentContext =
            `t=${state.time.toFixed(3)} phase=${state.phase} locomotion=${state.locomotionState} ` +
            `segment=${index} mount=${state.mountBlend.toFixed(3)} range=${state.distanceToBoard.toFixed(3)} ` +
            `contact=${state.obstacleContactId ?? 'none'} grip=${state.segmentGroundContacts[index]?.strength.toFixed(3) ?? 'none'} ` +
            `locomotionTime=${state.locomotionTime.toFixed(3)} ` +
            `delta=${(segment.x - previousSegments[index][0]).toFixed(3)}/` +
            `${(segment.y - previousSegments[index][1]).toFixed(3)}/` +
            `${(segment.z - previousSegments[index][2]).toFixed(3)} ` +
            `anchor=${(state.segmentGroundContacts[index]?.anchorX - segment.x).toFixed(3)}/` +
            `${(state.segmentGroundContacts[index]?.anchorZ - segment.z).toFixed(3)} ` +
            `clearance=${previousStaticClearance.toFixed(4)}->${currentStaticClearance.toFixed(4)} ` +
            `resource=${closestPreviousResource.id}:${closestPreviousResource.clearance.toFixed(4)} ` +
            `headingDelta=${wrapAngle(state.wormHeading - previousWormHeading).toFixed(3)}${waterTrace}`
        }
      })
      bodySpeeds.push(stepBodySpeed)
      if (stepBodySpeed > maximumBodySpeed) {
        maximumBodySpeed = stepBodySpeed
        maximumBodyContext =
          `t=${state.time.toFixed(3)} phase=${state.phase} locomotion=${state.locomotionState} ` +
          `segment=${stepBodySegment} mount=${state.mountBlend.toFixed(3)} contact=${state.obstacleContactId ?? 'none'}`
      }
      maximumSeparation = Math.max(maximumSeparation, state.distanceToBoard)
      maximumYawHeadingError = Math.max(
        maximumYawHeadingError,
        Math.abs(wrapAngle(state.boardYaw + state.boardHeading)),
      )
    }
    if (audit.geometry) {
      boardX.push(state.boardX)
      boardZ.push(state.boardZ)
      wormX.push(state.wormX)
      wormZ.push(state.wormZ)
      const xLimit = field.width * 0.5
      const zLimit = field.depth * 0.5
      const boardFootprintInBounds = boardProbes.every(
        probe => Math.abs(probe.x) + probe.radius <= xLimit && Math.abs(probe.z) + probe.radius <= zLimit,
      )
      const segmentFootprintsInBounds = state.segments.every(
        (segment, index) =>
          Math.abs(segment.x) + renderedSegmentRadius(index, anatomy) <= xLimit &&
          Math.abs(segment.z) + renderedSegmentRadius(index, anatomy) <= zLimit,
      )
      const detachedRootInBounds =
        !detachedRoot || (Math.abs(state.wormX) + 0.13 <= xLimit && Math.abs(state.wormZ) + 0.13 <= zLimit)
      inBounds &&= boardFootprintInBounds && segmentFootprintsInBounds && detachedRootInBounds
    }

    if (audit.trace && step % 10 === 0) {
      trace.push(
        [
          state.boardX,
          state.boardY,
          state.boardZ,
          state.wormX,
          state.wormY,
          state.wormZ,
          state.boardRoll,
          state.flipsLanded,
          state.locomotionState,
          state.needs.targetResourceId ?? 'none',
        ]
          .map(value => (typeof value === 'number' ? value.toFixed(6) : value))
          .join(','),
      )
    }
  }

  const sortedBodySpeeds = audit.motion ? bodySpeeds.toSorted((a, b) => a - b) : [0]
  const rightX = -Math.sin(state.wormHeading)
  const rightZ = Math.cos(state.wormHeading)
  const finalLaterals = state.segments.map(
    segment => (segment.x - state.wormX) * rightX + (segment.z - state.wormZ) * rightZ,
  )
  return {
    ateWhileNeurallyCrawling,
    boardRangeX: range(boardX),
    boardRangeZ: range(boardZ),
    bodySpeedP99:
      sortedBodySpeeds[Math.min(sortedBodySpeeds.length - 1, Math.floor(sortedBodySpeeds.length * 0.99))],
    crawlDistance: state.wormDistance,
    flipsLanded: state.flipsLanded,
    foodFulfillment: state.needs.fulfillment.hunger,
    finalDistanceToBoard: state.distanceToBoard,
    finalObstacleContact: state.obstacleContactId,
    obstacleContacts: [...obstacleContacts].sort(),
    finalLateralSpan: range(finalLaterals),
    firstNeuralFoodFeedingTick,
    firstNeuralFoodFeedingTime,
    hash: createHash('sha256').update(trace.join('\n')).digest('hex'),
    inBounds,
    locomotionStates,
    maximumBodySpeed,
    maximumBodyContext,
    maximumRootStep,
    maximumRootContext,
    maximumSegmentStep,
    maximumSegmentContext,
    maximumSeparation,
    maximumYawHeadingError,
    maximumBoardContactRatio,
    maximumBoardContactSegments,
    maximumBoardContactStableSeconds,
    maximumBoardContactHeadWeight,
    maximumBoardContactMidbodyWeight,
    maximumBoardContactTailWeight,
    maximumDismountClearance,
    minimumDistanceToBoard,
    minimumSegmentDistanceToBoard,
    maximumPlanarBoardSupportWeight,
    minimumStaticObstacleClearance,
    minimumStaticObstacleContext,
    neuralActionMax,
    needTargetProgress,
    observationsFinite,
    firstBoardContactTime,
    firstNeuralBoardingTime,
    firstAutonomousRideTime,
    firstAutonomousRideApplication,
    autonomousRideContact,
    neuralMountingTicks,
    nonNeuralMountingTicks,
    postRemountWellbeingFulfillment,
    remounted,
    drankWhileNeurallyCrawling,
    waterFulfillment: state.needs.fulfillment.thirst,
    wellbeingFulfillment: state.needs.fulfillment.wellbeing,
    wormRangeX: range(wormX),
    wormRangeZ: range(wormZ),
  }
}

function interveneLocomotion(
  rawAction: Float32Array,
  rawFeedback: ArrayLike<number>,
  intervention: LocomotionIntervention,
  locomotionStep: number,
  frozenAction: Float32Array,
  frozenFeedback: Float64Array,
  outputAction: Float32Array,
  outputFeedback: Float64Array,
) {
  if (intervention === 'zero') {
    outputAction.fill(0)
    outputFeedback.fill(0)
    return
  }
  if (intervention === 'frozen') {
    if (locomotionStep === 24) {
      frozenAction.set(rawAction)
      frozenFeedback.set(rawFeedback)
    }
    outputAction.set(locomotionStep < 24 ? rawAction : frozenAction)
    outputFeedback.set(locomotionStep < 24 ? rawFeedback : frozenFeedback)
    return
  }
  if (intervention === 'shuffled') {
    // Match the fixed causal ownership ablation used by evolution and the
    // authoritative isolated-plant verifier.
    for (let segment = 0; segment < SEGMENT_COUNT; segment += 1) {
      const source = SEGMENT_SHUFFLE[segment]
      outputAction[segment * 2] = rawAction[source * 2]
      outputAction[segment * 2 + 1] = rawAction[source * 2 + 1]
      outputFeedback[segment] = rawFeedback[source] ?? 0
    }
    return
  }
  outputAction.set(rawAction)
  outputFeedback.set(rawFeedback)
}

function range(values: number[]) {
  return Math.max(...values) - Math.min(...values)
}

function renderedSegmentRadius(index: number, anatomy: WurmAnatomy) {
  return 0.085 * (0.74 + Math.sin((index / (SEGMENT_COUNT - 1)) * Math.PI) * 0.28) * anatomy.thicknessScale
}

function round(value: number) {
  return Number(value.toFixed(6))
}

function boardingOutcome(rollout: ReturnType<typeof simulate>) {
  return JSON.stringify(boardingJson(rollout))
}

function boardingJson(rollout: ReturnType<typeof simulate>) {
  return {
    contact: rollout.firstBoardContactTime,
    neuralBoarding: rollout.firstNeuralBoardingTime,
    ride: rollout.firstAutonomousRideTime,
    rideContact: rollout.autonomousRideContact,
    neuralMountingTicks: rollout.neuralMountingTicks,
    nonNeuralMountingTicks: rollout.nonNeuralMountingTicks,
    stable: round(rollout.maximumBoardContactStableSeconds),
    states: rollout.locomotionStates,
  }
}

function firstFoodHandoffJson(rollout: ReturnType<typeof simulate>) {
  return rollout.firstNeuralFoodFeedingTime === null
    ? null
    : {
        simulationTime: round(rollout.firstNeuralFoodFeedingTime),
        neuralControllerTick: rollout.firstNeuralFoodFeedingTick,
      }
}

function foodHandoffLeads(full: ReturnType<typeof simulate>, intervention: ReturnType<typeof simulate>) {
  if (full.firstNeuralFoodFeedingTime === null) return false
  return (
    intervention.firstNeuralFoodFeedingTime === null ||
    intervention.firstNeuralFoodFeedingTime - full.firstNeuralFoodFeedingTime >= FOOD_HANDOFF_LEAD_SECONDS
  )
}

function formatFoodHandoff(rollout: ReturnType<typeof simulate>) {
  return rollout.firstNeuralFoodFeedingTime === null
    ? 'never'
    : `${rollout.firstNeuralFoodFeedingTime.toFixed(3)} s / neural tick ${rollout.firstNeuralFoodFeedingTick}`
}

function wrapAngle(angle: number) {
  return Math.atan2(Math.sin(angle), Math.cos(angle))
}

function expect(condition: boolean, message: string) {
  if (!condition) {
    if (candidateScreen) {
      candidateFailures.push(message)
      return
    }
    console.error(`Stunt motion verification failed: ${message}`)
    process.exit(1)
  }
}
