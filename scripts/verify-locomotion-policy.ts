import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { copyFile, mkdir, mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { deriveWurmAnatomy } from '../src/creature/anatomy'
import {
  EvolvedLocomotionPolicy,
  LOCOMOTION_INPUT_WEIGHT_NAMES,
  LOCOMOTION_OUTPUT_WEIGHT_NAMES,
  LOCOMOTION_PLANT_CONTRACT,
  LOCOMOTION_RECURRENT_WEIGHT_NAMES,
  LOCOMOTION_SENSOR_NAMES,
  parseLocomotionPolicy,
  type LocomotionPolicyArtifact,
  type LocomotionSensors,
} from '../src/policy/locomotionPolicy'
import { ACTION_SIZE, SEGMENT_COUNT, type PolicyAction, type SegmentSnapshot } from '../src/policy/types'
import { TERRAIN_GRID_RESOLUTION, type TerrainField } from '../src/scene/terrainField'
import { stepArticulatedWorm } from '../src/scene/wormDynamics'
import { createWormLocomotionPlant } from '../src/scene/wormLocomotion'

type Mode = 'full' | 'zero' | 'frozen' | 'shuffled'

type InterventionCheckpoint = {
  step: number
  root: number[]
  heading: number
  forwardSpeed: number
  angularSpeed: number
  joints: number[]
  jointVelocities: number[]
  commands: number[]
  plantCommands: number[]
}

type ClosedLoopInterventionTrace = {
  mode: Mode
  steps: number
  checkpoints: InterventionCheckpoint[]
}

type NeuralTransitionInput = {
  targetForward: number
  targetRight: number
  targetDistance: number
  forwardSpeed: number
  angularSpeed: number
  terrainFriction: number
  urgency: number
  joints: number[]
  jointVelocities: number[]
  previousCommands: number[]
  contactLoads: number[]
  slipSpeeds: number[]
  obstacleForward: number[]
  obstacleRight: number[]
}

type NeuralTransitionFixture = {
  name: string
  input: NeuralTransitionInput
  output: {
    hidden: number[]
    commands: number[]
    plantCommands: number[]
  }
}

type IsolatedPlantTransition = {
  input: {
    friction: number
    joints: number[]
    jointVelocities: number[]
    plantCommands: number[]
    bodyPositions: number[][]
    bodyVelocities: number[][]
  }
  output: {
    root: number[]
    heading: number
    forwardSpeed: number
    angularSpeed: number
    joints: number[]
    jointVelocities: number[]
    bodyPositions: number[][]
    bodyVelocities: number[][]
  }
}

type PythonInterventionTrace = ClosedLoopInterventionTrace & {
  neuralTransitions: NeuralTransitionFixture[]
  isolatedPlantTransition: IsolatedPlantTransition
}

type Scenario = {
  targetX: number
  targetZ: number
  friction: number
  urgency: number
}

type Rollout = {
  progress: number
  displacement: number
  pathLength: number
  temporalStd: number
  spatialStd: number
  meanContactRatio: number
  finalX: number
  finalZ: number
  finalHeading: number
  meanTargetAlignment: number
  meanVelocityAlongBodyAxis: number
  meanAbsoluteTransverseFraction: number
  diagnosticPathLength: number
  traceHash: string
}

const OBJECTIVE_V2 = 'articulated-contact-obstacle-recovery-v2'
const OBJECTIVE_V3 = 'articulated-head-leading-transient-recovery-v3'
const V2_DOMAINS = ['obstacles', 'spatial-friction', 'body-scale', 'target-switches', 'contact-loss'] as const
const V3_DOMAINS = [
  ...V2_DOMAINS,
  'transient-slick-window',
  'external-body-shove',
  'paired-shove-numb-low-traction-proxy',
] as const
const V3_RECIPE_SOURCE_PATHS = [
  'training/wurmkickflip_rl/evolve_locomotion_policy.py',
  'training/wurmkickflip_rl/articulated_locomotion.py',
  'training/wurmkickflip_rl/contracts.py',
  'contracts/locomotion-v2.json',
  'training/uv.lock',
  'training/pyproject.toml',
] as const
const V3_SELECTION_GATE_KEYS = [
  'nominalSignedHeadLeading',
  'nominalLateralMotion',
  'nominalTargetAlignment',
  'combinedControlProgress',
  'combinedPerturbedProgress',
  'combinedProgressRetention',
  'combinedLateSpeedRetention',
  'combinedControlLateSpeed',
  'recoveredSignedHeadLeading',
  'recoveredLateralMotion',
  'recoveredTargetAlignment',
  'causalOrderingGap',
  'causalProgressGap',
  'localCausalInitial',
  'localCausalRadiusTwo',
  'localCausalRadiusThree',
  'localCausalSettled',
] as const
const V3_ABLATION_GATE_KEYS = [
  'fullProgress',
  'zeroActionStationary',
  'fullBeatsFrozen',
  'fullBeatsShuffled',
  'noFrictionConservesCom',
] as const
const INTERVENTION_MODES = ['full', 'zero', 'frozen', 'shuffled'] as const satisfies readonly Mode[]
const INTERVENTION_CHECKPOINT_STEPS = [0, 1, 2, 23, 24, 59, 119, 239, 419] as const
const FROZEN_CAPTURE_STEP = 24
const SEGMENT_SHUFFLE = [0, 9, 2, 13, 4, 15, 6, 11, 8, 1, 10, 3, 12, 5, 14, 7] as const

const root = resolve(import.meta.dirname, '..')
const trainingRoot = resolve(root, 'training')
const artifactSource = process.env.WURMKICKFLIP_LOCOMOTION_ARTIFACT
  ? resolve(process.env.WURMKICKFLIP_LOCOMOTION_ARTIFACT)
  : new URL('../public/models/wurmkickflip_locomotion_policy.json', import.meta.url)
const candidateScreen = process.env.WURMKICKFLIP_CANDIDATE_SCREEN === '1'
const legacyUrl = new URL('../training/seeds/wurmkickflip_locomotion_warm_start_v1.json', import.meta.url)
const trainerUrl = new URL('../training/wurmkickflip_rl/evolve_locomotion_policy.py', import.meta.url)
const rawV2 = JSON.parse(await readFile(artifactSource, 'utf8')) as any
const rawV1 = JSON.parse(await readFile(legacyUrl, 'utf8')) as any
// Retained evolutionary checkpoints deliberately are not runtime-loadable V3
// artifacts. The explicit CLI-only screen strips the publication discriminator
// from a clone so behavior can be measured without weakening the production parser.
const candidateInput = candidateScreen ? structuredClone(rawV2) : rawV2
if (candidateScreen && candidateInput.training) delete candidateInput.training.objectiveVersion
const artifact = parseLocomotionPolicy(candidateInput)
const trainingSource = await readFile(trainerUrl, 'utf8')
const currentRecipeSourceSha256 = Object.fromEntries(
  await Promise.all(
    V3_RECIPE_SOURCE_PATHS.map(async path => {
      const source = await readFile(resolve(root, path), 'utf8')
      return [
        path,
        createHash('sha256')
          .update(source.replace(/\r\n?|\n/gu, '\n'))
          .digest('hex'),
      ]
    }),
  ),
)

if (!candidateScreen) {
  verifyContract(artifact, rawV2, trainingSource)
  verifyLegacyMigration(rawV1)
  verifyMalformedArtifacts(rawV2, rawV1)
}
verifyAntagonisticOutputs(artifact)
const interventionParity = candidateScreen
  ? { skipped: 'non-public candidate behavior screen' }
  : verifyInterventionParity(artifact)
const commandFeedback = verifyCommandFeedbackOverride(artifact)
const localSensorDelta = verifyLocalSensorEffects(artifact)
const steeringActionDelta = verifySteeringSensorEffect(artifact)

const scenarios: Scenario[] = [
  { targetX: 4.2, targetZ: 0, friction: 0.95, urgency: 0.72 },
  { targetX: 3.2, targetZ: 3.2, friction: 0.58, urgency: 0.88 },
  { targetX: 3.2, targetZ: -3.2, friction: 1.12, urgency: 0.64 },
  { targetX: 0.5, targetZ: 4.3, friction: 0.82, urgency: 1 },
  { targetX: 0.5, targetZ: -4.3, friction: 0.42, urgency: 0.92 },
  { targetX: 4.8, targetZ: 1.2, friction: 0.33, urgency: 1 },
]
const steps = finiteInteger(artifact.training.episodeSteps, 'training.episodeSteps')
const fullRollouts = scenarios.map(scenario => runScenario(artifact, scenario, steps, 'full'))
const full = aggregate(fullRollouts)
const repeated = aggregate(scenarios.map(scenario => runScenario(artifact, scenario, steps, 'full')))
const zero = aggregate(scenarios.map(scenario => runScenario(artifact, scenario, steps, 'zero')))
const frozen = aggregate(scenarios.map(scenario => runScenario(artifact, scenario, steps, 'frozen')))
const shuffled = aggregate(scenarios.map(scenario => runScenario(artifact, scenario, steps, 'shuffled')))
const frictionless = runScenario(
  artifact,
  { targetX: 4.2, targetZ: 1.1, friction: 0, urgency: 1 },
  steps,
  'full',
)

if (candidateScreen) {
  const promotion = verifyV3RuntimePromotion(artifact, fullRollouts, full, shuffled, {
    force: true,
    enforce: false,
  })
  console.log(
    JSON.stringify(
      {
        candidateScreen: true,
        modelVersion: artifact.modelVersion,
        targetProgress: round(full.progress),
        meanDisplacement: round(full.displacement),
        recurrentAblations: {
          zeroProgress: round(zero.progress),
          frozenProgress: round(frozen.progress),
          shuffledProgress: round(shuffled.progress),
        },
        zeroFrictionComDisplacement: frictionless.displacement,
        v3RuntimePromotion: promotion,
      },
      null,
      2,
    ),
  )
  process.exit(promotion.passed ? 0 : 1)
}

expect(full.traceHash === repeated.traceHash, 'articulated locomotion traces must be deterministic')
expect(full.displacement > 0.5, `active ground contact must produce meaningful travel (${full.displacement})`)
expect(full.progress > 0.45, `evolved controller target progress is too small (${full.progress})`)
expect(full.progress > zero.progress + 0.4, 'evolved controller must beat the zero-action ablation')
expect(full.progress > frozen.progress + 0.3, 'evolved controller must beat the frozen-output ablation')
expect(full.progress > shuffled.progress + 0.2, 'evolved controller must beat the segment-shuffle ablation')
expect(full.temporalStd > 0.2, 'recurrent actuator commands must vary over time')
expect(full.spatialStd > 0.08, 'segments must own meaningfully different simultaneous commands')
expect(full.meanContactRatio > 0.2, 'active locomotion must retain ground contact')
expect(
  zero.displacement <= 1e-12,
  `zero outputs must leave the articulated chain stationary (${zero.displacement})`,
)
expect(
  frictionless.displacement <= 1e-12,
  `obstacle-free zero-friction internal forces must exactly conserve planar COM (${frictionless.displacement})`,
)
expect(
  frictionless.temporalStd > 0.15,
  'the zero-friction invariant must be tested with active neural commands',
)

const left = runScenario(artifact, { targetX: 3.2, targetZ: 3.2, friction: 0.8, urgency: 0.9 }, steps, 'full')
const right = runScenario(
  artifact,
  { targetX: 3.2, targetZ: -3.2, friction: 0.8, urgency: 0.9 },
  steps,
  'full',
)
expect(left.finalZ > 0.1, `left target must steer toward positive z (${left.finalZ})`)
expect(right.finalZ < -0.1, `right target must steer toward negative z (${right.finalZ})`)
expect(left.finalZ - right.finalZ > 0.5, 'opposite target bearings must produce distinct articulated paths')
const v3Promotion = verifyV3RuntimePromotion(artifact, fullRollouts, full, shuffled)

const trainingRepro = await verifyTrainingReproducibility()

console.log(
  JSON.stringify(
    {
      modelVersion: artifact.modelVersion,
      objectiveVersion: artifact.training.objectiveVersion,
      plantVersion: artifact.plant.version,
      targetProgress: round(full.progress),
      meanDisplacement: round(full.displacement),
      perScenarioProgress: fullRollouts.map(rollout => round(rollout.progress)),
      perScenarioTravel: fullRollouts.map((rollout, index) => ({
        target: [scenarios[index].targetX, scenarios[index].targetZ],
        friction: scenarios[index].friction,
        urgency: scenarios[index].urgency,
        progress: round(rollout.progress),
        displacement: round(rollout.displacement),
        pathLength: round(rollout.pathLength),
        finalPosition: [round(rollout.finalX), round(rollout.finalZ)],
        finalHeading: round(rollout.finalHeading),
        meanContactRatio: round(rollout.meanContactRatio),
        temporalStd: round(rollout.temporalStd),
        spatialStd: round(rollout.spatialStd),
        meanTargetAlignment: round(rollout.meanTargetAlignment),
        meanVelocityAlongBodyAxis: round(rollout.meanVelocityAlongBodyAxis),
        meanAbsoluteTransverseFraction: round(rollout.meanAbsoluteTransverseFraction),
        traceHash: rollout.traceHash,
      })),
      recurrentAblations: {
        zero: {
          progress: round(zero.progress),
          displacement: round(zero.displacement),
          traceHash: zero.traceHash,
        },
        frozen: {
          progress: round(frozen.progress),
          displacement: round(frozen.displacement),
          traceHash: frozen.traceHash,
        },
        shuffled: {
          progress: round(shuffled.progress),
          displacement: round(shuffled.displacement),
          traceHash: shuffled.traceHash,
        },
      },
      commandVariation: {
        temporal: round(full.temporalStd),
        spatial: round(full.spatialStd),
      },
      bodyAxisDiagnostics: {
        meanTargetAlignment: round(full.meanTargetAlignment),
        meanVelocityAlongBodyAxis: round(full.meanVelocityAlongBodyAxis),
        meanAbsoluteTransverseFraction: round(full.meanAbsoluteTransverseFraction),
      },
      zeroFrictionComDisplacement: frictionless.displacement,
      steeringFinalZ: { left: round(left.finalZ), right: round(right.finalZ) },
      steeringActionDelta: round(steeringActionDelta),
      localSensorDelta: round(localSensorDelta),
      commandFeedback,
      interventionParity,
      deterministicTraceHash: full.traceHash,
      v3RuntimePromotion: v3Promotion,
      trainingReproducibility: trainingRepro,
    },
    null,
    2,
  ),
)
console.log('Articulated evolved locomotion policy verification passed.')

function verifyContract(model: LocomotionPolicyArtifact, raw: any, source: string) {
  expect(model.schemaVersion === 2, 'runtime locomotion schema must normalize to version 2')
  expect(model.segmentCount === SEGMENT_COUNT, `artifact must own ${SEGMENT_COUNT} segment neurons`)
  expect(model.actionSize === ACTION_SIZE, `artifact must emit ${ACTION_SIZE} muscle channels`)
  expect(model.architecture === 'segmental-recurrent-tanh', 'artifact architecture changed')
  expect(model.plant.version === 'articulated-contact-v2', 'artifact must target articulated-contact-v2')
  expect(exactArray(model.sensorNames, LOCOMOTION_SENSOR_NAMES), 'sensor contract ordering changed')
  expect(exactArray(model.weights.inputNames, LOCOMOTION_INPUT_WEIGHT_NAMES), 'input weight ordering changed')
  expect(
    exactArray(model.weights.recurrentNames, LOCOMOTION_RECURRENT_WEIGHT_NAMES),
    'recurrent weight ordering changed',
  )
  expect(
    exactArray(model.weights.outputNames, LOCOMOTION_OUTPUT_WEIGHT_NAMES),
    'output weight ordering changed',
  )
  expect(
    !model.sensorNames.some(name => /time|clock|phase|cycle/iu.test(name)),
    'locomotion sensors must not expose authored gait timing',
  )
  for (const [key, expected] of Object.entries(LOCOMOTION_PLANT_CONTRACT)) {
    expect(model.plant[key as keyof typeof model.plant] === expected, `plant.${key} contract mismatch`)
  }

  const training = model.training
  expect(training.algorithm === 'elitist-mutation-evolution', 'artifact must record evolutionary training')
  expect(training.actuatorPrecision === 'float32-plant-command', 'artifact actuator precision mismatch')
  if (training.objectiveVersion === OBJECTIVE_V2) {
    verifyV2TrainingContract(training)
    verifyV2TrainingEvidence(training)
  } else if (training.objectiveVersion === OBJECTIVE_V3) {
    verifyV3TrainingContract(model, training)
  } else {
    throw new Error(`artifact objectiveVersion mismatch (${String(training.objectiveVersion)})`)
  }
  verifyWarmStart(model, training)

  expect(raw.plant.version === 'articulated-contact-v2', 'raw artifact plant version mismatch')
  expect(source.includes('articulated_plant_step('), 'trainer must evaluate the articulated contact plant')
  expect(source.includes('contact_load'), 'trainer must feed back segment-local contact load')
  expect(source.includes('slip_speed'), 'trainer must feed back segment-local slip speed')
  expect(source.includes('obstacle_forward'), 'trainer must feed back local obstacle direction')
  expect(source.includes('obstacle_right'), 'trainer must feed back local obstacle side')
  expect(/astype\(np\.float32\)/u.test(source), 'trainer must quantize browser-bound actions to float32')
  expect(
    !/cpg_action|wave_frequency|phase_offset/iu.test(source),
    'trainer must not contain an authored gait generator',
  )
  expect(
    !/wave_gain|maximum_forward_acceleration|inverse_traction_drag|front_turn_base/iu.test(source),
    'trainer still references the removed scalar locomotion plant',
  )
  const warmStartLoader = source.slice(source.indexOf('def load_warm_start'), source.indexOf('\ndef evolve'))
  expect(
    !/["']path["']\s*:/u.test(warmStartLoader),
    'warm-start metadata must not serialize filesystem paths',
  )
}

function verifyV2TrainingEvidence(training: Record<string, unknown>) {
  expect(/^[a-f\d]{64}$/u.test(String(training.deterministicTraceHash)), 'training trace hash is malformed')
  const ablations = recordValue(training.ablations, 'training.ablations')
  for (const name of [
    'fullProgress',
    'zeroProgress',
    'frozenProgress',
    'shuffledProgress',
    'noFrictionDisplacement',
  ]) {
    expect(
      typeof ablations[name] === 'number' && Number.isFinite(ablations[name]),
      `training ablation ${name} is invalid`,
    )
  }
}

function verifyV3RuntimePromotion(
  model: LocomotionPolicyArtifact,
  rollouts: Rollout[],
  aggregateRollout: Rollout,
  shuffled: Rollout,
  options: { force?: boolean; enforce?: boolean } = {},
) {
  if (model.training.objectiveVersion !== OBJECTIVE_V3 && !options.force) return { required: false }

  const worstProgress = Math.min(...rollouts.map(rollout => rollout.progress))
  const worstTargetAlignment = Math.min(...rollouts.map(rollout => rollout.meanTargetAlignment))
  const worstBodyAxisVelocity = Math.min(...rollouts.map(rollout => rollout.meanVelocityAlongBodyAxis))
  const worstTransverseFraction = Math.max(...rollouts.map(rollout => rollout.meanAbsoluteTransverseFraction))
  const shuffledProgressGap = aggregateRollout.progress - shuffled.progress
  const criteria = {
    meanProgress: aggregateRollout.progress >= 3.5,
    worstScenarioProgress: worstProgress >= 2.8,
    shuffledProgressGap: shuffledProgressGap >= 0.35,
    meanTargetAlignment: aggregateRollout.meanTargetAlignment >= 0.55,
    worstTargetAlignment: worstTargetAlignment >= 0.4,
    meanBodyAxisVelocity: aggregateRollout.meanVelocityAlongBodyAxis >= 0.8,
    worstBodyAxisVelocity: worstBodyAxisVelocity >= 0.7,
    meanTransverseFraction: aggregateRollout.meanAbsoluteTransverseFraction <= 0.4,
    worstTransverseFraction: worstTransverseFraction <= 0.52,
  }
  const measurements = {
    meanProgress: round(aggregateRollout.progress),
    worstScenarioProgress: round(worstProgress),
    shuffledProgressGap: round(shuffledProgressGap),
    meanTargetAlignment: round(aggregateRollout.meanTargetAlignment),
    worstTargetAlignment: round(worstTargetAlignment),
    meanBodyAxisVelocity: round(aggregateRollout.meanVelocityAlongBodyAxis),
    worstBodyAxisVelocity: round(worstBodyAxisVelocity),
    meanTransverseFraction: round(aggregateRollout.meanAbsoluteTransverseFraction),
    worstTransverseFraction: round(worstTransverseFraction),
  }
  if (options.enforce !== false) {
    for (const [name, passed] of Object.entries(criteria)) {
      expect(
        passed,
        `v3 authoritative TypeScript promotion gate failed: ${name} ` +
          `(measured ${measurements[name as keyof typeof measurements]})`,
      )
    }
  }
  return {
    required: true,
    passed: Object.values(criteria).every(Boolean),
    criteria,
    measurements,
  }
}

function verifyV2TrainingContract(training: Record<string, unknown>) {
  expect(training.objectiveVersion === OBJECTIVE_V2, 'artifact objectiveVersion mismatch')
  expect(training.seed === 20260721, 'published refinement seed mismatch')
  expect(training.generations === 10, 'published refinement generation count mismatch')
  expect(training.populationSize === 64, 'published refinement population size mismatch')
  expect(training.eliteCount === 12, 'published refinement elite count mismatch')
  expect(training.episodeSteps === 480, 'published refinement episode length mismatch')
  expect(training.scenarioCount === 12, 'published scenario count mismatch')
  expect(
    exactArray(training.domainRandomization, V2_DOMAINS),
    'published domain randomization contract mismatch',
  )
}

function verifyV3TrainingContract(model: LocomotionPolicyArtifact, training: Record<string, unknown>) {
  training = exactRecord(training, 'training', [
    'ablationGateGuardBandResults',
    'ablationGateResults',
    'ablationProtocol',
    'actuatorPrecision',
    'algorithm',
    'allAblationGatesPassed',
    'allAblationGuardBandsPassed',
    'allGateGuardBandsPassed',
    'allPublicationGatesPassed',
    'allSelectionGatesPassed',
    'allSelectionGuardBandsPassed',
    'allSelectionMarginsFeasible',
    'canonicalGenomeRecipeHash',
    'causalScenarioCount',
    'combinedRecoveryEpisodeSteps',
    'combinedRecoveryMode',
    'combinedRecoveryScenarioCount',
    'controllerExposedTimingSignals',
    'controllerSensorClamps',
    'domainRandomization',
    'eliteCount',
    'episodeSteps',
    'gateDiagnosticDecimals',
    'generations',
    'genomeDecimals',
    'headLeadingMeasurement',
    'localCausalProbe',
    'objectiveVersion',
    'pairedHostReproduction',
    'perturbationSchedules',
    'populationSize',
    'recipeSourceSha256',
    'recoveryMeasurement',
    'robustnessScenarioCount',
    'scenarioCount',
    'seed',
    'selectionFitnessDecimals',
    'selectionGateGuardBandResults',
    'selectionGateResults',
    'selectionMargins',
    'warmStart',
  ])
  expect(training.objectiveVersion === OBJECTIVE_V3, 'v3 artifact objectiveVersion mismatch')
  expect(
    model.modelVersion === 'locomotion-articulated-head-leading-es-v3',
    'published v3 modelVersion mismatch',
  )
  expect(training.seed === 20260737, 'v3 published refinement seed mismatch')
  expect(training.generations === 1, 'v3 refinement generation count mismatch')
  expect(training.populationSize === 4, 'v3 refinement population size mismatch')
  expect(training.eliteCount === 1, 'v3 refinement elite count mismatch')
  expect(training.episodeSteps === 480, 'v3 refinement episode length mismatch')
  expect(training.scenarioCount === 12, 'v3 scenario count mismatch')
  expect(training.robustnessScenarioCount === 4, 'v3 robustness scenario count mismatch')
  expect(training.combinedRecoveryScenarioCount === 1, 'v3 combined recovery scenario count mismatch')
  expect(training.combinedRecoveryMode === 'bounded-480', 'v3 combined recovery mode mismatch')
  expect(training.combinedRecoveryEpisodeSteps === 480, 'v3 combined recovery episode length mismatch')
  expect(training.causalScenarioCount === 4, 'v3 causal scenario count mismatch')
  expect(exactArray(training.domainRandomization, V3_DOMAINS), 'v3 domain randomization contract mismatch')
  expect(training.genomeDecimals === 8, 'v3 serialized genome precision mismatch')
  expect(training.selectionFitnessDecimals === 4, 'v3 selection fitness precision mismatch')
  expect(training.gateDiagnosticDecimals === 8, 'v3 gate diagnostic precision mismatch')
  const recipeHashes = exactRecord(
    training.recipeSourceSha256,
    'training.recipeSourceSha256',
    V3_RECIPE_SOURCE_PATHS,
  )
  for (const path of V3_RECIPE_SOURCE_PATHS) {
    expect(/^[a-f\d]{64}$/u.test(String(recipeHashes[path])), `v3 source hash is malformed for ${path}`)
    expect(recipeHashes[path] === currentRecipeSourceSha256[path], `v3 source hash is stale for ${path}`)
  }
  expect(
    exactArray(training.controllerExposedTimingSignals, []),
    'v3 controller must expose no clock, phase, or perturbation timing signals',
  )
  const sensorClamps = exactRecord(training.controllerSensorClamps, 'training.controllerSensorClamps', [
    'angularSpeed',
    'contactLoad',
    'forwardSpeed',
    'obstacleForward',
    'obstacleRight',
    'slipSpeed',
    'targetDistance',
    'targetForward',
    'targetRight',
    'terrainFriction',
    'urgency',
  ])
  for (const [name, bounds] of Object.entries({
    targetForward: [-1, 1],
    targetRight: [-1, 1],
    targetDistance: [0, 1.5],
    forwardSpeed: [-2, 2],
    angularSpeed: [-3, 3],
    terrainFriction: [0, 1.2],
    urgency: [0, 1],
    contactLoad: [0, 1],
    slipSpeed: [0, 2],
    obstacleForward: [-1, 1],
    obstacleRight: [-1, 1],
  })) {
    expect(exactArray(sensorClamps[name], bounds), `v3 ${name} sensor clamp mismatch`)
  }
  const selectionMargins = exactRecord(training.selectionMargins, 'training.selectionMargins', [
    'causal',
    'combinedRecovery',
    'guardBands',
    'nominal',
  ])
  const selectionGuards = exactRecord(selectionMargins.guardBands, 'selectionMargins.guardBands', [
    'localRadiusThresholdDistance',
    'localSettledThresholdDistance',
    'scalarThresholdDistance',
  ])
  expect(selectionGuards.scalarThresholdDistance === 1e-3, 'v3 scalar guard band mismatch')
  expect(selectionGuards.localRadiusThresholdDistance === 1e-4, 'v3 local-radius guard mismatch')
  expect(selectionGuards.localSettledThresholdDistance === 1e-6, 'v3 settled guard mismatch')
  const nominalMargins = exactRecord(selectionMargins.nominal, 'selectionMargins.nominal', [
    'lateralMotionFractionAtMost',
    'movingTargetFacingAlignmentAtLeast',
    'signedHeadLeadingFractionAtLeast',
  ])
  expect(nominalMargins.signedHeadLeadingFractionAtLeast === 0.72, 'v3 nominal head margin mismatch')
  expect(nominalMargins.lateralMotionFractionAtMost === 0.4, 'v3 nominal lateral margin mismatch')
  expect(nominalMargins.movingTargetFacingAlignmentAtLeast === 0.62, 'v3 nominal target margin mismatch')
  const combinedMargins = exactRecord(
    selectionMargins.combinedRecovery,
    'selectionMargins.combinedRecovery',
    [
      'controlLateProgressSpeedAtLeast',
      'controlTargetProgressAtLeast',
      'lateSpeedRetentionAtLeast',
      'lateralMotionFractionAtMost',
      'movingTargetFacingAlignmentAtLeast',
      'perturbedProgressFloorRatio',
      'progressRetentionAtLeast',
      'signedHeadLeadingFractionAtLeast',
    ],
  )
  expect(combinedMargins.controlTargetProgressAtLeast === 4.36, 'v3 bounded progress margin mismatch')
  expect(combinedMargins.perturbedProgressFloorRatio === 0.85, 'v3 perturbed progress ratio mismatch')
  expect(combinedMargins.progressRetentionAtLeast === 0.88, 'v3 progress retention margin mismatch')
  expect(combinedMargins.lateSpeedRetentionAtLeast === 0.83, 'v3 late-speed retention margin mismatch')
  expect(combinedMargins.controlLateProgressSpeedAtLeast === 0.12, 'v3 control speed margin mismatch')
  expect(combinedMargins.signedHeadLeadingFractionAtLeast === 0.62, 'v3 recovery head margin mismatch')
  expect(combinedMargins.lateralMotionFractionAtMost === 0.48, 'v3 recovery lateral margin mismatch')
  expect(combinedMargins.movingTargetFacingAlignmentAtLeast === 0.52, 'v3 recovery target margin mismatch')
  const causalSelectionMargins = exactRecord(selectionMargins.causal, 'selectionMargins.causal', [
    'localInitialDeltaAtLeast',
    'localRadiusThreeDeltaAtLeast',
    'localRadiusTwoDeltaAtLeast',
    'localSettledDeltaAtMost',
    'orderingGapAtLeast',
    'progressGapAtLeast',
  ])
  expect(causalSelectionMargins.orderingGapAtLeast === 0.5, 'v3 causal ordering margin mismatch')
  expect(causalSelectionMargins.progressGapAtLeast === 0.05, 'v3 causal progress margin mismatch')
  expect(causalSelectionMargins.localInitialDeltaAtLeast === 0.12, 'v3 causal initial margin mismatch')
  expect(causalSelectionMargins.localRadiusTwoDeltaAtLeast === 2e-3, 'v3 radius-two margin mismatch')
  expect(causalSelectionMargins.localRadiusThreeDeltaAtLeast === 2e-3, 'v3 radius-three margin mismatch')
  expect(causalSelectionMargins.localSettledDeltaAtMost === 5e-6, 'v3 settled margin mismatch')

  const headLeading = exactRecord(training.headLeadingMeasurement, 'training.headLeadingMeasurement', [
    'axis',
    'lateralFraction',
    'selectionMargins',
    'signedFraction',
    'turnHandling',
  ])
  expect(
    headLeading.axis === 'tail-to-head segment COM axis; positive toward rendered head',
    'v3 head-leading axis contract mismatch',
  )
  expect(
    headLeading.signedFraction === 'sum(COM displacement dot body axis) / total COM path length',
    'v3 signed head-leading measurement mismatch',
  )
  expect(
    headLeading.lateralFraction === 'sum(abs(COM displacement dot body right)) / total COM path length',
    'v3 lateral-motion measurement mismatch',
  )
  expect(
    headLeading.turnHandling ===
      'all world headings are valid; target-facing alignment has a short grace window after deterministic target switches',
    'v3 turn-handling measurement mismatch',
  )
  const margins = exactRecord(headLeading.selectionMargins, 'headLeadingMeasurement.selectionMargins', [
    'combinedRecoveredMaximumLateralMotionFraction',
    'combinedRecoveredSignedHeadLeadingFraction',
    'combinedRecoveredTargetFacingAlignment',
    'maximumLateralMotionFraction',
    'movingTargetFacingAlignment',
    'signedHeadLeadingFraction',
  ])
  expect(margins.signedHeadLeadingFraction === 0.72, 'v3 signed head-leading selection margin mismatch')
  expect(margins.maximumLateralMotionFraction === 0.4, 'v3 lateral-motion selection margin mismatch')
  expect(margins.movingTargetFacingAlignment === 0.62, 'v3 target-facing selection margin mismatch')
  expect(
    margins.combinedRecoveredSignedHeadLeadingFraction === 0.62,
    'v3 combined-recovery head-leading margin mismatch',
  )
  expect(
    margins.combinedRecoveredMaximumLateralMotionFraction === 0.48,
    'v3 combined-recovery lateral margin mismatch',
  )
  expect(
    margins.combinedRecoveredTargetFacingAlignment === 0.52,
    'v3 combined-recovery target-facing margin mismatch',
  )

  const schedules = exactRecord(training.perturbationSchedules, 'training.perturbationSchedules', [
    'bodyShove',
    'combinedRecoveryProxy',
    'slickWindow',
  ])
  const slick = exactRecord(schedules.slickWindow, 'perturbationSchedules.slickWindow', [
    'activeSteps',
    'friction',
    'postSteps',
    'preSteps',
  ])
  expect(exactArray(slick.preSteps, [57, 115]), 'v3 slick pre-window mismatch')
  expect(exactArray(slick.activeSteps, [115, 182]), 'v3 slick active window mismatch')
  expect(exactArray(slick.postSteps, [182, 240]), 'v3 slick post-window mismatch')
  expect(slick.friction === 0.035, 'v3 slick friction mismatch')
  const shove = exactRecord(schedules.bodyShove, 'perturbationSchedules.bodyShove', [
    'alternatingLateralSpeed',
    'atStep',
    'backwardSpeed',
    'postSteps',
    'preSteps',
  ])
  expect(exactArray(shove.preSteps, [240, 297]), 'v3 shove pre-window mismatch')
  expect(shove.atStep === 297, 'v3 shove event step mismatch')
  expect(exactArray(shove.postSteps, [298, 393]), 'v3 shove post-window mismatch')
  expect(shove.alternatingLateralSpeed === 1.35, 'v3 shove lateral speed mismatch')
  expect(shove.backwardSpeed === 0.28, 'v3 shove backward speed mismatch')
  const combined = exactRecord(
    schedules.combinedRecoveryProxy,
    'perturbationSchedules.combinedRecoveryProxy',
    [
      'activeSteps',
      'controlTarget',
      'lateSpeedSteps',
      'lateralShoveSpeed',
      'numbSegment',
      'plantContractModified',
      'pythonArenaLimitedProxy',
      'tractionScale',
    ],
  )
  expect(exactArray(combined.controlTarget, [20, 0]), 'v3 combined recovery target mismatch')
  expect(exactArray(combined.activeSteps, [128, 176]), 'v3 combined active window mismatch')
  expect(exactArray(combined.lateSpeedSteps, [240, 336]), 'v3 combined late-speed window mismatch')
  expect(combined.lateralShoveSpeed === 1.1, 'v3 combined lateral shove mismatch')
  expect(combined.numbSegment === 7, 'v3 combined numb segment mismatch')
  expect(combined.tractionScale === 1 / 15, 'v3 combined traction scale mismatch')
  expect(combined.pythonArenaLimitedProxy === true, 'v3 arena-limited proxy provenance missing')
  expect(combined.plantContractModified === false, 'v3 combined proxy must preserve the plant contract')

  const recovery = exactRecord(training.recoveryMeasurement, 'training.recoveryMeasurement', [
    'ratio',
    'speed',
    'time',
  ])
  expect(
    recovery.speed === 'positive head-axis speed multiplied by positive target-facing alignment',
    'v3 recovery speed measurement mismatch',
  )
  expect(
    recovery.ratio === 'post-window mean speed / max(pre-window mean speed, 0.04 m/s)',
    'v3 recovery ratio measurement mismatch',
  )
  expect(
    recovery.time ===
      'first post-event step whose 0.18-EMA speed reaches max(55% of pre-window speed, 0.04 m/s) while target alignment is at least 0.1',
    'v3 recovery time measurement mismatch',
  )

  const causal = exactRecord(training.localCausalProbe, 'training.localCausalProbe', [
    'actionPrecision',
    'browserGates',
    'pulseSegment',
    'pulseTicks',
    'selectionMargins',
  ])
  expect(causal.pulseSegment === 7, 'v3 local causal pulse segment mismatch')
  expect(causal.pulseTicks === 1, 'v3 local causal pulse duration mismatch')
  expect(
    causal.actionPrecision === 'float32-returned-action-float64-recurrent-command',
    'v3 local causal action precision mismatch',
  )
  const browserGates = exactRecord(causal.browserGates, 'localCausalProbe.browserGates', [
    'initialLocalDeltaGreaterThan',
    'radiusThreeDeltaGreaterThan',
    'radiusTwoDeltaGreaterThan',
    'settledDeltaLessThan',
  ])
  expect(browserGates.initialLocalDeltaGreaterThan === 0.1, 'v3 initial causal browser gate mismatch')
  expect(browserGates.radiusTwoDeltaGreaterThan === 1e-3, 'v3 radius-two browser gate mismatch')
  expect(browserGates.radiusThreeDeltaGreaterThan === 1e-3, 'v3 radius-three browser gate mismatch')
  expect(browserGates.settledDeltaLessThan === 1e-5, 'v3 settled causal browser gate mismatch')
  const causalMargins = exactRecord(causal.selectionMargins, 'localCausalProbe.selectionMargins', [
    'initialLocalDelta',
    'radiusThreeDelta',
    'radiusTwoDelta',
    'settledDelta',
  ])
  expect(causalMargins.initialLocalDelta === 0.12, 'v3 initial causal selection margin mismatch')
  expect(causalMargins.radiusTwoDelta === 2e-3, 'v3 radius-two selection margin mismatch')
  expect(causalMargins.radiusThreeDelta === 2e-3, 'v3 radius-three selection margin mismatch')
  expect(causalMargins.settledDelta === 5e-6, 'v3 settled causal selection margin mismatch')

  const ablationProtocol = exactRecord(training.ablationProtocol, 'training.ablationProtocol', [
    'guardBands',
    'hostMagnitudesStoredInSummaryOnly',
    'runtimeSensorContract',
    'thresholds',
  ])
  expect(ablationProtocol.runtimeSensorContract === true, 'v3 ablations changed the runtime sensors')
  expect(
    ablationProtocol.hostMagnitudesStoredInSummaryOnly === true,
    'v3 artifact claims to contain host-sensitive ablation magnitudes',
  )
  const ablationGuards = exactRecord(ablationProtocol.guardBands, 'ablationProtocol.guardBands', [
    'conservationAtMost',
    'progressThresholdDistance',
  ])
  expect(ablationGuards.progressThresholdDistance === 0.01, 'v3 ablation progress guard mismatch')
  expect(ablationGuards.conservationAtMost === 1e-10, 'v3 ablation conservation guard mismatch')
  const ablationThresholds = exactRecord(ablationProtocol.thresholds, 'ablationProtocol.thresholds', [
    'fullMinusFrozenProgressAtLeast',
    'fullMinusShuffledProgressAtLeast',
    'fullProgressAtLeast',
    'noFrictionDisplacementAtMost',
    'zeroProgressAtMost',
  ])
  expect(ablationThresholds.fullProgressAtLeast === 0.45, 'v3 full-progress threshold mismatch')
  expect(ablationThresholds.zeroProgressAtMost === 1e-9, 'v3 zero-action threshold mismatch')
  expect(ablationThresholds.fullMinusFrozenProgressAtLeast === 0.3, 'v3 frozen gap mismatch')
  expect(ablationThresholds.fullMinusShuffledProgressAtLeast === 0.2, 'v3 shuffled gap mismatch')
  expect(ablationThresholds.noFrictionDisplacementAtMost === 1e-9, 'v3 no-friction threshold mismatch')

  requireAllTrueGateRecord(
    training.selectionGateResults,
    'training.selectionGateResults',
    V3_SELECTION_GATE_KEYS,
  )
  requireAllTrueGateRecord(
    training.selectionGateGuardBandResults,
    'training.selectionGateGuardBandResults',
    V3_SELECTION_GATE_KEYS,
  )
  requireAllTrueGateRecord(
    training.ablationGateResults,
    'training.ablationGateResults',
    V3_ABLATION_GATE_KEYS,
  )
  requireAllTrueGateRecord(
    training.ablationGateGuardBandResults,
    'training.ablationGateGuardBandResults',
    V3_ABLATION_GATE_KEYS,
  )
  for (const key of [
    'allSelectionGatesPassed',
    'allSelectionMarginsFeasible',
    'allSelectionGuardBandsPassed',
    'allAblationGatesPassed',
    'allAblationGuardBandsPassed',
    'allGateGuardBandsPassed',
    'allPublicationGatesPassed',
  ]) {
    expect(training[key] === true, `v3 publication aggregate ${key} must be true`)
  }

  expect(
    /^[a-f\d]{64}$/u.test(String(training.canonicalGenomeRecipeHash)),
    'v3 canonical genome/recipe hash is malformed',
  )
  const pairedHost = exactRecord(training.pairedHostReproduction, 'training.pairedHostReproduction', [
    'compared',
    'requiredForPublication',
    'trainerAttestation',
  ])
  expect(pairedHost.requiredForPublication === true, 'v3 paired-host publication proof is not required')
  expect(
    exactArray(pairedHost.compared, ['serializedGenome', 'artifactBytes']),
    'v3 paired-host comparison contract changed',
  )
  expect(pairedHost.trainerAttestation === 'none', 'trainer must not self-attest paired-host reproduction')

  const warmStart = exactRecord(training.warmStart, 'training.warmStart', ['modelVersion', 'sha256'])
  expect(
    warmStart.sha256 === '0c46518c424528337f3a2ad5a96912b7812dda473688d468556023be095551ba',
    'v3 retained warm-start digest mismatch',
  )
  expect(
    warmStart.modelVersion === 'locomotion-head-leading-es-v3-terrarium-causal-candidate-v1',
    'v3 retained warm-start model identity mismatch',
  )
}

function verifyWarmStart(model: LocomotionPolicyArtifact, training: Record<string, unknown>) {
  const warmStart = exactRecord(training.warmStart, 'training.warmStart', ['modelVersion', 'sha256'])
  expect(
    /^[a-f\d]{64}$/u.test(String(warmStart.sha256)),
    'warmStart.sha256 must be a lowercase SHA-256 digest',
  )
  expect(
    typeof warmStart.modelVersion === 'string' &&
      warmStart.modelVersion.length > 0 &&
      !/[\\/]/u.test(warmStart.modelVersion),
    'warmStart.modelVersion must be a location-independent model identifier',
  )
  expect(
    warmStart.modelVersion !== model.modelVersion,
    'warmStart.modelVersion must identify a distinct ancestor artifact',
  )
}

function verifyLegacyMigration(raw: any) {
  const migrated = parseLocomotionPolicy(structuredClone(raw))
  expect(migrated.schemaVersion === 2, 'schema-v1 artifacts must normalize to schema v2')
  expect(
    migrated.plant.version === 'articulated-contact-v2',
    'legacy artifacts must adopt the articulated plant',
  )
  expect(
    migrated.training.runtimeMigration === 'schema-1-zero-contact-weights',
    'legacy migration provenance is missing',
  )
  expect(
    migrated.weights.input.slice(13).every(value => value === 0),
    'schema-v1 migration must zero new local-sensor input weights',
  )
  expect(
    migrated.weights.output.slice(7).every(value => value === 0),
    'schema-v1 migration must zero new contact output weights',
  )
}

function verifyMalformedArtifacts(v2: any, v1: any) {
  expectReject(v2, candidate => (candidate.schemaVersion = 3), 'unknown schema version')
  expectReject(v2, candidate => (candidate.kind = 'wurmkickflip.other'), 'wrong artifact kind')
  expectReject(v2, candidate => candidate.sensorNames.reverse(), 'reversed v2 sensor order')
  expectReject(v2, candidate => candidate.weights.input.pop(), 'short v2 input vector')
  expectReject(v2, candidate => candidate.weights.output.push(0), 'long v2 output vector')
  expectReject(v2, candidate => (candidate.weights.recurrent[0] = Number.NaN), 'non-finite recurrent weight')
  expectReject(v2, candidate => (candidate.plant.version = 'scalar-wave-v1'), 'wrong v2 plant version')
  expectReject(v2, candidate => (candidate.plant.spacing *= 1.01), 'wrong v2 plant constant')
  expectReject(v2, candidate => (candidate.plant.maximumVerticalSpeed = 0), 'invalid v2 vertical speed')
  expectReject(v2, candidate => delete candidate.training, 'missing v2 training metadata')
  if (v2.training?.objectiveVersion === OBJECTIVE_V3) {
    expectReject(
      v2,
      candidate => (candidate.training.allPublicationGatesPassed = false),
      'failed v3 aggregate publication gate',
    )
    expectReject(
      v2,
      candidate => (candidate.training.selectionGateResults.nominalSignedHeadLeading = false),
      'failed v3 categorical selection gate',
    )
    expectReject(
      v2,
      candidate => delete candidate.training.selectionGateResults.causalProgressGap,
      'incomplete v3 selection gate map',
    )
    expectReject(
      v2,
      candidate => (candidate.training.selectionGateGuardBandResults.inventedGate = true),
      'extra v3 selection guard-band gate',
    )
    expectReject(
      v2,
      candidate => delete candidate.training.ablationGateResults.fullBeatsShuffled,
      'incomplete v3 ablation gate map',
    )
    expectReject(
      v2,
      candidate => (candidate.training.ablationGateGuardBandResults.inventedGate = true),
      'extra v3 ablation guard-band gate',
    )
    expectReject(
      v2,
      candidate => (candidate.training.canonicalGenomeRecipeHash = '0'.repeat(63)),
      'malformed v3 canonical recipe hash',
    )
    expectReject(
      v2,
      candidate => (candidate.training.pairedHostReproduction.trainerAttestation = 'verified'),
      'self-attested paired-host reproduction',
    )
    expectReject(
      v2,
      candidate => (candidate.training.objectiveVersion = OBJECTIVE_V2),
      'published v3 identity with a legacy objective',
    )
  }
  expectReject(v1, candidate => (candidate.schemaVersion = 0), 'unknown legacy schema')
  expectReject(v1, candidate => candidate.weights.input.pop(), 'short legacy input vector')
  expectReject(v1, candidate => (candidate.plant.jointStiffness += 0.1), 'wrong legacy joint contract')
}

function verifyAntagonisticOutputs(model: LocomotionPolicyArtifact) {
  const policy = new EvolvedLocomotionPolicy(model)
  const action = policy.run(baseSensors(), new Float64Array(SEGMENT_COUNT), new Float64Array(SEGMENT_COUNT))
  expect(action.length === ACTION_SIZE, 'runtime action length changed')
  expect(Array.from(action).every(Number.isFinite), 'runtime action must remain finite')
  for (let segment = 0; segment < SEGMENT_COUNT; segment += 1) {
    expect(
      Math.abs(action[segment * 2] + action[segment * 2 + 1]) <= 1e-7,
      `segment ${segment} must own an antagonistic actuator pair`,
    )
  }
}

function verifyCommandFeedbackOverride(model: LocomotionPolicyArtifact) {
  const baseline = new EvolvedLocomotionPolicy(model)
  const overridden = new EvolvedLocomotionPolicy(model)
  const sensors = baseSensors()
  const bends = Array.from({ length: SEGMENT_COUNT }, (_, segment) => (segment - 7.5) / 12)
  const velocities = Array.from({ length: SEGMENT_COUNT }, (_, segment) => (7.5 - segment) / 10)

  const baselineFirst = baseline.run(sensors, bends, velocities)
  const overriddenFirst = overridden.run(sensors, bends, velocities)
  expect(
    Array.from(baselineFirst).every((value, index) => value === overriddenFirst[index]),
    'command-feedback fixtures diverged before the override',
  )

  const beforeOverride = overridden.getTelemetry()
  const rawCommands = beforeOverride.commands
  expect(
    Math.max(...rawCommands.map(Math.abs)) > 0.1,
    'command-feedback fixture did not produce a material recurrent command',
  )
  const zeroFeedback = new Float64Array(SEGMENT_COUNT)
  overridden.commitCommandFeedback(zeroFeedback)
  zeroFeedback.fill(0.75)
  const committed = overridden.getTelemetry()
  expect(
    committed.commands.every(command => command === 0),
    'command feedback retained a mutable reference to the caller vector',
  )
  expect(
    committed.requestedCommands.every(
      (command, segment) => command === beforeOverride.requestedCommands[segment],
    ),
    'overriding applied command feedback rewrote requested-command telemetry',
  )

  baseline.run(sensors, bends, velocities)
  overridden.run(sensors, bends, velocities)
  const baselineSecond = baseline.getTelemetry()
  const overriddenSecond = overridden.getTelemetry()
  const previousCommandWeight = model.weights.input[12]
  let maximumDriveError = 0
  let maximumDriveDelta = 0
  for (let segment = 0; segment < SEGMENT_COUNT; segment += 1) {
    const expectedDelta = -previousCommandWeight * rawCommands[segment]
    const actualDelta = overriddenSecond.drives[segment] - baselineSecond.drives[segment]
    maximumDriveError = Math.max(maximumDriveError, Math.abs(actualDelta - expectedDelta))
    maximumDriveDelta = Math.max(maximumDriveDelta, Math.abs(actualDelta))
  }
  expect(maximumDriveDelta > 1e-3, 'command feedback override did not affect the next recurrent step')
  expect(
    maximumDriveError <= 1e-12,
    `command feedback entered more than the previous-command channel (${maximumDriveError})`,
  )

  let rejectedLength = false
  try {
    overridden.commitCommandFeedback(new Float64Array(SEGMENT_COUNT - 1))
  } catch {
    rejectedLength = true
  }
  expect(rejectedLength, 'command feedback accepted the wrong segment count')

  const nonFinite = new Float64Array(SEGMENT_COUNT)
  nonFinite[4] = Number.NaN
  let rejectedNonFinite = false
  try {
    overridden.commitCommandFeedback(nonFinite)
  } catch {
    rejectedNonFinite = true
  }
  expect(rejectedNonFinite, 'command feedback accepted a non-finite segment value')

  return {
    controllerPrecision: 'float64',
    detachedInput: true,
    maximumDriveDelta: round(maximumDriveDelta),
    maximumDriveError,
  }
}

function verifyLocalSensorEffects(model: LocomotionPolicyArtifact) {
  const baselinePolicy = new EvolvedLocomotionPolicy(model)
  const perturbedPolicy = new EvolvedLocomotionPolicy(model)
  const baselineSensors = baseSensors()
  const perturbedSensors = baseSensors()
  const contactLoads = new Float64Array(SEGMENT_COUNT).fill(1)
  const slipSpeeds = new Float64Array(SEGMENT_COUNT)
  const obstacleForward = new Float64Array(SEGMENT_COUNT)
  const obstacleRight = new Float64Array(SEGMENT_COUNT)
  perturbedSensors.contactLoads = contactLoads
  perturbedSensors.slipSpeeds = slipSpeeds
  perturbedSensors.obstacleForward = obstacleForward
  perturbedSensors.obstacleRight = obstacleRight
  const changedSegment = 7
  contactLoads[changedSegment] = 0
  slipSpeeds[changedSegment] = 1.7
  obstacleForward[changedSegment] = 0.8
  obstacleRight[changedSegment] = 0.8
  const bends = new Float64Array(SEGMENT_COUNT)
  const velocities = new Float64Array(SEGMENT_COUNT)
  const baseline = baselinePolicy.run(baselineSensors, bends, velocities)
  const perturbed = perturbedPolicy.run(perturbedSensors, bends, velocities)
  const deltas = Array.from({ length: SEGMENT_COUNT }, (_, segment) =>
    Math.abs(perturbed[segment * 2] - baseline[segment * 2]),
  )
  const localDelta = Math.max(...deltas.slice(changedSegment - 1, changedSegment + 2))
  const remoteDelta = Math.max(...deltas.filter((_, segment) => Math.abs(segment - changedSegment) > 1))
  expect(localDelta > 1e-3, 'segment-local contact and obstacle sensors must affect nearby commands')
  expect(remoteDelta <= 1e-7, 'one local sensor sample must not instantly teleport to remote segments')
  return localDelta
}

function verifySteeringSensorEffect(model: LocomotionPolicyArtifact) {
  const left = new EvolvedLocomotionPolicy(model).run(
    { ...baseSensors(), targetRight: 0.8 },
    new Float64Array(SEGMENT_COUNT),
    new Float64Array(SEGMENT_COUNT),
  )
  const right = new EvolvedLocomotionPolicy(model).run(
    { ...baseSensors(), targetRight: -0.8 },
    new Float64Array(SEGMENT_COUNT),
    new Float64Array(SEGMENT_COUNT),
  )
  const delta = mean(
    Array.from({ length: SEGMENT_COUNT }, (_, segment) => Math.abs(left[segment * 2] - right[segment * 2])),
  )
  expect(delta > 0.15, 'targetRight must materially change segment activations')
  return delta
}

type InterventionState = {
  frozenAction: Float32Array
  frozenFeedback: Float64Array
  frozenCaptured: boolean
}

function createInterventionState(): InterventionState {
  return {
    frozenAction: new Float32Array(ACTION_SIZE),
    frozenFeedback: new Float64Array(SEGMENT_COUNT),
    frozenCaptured: false,
  }
}

function applyControllerIntervention(
  mode: Mode,
  step: number,
  neuralAction: PolicyAction,
  neuralFeedback: ArrayLike<number>,
  state: InterventionState,
) {
  const action = new Float32Array(ACTION_SIZE)
  const feedback = new Float64Array(SEGMENT_COUNT)
  if (mode === 'zero') return { action: action as PolicyAction, feedback }

  if (mode === 'frozen') {
    if (step === FROZEN_CAPTURE_STEP) {
      state.frozenAction.set(neuralAction)
      state.frozenFeedback.set(neuralFeedback)
      state.frozenCaptured = true
    }
    if (step >= FROZEN_CAPTURE_STEP) {
      expect(state.frozenCaptured, 'frozen intervention reached its hold window without a captured command')
      action.set(state.frozenAction)
      feedback.set(state.frozenFeedback)
      return { action: action as PolicyAction, feedback }
    }
  }

  if (mode === 'shuffled') {
    for (let segment = 0; segment < SEGMENT_COUNT; segment += 1) {
      const source = SEGMENT_SHUFFLE[segment]
      action[segment * 2] = neuralAction[source * 2]
      action[segment * 2 + 1] = neuralAction[source * 2 + 1]
      feedback[segment] = neuralFeedback[source]
    }
    return { action: action as PolicyAction, feedback }
  }

  action.set(neuralAction)
  feedback.set(neuralFeedback)
  return { action: action as PolicyAction, feedback }
}

function verifyInterventionParity(model: LocomotionPolicyArtifact) {
  const modelPath = typeof artifactSource === 'string' ? artifactSource : fileURLToPath(artifactSource)
  const pythonTraces = new Map<Mode, PythonInterventionTrace>()
  for (const mode of INTERVENTION_MODES) {
    pythonTraces.set(mode, runPythonInterventionTrace(modelPath, mode))
  }
  const pythonReference = pythonTraces.get('full')
  expect(pythonReference, 'Python full intervention trace is missing')
  const neuralTransitions = verifyNeuralTransitionParity(model, pythonReference.neuralTransitions)
  const isolatedPlantTransition = verifyIsolatedPlantTransition(
    model,
    pythonReference.isolatedPlantTransition,
  )

  const machineTolerances = {
    commands: 1e-8,
    plantCommands: 6e-8,
    joints: 5e-10,
    jointVelocities: 3e-8,
    root: 1e-10,
    heading: 1e-8,
    forwardSpeed: 1e-9,
    angularSpeed: 1e-6,
  }
  const captureWindowTolerances = {
    commands: 0.012,
    plantCommands: 0.012,
    joints: 3e-4,
    jointVelocities: 8e-3,
    root: 2e-4,
    heading: 0.0032,
    forwardSpeed: 1e-3,
    angularSpeed: 0.02,
  }
  const modes = INTERVENTION_MODES.map(mode => {
    const python = pythonTraces.get(mode)
    expect(python, `Python ${mode} intervention trace is missing`)
    const typescript = runTypeScriptInterventionTrace(model, mode)
    expect(python.mode === mode, `Python intervention trace returned the wrong mode for ${mode}`)
    expect(python.steps === typescript.steps, `${mode} intervention trace step count differs across hosts`)
    const expectedSteps = Array.from(
      new Set([...INTERVENTION_CHECKPOINT_STEPS.filter(step => step < python.steps), python.steps - 1]),
    ).sort((left, right) => left - right)
    expect(
      exactArray(
        python.checkpoints.map(checkpoint => checkpoint.step),
        expectedSteps,
      ),
      `Python ${mode} intervention checkpoints changed`,
    )
    expect(
      exactArray(
        typescript.checkpoints.map(checkpoint => checkpoint.step),
        expectedSteps,
      ),
      `TypeScript ${mode} intervention checkpoints changed`,
    )

    const checkpointDiagnostics = python.checkpoints.map((expected, index) => ({
      step: expected.step,
      errors: interventionCheckpointErrors(expected, typescript.checkpoints[index]),
    }))
    for (const checkpoint of checkpointDiagnostics) {
      if (checkpoint.step <= 2 || mode === 'zero') {
        requireParityWithin(
          checkpoint.errors,
          machineTolerances,
          `${mode} closed-loop machine window at step ${checkpoint.step}`,
        )
      } else if (checkpoint.step === 23 || checkpoint.step === FROZEN_CAPTURE_STEP) {
        requireParityWithin(
          checkpoint.errors,
          captureWindowTolerances,
          `${mode} closed-loop capture window at step ${checkpoint.step}`,
        )
      }
    }
    const requiredDiagnostics = checkpointDiagnostics.filter(
      checkpoint =>
        checkpoint.step <= 2 ||
        checkpoint.step === 23 ||
        checkpoint.step === FROZEN_CAPTURE_STEP ||
        mode === 'zero',
    )
    return {
      mode,
      checkpoints: expectedSteps,
      requiredCheckpoints: requiredDiagnostics.map(checkpoint => checkpoint.step),
      maximumRequiredErrors: maximumParityErrors(requiredDiagnostics.map(checkpoint => checkpoint.errors)),
      longHorizonDiagnostics: {
        trajectoryIdentityRequired: false,
        maximumErrors: maximumParityErrors(checkpointDiagnostics.map(checkpoint => checkpoint.errors)),
        final: checkpointDiagnostics.at(-1),
      },
    }
  })
  return {
    precisionBoundary: 'float64-intervention-feedback/float32-plant-command',
    frozenCaptureStep: FROZEN_CAPTURE_STEP,
    segmentShuffle: [...SEGMENT_SHUFFLE],
    neuralTransitions,
    isolatedPlantTransition,
    closedLoopContract: {
      machineSteps: [0, 1, 2],
      captureSteps: [23, 24],
      machineTolerances,
      captureWindowTolerances,
      longHorizonTrajectoryIdentityRequired: false,
      longHorizonAuthority: 'independent behavior, ablation, recovery, and causal gates',
    },
    modes,
  }
}

function verifyNeuralTransitionParity(model: LocomotionPolicyArtifact, fixtures: NeuralTransitionFixture[]) {
  expect(fixtures.length >= 2, 'Python neural parity fixtures are incomplete')
  const tolerances = {
    hidden: 2e-12,
    commands: 2e-12,
    requestedCommands: 2e-12,
    plantCommands: 0,
  }
  const results = fixtures.map(fixture => {
    const policy = new EvolvedLocomotionPolicy(model)
    policy.commitCommandFeedback(fixture.input.previousCommands)
    const action = policy.run(
      {
        targetForward: fixture.input.targetForward,
        targetRight: fixture.input.targetRight,
        targetDistance: fixture.input.targetDistance,
        forwardSpeed: fixture.input.forwardSpeed,
        angularSpeed: fixture.input.angularSpeed,
        terrainFriction: fixture.input.terrainFriction,
        urgency: fixture.input.urgency,
        contactLoads: fixture.input.contactLoads,
        slipSpeeds: fixture.input.slipSpeeds,
        obstacleForward: fixture.input.obstacleForward,
        obstacleRight: fixture.input.obstacleRight,
      },
      fixture.input.joints,
      fixture.input.jointVelocities,
    )
    const telemetry = policy.getTelemetry()
    const plantCommands = Array.from({ length: SEGMENT_COUNT }, (_, segment) => action[segment * 2])
    const errors = {
      hidden: vectorError(fixture.output.hidden, telemetry.hidden),
      commands: vectorError(fixture.output.commands, telemetry.commands),
      requestedCommands: vectorError(fixture.output.commands, telemetry.requestedCommands),
      plantCommands: vectorError(fixture.output.plantCommands, plantCommands),
    }
    requireNumericRecordWithin(errors, tolerances, `identical-state neural fixture ${fixture.name}`)
    return { name: fixture.name, errors }
  })
  return { tolerances, fixtures: results }
}

function verifyIsolatedPlantTransition(model: LocomotionPolicyArtifact, fixture: IsolatedPlantTransition) {
  const plant = createWormLocomotionPlant()
  const segments = createSegments(model)
  expect(fixture.input.joints.length === SEGMENT_COUNT, 'isolated plant fixture joint count changed')
  expect(
    fixture.input.jointVelocities.length === SEGMENT_COUNT,
    'isolated plant fixture velocity count changed',
  )
  expect(fixture.input.plantCommands.length === SEGMENT_COUNT, 'isolated plant fixture command count changed')
  expect(
    fixture.input.bodyPositions.length === SEGMENT_COUNT &&
      fixture.input.bodyVelocities.length === SEGMENT_COUNT,
    'isolated plant fixture body count changed',
  )
  const inputMeanVx = mean(fixture.input.bodyVelocities.map(velocity => velocity[0]))
  const inputMeanVz = mean(fixture.input.bodyVelocities.map(velocity => velocity[1]))
  expect(
    Math.hypot(inputMeanVx, inputMeanVz) > model.plant.maximumSpeed,
    'isolated plant fixture must exercise the COM velocity cap',
  )
  plant.joints.splice(0, SEGMENT_COUNT, ...fixture.input.joints)
  plant.jointVelocities.splice(0, SEGMENT_COUNT, ...fixture.input.jointVelocities)
  for (let index = 0; index < SEGMENT_COUNT; index += 1) {
    const position = fixture.input.bodyPositions[index]
    const velocity = fixture.input.bodyVelocities[index]
    expect(position.length === 2 && velocity.length === 2, 'isolated plant fixture vectors must be planar')
    segments[index].x = position[0]
    segments[index].z = position[1]
    segments[index].vx = velocity[0]
    segments[index].vz = velocity[1]
  }
  const action = new Float32Array(ACTION_SIZE)
  for (let segment = 0; segment < SEGMENT_COUNT; segment += 1) {
    const command = fixture.input.plantCommands[segment]
    action[segment * 2] = command
    action[segment * 2 + 1] = -command
  }
  const result = stepArticulatedWorm(
    plant,
    segments,
    action as PolicyAction,
    model.plant.timestep,
    flatField(fixture.input.friction),
    null,
    deriveWurmAnatomy(null),
  )
  const errors = {
    joints: vectorError(fixture.output.joints, plant.joints),
    jointVelocities: vectorError(fixture.output.jointVelocities, plant.jointVelocities),
    bodyPositions: matrixError(
      fixture.output.bodyPositions,
      segments.map(segment => [segment.x, segment.z]),
    ),
    bodyVelocities: matrixError(
      fixture.output.bodyVelocities,
      segments.map(segment => [segment.vx, segment.vz]),
    ),
    root: vectorError(fixture.output.root, [result.root.x, result.root.z]),
    heading: angleError(fixture.output.heading, result.root.heading),
    forwardSpeed: Math.abs(fixture.output.forwardSpeed - plant.forwardSpeed),
    angularSpeed: Math.abs(fixture.output.angularSpeed - plant.angularSpeed),
  }
  const tolerances = {
    joints: 1e-12,
    jointVelocities: 1e-12,
    bodyPositions: 2e-10,
    bodyVelocities: 2e-8,
    root: 2e-10,
    heading: 2e-10,
    forwardSpeed: 2e-8,
    angularSpeed: 2e-8,
  }
  requireNumericRecordWithin(errors, tolerances, 'isolated asymmetric plant transition')
  return {
    catchesSceneEdgeMapping: 'controller joints 15..1 map onto tail-to-head scene edges',
    exercisesVelocityCap: true,
    errors,
    tolerances,
  }
}

function interventionCheckpointErrors(expected: InterventionCheckpoint, actual: InterventionCheckpoint) {
  expect(expected.step === actual.step, 'intervention checkpoint steps differ')
  return {
    commands: vectorError(expected.commands, actual.commands),
    plantCommands: vectorError(expected.plantCommands, actual.plantCommands),
    joints: vectorError(expected.joints, actual.joints),
    jointVelocities: vectorError(expected.jointVelocities, actual.jointVelocities),
    root: vectorError(expected.root, actual.root),
    heading: angleError(expected.heading, actual.heading),
    forwardSpeed: Math.abs(expected.forwardSpeed - actual.forwardSpeed),
    angularSpeed: Math.abs(expected.angularSpeed - actual.angularSpeed),
  }
}

type ParityErrors = ReturnType<typeof interventionCheckpointErrors>

function maximumParityErrors(records: ParityErrors[]): ParityErrors {
  const maximum: ParityErrors = {
    commands: 0,
    plantCommands: 0,
    joints: 0,
    jointVelocities: 0,
    root: 0,
    heading: 0,
    forwardSpeed: 0,
    angularSpeed: 0,
  }
  for (const errors of records) {
    for (const key of Object.keys(maximum) as Array<keyof ParityErrors>) {
      maximum[key] = Math.max(maximum[key], errors[key])
    }
  }
  return maximum
}

function requireParityWithin(errors: ParityErrors, tolerances: ParityErrors, label: string) {
  requireNumericRecordWithin(errors, tolerances, label)
}

function requireNumericRecordWithin<Keys extends string>(
  errors: Record<Keys, number>,
  tolerances: Record<Keys, number>,
  label: string,
) {
  for (const key of Object.keys(tolerances) as Keys[]) {
    expect(errors[key] <= tolerances[key], `${label} ${key} drifted (${errors[key]} > ${tolerances[key]})`)
  }
}

function runPythonInterventionTrace(modelPath: string, mode: Mode): PythonInterventionTrace {
  const result = spawnSync(
    'uv',
    [
      'run',
      '--locked',
      'python',
      '-m',
      'wurmkickflip_rl.locomotion_parity_trace',
      '--model',
      modelPath,
      '--mode',
      mode,
    ],
    { cwd: trainingRoot, encoding: 'utf8', maxBuffer: 4 * 1024 * 1024 },
  )
  if (result.status !== 0) {
    throw new Error(`Python ${mode} intervention trace failed:\n${result.stderr || result.stdout}`)
  }
  try {
    return JSON.parse(result.stdout) as PythonInterventionTrace
  } catch (error) {
    throw new Error(`Python ${mode} intervention trace emitted invalid JSON`, { cause: error })
  }
}

function runTypeScriptInterventionTrace(
  model: LocomotionPolicyArtifact,
  mode: Mode,
): ClosedLoopInterventionTrace {
  const steps = finiteInteger(model.training.episodeSteps, 'training.episodeSteps')
  const scenario = { targetX: 3.2, targetZ: 3.2, friction: 0.8, urgency: 0.9 }
  const policy = new EvolvedLocomotionPolicy(model)
  const plant = createWormLocomotionPlant()
  const segments = createSegments(model)
  const anatomy = deriveWurmAnatomy(null)
  const field = flatField(scenario.friction)
  const state = createInterventionState()
  const checkpointSteps = new Set<number>([...INTERVENTION_CHECKPOINT_STEPS, steps - 1])
  const checkpoints: InterventionCheckpoint[] = []

  for (let step = 0; step < steps; step += 1) {
    const current = center(segments)
    const heading = bodyHeading(segments)
    const forwardX = Math.cos(heading)
    const forwardZ = Math.sin(heading)
    const dx = scenario.targetX - current.x
    const dz = scenario.targetZ - current.z
    const distance = Math.max(Math.hypot(dx, dz), 1e-9)
    const sensors: LocomotionSensors = {
      targetForward: (dx * forwardX + dz * forwardZ) / distance,
      targetRight: (dx * -forwardZ + dz * forwardX) / distance,
      targetDistance: Math.min(distance / 5, 1.5),
      forwardSpeed: plant.forwardSpeed,
      angularSpeed: plant.angularSpeed,
      terrainFriction: scenario.friction,
      urgency: scenario.urgency,
      contactLoads: plant.contactLoads,
      slipSpeeds: plant.slipSpeeds,
      obstacleForward: plant.obstacleForward,
      obstacleRight: plant.obstacleRight,
    }
    const neuralAction = policy.run(sensors, plant.joints, plant.jointVelocities)
    const neuralFeedback = policy.getTelemetry().commands
    const intervention = applyControllerIntervention(mode, step, neuralAction, neuralFeedback, state)
    policy.commitCommandFeedback(intervention.feedback)
    const result = stepArticulatedWorm(
      plant,
      segments,
      intervention.action,
      model.plant.timestep,
      field,
      null,
      anatomy,
    )
    if (checkpointSteps.has(step)) {
      checkpoints.push({
        step,
        root: [result.root.x, result.root.z],
        heading: result.root.heading,
        forwardSpeed: plant.forwardSpeed,
        angularSpeed: plant.angularSpeed,
        joints: [...plant.joints],
        jointVelocities: [...plant.jointVelocities],
        commands: Array.from(intervention.feedback),
        plantCommands: Array.from(
          { length: SEGMENT_COUNT },
          (_, segment) => intervention.action[segment * 2],
        ),
      })
    }
  }
  return { mode, steps, checkpoints }
}

function vectorError(expected: readonly number[], actual: readonly number[]) {
  expect(expected.length === actual.length, 'intervention parity vector lengths differ')
  let maximum = 0
  for (let index = 0; index < expected.length; index += 1) {
    maximum = Math.max(maximum, Math.abs(expected[index] - actual[index]))
  }
  return maximum
}

function matrixError(expected: readonly number[][], actual: readonly number[][]) {
  expect(expected.length === actual.length, 'intervention parity matrix lengths differ')
  let maximum = 0
  for (let index = 0; index < expected.length; index += 1) {
    maximum = Math.max(maximum, vectorError(expected[index], actual[index]))
  }
  return maximum
}

function angleError(expected: number, actual: number) {
  return Math.abs(Math.atan2(Math.sin(expected - actual), Math.cos(expected - actual)))
}

function runScenario(
  model: LocomotionPolicyArtifact,
  scenario: Scenario,
  steps: number,
  mode: Mode,
): Rollout {
  const policy = new EvolvedLocomotionPolicy(model)
  const plant = createWormLocomotionPlant()
  const segments = createSegments(model)
  const anatomy = deriveWurmAnatomy(null)
  const field = flatField(scenario.friction)
  const start = center(segments)
  const initialDistance = Math.hypot(scenario.targetX - start.x, scenario.targetZ - start.z)
  const interventionState = createInterventionState()
  const commands: number[][] = []
  const checkpoints: unknown[] = []
  const checkpointSteps = new Set([0, 1, 2, 23, 24, 59, 119, 239, steps - 1])
  let pathLength = 0
  let contactTotal = 0
  let targetAlignmentTotal = 0
  let bodyAxisVelocityTotal = 0
  let transverseFractionTotal = 0
  let diagnosticPathLength = 0
  let finalRoot = {
    x: start.x,
    y: model.plant.baseGroundClearance,
    z: start.z,
    vx: 0,
    vz: 0,
    heading: 0,
  }

  for (let step = 0; step < steps; step += 1) {
    const current = center(segments)
    const heading = bodyHeading(segments)
    const forwardX = Math.cos(heading)
    const forwardZ = Math.sin(heading)
    const dx = scenario.targetX - current.x
    const dz = scenario.targetZ - current.z
    const distance = Math.max(Math.hypot(dx, dz), 1e-9)
    const sensors: LocomotionSensors = {
      targetForward: (dx * forwardX + dz * forwardZ) / distance,
      targetRight: (dx * -forwardZ + dz * forwardX) / distance,
      targetDistance: Math.min(distance / 5, 1.5),
      forwardSpeed: plant.forwardSpeed,
      angularSpeed: plant.angularSpeed,
      terrainFriction: scenario.friction,
      urgency: scenario.urgency,
      contactLoads: plant.contactLoads,
      slipSpeeds: plant.slipSpeeds,
      obstacleForward: plant.obstacleForward,
      obstacleRight: plant.obstacleRight,
    }
    const neuralAction = policy.run(sensors, plant.joints, plant.jointVelocities)
    const neuralFeedback = policy.getTelemetry().commands
    const intervention = applyControllerIntervention(
      mode,
      step,
      neuralAction,
      neuralFeedback,
      interventionState,
    )
    const applied = intervention.action
    policy.commitCommandFeedback(intervention.feedback)
    commands.push(Array.from({ length: SEGMENT_COUNT }, (_, segment) => applied[segment * 2]))
    const result = stepArticulatedWorm(
      plant,
      segments,
      applied as PolicyAction,
      model.plant.timestep,
      field,
      null,
      anatomy,
    )
    finalRoot = result.root
    pathLength += result.distance
    contactTotal += result.contactRatio
    if (step >= Math.min(60, Math.floor(steps * 0.25))) {
      const targetDx = scenario.targetX - result.root.x
      const targetDz = scenario.targetZ - result.root.z
      const targetDistance = Math.max(Math.hypot(targetDx, targetDz), 1e-9)
      const bodyForwardX = Math.cos(result.root.heading)
      const bodyForwardZ = Math.sin(result.root.heading)
      const bodyRightX = -bodyForwardZ
      const bodyRightZ = bodyForwardX
      const rootDx = result.root.x - current.x
      const rootDz = result.root.z - current.z
      const stepDistance = Math.hypot(rootDx, rootDz)
      if (stepDistance > 1e-12) {
        targetAlignmentTotal +=
          ((targetDx * bodyForwardX + targetDz * bodyForwardZ) / targetDistance) * stepDistance
        bodyAxisVelocityTotal += rootDx * bodyForwardX + rootDz * bodyForwardZ
        transverseFractionTotal += Math.abs(rootDx * bodyRightX + rootDz * bodyRightZ)
        diagnosticPathLength += stepDistance
      }
    }
    if (checkpointSteps.has(step)) {
      checkpoints.push({
        step,
        root: [round(result.root.x), round(result.root.z), round(result.root.heading)],
        joints: plant.joints.map(round),
        commands: commands[commands.length - 1].map(round),
        contactRatio: round(result.contactRatio),
      })
    }
  }

  const finalDistance = Math.hypot(scenario.targetX - finalRoot.x, scenario.targetZ - finalRoot.z)
  const temporalStd = mean(
    Array.from({ length: SEGMENT_COUNT }, (_, segment) =>
      standardDeviation(commands.map(command => command[segment])),
    ),
  )
  const spatialStd = mean(commands.map(standardDeviation))
  const displacement = Math.hypot(finalRoot.x - start.x, finalRoot.z - start.z)
  const traceHash = createHash('sha256').update(JSON.stringify(checkpoints)).digest('hex')
  return {
    progress: initialDistance - finalDistance,
    displacement,
    pathLength,
    temporalStd,
    spatialStd,
    meanContactRatio: contactTotal / steps,
    finalX: finalRoot.x,
    finalZ: finalRoot.z,
    finalHeading: finalRoot.heading,
    meanTargetAlignment: targetAlignmentTotal / Math.max(1e-9, diagnosticPathLength),
    meanVelocityAlongBodyAxis: bodyAxisVelocityTotal / Math.max(1e-9, diagnosticPathLength),
    meanAbsoluteTransverseFraction: transverseFractionTotal / Math.max(1e-9, diagnosticPathLength),
    diagnosticPathLength,
    traceHash,
  }
}

async function verifyTrainingReproducibility() {
  const workspace = await mkdtemp(join(tmpdir(), 'wurmkickflip-locomotion-smoke-'))
  try {
    const firstDirectory = join(workspace, 'location-a')
    const secondDirectory = join(workspace, 'location-b')
    const thirdDirectory = join(workspace, 'different-seed')
    await Promise.all(
      [firstDirectory, secondDirectory, thirdDirectory].map(directory =>
        mkdir(directory, { recursive: true }),
      ),
    )
    const firstWarm = join(firstDirectory, 'warm.json')
    const secondWarm = join(secondDirectory, 'renamed-warm.json')
    const thirdWarm = join(thirdDirectory, 'warm.json')
    await Promise.all([
      copyFile(legacyUrl, firstWarm),
      copyFile(legacyUrl, secondWarm),
      copyFile(legacyUrl, thirdWarm),
    ])
    const first = runTrainingSmoke(firstDirectory, firstWarm, 991)
    const second = runTrainingSmoke(secondDirectory, secondWarm, 991)
    const third = runTrainingSmoke(thirdDirectory, thirdWarm, 992)
    const [firstArtifact, secondArtifact, thirdArtifact, firstSummary, secondSummary] = await Promise.all([
      readFile(first.artifact),
      readFile(second.artifact),
      readFile(third.artifact),
      readFile(first.summary),
      readFile(second.summary),
    ])
    expect(
      firstArtifact.equals(secondArtifact),
      'same-seed artifacts must be byte-identical across warm paths',
    )
    expect(firstSummary.equals(secondSummary), 'same-seed summaries must be byte-identical across warm paths')
    expect(!firstArtifact.equals(thirdArtifact), 'different evolution seeds must produce different artifacts')
    const parsed = JSON.parse(firstArtifact.toString('utf8')) as any
    const warmStart = recordValue(parsed.training?.warmStart, 'smoke training.warmStart')
    expect(
      exactArray(Object.keys(warmStart).sort(), ['modelVersion', 'sha256']),
      'generated warm-start metadata must be path-independent',
    )
    return {
      seed: 991,
      canonicalSha256: createHash('sha256').update(firstArtifact).digest('hex'),
      locationIndependent: true,
    }
  } finally {
    await rm(workspace, { recursive: true, force: true })
  }
}

function runTrainingSmoke(directory: string, warmStart: string, seed: number) {
  const artifact = join(directory, 'model.json')
  const summary = join(directory, 'summary.json')
  const result = spawnSync(
    'uv',
    [
      'run',
      '--locked',
      'python',
      '-m',
      'wurmkickflip_rl.evolve_locomotion_policy',
      '--seed',
      String(seed),
      '--generations',
      '1',
      '--population-size',
      '4',
      '--elite-count',
      '2',
      '--episode-steps',
      '60',
      '--model-version',
      'locomotion-articulated-smoke-v2',
      '--warm-start',
      warmStart,
      '--out',
      artifact,
      '--summary',
      summary,
    ],
    { cwd: trainingRoot, encoding: 'utf8', maxBuffer: 4 * 1024 * 1024 },
  )
  if (result.status !== 0) {
    throw new Error(`locomotion evolution smoke failed:\n${result.stderr || result.stdout}`)
  }
  return { artifact, summary }
}

function createSegments(model: LocomotionPolicyArtifact): SegmentSnapshot[] {
  return Array.from({ length: SEGMENT_COUNT }, (_, index) => ({
    x: (index - (SEGMENT_COUNT - 1) * 0.5) * model.plant.spacing,
    y: model.plant.baseGroundClearance,
    z: 0,
    vx: 0,
    vy: 0,
    vz: 0,
    pitch: 0,
    yaw: 0,
  }))
}

function flatField(friction: number): TerrainField {
  return {
    width: 100,
    depth: 100,
    minimumHeight: 0,
    maximumHeight: 0,
    gridResolution: TERRAIN_GRID_RESOLUTION,
    heightAtGridVertex: () => 0,
    waypoints: [],
    sample: () => ({ height: 0, friction, normal: [0, 1, 0], surface: 'sand' }),
  }
}

function baseSensors(): LocomotionSensors {
  return {
    targetForward: 0.7,
    targetRight: -0.25,
    targetDistance: 0.8,
    forwardSpeed: 0,
    angularSpeed: 0,
    terrainFriction: 0.9,
    urgency: 0.8,
    contactLoads: new Float64Array(SEGMENT_COUNT).fill(1),
    slipSpeeds: new Float64Array(SEGMENT_COUNT),
    obstacleForward: new Float64Array(SEGMENT_COUNT),
    obstacleRight: new Float64Array(SEGMENT_COUNT),
  }
}

function aggregate(rollouts: Rollout[]): Rollout {
  const diagnosticPathLength = rollouts.reduce((total, rollout) => total + rollout.diagnosticPathLength, 0)
  const pathWeighted = (read: (rollout: Rollout) => number) =>
    rollouts.reduce((total, rollout) => total + read(rollout) * rollout.diagnosticPathLength, 0) /
    Math.max(1e-9, diagnosticPathLength)
  return {
    progress: mean(rollouts.map(rollout => rollout.progress)),
    displacement: mean(rollouts.map(rollout => rollout.displacement)),
    pathLength: mean(rollouts.map(rollout => rollout.pathLength)),
    temporalStd: mean(rollouts.map(rollout => rollout.temporalStd)),
    spatialStd: mean(rollouts.map(rollout => rollout.spatialStd)),
    meanContactRatio: mean(rollouts.map(rollout => rollout.meanContactRatio)),
    finalX: mean(rollouts.map(rollout => rollout.finalX)),
    finalZ: mean(rollouts.map(rollout => rollout.finalZ)),
    finalHeading: mean(rollouts.map(rollout => rollout.finalHeading)),
    meanTargetAlignment: pathWeighted(rollout => rollout.meanTargetAlignment),
    meanVelocityAlongBodyAxis: pathWeighted(rollout => rollout.meanVelocityAlongBodyAxis),
    meanAbsoluteTransverseFraction: pathWeighted(rollout => rollout.meanAbsoluteTransverseFraction),
    diagnosticPathLength,
    traceHash: createHash('sha256')
      .update(rollouts.map(rollout => rollout.traceHash).join(':'))
      .digest('hex'),
  }
}

function expectReject(source: any, mutate: (candidate: any) => void, label: string) {
  const candidate = structuredClone(source)
  mutate(candidate)
  let rejected = false
  try {
    parseLocomotionPolicy(candidate)
  } catch {
    rejected = true
  }
  expect(rejected, `parser must reject ${label}`)
}

function bodyHeading(segments: readonly SegmentSnapshot[]) {
  const tail = segments[0]
  const head = segments[segments.length - 1]
  return Math.atan2(head.z - tail.z, head.x - tail.x)
}

function center(segments: readonly SegmentSnapshot[]) {
  return segments.reduce(
    (result, segment) => ({
      x: result.x + segment.x / segments.length,
      z: result.z + segment.z / segments.length,
    }),
    { x: 0, z: 0 },
  )
}

function recordValue(value: unknown, label: string): Record<string, any> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object`)
  }
  return value as Record<string, any>
}

function exactRecord(value: unknown, label: string, keys: readonly string[]) {
  const record = recordValue(value, label)
  expect(exactArray(Object.keys(record).sort(), [...keys].sort()), `${label} keys changed`)
  return record
}

function requireAllTrueGateRecord(value: unknown, label: string, keys: readonly string[]) {
  const gates = exactRecord(value, label, keys)
  for (const key of keys) expect(gates[key] === true, `${label}.${key} must be true`)
  return gates
}

function finiteInteger(value: unknown, label: string) {
  if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer`)
  }
  return value
}

function exactArray(value: unknown, expected: readonly unknown[]) {
  return (
    Array.isArray(value) &&
    value.length === expected.length &&
    value.every((item, index) => item === expected[index])
  )
}

function standardDeviation(values: number[]) {
  const average = mean(values)
  return Math.sqrt(mean(values.map(value => (value - average) ** 2)))
}

function mean(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1)
}

function round(value: number) {
  return Math.round(value * 1e8) / 1e8
}

function expect(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message)
}
