import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { isDeepStrictEqual } from 'node:util'

const root = resolve(import.meta.dirname, '..')
const failures = []
const read = path => readFile(resolve(root, path), 'utf8')
const locomotionArtifactPath =
  process.env.WURMKICKFLIP_LOCOMOTION_ARTIFACT ?? 'public/models/wurmkickflip_locomotion_policy.json'
const assert = (condition, message) => {
  if (!condition) failures.push(message)
}
const equal = (actual, expected, message) => {
  assert(JSON.stringify(actual) === JSON.stringify(expected), message)
}
const record = (value, label) => {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    failures.push(`${label} must be an object.`)
    return {}
  }
  return value
}

const V3_RECIPE_SOURCE_PATHS = [
  'training/wurmkickflip_rl/evolve_locomotion_policy.py',
  'training/wurmkickflip_rl/articulated_locomotion.py',
  'training/wurmkickflip_rl/contracts.py',
  'contracts/locomotion-v2.json',
  'training/uv.lock',
  'training/pyproject.toml',
]

const V3_ARTIFACT_KEYS = [
  'actionSize',
  'architecture',
  'initialState',
  'kind',
  'modelVersion',
  'plant',
  'schemaVersion',
  'segmentCount',
  'segmentPositions',
  'sensorNames',
  'training',
  'weights',
]

const V3_TRAINING_KEYS = [
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
]

const V3_SELECTION_GATE_KEYS = [
  'causalOrderingGap',
  'causalProgressGap',
  'combinedControlLateSpeed',
  'combinedControlProgress',
  'combinedLateSpeedRetention',
  'combinedPerturbedProgress',
  'combinedProgressRetention',
  'localCausalInitial',
  'localCausalRadiusThree',
  'localCausalRadiusTwo',
  'localCausalSettled',
  'nominalLateralMotion',
  'nominalSignedHeadLeading',
  'nominalTargetAlignment',
  'recoveredLateralMotion',
  'recoveredSignedHeadLeading',
  'recoveredTargetAlignment',
]

const V3_ABLATION_GATE_KEYS = [
  'fullBeatsFrozen',
  'fullBeatsShuffled',
  'fullProgress',
  'noFrictionConservesCom',
  'zeroActionStationary',
]

const V3_SENSOR_CLAMPS = {
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
}

const V3_SELECTION_MARGINS = {
  guardBands: {
    scalarThresholdDistance: 0.001,
    localRadiusThresholdDistance: 0.0001,
    localSettledThresholdDistance: 0.000001,
  },
  nominal: {
    signedHeadLeadingFractionAtLeast: 0.72,
    lateralMotionFractionAtMost: 0.4,
    movingTargetFacingAlignmentAtLeast: 0.62,
  },
  combinedRecovery: {
    controlTargetProgressAtLeast: 4.36,
    perturbedProgressFloorRatio: 0.85,
    progressRetentionAtLeast: 0.88,
    lateSpeedRetentionAtLeast: 0.83,
    controlLateProgressSpeedAtLeast: 0.12,
    signedHeadLeadingFractionAtLeast: 0.62,
    lateralMotionFractionAtMost: 0.48,
    movingTargetFacingAlignmentAtLeast: 0.52,
  },
  causal: {
    orderingGapAtLeast: 0.5,
    progressGapAtLeast: 0.05,
    localInitialDeltaAtLeast: 0.12,
    localRadiusTwoDeltaAtLeast: 0.002,
    localRadiusThreeDeltaAtLeast: 0.002,
    localSettledDeltaAtMost: 0.000005,
  },
}

const V3_DOMAIN_RANDOMIZATION = [
  'obstacles',
  'spatial-friction',
  'body-scale',
  'target-switches',
  'contact-loss',
  'transient-slick-window',
  'external-body-shove',
  'paired-shove-numb-low-traction-proxy',
]

const [
  contractText,
  artifactText,
  legacyText,
  retainedCrossoverSeedText,
  retainedCrossoverSeedSummaryText,
  headLeadingSeedText,
  headLeadingSeedSummaryText,
  typesSource,
  runtimeContractSource,
  policySource,
  dynamicsSource,
  pythonContractsSource,
  pythonPlantSource,
  pythonTrainerSource,
  policyDocs,
  locomotionDocs,
] = await Promise.all([
  read('contracts/locomotion-v2.json'),
  read(locomotionArtifactPath),
  read('training/seeds/wurmkickflip_locomotion_warm_start_v1.json'),
  read('training/seeds/wurmkickflip_locomotion_four_solve_swap_1_3_warm_start_v3.json'),
  read('training/seeds/wurmkickflip_locomotion_four_solve_swap_1_3_summary_v3.json'),
  read('training/seeds/wurmkickflip_locomotion_terrarium_causal_candidate_warm_start_v3.json'),
  read('training/seeds/wurmkickflip_locomotion_terrarium_causal_candidate_summary_v3.json'),
  read('src/policy/types.ts'),
  read('src/policy/locomotionContract.ts'),
  read('src/policy/locomotionPolicy.ts'),
  read('src/scene/wormDynamics.ts'),
  read('training/wurmkickflip_rl/contracts.py'),
  read('training/wurmkickflip_rl/articulated_locomotion.py'),
  read('training/wurmkickflip_rl/evolve_locomotion_policy.py'),
  read('docs/policy-contract.md'),
  read('training/LOCOMOTION_POLICY.md'),
])

const contract = record(JSON.parse(contractText), 'locomotion contract')
const artifact = record(JSON.parse(artifactText), 'published locomotion artifact')
const legacy = record(JSON.parse(legacyText), 'legacy locomotion seed')
const retainedCrossoverSeed = record(
  JSON.parse(retainedCrossoverSeedText),
  'retained crossover locomotion seed',
)
const retainedCrossoverSeedSummary = record(
  JSON.parse(retainedCrossoverSeedSummaryText),
  'retained crossover locomotion seed summary',
)
const headLeadingSeed = record(JSON.parse(headLeadingSeedText), 'head-leading locomotion seed')
const headLeadingSeedSummary = record(
  JSON.parse(headLeadingSeedSummaryText),
  'head-leading locomotion seed summary',
)
const joint = record(contract.joint, 'contract.joint')
const dynamics = record(contract.dynamics, 'contract.dynamics')
const weights = record(artifact.weights, 'artifact.weights')
const plant = record(artifact.plant, 'artifact.plant')
const recipeSourceSha256 = Object.fromEntries(
  await Promise.all(
    V3_RECIPE_SOURCE_PATHS.map(async path => {
      const source = await read(path)
      return [path, sha256Text(canonicalLineEndings(source))]
    }),
  ),
)

assert(contract.schemaVersion === 2, 'Locomotion contract schemaVersion must equal 2.')
assert(contract.kind === 'wurmkickflip.locomotionContract', 'Locomotion contract kind mismatch.')
assert(contract.plantVersion === 'articulated-contact-v2', 'Locomotion plantVersion mismatch.')
assert(contract.segmentCount === 16, 'Locomotion segmentCount must equal 16.')
assert(contract.muscleChannelCount === 32, 'Locomotion muscleChannelCount must equal 32.')
assert(
  contract.muscleChannelCount === contract.segmentCount * 2,
  'Locomotion muscle channel count must be two antagonistic channels per segment.',
)
assert(contract.timestep === 1 / 60, 'Locomotion timestep must equal 1/60.')
assert(
  contract.postIntegrationRelaxationPasses === 3,
  'Locomotion post-integration relaxation pass count must equal 3.',
)
equal(
  contract.sensorNames,
  [
    'targetForward',
    'targetRight',
    'targetDistance',
    'forwardSpeed',
    'angularSpeed',
    'terrainFriction',
    'urgency',
    'contactLoad',
    'slipSpeed',
    'obstacleForward',
    'obstacleRight',
  ],
  'Locomotion sensor ordering mismatch.',
)
equal(
  contract.inputWeightNames,
  [
    'bias',
    'segmentPosition',
    'targetForward',
    'targetRight',
    'targetRightByPosition',
    'targetDistance',
    'forwardSpeed',
    'angularSpeed',
    'terrainFriction',
    'urgency',
    'segmentBend',
    'segmentBendVelocity',
    'previousCommand',
    'contactLoad',
    'slipSpeed',
    'obstacleForward',
    'obstacleRight',
  ],
  'Locomotion input weight ordering mismatch.',
)
equal(
  contract.recurrentWeightNames,
  ['self', 'anteriorNeighbor', 'posteriorNeighbor'],
  'Locomotion recurrent weight ordering mismatch.',
)
equal(
  contract.outputWeightNames,
  [
    'bias',
    'self',
    'anteriorNeighbor',
    'posteriorNeighbor',
    'targetRight',
    'targetRightByPosition',
    'segmentBend',
    'obstacleRight',
    'contactLoad',
  ],
  'Locomotion output weight ordering mismatch.',
)
equal(joint, { stiffness: 22, damping: 7, limit: 1.15 }, 'Locomotion joint contract mismatch.')
equal(
  dynamics,
  {
    spacing: 0.102,
    actuatorStiffness: 78,
    actuatorDamping: 8.5,
    shapeBendScale: 0.19,
    constraintIterations: 6,
    constraintCompliance: 0.035,
    longitudinalFriction: 1.15,
    lateralFriction: 17.5,
    freeVelocityDrag: 0.42,
    maximumSpeed: 2.4,
    maximumVerticalSpeed: 2.4,
    baseRadius: 0.085,
    baseGroundClearance: 0.105,
  },
  'Locomotion articulated dynamics contract mismatch.',
)
assert(
  !contract.sensorNames.some(name => /time|clock|phase|cycle/i.test(name)),
  'Locomotion contract must remain clock-free.',
)
assert(
  !pythonTrainerSource.includes('DEFAULT_ARTIFACT = ROOT / "public/models/'),
  'The trainer must never target the published model through an implicit output path.',
)

assert(artifact.schemaVersion === 2, 'Published locomotion schemaVersion must equal 2.')
assert(artifact.kind === 'wurmkickflip.locomotionPolicy', 'Published locomotion artifact kind mismatch.')
assert(artifact.architecture === 'segmental-recurrent-tanh', 'Published locomotion architecture mismatch.')
assert(artifact.segmentCount === contract.segmentCount, 'Published locomotion segmentCount mismatch.')
assert(artifact.actionSize === contract.muscleChannelCount, 'Published locomotion actionSize mismatch.')
equal(artifact.sensorNames, contract.sensorNames, 'Published locomotion sensor names mismatch.')
equal(weights.inputNames, contract.inputWeightNames, 'Published locomotion input names mismatch.')
equal(weights.recurrentNames, contract.recurrentWeightNames, 'Published recurrent names mismatch.')
equal(weights.outputNames, contract.outputWeightNames, 'Published output names mismatch.')
assert(
  Array.isArray(artifact.initialState) && artifact.initialState.length === 16,
  'Initial state length mismatch.',
)
assert(Array.isArray(weights.input) && weights.input.length === 17, 'Input weight length mismatch.')
assert(
  Array.isArray(weights.recurrent) && weights.recurrent.length === 3,
  'Recurrent weight length mismatch.',
)
assert(Array.isArray(weights.output) && weights.output.length === 9, 'Output weight length mismatch.')
assert(plant.version === contract.plantVersion, 'Published plant version mismatch.')
assert(plant.timestep === contract.timestep, 'Published plant timestep mismatch.')
assert(plant.jointStiffness === joint.stiffness, 'Published joint stiffness mismatch.')
assert(plant.jointDamping === joint.damping, 'Published joint damping mismatch.')
assert(plant.jointLimit === joint.limit, 'Published joint limit mismatch.')
for (const [name, expected] of Object.entries(dynamics)) {
  const actual = name === 'maximumVerticalSpeed' && plant[name] === undefined ? expected : plant[name]
  assert(actual === expected, `Published plant.${name} mismatch.`)
}
const headLeadingSeedSha256 = createHash('sha256')
  .update(headLeadingSeedText.replace(/\r\n?/g, '\n'))
  .digest('hex')
assert(
  headLeadingSeedSha256 === '0c46518c424528337f3a2ad5a96912b7812dda473688d468556023be095551ba',
  'Tracked head-leading search seed bytes changed.',
)
assert(legacy.schemaVersion === 1, 'Tracked legacy warm start must remain schema v1.')
equal(
  legacy.sensorNames,
  contract.sensorNames.slice(0, 7),
  'Legacy sensor names must be the v2 global-sensor prefix.',
)
const legacyWeights = record(legacy.weights, 'legacy.weights')
equal(
  legacyWeights.inputNames,
  contract.inputWeightNames.slice(0, 13),
  'Legacy input names must be the v2 input prefix.',
)
equal(
  legacyWeights.outputNames,
  contract.outputWeightNames.slice(0, 7),
  'Legacy output names must be the v2 output prefix.',
)

assert(
  runtimeContractSource.includes("import contractJson from '../../contracts/locomotion-v2.json'"),
  'TypeScript locomotion runtime must import the canonical JSON contract.',
)
assert(
  typesSource.includes('LOCOMOTION_CONTRACT.segmentCount') &&
    typesSource.includes('LOCOMOTION_CONTRACT.muscleChannelCount') &&
    typesSource.includes('LOCOMOTION_CONTRACT.timestep'),
  'TypeScript policy dimensions and timestep must derive from the locomotion contract.',
)
assert(
  policySource.includes('LOCOMOTION_CONTRACT.sensorNames') &&
    policySource.includes('LOCOMOTION_CONTRACT.inputWeightNames') &&
    policySource.includes('LOCOMOTION_CONTRACT.outputWeightNames'),
  'TypeScript locomotion name vectors must derive from the canonical contract.',
)
assert(
  dynamicsSource.includes("import { LOCOMOTION_CONTRACT } from '../policy/locomotionContract'") &&
    dynamicsSource.includes('LOCOMOTION_CONTRACT.postIntegrationRelaxationPasses'),
  'Browser articulated dynamics and its relaxation pass count must derive from the canonical contract.',
)
assert(dynamicsSource.includes('stepArticulatedWorm'), 'Browser dynamics must expose the articulated plant.')
assert(
  pythonContractsSource.includes('contracts" / "locomotion-v2.json'),
  'Python mounted constants must derive dimensions and timestep from the canonical contract.',
)
assert(
  pythonPlantSource.includes('CONTRACT_PATH = ROOT / "contracts" / "locomotion-v2.json"') &&
    pythonPlantSource.includes('CONTRACT["postIntegrationRelaxationPasses"]'),
  'Python articulated plant and its relaxation pass count must derive from the canonical contract.',
)
assert(
  pythonTrainerSource.includes('SENSOR_NAMES = tuple(CONTRACT["sensorNames"])') &&
    pythonTrainerSource.includes('INPUT_WEIGHT_NAMES = tuple(CONTRACT["inputWeightNames"])'),
  'Python evolution name vectors must derive from the canonical contract.',
)
assert(
  pythonTrainerSource.includes('articulated_plant_step(') &&
    !/wave_gain|maximum_forward_acceleration|inverse_traction_drag|front_turn_base/i.test(
      pythonTrainerSource,
    ),
  'Python evolution must use the articulated plant without removed scalar fields.',
)
const warmLoader = pythonTrainerSource.slice(
  pythonTrainerSource.indexOf('def load_warm_start'),
  pythonTrainerSource.indexOf('\ndef evolve'),
)
assert(!/["']path["']\s*:/.test(warmLoader), 'Warm-start metadata must not include filesystem paths.')

for (const [name, text] of [
  ['docs/policy-contract.md', policyDocs],
  ['training/LOCOMOTION_POLICY.md', locomotionDocs],
]) {
  assert(text.includes('16'), `${name} must document the 16-segment locomotion lattice.`)
  assert(text.includes('32'), `${name} must document the 32-channel action contract.`)
}

function verifyPublishedV3Artifact(candidate, expectedSourceHashes, seedArtifact) {
  requireExactKeys(candidate, V3_ARTIFACT_KEYS, 'artifact')
  requireEqual(candidate.schemaVersion, 2, 'artifact.schemaVersion')
  requireEqual(candidate.kind, 'wurmkickflip.locomotionPolicy', 'artifact.kind')
  requireEqual(candidate.architecture, 'segmental-recurrent-tanh', 'artifact.architecture')
  requireEqual(candidate.segmentCount, 16, 'artifact.segmentCount')
  requireEqual(candidate.actionSize, 32, 'artifact.actionSize')
  requireEqual(
    candidate.modelVersion,
    'locomotion-articulated-head-leading-es-v3',
    'Published locomotion model version',
  )
  const candidateTraining = requireRecord(candidate.training, 'artifact.training')
  requireExactKeys(candidateTraining, V3_TRAINING_KEYS, 'artifact.training')

  for (const [name, expected] of Object.entries({
    algorithm: 'elitist-mutation-evolution',
    seed: 20260737,
    generations: 1,
    populationSize: 4,
    eliteCount: 1,
    episodeSteps: 480,
    scenarioCount: 12,
    robustnessScenarioCount: 4,
    combinedRecoveryScenarioCount: 1,
    combinedRecoveryMode: 'bounded-480',
    combinedRecoveryEpisodeSteps: 480,
    causalScenarioCount: 4,
    objectiveVersion: 'articulated-head-leading-transient-recovery-v3',
    actuatorPrecision: 'float32-plant-command',
    genomeDecimals: 8,
    selectionFitnessDecimals: 4,
    gateDiagnosticDecimals: 8,
  })) {
    requireEqual(candidateTraining[name], expected, `artifact.training.${name}`)
  }

  requireDeepEqual(
    candidateTraining.domainRandomization,
    V3_DOMAIN_RANDOMIZATION,
    'artifact.training.domainRandomization',
  )
  requireDeepEqual(
    candidateTraining.controllerExposedTimingSignals,
    [],
    'artifact.training.controllerExposedTimingSignals',
  )
  requireDeepEqual(
    candidateTraining.controllerSensorClamps,
    V3_SENSOR_CLAMPS,
    'artifact.training.controllerSensorClamps',
  )
  requireDeepEqual(
    candidateTraining.selectionMargins,
    V3_SELECTION_MARGINS,
    'artifact.training.selectionMargins',
  )
  requireDeepEqual(
    candidateTraining.headLeadingMeasurement,
    {
      axis: 'tail-to-head segment COM axis; positive toward rendered head',
      signedFraction: 'sum(COM displacement dot body axis) / total COM path length',
      lateralFraction: 'sum(abs(COM displacement dot body right)) / total COM path length',
      turnHandling:
        'all world headings are valid; target-facing alignment has a short grace window after deterministic target switches',
      selectionMargins: {
        signedHeadLeadingFraction: 0.72,
        maximumLateralMotionFraction: 0.4,
        movingTargetFacingAlignment: 0.62,
        combinedRecoveredSignedHeadLeadingFraction: 0.62,
        combinedRecoveredMaximumLateralMotionFraction: 0.48,
        combinedRecoveredTargetFacingAlignment: 0.52,
      },
    },
    'artifact.training.headLeadingMeasurement',
  )
  requireDeepEqual(
    candidateTraining.perturbationSchedules,
    {
      slickWindow: {
        preSteps: [57, 115],
        activeSteps: [115, 182],
        postSteps: [182, 240],
        friction: 0.035,
      },
      bodyShove: {
        preSteps: [240, 297],
        atStep: 297,
        postSteps: [298, 393],
        alternatingLateralSpeed: 1.35,
        backwardSpeed: 0.28,
      },
      combinedRecoveryProxy: {
        controlTarget: [20, 0],
        activeSteps: [128, 176],
        lateSpeedSteps: [240, 336],
        lateralShoveSpeed: 1.1,
        numbSegment: 7,
        tractionScale: 1 / 15,
        pythonArenaLimitedProxy: true,
        plantContractModified: false,
      },
    },
    'artifact.training.perturbationSchedules',
  )
  requireDeepEqual(
    candidateTraining.recoveryMeasurement,
    {
      speed: 'positive head-axis speed multiplied by positive target-facing alignment',
      ratio: 'post-window mean speed / max(pre-window mean speed, 0.04 m/s)',
      time: 'first post-event step whose 0.18-EMA speed reaches max(55% of pre-window speed, 0.04 m/s) while target alignment is at least 0.1',
    },
    'artifact.training.recoveryMeasurement',
  )
  requireDeepEqual(
    candidateTraining.localCausalProbe,
    {
      pulseSegment: 7,
      pulseTicks: 1,
      browserGates: {
        initialLocalDeltaGreaterThan: 0.1,
        radiusTwoDeltaGreaterThan: 0.001,
        radiusThreeDeltaGreaterThan: 0.001,
        settledDeltaLessThan: 0.00001,
      },
      selectionMargins: {
        initialLocalDelta: 0.12,
        radiusTwoDelta: 0.002,
        radiusThreeDelta: 0.002,
        settledDelta: 0.000005,
      },
      actionPrecision: 'float32-returned-action-float64-recurrent-command',
    },
    'artifact.training.localCausalProbe',
  )
  requireDeepEqual(
    candidateTraining.ablationProtocol,
    {
      runtimeSensorContract: true,
      hostMagnitudesStoredInSummaryOnly: true,
      guardBands: {
        progressThresholdDistance: 0.01,
        conservationAtMost: 1e-10,
      },
      thresholds: {
        fullProgressAtLeast: 0.45,
        zeroProgressAtMost: 1e-9,
        fullMinusFrozenProgressAtLeast: 0.3,
        fullMinusShuffledProgressAtLeast: 0.2,
        noFrictionDisplacementAtMost: 1e-9,
      },
    },
    'artifact.training.ablationProtocol',
  )

  const passedSelectionGates = Object.fromEntries(V3_SELECTION_GATE_KEYS.map(name => [name, true]))
  const passedAblationGates = Object.fromEntries(V3_ABLATION_GATE_KEYS.map(name => [name, true]))
  for (const name of ['selectionGateResults', 'selectionGateGuardBandResults']) {
    requireDeepEqual(candidateTraining[name], passedSelectionGates, `artifact.training.${name}`)
  }
  for (const name of ['ablationGateResults', 'ablationGateGuardBandResults']) {
    requireDeepEqual(candidateTraining[name], passedAblationGates, `artifact.training.${name}`)
  }
  for (const name of [
    'allSelectionGatesPassed',
    'allSelectionMarginsFeasible',
    'allSelectionGuardBandsPassed',
    'allAblationGatesPassed',
    'allAblationGuardBandsPassed',
    'allGateGuardBandsPassed',
    'allPublicationGatesPassed',
  ]) {
    requireEqual(candidateTraining[name], true, `artifact.training.${name}`)
  }

  const warmStart = requireRecord(candidateTraining.warmStart, 'artifact.training.warmStart')
  requireExactKeys(warmStart, ['modelVersion', 'sha256'], 'artifact.training.warmStart')
  requireEqual(warmStart.modelVersion, seedArtifact.modelVersion, 'artifact.training.warmStart.modelVersion')
  requireEqual(
    warmStart.modelVersion,
    'locomotion-head-leading-es-v3-terrarium-causal-candidate-v1',
    'artifact.training.warmStart.modelVersion',
  )
  requireEqual(
    warmStart.sha256,
    '0c46518c424528337f3a2ad5a96912b7812dda473688d468556023be095551ba',
    'artifact.training.warmStart.sha256',
  )

  requireDeepEqual(
    candidateTraining.pairedHostReproduction,
    {
      requiredForPublication: true,
      compared: ['serializedGenome', 'artifactBytes'],
      trainerAttestation: 'none',
    },
    'artifact.training.pairedHostReproduction',
  )

  const sourceHashes = requireRecord(
    candidateTraining.recipeSourceSha256,
    'artifact.training.recipeSourceSha256',
  )
  requireExactKeys(sourceHashes, V3_RECIPE_SOURCE_PATHS, 'artifact.training.recipeSourceSha256')
  for (const path of V3_RECIPE_SOURCE_PATHS) {
    if (!/^[a-f\d]{64}$/u.test(String(sourceHashes[path]))) {
      throw new Error(`artifact.training.recipeSourceSha256[${JSON.stringify(path)}] is malformed`)
    }
    requireEqual(
      sourceHashes[path],
      expectedSourceHashes[path],
      `artifact.training.recipeSourceSha256[${JSON.stringify(path)}]`,
    )
  }

  if (!/^[a-f\d]{64}$/u.test(String(candidateTraining.canonicalGenomeRecipeHash))) {
    throw new Error('artifact.training.canonicalGenomeRecipeHash is malformed')
  }
  requireEqual(
    candidateTraining.canonicalGenomeRecipeHash,
    canonicalGenomeRecipeHash(candidate, candidateTraining),
    'artifact.training.canonicalGenomeRecipeHash',
  )
}

function runV3ContractMutationChecks(candidate, expectedSourceHashes, seedArtifact) {
  const mutationCases = [
    [
      'host-sensitive fitness magnitude',
      clone => {
        clone.training.fitness = 123
      },
    ],
    [
      'host-sensitive causal-progress magnitude',
      clone => {
        clone.training.causalProgressGap = 0.5
      },
    ],
    [
      'host-sensitive recovery magnitudes',
      clone => {
        clone.training.perturbationRecovery = {}
      },
    ],
    [
      'host-sensitive ablation magnitudes',
      clone => {
        clone.training.ablations = {}
      },
    ],
    [
      'host-sensitive deterministic trace',
      clone => {
        clone.training.deterministicTraceHash = '0'.repeat(64)
      },
    ],
    [
      'top-level host diagnostics',
      clone => {
        clone.hostDiagnostics = {}
      },
    ],
    [
      'false publication aggregate',
      clone => {
        clone.training.allPublicationGatesPassed = false
      },
    ],
    [
      'missing causal-progress gate',
      clone => {
        delete clone.training.selectionGateResults.causalProgressGap
      },
    ],
    [
      'false guard-band gate',
      clone => {
        clone.training.selectionGateGuardBandResults.localCausalSettled = false
      },
    ],
    [
      'selection precision drift',
      clone => {
        clone.training.selectionFitnessDecimals = 8
      },
    ],
    [
      'nested selection-margin drift',
      clone => {
        clone.training.selectionMargins.causal.localSettledDeltaAtMost = 0.00001
      },
    ],
    [
      'recipe source substitution',
      clone => {
        clone.training.recipeSourceSha256['training/uv.lock'] = '0'.repeat(64)
      },
    ],
    [
      'self-attested paired-host claim',
      clone => {
        clone.training.pairedHostReproduction.trainerAttestation = 'verified'
      },
    ],
    [
      'extra paired-host status',
      clone => {
        clone.training.pairedHostReproduction.status = 'verified'
      },
    ],
    [
      'genome mutation without recipe rehash',
      clone => {
        clone.initialState[0] = Number((clone.initialState[0] + 0.00000001).toFixed(8))
      },
    ],
    [
      'forged canonical recipe hash',
      clone => {
        clone.training.canonicalGenomeRecipeHash = 'f'.repeat(64)
      },
    ],
  ]

  for (const [label, mutate] of mutationCases) {
    const clone = structuredClone(candidate)
    mutate(clone)
    let rejected = false
    try {
      verifyPublishedV3Artifact(clone, expectedSourceHashes, seedArtifact)
    } catch {
      rejected = true
    }
    assert(rejected, `V3 contract mutation was accepted: ${label}.`)
  }
}

function canonicalGenomeRecipeHash(candidate, candidateTraining) {
  const candidateWeights = requireRecord(candidate.weights, 'artifact.weights')
  const genome = [
    ...requireFiniteNumbers(candidate.initialState, 16, 'artifact.initialState'),
    ...requireFiniteNumbers(candidateWeights.input, 17, 'artifact.weights.input'),
    ...requireFiniteNumbers(candidateWeights.recurrent, 3, 'artifact.weights.recurrent'),
    ...requireFiniteNumbers(candidateWeights.output, 9, 'artifact.weights.output'),
  ]
  for (const [index, value] of genome.entries()) {
    if (Number(value.toFixed(8)) !== value) {
      throw new Error(`artifact genome value ${index} is not serialized to eight decimals`)
    }
  }

  const payload = {
    identitySchema: 'wurmkickflip.canonicalGenomeRecipe.v2',
    genome: genome.map(value => new PythonFloat(value)),
    genomeDecimals: 8,
    modelVersion: candidate.modelVersion,
    sourceSha256: candidateTraining.recipeSourceSha256,
    recipe: {
      algorithm: candidateTraining.algorithm,
      objectiveVersion: candidateTraining.objectiveVersion,
      seed: candidateTraining.seed,
      generations: candidateTraining.generations,
      populationSize: candidateTraining.populationSize,
      eliteCount: candidateTraining.eliteCount,
      episodeSteps: candidateTraining.episodeSteps,
      combinedRecoveryMode: candidateTraining.combinedRecoveryMode,
      warmStart: candidateTraining.warmStart,
      genomeDecimals: candidateTraining.genomeDecimals,
      selectionFitnessDecimals: candidateTraining.selectionFitnessDecimals,
      gateDiagnosticDecimals: candidateTraining.gateDiagnosticDecimals,
      controllerSensorClamps: pythonFloatTree(candidateTraining.controllerSensorClamps),
      selectionMargins: pythonFloatTree(candidateTraining.selectionMargins),
    },
  }
  return sha256Text(pythonCanonicalJson(payload))
}

function verifyRetainedCrossoverSeed(seedArtifact, seedSummary) {
  requireExactKeys(seedArtifact, V3_ARTIFACT_KEYS, 'retained crossover seed')
  requireEqual(seedArtifact.schemaVersion, 2, 'retained crossover seed.schemaVersion')
  requireEqual(seedArtifact.kind, 'wurmkickflip.locomotionPolicy', 'retained crossover seed.kind')
  requireEqual(
    seedArtifact.modelVersion,
    'locomotion-head-leading-es-v3-four-solve-swap-1-3-seed-v1',
    'retained crossover seed.modelVersion',
  )
  requireEqual(seedArtifact.architecture, 'segmental-recurrent-tanh', 'retained crossover seed.architecture')
  requireEqual(seedArtifact.segmentCount, 16, 'retained crossover seed.segmentCount')
  requireEqual(seedArtifact.actionSize, 32, 'retained crossover seed.actionSize')

  const seedTraining = requireRecord(seedArtifact.training, 'retained crossover seed.training')
  requireExactKeys(
    seedTraining,
    [
      'ablationGateGuardBandResults',
      'ablationGateResults',
      'algorithm',
      'allAblationGatesPassed',
      'allAblationGuardBandsPassed',
      'allSelectionGatesPassed',
      'allSelectionGuardBandsPassed',
      'combinedRecoveryMode',
      'derivationRecipe',
      'episodeSteps',
      'objectiveVersion',
      'publicRuntimeModel',
      'recipeSourceSha256',
      'retainedSeedRecipeHash',
      'selectionGateGuardBandResults',
      'selectionGateResults',
    ],
    'retained crossover seed.training',
  )
  requireEqual(
    seedTraining.algorithm,
    'deterministic-module-crossover-retained-seed',
    'retained crossover seed.training.algorithm',
  )
  requireEqual(seedTraining.episodeSteps, 480, 'retained crossover seed.training.episodeSteps')
  requireEqual(
    seedTraining.combinedRecoveryMode,
    'bounded-480',
    'retained crossover seed.training.combinedRecoveryMode',
  )
  requireEqual(seedTraining.publicRuntimeModel, false, 'retained crossover seed.training.publicRuntimeModel')

  const weights = requireRecord(seedArtifact.weights, 'retained crossover seed.weights')
  const serializedGenome = [
    ...requireFiniteNumbers(seedArtifact.initialState, 16, 'retained crossover seed.initialState'),
    ...requireFiniteNumbers(weights.input, 17, 'retained crossover seed.weights.input'),
    ...requireFiniteNumbers(weights.recurrent, 3, 'retained crossover seed.weights.recurrent'),
    ...requireFiniteNumbers(weights.output, 9, 'retained crossover seed.weights.output'),
  ]
  for (const [index, value] of serializedGenome.entries()) {
    if (Number(value.toFixed(8)) !== value) {
      throw new Error(`retained crossover seed genome value ${index} is not serialized to eight decimals`)
    }
  }

  const derivationRecipe = requireRecord(
    seedTraining.derivationRecipe,
    'retained crossover seed.training.derivationRecipe',
  )
  requireExactKeys(
    derivationRecipe,
    [
      'kind',
      'operation',
      'parents',
      'recipeSourceSha256',
      'replacementGenomeRanges',
      'replacementInputNames',
      'schemaVersion',
    ],
    'retained crossover seed.training.derivationRecipe',
  )
  requireEqual(derivationRecipe.schemaVersion, 1, 'retained crossover derivation.schemaVersion')
  requireEqual(
    derivationRecipe.kind,
    'wurmkickflip.deterministicModuleCrossoverRecipe',
    'retained crossover derivation.kind',
  )
  requireEqual(
    derivationRecipe.operation,
    'copy base genome, then replace half-open genome ranges from donor',
    'retained crossover derivation.operation',
  )
  requireDeepEqual(
    derivationRecipe.replacementGenomeRanges,
    [
      [16, 18],
      [22, 26],
    ],
    'retained crossover derivation.replacementGenomeRanges',
  )
  requireDeepEqual(
    derivationRecipe.replacementInputNames,
    [
      ['bias', 'segmentPosition'],
      ['forwardSpeed', 'angularSpeed', 'terrainFriction', 'urgency'],
    ],
    'retained crossover derivation.replacementInputNames',
  )

  if (!Array.isArray(derivationRecipe.parents) || derivationRecipe.parents.length !== 2) {
    throw new Error('retained crossover derivation must contain exactly two parents')
  }
  const [baseParent, donorParent] = derivationRecipe.parents.map((parent, index) =>
    requireRecord(parent, `retained crossover derivation.parents[${index}]`),
  )
  const parentKeys = [
    'role',
    'serializedGenome',
    'sourceModelVersion',
    'sourceRecipe',
    'sourceSummary',
    'sourceSummarySha256',
  ]
  const expectedParents = [
    {
      role: 'base',
      sourceSummary: 'training/runs/locomotion_head_leading_v3/explore-four-solve-final-480-summary.json',
      sourceSummarySha256: 'cf70b52b859539f61c06d34d278a82093dc4579a87a15bb73551c8523982adcd',
      sourceModelVersion: 'locomotion-exploratory-four-solve-final-480-v1',
      sourceRecipe: {
        seed: 20260757,
        generations: 4,
        populationSize: 24,
        eliteCount: 6,
        episodeSteps: 480,
        warmStart: {
          sha256: '7c0a52a00d14c0c5c4d4806ea9b04a7a4c625d01e00a1f21c6bae6e2ee065c58',
          modelVersion: 'locomotion-head-leading-es-v3-final-probe',
        },
      },
    },
    {
      role: 'donor',
      sourceSummary: 'training/runs/locomotion_head_leading_v3/explore-four-solve-guarded-480-summary.json',
      sourceSummarySha256: '2a17698f1dd18b52ccb15e34ef6c084fc8b14510f6f16cf3e48517e710ceba86',
      sourceModelVersion: 'locomotion-exploratory-four-solve-guarded-480-v1',
      sourceRecipe: {
        seed: 20260756,
        generations: 4,
        populationSize: 24,
        eliteCount: 6,
        episodeSteps: 480,
        warmStart: {
          sha256: '64648b58b50e93ed00f05d088d0054cd52aa1479f9c00c24ea4aeb3564ff5d21',
          modelVersion: 'locomotion-head-leading-es-v3-guarded-search-seed-v2',
        },
      },
    },
  ]
  for (const [index, parent] of [baseParent, donorParent].entries()) {
    requireExactKeys(parent, parentKeys, `retained crossover derivation.parents[${index}]`)
    for (const name of ['role', 'sourceSummary', 'sourceSummarySha256', 'sourceModelVersion']) {
      requireEqual(
        parent[name],
        expectedParents[index][name],
        `retained crossover derivation.parents[${index}].${name}`,
      )
    }
    requireDeepEqual(
      parent.sourceRecipe,
      expectedParents[index].sourceRecipe,
      `retained crossover derivation.parents[${index}].sourceRecipe`,
    )
  }
  const baseGenome = requireFiniteNumbers(
    baseParent.serializedGenome,
    45,
    'retained crossover derivation base genome',
  )
  const donorGenome = requireFiniteNumbers(
    donorParent.serializedGenome,
    45,
    'retained crossover derivation donor genome',
  )
  const reconstructedGenome = [...baseGenome]
  for (const [start, end] of derivationRecipe.replacementGenomeRanges) {
    reconstructedGenome.splice(start, end - start, ...donorGenome.slice(start, end))
  }
  requireDeepEqual(reconstructedGenome, serializedGenome, 'retained crossover seed reconstructed genome')

  const sourceHashes = requireRecord(
    seedTraining.recipeSourceSha256,
    'retained crossover seed.training.recipeSourceSha256',
  )
  requireExactKeys(sourceHashes, V3_RECIPE_SOURCE_PATHS, 'retained crossover seed source hashes')
  requireDeepEqual(
    derivationRecipe.recipeSourceSha256,
    sourceHashes,
    'retained crossover derivation source hashes',
  )
  for (const [path, hash] of Object.entries(sourceHashes)) {
    if (!/^[a-f\d]{64}$/u.test(String(hash))) {
      throw new Error(`retained crossover source hash is malformed for ${path}`)
    }
  }

  const expectedRecipeHash = sha256Text(canonicalSortedJson({ derivationRecipe, serializedGenome }))
  requireEqual(
    seedTraining.retainedSeedRecipeHash,
    expectedRecipeHash,
    'retained crossover seed.training.retainedSeedRecipeHash',
  )
  const passedSelectionGates = Object.fromEntries(V3_SELECTION_GATE_KEYS.map(name => [name, true]))
  const passedAblationGates = Object.fromEntries(V3_ABLATION_GATE_KEYS.map(name => [name, true]))
  for (const name of ['selectionGateResults', 'selectionGateGuardBandResults']) {
    requireDeepEqual(seedTraining[name], passedSelectionGates, `retained crossover seed.training.${name}`)
  }
  for (const name of ['ablationGateResults', 'ablationGateGuardBandResults']) {
    requireDeepEqual(seedTraining[name], passedAblationGates, `retained crossover seed.training.${name}`)
  }
  for (const name of [
    'allSelectionGatesPassed',
    'allSelectionGuardBandsPassed',
    'allAblationGatesPassed',
    'allAblationGuardBandsPassed',
  ]) {
    requireEqual(seedTraining[name], true, `retained crossover seed.training.${name}`)
  }

  requireExactKeys(
    seedSummary,
    [
      'ablationGateGuardBandResults',
      'ablationGateResults',
      'derivationRecipe',
      'hostDiagnostics',
      'kind',
      'modelVersion',
      'retainedSeedRecipeHash',
      'schemaVersion',
      'selectionGateGuardBandResults',
      'selectionGateResults',
      'serializedGenome',
    ],
    'retained crossover seed summary',
  )
  requireEqual(seedSummary.schemaVersion, 1, 'retained crossover seed summary.schemaVersion')
  requireEqual(
    seedSummary.kind,
    'wurmkickflip.retainedCrossoverSeedSummary',
    'retained crossover seed summary.kind',
  )
  requireEqual(
    seedSummary.modelVersion,
    seedArtifact.modelVersion,
    'retained crossover seed summary.modelVersion',
  )
  requireDeepEqual(
    seedSummary.serializedGenome,
    serializedGenome,
    'retained crossover seed summary.serializedGenome',
  )
  requireDeepEqual(
    seedSummary.derivationRecipe,
    derivationRecipe,
    'retained crossover seed summary.derivationRecipe',
  )
  requireEqual(
    seedSummary.retainedSeedRecipeHash,
    expectedRecipeHash,
    'retained crossover seed summary.retainedSeedRecipeHash',
  )
  for (const name of [
    'selectionGateResults',
    'selectionGateGuardBandResults',
    'ablationGateResults',
    'ablationGateGuardBandResults',
  ]) {
    requireDeepEqual(seedSummary[name], seedTraining[name], `retained crossover seed summary.${name}`)
  }
  requireRecord(seedSummary.hostDiagnostics, 'retained crossover seed summary.hostDiagnostics')
}

function verifyRetainedTerrariumCausalSeed(seedArtifact, seedSummary) {
  requireExactKeys(seedArtifact, V3_ARTIFACT_KEYS, 'retained terrarium-causal seed')
  requireEqual(seedArtifact.schemaVersion, 2, 'retained terrarium-causal seed.schemaVersion')
  requireEqual(seedArtifact.kind, 'wurmkickflip.locomotionPolicy', 'retained terrarium-causal seed.kind')
  requireEqual(
    seedArtifact.modelVersion,
    'locomotion-head-leading-es-v3-terrarium-causal-candidate-v1',
    'retained terrarium-causal seed.modelVersion',
  )
  requireEqual(seedArtifact.architecture, 'segmental-recurrent-tanh', 'retained seed architecture')
  requireEqual(seedArtifact.segmentCount, 16, 'retained terrarium-causal seed.segmentCount')
  requireEqual(seedArtifact.actionSize, 32, 'retained terrarium-causal seed.actionSize')

  const weights = requireRecord(seedArtifact.weights, 'retained terrarium-causal seed.weights')
  const serializedGenome = [
    ...requireFiniteNumbers(seedArtifact.initialState, 16, 'retained terrarium-causal seed.initialState'),
    ...requireFiniteNumbers(weights.input, 17, 'retained terrarium-causal seed.weights.input'),
    ...requireFiniteNumbers(weights.recurrent, 3, 'retained terrarium-causal seed.weights.recurrent'),
    ...requireFiniteNumbers(weights.output, 9, 'retained terrarium-causal seed.weights.output'),
  ]
  for (const [index, value] of serializedGenome.entries()) {
    if (Number(value.toFixed(8)) !== value) {
      throw new Error(`retained terrarium-causal genome value ${index} is not serialized to eight decimals`)
    }
  }

  const seedTraining = requireRecord(seedArtifact.training, 'retained terrarium-causal seed.training')
  requireExactKeys(
    seedTraining,
    [
      'ablationGateGuardBandResults',
      'ablationGateResults',
      'algorithm',
      'allAblationGatesPassed',
      'allAblationGuardBandsPassed',
      'allSelectionGatesPassed',
      'allSelectionGuardBandsPassed',
      'combinedRecoveryMode',
      'derivationRecipe',
      'episodeSteps',
      'objectiveVersion',
      'publicationStatus',
      'publicRuntimeModel',
      'recipeSourceSha256',
      'retainedCandidateRecipeHash',
      'selectionGateGuardBandResults',
      'selectionGateResults',
    ],
    'retained terrarium-causal seed.training',
  )
  requireEqual(
    seedTraining.algorithm,
    'deterministic-coordinate-composition-retained-candidate',
    'retained terrarium-causal seed.training.algorithm',
  )
  requireEqual(seedTraining.episodeSteps, 480, 'retained terrarium-causal seed.training.episodeSteps')
  requireEqual(
    seedTraining.combinedRecoveryMode,
    'bounded-480',
    'retained terrarium-causal seed.training.combinedRecoveryMode',
  )
  requireEqual(seedTraining.publicRuntimeModel, false, 'retained terrarium-causal seed public flag')

  const retainedSourceHashes = {
    'training/wurmkickflip_rl/evolve_locomotion_policy.py':
      '6190e14bd1e887231abf96c95413057b6730db1b21086f635055cb8edf1a31a9',
    'training/wurmkickflip_rl/articulated_locomotion.py':
      '8892268f98d25839c4bb35d911e2def52efb50da65d43f2a21f1c8218929cf72',
    'training/wurmkickflip_rl/contracts.py':
      'ea7b182d9f753c3dcd6d569150adfd08496b3ec2af01bd6cb73d15c6be901a91',
    'contracts/locomotion-v2.json': '50acf6e504a4c98975f1983ed3710343d65045c818787108a885ee9cdd08f72e',
    'training/uv.lock': 'd1e602609b18ae9547455ec9d128b10356487e6665a698e3963090e5b6451819',
    'training/pyproject.toml': '701bd00f81d7bc36c2c3cc328120ebd544cdb3a70034238a3f153cd62f04a0fb',
  }
  requireDeepEqual(
    seedTraining.recipeSourceSha256,
    retainedSourceHashes,
    'retained terrarium-causal historical recipe source hashes',
  )

  const derivationRecipe = requireRecord(
    seedTraining.derivationRecipe,
    'retained terrarium-causal seed.training.derivationRecipe',
  )
  requireExactKeys(
    derivationRecipe,
    [
      'cleanCloneReconstruction',
      'coordinateDeltas',
      'diagnosticArtifact',
      'genomeDecimals',
      'historicalDiscoverySourceArtifact',
      'kind',
      'operation',
      'recipeSourceSha256',
      'schemaVersion',
    ],
    'retained terrarium-causal derivation',
  )
  requireEqual(derivationRecipe.schemaVersion, 1, 'retained terrarium-causal derivation.schemaVersion')
  requireEqual(
    derivationRecipe.kind,
    'wurmkickflip.deterministicCoordinateCompositionRecipe',
    'retained terrarium-causal derivation.kind',
  )
  requireEqual(
    derivationRecipe.operation,
    'copy the exact historical discovery genome, then add three bounded coordinate deltas',
    'retained terrarium-causal derivation.operation',
  )
  requireEqual(derivationRecipe.genomeDecimals, 8, 'retained terrarium-causal genome decimals')
  requireDeepEqual(
    derivationRecipe.recipeSourceSha256,
    retainedSourceHashes,
    'retained terrarium-causal derivation source hashes',
  )

  const historicalSource = requireRecord(
    derivationRecipe.historicalDiscoverySourceArtifact,
    'retained terrarium-causal historical discovery source',
  )
  requireExactKeys(
    historicalSource,
    [
      'canonicalGenomeRecipeHash',
      'declaredRecipe',
      'genomeDecimals',
      'modelVersion',
      'path',
      'recipeSourceSha256',
      'serializedGenome',
      'sha256',
      'summaryPath',
      'summarySha256',
      'warmStart',
    ],
    'retained terrarium-causal historical discovery source',
  )
  for (const [name, expected] of Object.entries({
    path: 'training/runs/locomotion_head_leading_v3/final-v3-windows-four-solve.json',
    sha256: '87b9e3444b1fa1366d21df0be4ea72f1441665424b96a196751c119f820bbe67',
    summaryPath: 'training/runs/locomotion_head_leading_v3/final-v3-windows-four-solve-summary.json',
    summarySha256: '6dab1afb5d4b396b6ea523c6a85c1bdaf1c288250a72bb6f8c67e1664aed79a9',
    modelVersion: 'locomotion-articulated-head-leading-es-v3',
    canonicalGenomeRecipeHash: 'd8ff50d99a66837ff218561fe2979acf85d84e0e470507bb9736d06473b6cf42',
    genomeDecimals: 8,
  })) {
    requireEqual(historicalSource[name], expected, `historical discovery source.${name}`)
  }
  requireDeepEqual(
    historicalSource.declaredRecipe,
    {
      seed: 20260737,
      generations: 10,
      populationSize: 64,
      eliteCount: 12,
      episodeSteps: 480,
      combinedRecoveryMode: 'bounded-480',
    },
    'historical discovery source declared recipe',
  )
  requireDeepEqual(
    historicalSource.warmStart,
    {
      sha256: '5a4e670acb45f69cdfc30ee9c2be5d4e12563e6b37d960264890f44658b656b8',
      modelVersion: 'locomotion-head-leading-es-v3-four-solve-swap-1-3-seed-v1',
    },
    'historical discovery source warm start',
  )
  requireDeepEqual(
    historicalSource.recipeSourceSha256,
    retainedSourceHashes,
    'historical discovery source hashes',
  )

  const historicalGenome = requireFiniteNumbers(
    historicalSource.serializedGenome,
    45,
    'historical discovery serialized genome',
  )
  const coordinateDeltas = [
    { genomeIndex: 24, section: 'weights.input', name: 'terrainFriction', delta: 0.1 },
    { genomeIndex: 40, section: 'weights.output', name: 'targetRight', delta: 0.05 },
    { genomeIndex: 43, section: 'weights.output', name: 'obstacleRight', delta: -0.01 },
  ]
  requireDeepEqual(
    derivationRecipe.coordinateDeltas,
    coordinateDeltas,
    'retained terrarium-causal coordinate deltas',
  )
  const reconstructedGenome = [...historicalGenome]
  for (const { genomeIndex, delta } of coordinateDeltas) {
    reconstructedGenome[genomeIndex] = Number((reconstructedGenome[genomeIndex] + delta).toFixed(8))
  }
  requireDeepEqual(reconstructedGenome, serializedGenome, 'retained terrarium-causal reconstructed genome')
  requireDeepEqual(
    derivationRecipe.diagnosticArtifact,
    {
      path: 'training/runs/locomotion_candidate_screen/exact-local/exact-p10-g43m01-g40p05.json',
      sha256: '420744ebb7a7860d45e463b448c700c8dd1bbdaae0640ac1ab6220a6e3932827',
      modelVersion: 'locomotion-diagnostic-exact-p10-g43m01-g40p05-v1',
    },
    'retained terrarium-causal diagnostic artifact',
  )

  const expectedRecipeHash = sha256Text(canonicalSortedJson({ derivationRecipe, serializedGenome }))
  requireEqual(
    seedTraining.retainedCandidateRecipeHash,
    expectedRecipeHash,
    'retained terrarium-causal recipe hash',
  )
  const passedSelectionGates = Object.fromEntries(V3_SELECTION_GATE_KEYS.map(name => [name, true]))
  const passedAblationGates = Object.fromEntries(V3_ABLATION_GATE_KEYS.map(name => [name, true]))
  for (const name of ['selectionGateResults', 'selectionGateGuardBandResults']) {
    requireDeepEqual(seedTraining[name], passedSelectionGates, `retained terrarium-causal seed.${name}`)
  }
  for (const name of ['ablationGateResults', 'ablationGateGuardBandResults']) {
    requireDeepEqual(seedTraining[name], passedAblationGates, `retained terrarium-causal seed.${name}`)
  }
  for (const name of [
    'allSelectionGatesPassed',
    'allSelectionGuardBandsPassed',
    'allAblationGatesPassed',
    'allAblationGuardBandsPassed',
  ]) {
    requireEqual(seedTraining[name], true, `retained terrarium-causal seed.${name}`)
  }

  requireExactKeys(
    seedSummary,
    [
      'derivationRecipe',
      'forecastStatus',
      'kind',
      'modelVersion',
      'publicRuntimeModel',
      'pythonHostDiagnostics',
      'retainedCandidateRecipeHash',
      'schemaVersion',
      'seed',
      'serializedGenome',
      'typescriptHostDiagnostics',
    ],
    'retained terrarium-causal seed summary',
  )
  requireEqual(seedSummary.schemaVersion, 1, 'retained terrarium-causal seed summary.schemaVersion')
  requireEqual(
    seedSummary.kind,
    'wurmkickflip.retainedTerrariumCausalCandidateSummary',
    'retained terrarium-causal seed summary.kind',
  )
  requireEqual(seedSummary.modelVersion, seedArtifact.modelVersion, 'retained seed summary model version')
  requireDeepEqual(seedSummary.serializedGenome, serializedGenome, 'retained seed summary genome')
  requireDeepEqual(seedSummary.derivationRecipe, derivationRecipe, 'retained seed summary derivation')
  requireEqual(
    seedSummary.retainedCandidateRecipeHash,
    expectedRecipeHash,
    'retained seed summary recipe hash',
  )
  requireDeepEqual(
    seedSummary.seed,
    {
      path: 'training/seeds/wurmkickflip_locomotion_terrarium_causal_candidate_warm_start_v3.json',
      sha256: headLeadingSeedSha256,
    },
    'retained seed summary identity',
  )
  requireEqual(seedSummary.publicRuntimeModel, false, 'retained seed summary public flag')
  const pythonDiagnostics = requireRecord(
    seedSummary.pythonHostDiagnostics,
    'retained terrarium-causal Python diagnostics',
  )
  for (const name of ['selectionGateResults', 'selectionGateGuardBandResults']) {
    requireDeepEqual(pythonDiagnostics[name], passedSelectionGates, `retained seed summary Python ${name}`)
  }
  for (const name of ['ablationGateResults', 'ablationGateGuardBandResults']) {
    requireDeepEqual(pythonDiagnostics[name], passedAblationGates, `retained seed summary Python ${name}`)
  }
  const typescriptDiagnostics = requireRecord(
    seedSummary.typescriptHostDiagnostics,
    'retained terrarium-causal TypeScript diagnostics',
  )
  requireDeepEqual(
    typescriptDiagnostics.strictPolicyContract,
    {
      passed: true,
      artifactSha256: '558d78d6fbe9bf4915d7bba71b332723101a8337597e87e33e8a561c64183117',
      canonicalGenomeRecipeHash: '184075f01dafb9e69696a954a25a1d1cf25e119ef65558d49e5bae4292551c28',
      recipeSourceTrainerSha256: '33e70cac45410b35cdbf299749bcb646c022eb6da4f652acdb4420662647b873',
      pairedHostArtifactBytesEqual: true,
      platforms: ['Windows', 'Linux'],
    },
    'retained seed summary strict policy contract',
  )
  for (const name of ['locomotion', 'gaitMicroscope', 'integratedMotion']) {
    requireEqual(
      requireRecord(typescriptDiagnostics[name], `retained seed summary TypeScript ${name}`).passed,
      true,
      `retained seed summary TypeScript ${name}.passed`,
    )
  }
}

function canonicalSortedJson(value) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value)
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error('canonical crossover JSON contains a non-finite number')
    return JSON.stringify(value)
  }
  if (Array.isArray(value)) return `[${value.map(canonicalSortedJson).join(',')}]`
  if (typeof value === 'object') {
    return `{${Object.keys(value)
      .sort()
      .map(key => `${JSON.stringify(key)}:${canonicalSortedJson(value[key])}`)
      .join(',')}}`
  }
  throw new Error(`unsupported canonical crossover JSON value: ${String(value)}`)
}

class PythonFloat {
  constructor(value) {
    this.value = value
  }
}

function pythonFloatTree(value) {
  if (typeof value === 'number') return new PythonFloat(value)
  if (Array.isArray(value)) return value.map(pythonFloatTree)
  if (typeof value === 'object' && value !== null) {
    return Object.fromEntries(Object.entries(value).map(([key, child]) => [key, pythonFloatTree(child)]))
  }
  return value
}

function pythonCanonicalJson(value) {
  if (value instanceof PythonFloat) return pythonFloatLiteral(value.value)
  if (value === null) return 'null'
  if (typeof value === 'string') return JSON.stringify(value)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'number') {
    if (!Number.isSafeInteger(value)) throw new Error(`canonical recipe integer is invalid: ${value}`)
    return String(value)
  }
  if (Array.isArray(value)) return `[${value.map(pythonCanonicalJson).join(',')}]`
  if (typeof value === 'object') {
    return `{${Object.keys(value)
      .sort()
      .map(key => `${JSON.stringify(key)}:${pythonCanonicalJson(value[key])}`)
      .join(',')}}`
  }
  throw new Error(`unsupported canonical JSON value: ${String(value)}`)
}

function pythonFloatLiteral(value) {
  if (!Number.isFinite(value)) throw new Error(`canonical recipe float is not finite: ${value}`)
  if (Object.is(value, -0)) return '-0.0'
  if (value === 0) return '0.0'

  const sign = value < 0 ? '-' : ''
  const magnitude = Math.abs(value)
  let literal = String(magnitude)
  if (!literal.includes('e') && magnitude < 0.0001) {
    const fraction = literal.slice(literal.indexOf('.') + 1)
    const firstSignificant = fraction.search(/[1-9]/u)
    const digits = fraction.slice(firstSignificant)
    literal = `${digits[0]}${digits.length > 1 ? `.${digits.slice(1)}` : ''}e${firstSignificant + 1}`
    return `${sign}${normalizePythonExponent(literal, true)}`
  }
  if (!literal.includes('e') && magnitude >= 1e16) literal = magnitude.toExponential()
  if (literal.includes('e')) return `${sign}${normalizePythonExponent(literal, false)}`
  return `${sign}${Number.isInteger(magnitude) ? `${literal}.0` : literal}`
}

function normalizePythonExponent(literal, negativeExponent) {
  const [mantissa, rawExponent] = literal.toLowerCase().split('e')
  const exponent = negativeExponent ? -Number(rawExponent) : Number(rawExponent)
  const exponentSign = exponent < 0 ? '-' : '+'
  return `${mantissa}e${exponentSign}${Math.abs(exponent).toString().padStart(2, '0')}`
}

function requireRecord(value, label) {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object`)
  }
  return value
}

function requireFiniteNumbers(value, expectedLength, label) {
  if (
    !Array.isArray(value) ||
    value.length !== expectedLength ||
    value.some(entry => typeof entry !== 'number' || !Number.isFinite(entry))
  ) {
    throw new Error(`${label} must contain ${expectedLength} finite numbers`)
  }
  return value
}

function requireExactKeys(value, expectedKeys, label) {
  const actual = Object.keys(value).sort()
  const expected = [...expectedKeys].sort()
  if (!isDeepStrictEqual(actual, expected)) {
    throw new Error(`${label} keys mismatch: expected ${expected.join(', ')}, got ${actual.join(', ')}`)
  }
}

function requireEqual(actual, expected, label) {
  if (actual !== expected) throw new Error(`${label} mismatch: expected ${expected}, got ${String(actual)}`)
}

function requireDeepEqual(actual, expected, label) {
  if (!isDeepStrictEqual(actual, expected)) throw new Error(`${label} does not match the V3 contract`)
}

function canonicalLineEndings(value) {
  return value.replace(/\r\n?/gu, '\n')
}

function sha256Text(value) {
  return createHash('sha256').update(value, 'utf8').digest('hex')
}

try {
  verifyRetainedCrossoverSeed(retainedCrossoverSeed, retainedCrossoverSeedSummary)
  verifyRetainedTerrariumCausalSeed(headLeadingSeed, headLeadingSeedSummary)
} catch (error) {
  failures.push(error instanceof Error ? error.message : String(error))
}

let publishedV3ContractValid = true
try {
  verifyPublishedV3Artifact(artifact, recipeSourceSha256, headLeadingSeed)
} catch (error) {
  publishedV3ContractValid = false
  failures.push(error instanceof Error ? error.message : String(error))
}
if (publishedV3ContractValid) {
  runV3ContractMutationChecks(artifact, recipeSourceSha256, headLeadingSeed)
}

if (failures.length > 0) {
  console.error('Policy contract verification failed:')
  for (const failure of failures) console.error(`- ${failure}`)
  process.exit(1)
}

console.log('Policy contracts verified from the canonical articulated-contact-v2 JSON contract.')
