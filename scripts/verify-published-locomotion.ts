import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { copyFile, mkdir, mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { isDeepStrictEqual } from 'node:util'

type JsonRecord = Record<string, unknown>

const root = resolve(import.meta.dirname, '..')
const trainingRoot = resolve(root, 'training')
const retainedSeedSha256 = '0c46518c424528337f3a2ad5a96912b7812dda473688d468556023be095551ba'
const retainedSeedModelVersion = 'locomotion-head-leading-es-v3-terrarium-causal-candidate-v1'
const retainedSeedPath = 'seeds/wurmkickflip_locomotion_terrarium_causal_candidate_warm_start_v3.json'
const v3RecipeSourcePaths = [
  'training/wurmkickflip_rl/evolve_locomotion_policy.py',
  'training/wurmkickflip_rl/articulated_locomotion.py',
  'training/wurmkickflip_rl/contracts.py',
  'contracts/locomotion-v2.json',
  'training/uv.lock',
  'training/pyproject.toml',
] as const

const v3ArtifactKeys = [
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
] as const

const v3TrainingKeys = [
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
] as const

const v3SelectionGateKeys = [
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
] as const

const v3AblationGateKeys = [
  'fullBeatsFrozen',
  'fullBeatsShuffled',
  'fullProgress',
  'noFrictionConservesCom',
  'zeroActionStationary',
] as const

const v3SensorClamps = {
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

const v3SelectionMargins = {
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

const v3DomainRandomization = [
  'obstacles',
  'spatial-friction',
  'body-scale',
  'target-switches',
  'contact-loss',
  'transient-slick-window',
  'external-body-shove',
  'paired-shove-numb-low-traction-proxy',
]

async function main() {
  const retainedWorkspace = process.env.WURMKICKFLIP_REPRO_WORKSPACE
  const workspace = retainedWorkspace
    ? resolve(retainedWorkspace)
    : await mkdtemp(join(tmpdir(), 'wurmkickflip-published-locomotion-'))
  if (retainedWorkspace) await mkdir(workspace, { recursive: false })

  try {
    const stagedSeedDirectory = join(workspace, 'seeds')
    await mkdir(stagedSeedDirectory, { recursive: true })
    const stagedSeed = join(stagedSeedDirectory, 'head-leading-v3.json')
    await copyFile(resolve(trainingRoot, retainedSeedPath), stagedSeed)

    const reproducedOutput = join(workspace, 'published-model.json')
    const reproducedSummary = join(workspace, 'published-summary.json')
    runEvolution([
      '--objective-version',
      'v3',
      '--combined-recovery-mode',
      'bounded-480',
      '--seed',
      '20260737',
      '--generations',
      '1',
      '--population-size',
      '4',
      '--elite-count',
      '1',
      '--episode-steps',
      '480',
      '--model-version',
      'locomotion-articulated-head-leading-es-v3',
      '--warm-start',
      stagedSeed,
      '--out',
      reproducedOutput,
      '--summary',
      reproducedSummary,
    ])

    const trackedPath = resolve(root, 'public/models/wurmkickflip_locomotion_policy.json')
    const reproducedBytes = await readFile(reproducedOutput)
    const trackedBytes = await readFile(trackedPath)
    const reproduced = asRecord(JSON.parse(reproducedBytes.toString('utf8')), 'reproduced artifact')
    const tracked = asRecord(JSON.parse(trackedBytes.toString('utf8')), 'tracked artifact')

    const sourceHashes = await canonicalRecipeSourceSha256()
    verifyPublishedV3Artifact(reproduced, 'reproduced artifact', sourceHashes)
    verifyPublishedV3Artifact(tracked, 'tracked artifact', sourceHashes)

    const seedBytes = await readFile(stagedSeed)
    const refinedWarmStart = asRecord(
      asRecord(reproduced.training, 'reproduced.training').warmStart,
      'reproduced.training.warmStart',
    )
    const seedSha256 = createHash('sha256').update(canonicalLineEndings(seedBytes)).digest('hex')
    if (seedSha256 !== retainedSeedSha256) {
      throw new Error(`retained search seed SHA changed: expected ${retainedSeedSha256}, got ${seedSha256}`)
    }
    if (refinedWarmStart.sha256 !== seedSha256) {
      throw new Error('refined artifact warmStart.sha256 does not identify the retained search seed')
    }
    if (refinedWarmStart.modelVersion !== retainedSeedModelVersion) {
      throw new Error('refined artifact warmStart.modelVersion does not identify the retained search seed')
    }

    if (!trackedBytes.equals(reproducedBytes)) {
      const trackedWarm = asRecord(
        asRecord(tracked.training, 'tracked.training').warmStart,
        'tracked.training.warmStart',
      )
      throw new Error(
        [
          'published locomotion artifact is not reproduced by the documented retained-seed refinement',
          `tracked warm SHA: ${String(trackedWarm?.sha256 ?? 'missing')}`,
          `reproduced warm SHA: ${String(refinedWarmStart.sha256)}`,
          `tracked artifact SHA: ${sha256(trackedBytes)}`,
          `reproduced artifact SHA: ${sha256(reproducedBytes)}`,
          'Publish only after reviewing the newly reproduced artifact with the locomotion and gait audits.',
        ].join('\n'),
      )
    }

    console.log(
      JSON.stringify(
        {
          modelVersion: tracked.modelVersion,
          artifactSha256: sha256(trackedBytes),
          refinement: {
            seed: 20260737,
            generations: 1,
            populationSize: 4,
            eliteCount: 1,
            episodeSteps: 480,
            combinedRecoveryMode: 'bounded-480',
          },
          warmStartSha256: refinedWarmStart.sha256,
        },
        null,
        2,
      ),
    )
    console.log('Published head-leading evolution reproduced exactly in an isolated workspace.')
  } finally {
    if (!retainedWorkspace) await rm(workspace, { recursive: true, force: true })
  }
}

function runEvolution(arguments_: string[]) {
  const result = spawnSync(
    'uv',
    ['run', '--locked', 'python', '-m', 'wurmkickflip_rl.evolve_locomotion_policy', ...arguments_],
    { cwd: trainingRoot, encoding: 'utf8', maxBuffer: 8 * 1024 * 1024 },
  )
  if (result.status !== 0) {
    throw new Error(`published locomotion evolution failed:\n${result.stderr || result.stdout}`)
  }
}

function verifyPublishedV3Artifact(
  artifact: JsonRecord,
  label: string,
  expectedSourceHashes: Record<string, string>,
) {
  requireExactKeys(artifact, v3ArtifactKeys, label)
  requireEqual(artifact.schemaVersion, 2, `${label}.schemaVersion`)
  requireEqual(artifact.kind, 'wurmkickflip.locomotionPolicy', `${label}.kind`)
  requireEqual(artifact.architecture, 'segmental-recurrent-tanh', `${label}.architecture`)
  requireEqual(artifact.segmentCount, 16, `${label}.segmentCount`)
  requireEqual(artifact.actionSize, 32, `${label}.actionSize`)
  requireEqual(artifact.modelVersion, 'locomotion-articulated-head-leading-es-v3', `${label}.modelVersion`)
  const training = asRecord(artifact.training, `${label}.training`)
  requireExactKeys(training, v3TrainingKeys, `${label}.training`)

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
    requireEqual(training[name], expected, `${label}.training.${name}`)
  }

  requireDeepEqual(
    training.domainRandomization,
    v3DomainRandomization,
    `${label}.training.domainRandomization`,
  )
  requireDeepEqual(
    training.controllerExposedTimingSignals,
    [],
    `${label}.training.controllerExposedTimingSignals`,
  )
  requireDeepEqual(
    training.controllerSensorClamps,
    v3SensorClamps,
    `${label}.training.controllerSensorClamps`,
  )
  requireDeepEqual(training.selectionMargins, v3SelectionMargins, `${label}.training.selectionMargins`)
  requireDeepEqual(
    training.headLeadingMeasurement,
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
    `${label}.training.headLeadingMeasurement`,
  )
  requireDeepEqual(
    training.perturbationSchedules,
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
    `${label}.training.perturbationSchedules`,
  )
  requireDeepEqual(
    training.recoveryMeasurement,
    {
      speed: 'positive head-axis speed multiplied by positive target-facing alignment',
      ratio: 'post-window mean speed / max(pre-window mean speed, 0.04 m/s)',
      time: 'first post-event step whose 0.18-EMA speed reaches max(55% of pre-window speed, 0.04 m/s) while target alignment is at least 0.1',
    },
    `${label}.training.recoveryMeasurement`,
  )
  requireDeepEqual(
    training.localCausalProbe,
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
    `${label}.training.localCausalProbe`,
  )
  requireDeepEqual(
    training.ablationProtocol,
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
    `${label}.training.ablationProtocol`,
  )

  const passedSelectionGates = Object.fromEntries(v3SelectionGateKeys.map(name => [name, true]))
  const passedAblationGates = Object.fromEntries(v3AblationGateKeys.map(name => [name, true]))
  for (const name of ['selectionGateResults', 'selectionGateGuardBandResults']) {
    requireDeepEqual(training[name], passedSelectionGates, `${label}.training.${name}`)
  }
  for (const name of ['ablationGateResults', 'ablationGateGuardBandResults']) {
    requireDeepEqual(training[name], passedAblationGates, `${label}.training.${name}`)
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
    requireEqual(training[name], true, `${label}.training.${name}`)
  }

  const warmStart = asRecord(training.warmStart, `${label}.training.warmStart`)
  requireExactKeys(warmStart, ['modelVersion', 'sha256'], `${label}.training.warmStart`)
  requireEqual(warmStart.modelVersion, retainedSeedModelVersion, `${label}.training.warmStart.modelVersion`)
  requireEqual(warmStart.sha256, retainedSeedSha256, `${label}.training.warmStart.sha256`)
  requireDeepEqual(
    training.pairedHostReproduction,
    {
      requiredForPublication: true,
      compared: ['serializedGenome', 'artifactBytes'],
      trainerAttestation: 'none',
    },
    `${label}.training.pairedHostReproduction`,
  )

  const sourceHashes = asRecord(training.recipeSourceSha256, `${label}.training.recipeSourceSha256`)
  requireExactKeys(sourceHashes, v3RecipeSourcePaths, `${label}.training.recipeSourceSha256`)
  for (const path of v3RecipeSourcePaths) {
    if (!/^[a-f\d]{64}$/u.test(String(sourceHashes[path]))) {
      throw new Error(`${label}.training.recipeSourceSha256[${JSON.stringify(path)}] is malformed`)
    }
    requireEqual(
      sourceHashes[path],
      expectedSourceHashes[path],
      `${label}.training.recipeSourceSha256[${JSON.stringify(path)}]`,
    )
  }

  if (!/^[a-f\d]{64}$/u.test(String(training.canonicalGenomeRecipeHash))) {
    throw new Error(`${label}.training.canonicalGenomeRecipeHash is malformed`)
  }
  requireEqual(
    training.canonicalGenomeRecipeHash,
    canonicalGenomeRecipeHash(artifact, training),
    `${label}.training.canonicalGenomeRecipeHash`,
  )
}

async function canonicalRecipeSourceSha256() {
  return Object.fromEntries(
    await Promise.all(
      v3RecipeSourcePaths.map(async path => {
        const source = await readFile(resolve(root, path))
        return [path, sha256(canonicalLineEndings(source))]
      }),
    ),
  )
}

function canonicalGenomeRecipeHash(artifact: JsonRecord, training: JsonRecord) {
  const weights = asRecord(artifact.weights, 'artifact.weights')
  const genome = [
    ...asFiniteNumbers(artifact.initialState, 16, 'artifact.initialState'),
    ...asFiniteNumbers(weights.input, 17, 'artifact.weights.input'),
    ...asFiniteNumbers(weights.recurrent, 3, 'artifact.weights.recurrent'),
    ...asFiniteNumbers(weights.output, 9, 'artifact.weights.output'),
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
    modelVersion: artifact.modelVersion,
    sourceSha256: training.recipeSourceSha256,
    recipe: {
      algorithm: training.algorithm,
      objectiveVersion: training.objectiveVersion,
      seed: training.seed,
      generations: training.generations,
      populationSize: training.populationSize,
      eliteCount: training.eliteCount,
      episodeSteps: training.episodeSteps,
      combinedRecoveryMode: training.combinedRecoveryMode,
      warmStart: training.warmStart,
      genomeDecimals: training.genomeDecimals,
      selectionFitnessDecimals: training.selectionFitnessDecimals,
      gateDiagnosticDecimals: training.gateDiagnosticDecimals,
      controllerSensorClamps: pythonFloatTree(training.controllerSensorClamps),
      selectionMargins: pythonFloatTree(training.selectionMargins),
    },
  }
  return createHash('sha256').update(pythonCanonicalJson(payload), 'utf8').digest('hex')
}

class PythonFloat {
  readonly value: number

  constructor(value: number) {
    this.value = value
  }
}

function pythonFloatTree(value: unknown): unknown {
  if (typeof value === 'number') return new PythonFloat(value)
  if (Array.isArray(value)) return value.map(pythonFloatTree)
  if (typeof value === 'object' && value !== null) {
    return Object.fromEntries(Object.entries(value).map(([key, child]) => [key, pythonFloatTree(child)]))
  }
  return value
}

function pythonCanonicalJson(value: unknown): string {
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
    const values = value as JsonRecord
    return `{${Object.keys(values)
      .sort()
      .map(key => `${JSON.stringify(key)}:${pythonCanonicalJson(values[key])}`)
      .join(',')}}`
  }
  throw new Error(`unsupported canonical JSON value: ${String(value)}`)
}

function pythonFloatLiteral(value: number) {
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

function normalizePythonExponent(literal: string, negativeExponent: boolean) {
  const [mantissa, rawExponent] = literal.toLowerCase().split('e')
  const exponent = negativeExponent ? -Number(rawExponent) : Number(rawExponent)
  const exponentSign = exponent < 0 ? '-' : '+'
  return `${mantissa}e${exponentSign}${Math.abs(exponent).toString().padStart(2, '0')}`
}

function asFiniteNumbers(value: unknown, expectedLength: number, label: string): number[] {
  if (
    !Array.isArray(value) ||
    value.length !== expectedLength ||
    value.some(entry => typeof entry !== 'number' || !Number.isFinite(entry))
  ) {
    throw new Error(`${label} must contain ${expectedLength} finite numbers`)
  }
  return value as number[]
}

function requireExactKeys(value: JsonRecord, expectedKeys: readonly string[], label: string) {
  const actual = Object.keys(value).sort()
  const expected = [...expectedKeys].sort()
  if (!isDeepStrictEqual(actual, expected)) {
    throw new Error(`${label} keys mismatch: expected ${expected.join(', ')}, got ${actual.join(', ')}`)
  }
}

function requireEqual(actual: unknown, expected: unknown, label: string) {
  if (actual !== expected)
    throw new Error(`${label} mismatch: expected ${String(expected)}, got ${String(actual)}`)
}

function requireDeepEqual(actual: unknown, expected: unknown, label: string) {
  if (!isDeepStrictEqual(actual, expected)) throw new Error(`${label} does not match the V3 contract`)
}

function canonicalLineEndings(value: Buffer) {
  return Buffer.from(value.toString('utf8').replace(/\r\n?/gu, '\n'))
}

function sha256(value: Buffer) {
  return createHash('sha256').update(value).digest('hex')
}

function asRecord(value: unknown, label: string): JsonRecord {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object`)
  }
  return value as JsonRecord
}

await main()
