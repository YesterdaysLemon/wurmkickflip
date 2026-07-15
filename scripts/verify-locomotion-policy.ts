import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { copyFile, mkdir, mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
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
  traceHash: string
}

const root = resolve(import.meta.dirname, '..')
const trainingRoot = resolve(root, 'training')
const artifactSource = process.env.WURMKICKFLIP_LOCOMOTION_ARTIFACT
  ? resolve(process.env.WURMKICKFLIP_LOCOMOTION_ARTIFACT)
  : new URL('../public/models/wurmkickflip_locomotion_policy.json', import.meta.url)
const legacyUrl = new URL('../training/seeds/wurmkickflip_locomotion_warm_start_v1.json', import.meta.url)
const trainerUrl = new URL('../training/wurmkickflip_rl/evolve_locomotion_policy.py', import.meta.url)
const rawV2 = JSON.parse(await readFile(artifactSource, 'utf8')) as any
const rawV1 = JSON.parse(await readFile(legacyUrl, 'utf8')) as any
const artifact = parseLocomotionPolicy(rawV2)
const trainingSource = await readFile(trainerUrl, 'utf8')

verifyContract(artifact, rawV2, trainingSource)
verifyLegacyMigration(rawV1)
verifyMalformedArtifacts(rawV2, rawV1)
verifyAntagonisticOutputs(artifact)
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

const trainingRepro = await verifyTrainingReproducibility()

console.log(
  JSON.stringify(
    {
      modelVersion: artifact.modelVersion,
      plantVersion: artifact.plant.version,
      targetProgress: round(full.progress),
      meanDisplacement: round(full.displacement),
      perScenarioProgress: fullRollouts.map(rollout => round(rollout.progress)),
      recurrentAblations: {
        zero: round(zero.progress),
        frozen: round(frozen.progress),
        shuffled: round(shuffled.progress),
      },
      commandVariation: {
        temporal: round(full.temporalStd),
        spatial: round(full.spatialStd),
      },
      zeroFrictionComDisplacement: frictionless.displacement,
      steeringFinalZ: { left: round(left.finalZ), right: round(right.finalZ) },
      steeringActionDelta: round(steeringActionDelta),
      localSensorDelta: round(localSensorDelta),
      deterministicTraceHash: full.traceHash,
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
  expect(
    training.objectiveVersion === 'articulated-contact-obstacle-recovery-v2',
    'artifact objectiveVersion mismatch',
  )
  expect(training.actuatorPrecision === 'float32-plant-command', 'artifact actuator precision mismatch')
  expect(training.seed === 20260721, 'published refinement seed mismatch')
  expect(training.generations === 10, 'published refinement generation count mismatch')
  expect(training.populationSize === 64, 'published refinement population size mismatch')
  expect(training.eliteCount === 12, 'published refinement elite count mismatch')
  expect(training.episodeSteps === 480, 'published refinement episode length mismatch')
  expect(training.scenarioCount === 12, 'published scenario count mismatch')
  expect(
    exactArray(training.domainRandomization, [
      'obstacles',
      'spatial-friction',
      'body-scale',
      'target-switches',
      'contact-loss',
    ]),
    'published domain randomization contract mismatch',
  )
  const warmStart = recordValue(training.warmStart, 'training.warmStart')
  expect(
    exactArray(Object.keys(warmStart).sort(), ['modelVersion', 'sha256']),
    'published warmStart metadata must be location-independent',
  )
  expect(/^[a-f\d]{64}$/u.test(String(warmStart.sha256)), 'warmStart.sha256 must be a SHA-256 digest')
  expect(typeof warmStart.modelVersion === 'string', 'warmStart.modelVersion must be recorded')
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
  const frozen = new Float32Array(ACTION_SIZE)
  const shuffle = [0, 9, 2, 13, 4, 15, 6, 11, 8, 1, 10, 3, 12, 5, 14, 7]
  const commands: number[][] = []
  const checkpoints: unknown[] = []
  const checkpointSteps = new Set([0, 1, 2, 23, 24, 59, 119, 239, steps - 1])
  let pathLength = 0
  let contactTotal = 0
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
    const applied = new Float32Array(ACTION_SIZE)
    if (mode === 'zero') {
      applied.fill(0)
    } else if (mode === 'frozen') {
      if (step === 24) frozen.set(neuralAction)
      applied.set(step >= 24 ? frozen : neuralAction)
    } else if (mode === 'shuffled') {
      for (let segment = 0; segment < SEGMENT_COUNT; segment += 1) {
        const source = shuffle[segment]
        applied[segment * 2] = neuralAction[source * 2]
        applied[segment * 2 + 1] = neuralAction[source * 2 + 1]
      }
    } else {
      applied.set(neuralAction)
    }
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
