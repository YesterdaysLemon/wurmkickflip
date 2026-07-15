import { createHash } from 'node:crypto'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { spawnSync } from 'node:child_process'
import {
  EvolvedLocomotionPolicy,
  parseLocomotionPolicy,
  type LocomotionPolicyArtifact,
  type LocomotionSensors,
} from '../src/policy/locomotionPolicy'
import { ACTION_SIZE, SEGMENT_COUNT } from '../src/policy/types'
import { createWormLocomotionPlant, deriveWormLocalPose, stepWormLocomotion } from '../src/scene/wormLocomotion'
import type { TerrainField } from '../src/scene/terrainField'

type Mode = 'full' | 'zero' | 'frozen' | 'shuffled'

type Scenario = {
  targetX: number
  targetZ: number
  friction: number
  plantFriction?: number
  urgency: number
}

type Rollout = {
  progress: number
  displacement: number
  temporalStd: number
  spatialStd: number
  finalX: number
  finalZ: number
  traceHash: string
}

type ParityCheckpoint = {
  step: number
  position: number[]
  heading: number
  forwardSpeed: number
  angularSpeed: number
  joints: number[]
  jointVelocities: number[]
  commands: number[]
}

const root = resolve(import.meta.dirname, '..')
const artifactUrl = new URL('../public/models/wurmkickflip_locomotion_policy.json', import.meta.url)
const sourceUrl = new URL('../training/wurmkickflip_rl/evolve_locomotion_policy.py', import.meta.url)
const warmStartUrl = new URL('../training/seeds/wurmkickflip_locomotion_warm_start_v1.json', import.meta.url)
const artifact = parseLocomotionPolicy(JSON.parse(await readFile(artifactUrl, 'utf8')))
const trainingSource = await readFile(sourceUrl, 'utf8')
const warmStartPayload = await readFile(warmStartUrl)
const canonicalWarmStartPayload = Buffer.from(warmStartPayload.toString('utf8').replace(/\r\n?/g, '\n'))

const scenarios: Scenario[] = [
  { targetX: 4.2, targetZ: 0, friction: 0.95, urgency: 0.72 },
  { targetX: 3.2, targetZ: 3.2, friction: 0.58, urgency: 0.88 },
  { targetX: 3.2, targetZ: -3.2, friction: 1.12, urgency: 0.64 },
  { targetX: 0.5, targetZ: 4.3, friction: 0.82, urgency: 1 },
  { targetX: 0.5, targetZ: -4.3, friction: 0.42, urgency: 0.92 },
  { targetX: -2.2, targetZ: 3.5, friction: 1.05, urgency: 0.78 },
  { targetX: -2.2, targetZ: -3.5, friction: 0.67, urgency: 0.84 },
  { targetX: 4.8, targetZ: 1.2, friction: 0.33, urgency: 1 },
]

verifyContract(artifact, trainingSource)
verifyPlantContractRejection(artifact)
verifyAntagonisticOutputs(artifact)

const steps = finiteInteger(artifact.training.episodeSteps, 'training.episodeSteps')
const fullRollouts = scenarios.map((scenario) => runScenario(artifact, scenario, steps, 'full'))
const full = aggregate(fullRollouts)
const repeated = aggregate(scenarios.map((scenario) => runScenario(artifact, scenario, steps, 'full')))
const zero = aggregate(scenarios.map((scenario) => runScenario(artifact, scenario, steps, 'zero')))
const frozen = aggregate(scenarios.map((scenario) => runScenario(artifact, scenario, steps, 'frozen')))
const shuffled = aggregate(scenarios.map((scenario) => runScenario(artifact, scenario, steps, 'shuffled')))
const noFriction = aggregate(
  scenarios.map((scenario) => runScenario(artifact, { ...scenario, plantFriction: 0 }, steps, 'full')),
)

expect(full.traceHash === repeated.traceHash, 'runtime locomotion trace must be deterministic')
expect(full.progress > 0.8, `evolved controller target progress is too small: ${full.progress}`)
expect(
  Math.min(...fullRollouts.map((rollout) => rollout.progress)) > 0.12,
  'evolved controller must make positive progress from every published approach direction',
)
const reachedScenarioCount = fullRollouts.filter(
  (rollout, index) => Math.hypot(scenarios[index].targetX, scenarios[index].targetZ) - rollout.progress < 0.72,
).length
expect(reachedScenarioCount >= 2, 'evolved controller reached too few published target scenarios')
expect(full.progress > zero.progress + 0.8, 'evolved controller must beat zero-action progress')
expect(full.progress > frozen.progress + 0.8, 'evolved controller must beat a frozen-pose ablation')
expect(full.progress > shuffled.progress * 1.2, 'evolved controller must beat deterministic segment shuffling')
expect(noFriction.displacement < 1e-10, 'zero traction must prevent actuator-driven translation from rest')
expect(noFriction.temporalStd > 0.25, 'zero-traction ablation must retain active neural commands')
expect(Math.abs(zero.displacement) < 1e-12, 'zero actions must leave the plant exactly stationary')
expect(full.temporalStd > 0.25, 'recurrent actuators must vary over time')
expect(full.spatialStd > 0.18, 'segments must produce meaningfully different simultaneous activations')

const left = runScenario(artifact, { targetX: 3.2, targetZ: 3.2, friction: 0.8, urgency: 0.9 }, steps, 'full')
const right = runScenario(artifact, { targetX: 3.2, targetZ: -3.2, friction: 0.8, urgency: 0.9 }, steps, 'full')
expect(left.finalZ > 0.15, 'positive local-right target must produce positive steering displacement')
expect(right.finalZ < -0.15, 'negative local-right target must produce negative steering displacement')
const browserPlant = runBrowserPlantScenario(
  artifact,
  { targetX: 3.2, targetZ: 3.2, friction: 0.8, urgency: 0.9 },
  steps,
)
const pythonParity = readPythonParityTrace()
const parityMaximumDelta = compareParityCheckpoints(browserPlant.checkpoints, pythonParity.checkpoints)
expect(Math.abs(browserPlant.finalX - left.finalX) < 1e-5, 'browser plant x motion drifted from evolved plant contract')
expect(Math.abs(browserPlant.finalZ - left.finalZ) < 1e-5, 'browser plant z motion drifted from evolved plant contract')
expect(browserPlant.maximumLateralSpan > 0.12, 'evolved joint motion must visibly articulate the body chain')
expect(parityMaximumDelta < 0.01, `Python/browser checkpoint parity drifted by ${parityMaximumDelta}`)

await verifyTrainingReproducibility()

console.log(
  JSON.stringify(
    {
      modelVersion: artifact.modelVersion,
      architecture: artifact.architecture,
      training: {
        algorithm: artifact.training.algorithm,
        seed: artifact.training.seed,
        generations: artifact.training.generations,
        populationSize: artifact.training.populationSize,
      },
      targetProgress: round(full.progress),
      perScenarioProgress: fullRollouts.map((rollout) => round(rollout.progress)),
      reachedScenarioCount,
      actuatorAblations: {
        zero: round(zero.progress),
        frozen: round(frozen.progress),
        shuffled: round(shuffled.progress),
        noFrictionDisplacement: round(noFriction.displacement),
      },
      actuatorTemporalStd: round(full.temporalStd),
      actuatorSpatialStd: round(full.spatialStd),
      steeringFinalZ: { left: round(left.finalZ), right: round(right.finalZ) },
      browserPoseLateralSpan: round(browserPlant.maximumLateralSpan),
      pythonBrowserParityMaximumDelta: round(parityMaximumDelta),
      deterministicTraceHash: full.traceHash,
    },
    null,
    2,
  ),
)
console.log('Evolved locomotion policy verification passed.')

function verifyContract(model: LocomotionPolicyArtifact, source: string) {
  expect(model.segmentCount === SEGMENT_COUNT, `artifact must own ${SEGMENT_COUNT} segment neurons`)
  expect(model.actionSize === ACTION_SIZE, `artifact must emit ${ACTION_SIZE} muscle activations`)
  expect(model.architecture === 'segmental-recurrent-tanh', 'artifact must use the recurrent segmental architecture')
  expect(
    !model.sensorNames.some((name) => /time|clock|phase|cycle/i.test(name)),
    'locomotion sensors must not expose a clock or gait phase',
  )
  expect(model.sensorNames.includes('targetRight'), 'locomotion controller must sense steering error')
  expect(model.sensorNames.includes('terrainFriction'), 'locomotion controller must sense terrain friction')
  expect(model.sensorNames.includes('urgency'), 'locomotion controller must sense need urgency')
  expect(model.training.algorithm === 'elitist-mutation-evolution', 'artifact must record evolutionary training')
  expect(
    model.training.objectiveVersion === 'risk-sensitive-bottom-two-v1',
    'published artifact must record the risk-sensitive objective',
  )
  expect(
    model.training.actuatorPrecision === 'float32-plant-command',
    'published artifact must record browser-equivalent plant action precision',
  )
  const warmStart = recordValue(model.training.warmStart, 'training.warmStart')
  expect(
    warmStart.sha256 === createHash('sha256').update(canonicalWarmStartPayload).digest('hex'),
    'published warm-start hash does not match the tracked base artifact',
  )
  expect(warmStart.modelVersion === 'locomotion-segmental-es-v1', 'unexpected warm-start model version')
  expect(finiteInteger(model.training.generations, 'training.generations') >= 50, 'published model needs a substantial evolution run')
  expect(finiteInteger(model.training.populationSize, 'training.populationSize') >= 64, 'published model population is too small')
  expect(!/np\.(?:sin|cos)\s*\(/.test(source), 'trainer must not contain a trigonometric gait generator')
  expect(!/cpg_action|wave_frequency|phase_offset/.test(source), 'trainer must not evolve a disguised CPG recipe')
  expect(/wave_work/.test(source), 'trainer must derive translation from inter-segment actuator work')
  expect(/bottom_two_progress/.test(source), 'trainer must retain its risk-sensitive worst-case objective')
  expect(/--warm-start/.test(source), 'trainer must retain explicit warm-start provenance')
  expect(/astype\(np\.float32\)/.test(source), 'trainer must quantize plant-bound actions like the browser')
}

function verifyPlantContractRejection(model: LocomotionPolicyArtifact) {
  const changedPlant = {
    ...model,
    plant: { ...model.plant, jointStiffness: model.plant.jointStiffness + 0.25 },
  }
  let rejected = false
  try {
    parseLocomotionPolicy(changedPlant)
  } catch {
    rejected = true
  }
  expect(rejected, 'artifact parsing must reject plant constants that diverge from the browser contract')
}

function verifyAntagonisticOutputs(model: LocomotionPolicyArtifact) {
  const policy = new EvolvedLocomotionPolicy(model)
  const action = policy.run(
    {
      targetForward: 0.6,
      targetRight: -0.35,
      targetDistance: 0.8,
      forwardSpeed: 0,
      angularSpeed: 0,
      terrainFriction: 0.9,
      urgency: 0.75,
    },
    new Float64Array(SEGMENT_COUNT),
    new Float64Array(SEGMENT_COUNT),
  )
  expect(action.length === ACTION_SIZE, 'runtime action length changed')
  expect(Array.from(action).every(Number.isFinite), 'runtime actions must be finite')
  for (let segment = 0; segment < SEGMENT_COUNT; segment += 1) {
    expect(
      Math.abs(action[segment * 2] + action[segment * 2 + 1]) < 1e-7,
      `segment ${segment} must own an antagonistic actuator pair`,
    )
  }
}

function runScenario(model: LocomotionPolicyArtifact, scenario: Scenario, steps: number, mode: Mode): Rollout {
  const policy = new EvolvedLocomotionPolicy(model)
  const joints = new Float64Array(SEGMENT_COUNT)
  const jointVelocities = new Float64Array(SEGMENT_COUNT)
  const command = new Float64Array(SEGMENT_COUNT)
  const frozen = new Float64Array(SEGMENT_COUNT)
  const shuffle = [0, 9, 2, 13, 4, 15, 6, 11, 8, 1, 10, 3, 12, 5, 14, 7]
  const traces: number[][] = []
  const initialDistance = Math.hypot(scenario.targetX, scenario.targetZ)
  let x = 0
  let z = 0
  let forwardX = 1
  let forwardZ = 0
  let forwardSpeed = 0
  let angularSpeed = 0
  let pathLength = 0

  for (let step = 0; step < steps; step += 1) {
    const dx = scenario.targetX - x
    const dz = scenario.targetZ - z
    const distance = Math.max(1e-9, Math.hypot(dx, dz))
    const sensors: LocomotionSensors = {
      targetForward: (dx * forwardX + dz * forwardZ) / distance,
      targetRight: (dx * -forwardZ + dz * forwardX) / distance,
      targetDistance: Math.min(distance / 5, 1.5),
      forwardSpeed,
      angularSpeed,
      terrainFriction: scenario.friction,
      urgency: scenario.urgency,
    }
    const neuralAction = policy.run(sensors, joints, jointVelocities)
    for (let segment = 0; segment < SEGMENT_COUNT; segment += 1) {
      const neuralCommand = neuralAction[segment * 2]
      if (mode === 'zero') command[segment] = 0
      else if (mode === 'frozen' && step >= 24) command[segment] = frozen[segment]
      else if (mode === 'shuffled') command[segment] = neuralAction[shuffle[segment] * 2]
      else command[segment] = neuralCommand
      if (mode === 'frozen' && step === 24) frozen[segment] = neuralCommand
    }

    const dt = model.plant.timestep
    for (let segment = 0; segment < SEGMENT_COUNT; segment += 1) {
      jointVelocities[segment] +=
        ((command[segment] - joints[segment]) * model.plant.jointStiffness -
          jointVelocities[segment] * model.plant.jointDamping) *
        dt
      joints[segment] = clamp(
        joints[segment] + jointVelocities[segment] * dt,
        -model.plant.jointLimit,
        model.plant.jointLimit,
      )
    }

    let waveWork = 0
    for (let segment = 0; segment + 1 < SEGMENT_COUNT; segment += 1) {
      waveWork +=
        (joints[segment + 1] - joints[segment]) *
        0.5 *
        (jointVelocities[segment + 1] + jointVelocities[segment])
    }
    waveWork /= SEGMENT_COUNT - 1
    const traction = clamp(scenario.plantFriction ?? scenario.friction, 0, 1.2)
    const acceleration =
      clamp(
        model.plant.waveGain * waveWork,
        -model.plant.maximumForwardAcceleration,
        model.plant.maximumForwardAcceleration,
      ) * traction
    forwardSpeed +=
      (acceleration -
        (model.plant.linearDrag + model.plant.inverseTractionDrag / Math.max(traction, 0.12)) * forwardSpeed) *
      dt
    forwardSpeed = clamp(forwardSpeed, -1.45, 1.45)

    let frontBias = 0
    for (let segment = 0; segment < SEGMENT_COUNT; segment += 1) {
      frontBias += joints[segment] * (1 - segment / (SEGMENT_COUNT - 1))
    }
    frontBias /= SEGMENT_COUNT
    angularSpeed +=
      (frontBias * (model.plant.frontTurnBase + model.plant.frontTurnSpeedGain * Math.abs(forwardSpeed)) -
        model.plant.angularDrag * angularSpeed) *
      dt
    angularSpeed = clamp(angularSpeed, -3, 3)

    const rightX = -forwardZ
    const rightZ = forwardX
    forwardX += rightX * angularSpeed * dt
    forwardZ += rightZ * angularSpeed * dt
    const headingLength = Math.max(1e-9, Math.hypot(forwardX, forwardZ))
    forwardX /= headingLength
    forwardZ /= headingLength
    const deltaX = forwardX * forwardSpeed * dt
    const deltaZ = forwardZ * forwardSpeed * dt
    x += deltaX
    z += deltaZ
    pathLength += Math.hypot(deltaX, deltaZ)
    traces.push(Array.from(command))
  }

  const finalDistance = Math.hypot(scenario.targetX - x, scenario.targetZ - z)
  const temporalStd = mean(
    Array.from({ length: SEGMENT_COUNT }, (_, segment) => standardDeviation(traces.map((trace) => trace[segment]))),
  )
  const spatialStd = mean(traces.map((trace) => standardDeviation(trace)))
  const traceHash = createHash('sha256')
    .update(JSON.stringify([round(x), round(z), round(pathLength), round(temporalStd), round(spatialStd)]))
    .digest('hex')
  return {
    progress: initialDistance - finalDistance,
    displacement: Math.hypot(x, z),
    temporalStd,
    spatialStd,
    finalX: x,
    finalZ: z,
    traceHash,
  }
}

function runBrowserPlantScenario(model: LocomotionPolicyArtifact, scenario: Scenario, steps: number) {
  const policy = new EvolvedLocomotionPolicy(model)
  const plant = createWormLocomotionPlant()
  const rootState = { x: 0, z: 0, heading: 0, vx: 0, vz: 0 }
  const field: TerrainField = {
    width: 100,
    depth: 100,
    minimumHeight: 0,
    maximumHeight: 0,
    waypoints: [],
    sample: () => ({ height: 0, friction: scenario.friction, normal: [0, 1, 0], surface: 'sand' }),
  }
  let maximumLateralSpan = 0
  const checkpoints: ParityCheckpoint[] = []
  const checkpointSteps = new Set([0, 1, 2, 23, 59, 119, 239, 419, steps - 1])

  for (let step = 0; step < steps; step += 1) {
    const dx = scenario.targetX - rootState.x
    const dz = scenario.targetZ - rootState.z
    const distance = Math.max(1e-9, Math.hypot(dx, dz))
    const forwardX = Math.cos(rootState.heading)
    const forwardZ = Math.sin(rootState.heading)
    const action = policy.run(
      {
        targetForward: (dx * forwardX + dz * forwardZ) / distance,
        targetRight: (dx * -forwardZ + dz * forwardX) / distance,
        targetDistance: Math.min(distance / 5, 1.5),
        forwardSpeed: plant.forwardSpeed,
        angularSpeed: plant.angularSpeed,
        terrainFriction: scenario.friction,
        urgency: scenario.urgency,
      },
      plant.joints,
      plant.jointVelocities,
    )
    stepWormLocomotion(plant, rootState, action, model.plant.timestep, field, 0)
    const pose = deriveWormLocalPose(plant, action)
    const lateral = pose.map((segment) => segment.lateral)
    maximumLateralSpan = Math.max(maximumLateralSpan, Math.max(...lateral) - Math.min(...lateral))
    if (checkpointSteps.has(step)) {
      checkpoints.push({
        step,
        position: [rootState.x, rootState.z],
        heading: rootState.heading,
        forwardSpeed: plant.forwardSpeed,
        angularSpeed: plant.angularSpeed,
        joints: [...plant.joints],
        jointVelocities: [...plant.jointVelocities],
        commands: Array.from({ length: SEGMENT_COUNT }, (_, segment) => action[segment * 2]),
      })
    }
  }
  return { finalX: rootState.x, finalZ: rootState.z, maximumLateralSpan, checkpoints }
}

function readPythonParityTrace(): { steps: number; checkpoints: ParityCheckpoint[] } {
  const result = spawnSync(
    'uv',
    [
      'run',
      'python',
      '-m',
      'wurmkickflip_rl.locomotion_parity_trace',
      '--model',
      resolve(root, 'public/models/wurmkickflip_locomotion_policy.json'),
    ],
    { cwd: resolve(root, 'training'), encoding: 'utf8' },
  )
  expect(result.status === 0, `Python locomotion parity trace failed: ${result.stderr || result.stdout}`)
  const value = JSON.parse(result.stdout) as { steps?: unknown; checkpoints?: unknown }
  expect(value.steps === artifact.training.episodeSteps, 'Python parity trace used the wrong episode length')
  expect(Array.isArray(value.checkpoints), 'Python parity trace did not return checkpoints')
  return value as { steps: number; checkpoints: ParityCheckpoint[] }
}

function compareParityCheckpoints(browser: ParityCheckpoint[], python: ParityCheckpoint[]) {
  expect(browser.length === python.length, 'Python/browser parity checkpoint counts differ')
  let maximumDelta = 0
  for (let index = 0; index < browser.length; index += 1) {
    const browserCheckpoint = browser[index]
    const pythonCheckpoint = python[index]
    expect(browserCheckpoint.step === pythonCheckpoint.step, 'Python/browser parity checkpoint steps differ')
    for (const key of ['position', 'joints', 'jointVelocities', 'commands'] as const) {
      expect(browserCheckpoint[key].length === pythonCheckpoint[key].length, `parity ${key} lengths differ`)
      for (let valueIndex = 0; valueIndex < browserCheckpoint[key].length; valueIndex += 1) {
        maximumDelta = Math.max(
          maximumDelta,
          Math.abs(browserCheckpoint[key][valueIndex] - pythonCheckpoint[key][valueIndex]),
        )
      }
    }
    maximumDelta = Math.max(
      maximumDelta,
      Math.abs(browserCheckpoint.heading - pythonCheckpoint.heading),
      Math.abs(browserCheckpoint.forwardSpeed - pythonCheckpoint.forwardSpeed),
      Math.abs(browserCheckpoint.angularSpeed - pythonCheckpoint.angularSpeed),
    )
  }
  return maximumDelta
}

function aggregate(rollouts: Rollout): Rollout
function aggregate(rollouts: Rollout[]): Rollout
function aggregate(rollouts: Rollout | Rollout[]): Rollout {
  const values = Array.isArray(rollouts) ? rollouts : [rollouts]
  return {
    progress: mean(values.map((value) => value.progress)),
    displacement: mean(values.map((value) => value.displacement)),
    temporalStd: mean(values.map((value) => value.temporalStd)),
    spatialStd: mean(values.map((value) => value.spatialStd)),
    finalX: mean(values.map((value) => value.finalX)),
    finalZ: mean(values.map((value) => value.finalZ)),
    traceHash: createHash('sha256').update(values.map((value) => value.traceHash).join(':')).digest('hex'),
  }
}

async function verifyTrainingReproducibility() {
  const temp = await mkdtemp(join(tmpdir(), 'wurmkickflip-locomotion-'))
  try {
    const hashes: string[] = []
    for (const [suffix, seed] of [['a', '407'], ['b', '407'], ['different-seed', '408']] as const) {
      const output = join(temp, `model-${suffix}.json`)
      const summary = join(temp, `summary-${suffix}.json`)
      const result = spawnSync(
        'uv',
        [
          'run',
          'python',
          '-m',
          'wurmkickflip_rl.evolve_locomotion_policy',
          '--seed',
          seed,
          '--generations',
          '3',
          '--population-size',
          '10',
          '--elite-count',
          '2',
          '--episode-steps',
          '60',
          '--warm-start',
          resolve(root, 'training/seeds/wurmkickflip_locomotion_warm_start_v1.json'),
          '--out',
          output,
          '--summary',
          summary,
          '--model-version',
          'locomotion-repro-smoke',
        ],
        { cwd: resolve(root, 'training'), encoding: 'utf8' },
      )
      expect(result.status === 0, `locomotion reproducibility training failed: ${result.stderr || result.stdout}`)
      hashes.push(createHash('sha256').update(await readFile(output)).digest('hex'))
    }
    expect(hashes[0] === hashes[1], 'same-seed locomotion training must export byte-identical artifacts')
    expect(hashes[0] !== hashes[2], 'different-seed locomotion training must produce a different artifact')
  } finally {
    await rm(temp, { recursive: true, force: true })
  }
}

function finiteInteger(value: unknown, label: string) {
  expect(typeof value === 'number' && Number.isInteger(value), `${label} must be an integer`)
  return value
}

function recordValue(value: unknown, label: string): Record<string, unknown> {
  expect(typeof value === 'object' && value !== null && !Array.isArray(value), `${label} must be an object`)
  return value as Record<string, unknown>
}

function standardDeviation(values: number[]) {
  const average = mean(values)
  return Math.sqrt(mean(values.map((value) => (value - average) ** 2)))
}

function mean(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1)
}

function round(value: number) {
  return Math.round(value * 1_000_000) / 1_000_000
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value))
}

function expect(condition: boolean, message: string): asserts condition {
  if (!condition) throw new Error(message)
}
