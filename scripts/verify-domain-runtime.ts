import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import type { EnvironmentConfig } from '../src/creature/types'
import { deriveWurmAnatomy } from '../src/creature/anatomy'
import {
  actuatorLatencyTicks,
  deterministicNoise,
  DomainActionPipeline,
  handoffDomainActionOwner,
  perturbLocomotionInputs,
  perturbPolicyObservation,
} from '../src/environment/domainRuntime'
import { materializeDomainEnvironment, sampleDomain } from '../src/environment/seedForge'
import { ACTION_SIZE, OBSERVATION_SIZE, POLICY_TIMESTEP } from '../src/policy/types'
import { advanceStunt, createStuntState } from '../src/scene/terrariumSimulation'
import { createTerrainField } from '../src/scene/terrainField'

const environment = JSON.parse(
  await readFile(
    resolve(import.meta.dirname, '../public/configs/environments/adaptive-skate-terrarium.json'),
    'utf8',
  ),
) as EnvironmentConfig

const immediate = new DomainActionPipeline({ actuatorLatencyMs: 0, actuatorStrength: 1.5 })
const command = new Float32Array(ACTION_SIZE)
command[0] = 0.5
command[1] = -0.8
const immediateOutput = Float32Array.from(immediate.step(command, 'locomotion'))
expectClose(immediateOutput[0], 0.75, 'strength scaling')
expectClose(immediateOutput[1], -1, 'strength clamp')

const latencyMs = POLICY_TIMESTEP * 1000 * 2
expect(actuatorLatencyTicks(latencyMs) === 2, 'latency did not quantize to two fixed steps')
const delayed = new DomainActionPipeline({ actuatorLatencyMs: latencyMs, actuatorStrength: 1 })
const first = Float32Array.from(delayed.step(command, 'locomotion'))
const second = Float32Array.from(delayed.step(new Float32Array(ACTION_SIZE), 'locomotion'))
const third = Float32Array.from(delayed.step(new Float32Array(ACTION_SIZE), 'locomotion'))
expect(
  first.every(value => value === 0),
  'delayed pipeline emitted on its first tick',
)
expect(
  second.every(value => value === 0),
  'delayed pipeline emitted one tick early',
)
expectClose(third[0], 0.5, 'delayed pipeline did not emit the original command')
delayed.reset()
expect(
  delayed.step(new Float32Array(ACTION_SIZE), 'locomotion').every(value => value === 0),
  'reset leaked a delayed command',
)

const ownershipBoundary = new DomainActionPipeline({
  actuatorLatencyMs: latencyMs,
  actuatorStrength: 1,
})
const staleApplied = new Float32Array(ACTION_SIZE).fill(0.65)
const staleMounted = new Float32Array(ACTION_SIZE).fill(-0.45)
ownershipBoundary.step(staleApplied, 'locomotion')
ownershipBoundary.step(staleApplied, 'locomotion')
expect(
  handoffDomainActionOwner('locomotion', 'mounted', ownershipBoundary, staleApplied, staleMounted),
  'mounting-to-riding ownership transition was not detected',
)
expect(
  staleApplied.every(value => value === 0) && staleMounted.every(value => value === 0),
  'mounting-to-riding ownership transition retained a stale smoothing buffer',
)
const mountedCommand = new Float32Array(ACTION_SIZE)
mountedCommand[0] = -0.35
const firstMounted = Float32Array.from(ownershipBoundary.step(mountedCommand, 'mounted'))
const secondMounted = Float32Array.from(ownershipBoundary.step(new Float32Array(ACTION_SIZE), 'mounted'))
const thirdMounted = Float32Array.from(ownershipBoundary.step(new Float32Array(ACTION_SIZE), 'mounted'))
expect(
  firstMounted.every(value => value === 0) && secondMounted.every(value => value === 0),
  'ownership transition leaked queued locomotion commands into mounted action order',
)
expectClose(thirdMounted[0], -0.35, 'mounted command did not survive its own latency queue')

const noiseA = deterministicNoise(1337, 12, 4)
const noiseB = deterministicNoise(1337, 12, 4)
expect(noiseA === noiseB, 'sensor noise is not deterministic')
expect(noiseA >= -1 && noiseA < 1, 'sensor noise escaped the signed unit interval')
expect(
  noiseA !== deterministicNoise(1338, 12, 4) && noiseA !== deterministicNoise(1337, 13, 4),
  'sensor noise did not respond to seed and simulation step',
)

const observation = new Float32Array(OBSERVATION_SIZE).fill(0.25)
const quietObservation = perturbPolicyObservation(observation, { seed: 7, sensorNoise: 0 }, 3)
expect(quietObservation === observation, 'zero-noise observation path allocated or changed data')
const noisyObservation = perturbPolicyObservation(observation, { seed: 7, sensorNoise: 0.05 }, 3)
expect(noisyObservation !== observation, 'noisy observation path reused mutable source data')
expect(
  noisyObservation.every(value => Math.abs(value - 0.25) <= 0.050001),
  'observation noise exceeded its configured amplitude',
)
expect(
  observation.every(value => value === 0.25),
  'observation perturbation mutated its input',
)

const sensorArrays = {
  targetForward: 1,
  targetRight: 0,
  targetDistance: 0.5,
  forwardSpeed: 0.2,
  angularSpeed: -0.1,
  terrainFriction: 0.8,
  urgency: 0.6,
  contactLoads: [0.2, 0.4],
  slipSpeeds: [0.1, 0.3],
  obstacleForward: [0, 0.5],
  obstacleRight: [-0.2, 0],
}
const bends = [0.1, -0.1]
const velocities = [0.2, -0.2]
const firstNoisyInputs = perturbLocomotionInputs(
  sensorArrays,
  bends,
  velocities,
  { seed: 99, sensorNoise: 0.02 },
  4,
)
const secondNoisyInputs = perturbLocomotionInputs(
  sensorArrays,
  bends,
  velocities,
  { seed: 99, sensorNoise: 0.02 },
  4,
)
expect(
  JSON.stringify(firstNoisyInputs) === JSON.stringify(secondNoisyInputs),
  'locomotion sensor perturbation is not reproducible',
)
expect(sensorArrays.targetForward === 1 && bends[0] === 0.1, 'locomotion perturbation mutated source data')

const sampledDomain = sampleDomain(environment, 1337)
const sampledEnvironment = materializeDomainEnvironment(environment, sampledDomain)
const sampledField = createTerrainField(sampledEnvironment)
const spawned = createStuntState(sampledField, sampledEnvironment, deriveWurmAnatomy(null), sampledDomain)
expectClose(spawned.boardX, sampledDomain.skateboardSpawnX, 'sampled skateboard spawn X')
expectClose(spawned.boardZ, sampledDomain.skateboardSpawnZ, 'sampled skateboard spawn Z')
expectClose(spawned.boardHeading, (sampledDomain.spawnYawDegrees * Math.PI) / 180, 'sampled spawn yaw')

const noDragEnvironment = {
  ...sampledEnvironment,
  world: { ...sampledEnvironment.world, airDrag: 0 },
}
const highDragEnvironment = {
  ...sampledEnvironment,
  world: { ...sampledEnvironment.world, airDrag: 10 },
}
const noDragState = createRidingState(noDragEnvironment)
const highDragState = createRidingState(highDragEnvironment)
advanceStunt(
  noDragState,
  new Float32Array(ACTION_SIZE),
  POLICY_TIMESTEP,
  9.81,
  'kickflip',
  createTerrainField(noDragEnvironment),
  noDragEnvironment,
)
advanceStunt(
  highDragState,
  new Float32Array(ACTION_SIZE),
  POLICY_TIMESTEP,
  9.81,
  'kickflip',
  createTerrainField(highDragEnvironment),
  highDragEnvironment,
)
expect(
  Math.hypot(highDragState.boardVx, highDragState.boardVz) <
    Math.hypot(noDragState.boardVx, noDragState.boardVz),
  'sampled air drag did not damp skateboard velocity',
)

console.log(
  JSON.stringify({
    latencyTicks: delayed.latencyTicks,
    deterministicProbe: Number(noiseA.toFixed(9)),
    observationMaximumDelta: Number(
      Math.max(...Array.from(noisyObservation, value => Math.abs(value - 0.25))).toFixed(9),
    ),
  }),
)
console.log('Domain runtime verification passed.')

function expect(condition: boolean, message: string): asserts condition {
  if (!condition) throw new Error(message)
}

function expectClose(actual: number, expected: number, label: string) {
  expect(Math.abs(actual - expected) < 1e-6, `${label}: expected ${expected}, received ${actual}`)
}

function createRidingState(domainEnvironment: EnvironmentConfig) {
  const field = createTerrainField(domainEnvironment)
  const state = createStuntState(field, domainEnvironment)
  state.locomotionState = 'riding'
  state.mountBlend = 1
  state.boardContactRatio = 1
  state.boardContactSegmentCount = state.segments.length
  state.boardContactStableSeconds = 1
  state.boardContactHeadWeight = 1
  state.boardContactMidbodyWeight = 1
  state.boardContactTailWeight = 1
  state.contactRatio = 1
  state.boardSpeed = 1
  state.boardVx = Math.cos(state.boardHeading)
  state.boardVz = Math.sin(state.boardHeading)
  return state
}
