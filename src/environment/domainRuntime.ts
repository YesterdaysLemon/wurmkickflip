import type { LocomotionSensors } from '../policy/locomotionPolicy'
import { ACTION_SIZE, POLICY_TIMESTEP, type PolicyAction, type PolicyObservation } from '../policy/types'
import type { DomainSample } from './seedForge'

export type NoisyLocomotionInputs = {
  sensors: LocomotionSensors
  segmentBends: ArrayLike<number>
  segmentBendVelocities: ArrayLike<number>
}

export type DomainActionOwner = 'locomotion' | 'mounted' | 'authored'

/**
 * Fixed-step actuator transport for a sampled domain.
 *
 * Latency is represented as whole policy ticks and strength is applied only
 * after the delayed command is selected. The class owns its buffers so a reset
 * cannot leak commands from the previous forged world.
 */
export class DomainActionPipeline {
  readonly latencyTicks: number
  private readonly strength: number
  private readonly pending: PolicyAction[] = []
  private readonly output = new Float32Array(ACTION_SIZE)
  private owner: DomainActionOwner | null = null

  constructor(sample: Pick<DomainSample, 'actuatorLatencyMs' | 'actuatorStrength'>) {
    this.latencyTicks = actuatorLatencyTicks(sample.actuatorLatencyMs)
    this.strength = finiteOr(sample.actuatorStrength, 1)
  }

  step(action: ArrayLike<number>, owner: DomainActionOwner): PolicyAction {
    if (this.owner !== null && this.owner !== owner) this.reset()
    this.owner = owner
    const command = new Float32Array(ACTION_SIZE)
    for (let channel = 0; channel < ACTION_SIZE; channel += 1) {
      command[channel] = clamp(finiteOr(action[channel], 0), -1, 1)
    }
    this.pending.push(command)

    const delayed = this.pending.length > this.latencyTicks ? this.pending.shift() : undefined
    for (let channel = 0; channel < ACTION_SIZE; channel += 1) {
      this.output[channel] = clamp((delayed?.[channel] ?? 0) * this.strength, -1, 1)
    }
    return this.output
  }

  reset() {
    this.pending.length = 0
    this.output.fill(0)
    this.owner = null
  }
}

export function handoffDomainActionOwner(
  previousOwner: DomainActionOwner,
  nextOwner: DomainActionOwner,
  pipeline: DomainActionPipeline,
  appliedAction: PolicyAction,
  latestMountedAction: PolicyAction,
) {
  if (previousOwner === nextOwner) return false
  pipeline.reset()
  appliedAction.fill(0)
  latestMountedAction.fill(0)
  return true
}

export function actuatorLatencyTicks(latencyMs: number) {
  return Math.max(0, Math.round(finiteOr(latencyMs, 0) / (POLICY_TIMESTEP * 1000)))
}

export function perturbPolicyObservation(
  observation: PolicyObservation,
  sample: Pick<DomainSample, 'seed' | 'sensorNoise'>,
  simulationStep: number,
): PolicyObservation {
  const amplitude = Math.max(0, finiteOr(sample.sensorNoise, 0))
  if (amplitude === 0) return observation
  const perturbed = new Float32Array(observation.length)
  for (let channel = 0; channel < observation.length; channel += 1) {
    perturbed[channel] =
      finiteOr(observation[channel], 0) + deterministicNoise(sample.seed, simulationStep, channel) * amplitude
  }
  return perturbed
}

export function perturbLocomotionInputs(
  sensors: LocomotionSensors,
  segmentBends: ArrayLike<number>,
  segmentBendVelocities: ArrayLike<number>,
  sample: Pick<DomainSample, 'seed' | 'sensorNoise'>,
  simulationStep: number,
): NoisyLocomotionInputs {
  const amplitude = Math.max(0, finiteOr(sample.sensorNoise, 0))
  if (amplitude === 0) return { sensors, segmentBends, segmentBendVelocities }

  let channel = 0
  const noisy = (value: number | undefined) => {
    const result = finiteOr(value, 0) + deterministicNoise(sample.seed, simulationStep, channel) * amplitude
    channel += 1
    return result
  }
  const noisyArray = (values: ArrayLike<number> | undefined) => {
    if (!values) return undefined
    return Float64Array.from({ length: values.length }, (_, index) => noisy(values[index]))
  }

  const noisySensors: LocomotionSensors = {
    targetForward: noisy(sensors.targetForward),
    targetRight: noisy(sensors.targetRight),
    targetDistance: noisy(sensors.targetDistance),
    forwardSpeed: noisy(sensors.forwardSpeed),
    angularSpeed: noisy(sensors.angularSpeed),
    terrainFriction: noisy(sensors.terrainFriction),
    urgency: noisy(sensors.urgency),
    contactLoads: noisyArray(sensors.contactLoads),
    slipSpeeds: noisyArray(sensors.slipSpeeds),
    obstacleForward: noisyArray(sensors.obstacleForward),
    obstacleRight: noisyArray(sensors.obstacleRight),
  }

  return {
    sensors: noisySensors,
    segmentBends: noisyArray(segmentBends) ?? new Float64Array(0),
    segmentBendVelocities: noisyArray(segmentBendVelocities) ?? new Float64Array(0),
  }
}

/**
 * Stateless integer hash: identical inputs always produce the same signed unit
 * sample, independent of render cadence or prior calls.
 */
export function deterministicNoise(seed: number, simulationStep: number, channel: number) {
  let value =
    normalizeUint32(seed) ^
    Math.imul(normalizeUint32(simulationStep + 1), 0x9e3779b1) ^
    Math.imul(normalizeUint32(channel + 1), 0x85ebca6b)
  value ^= value >>> 16
  value = Math.imul(value, 0x7feb352d)
  value ^= value >>> 15
  value = Math.imul(value, 0x846ca68b)
  value ^= value >>> 16
  return ((value >>> 0) / 0x100000000) * 2 - 1
}

function normalizeUint32(value: number) {
  return Math.trunc(finiteOr(value, 0)) >>> 0
}

function finiteOr(value: number | undefined, fallback: number) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value))
}
