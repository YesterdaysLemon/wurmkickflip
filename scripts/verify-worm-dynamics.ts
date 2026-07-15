import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { deriveWurmAnatomy } from '../src/creature/anatomy'
import { EvolvedLocomotionPolicy, parseLocomotionPolicy } from '../src/policy/locomotionPolicy'
import { SEGMENT_COUNT, type PolicyAction, type SegmentSnapshot } from '../src/policy/types'
import { createTerrainField } from '../src/scene/terrainField'
import { createTerrariumArenaBounds, prepareTerrariumCollisionWorld } from '../src/scene/terrariumCollisions'
import { stepArticulatedWorm } from '../src/scene/wormDynamics'
import { createWormLocomotionPlant } from '../src/scene/wormLocomotion'

const field = createTerrainField(null)
const anatomy = deriveWurmAnatomy(null)
const world = prepareTerrariumCollisionWorld({
  bounds: createTerrariumArenaBounds(field.width, field.depth, 0.82),
  obstacles: [],
})

const zeroSegments = createSegments()
const zeroPlant = createWormLocomotionPlant()
const zeroAction = new Float32Array(SEGMENT_COUNT * 2)
const zeroStart = center(zeroSegments)
for (let step = 0; step < 360; step += 1) {
  stepArticulatedWorm(zeroPlant, zeroSegments, zeroAction, 1 / 60, field, world, anatomy)
}
const zeroEnd = center(zeroSegments)
assert.ok(
  Math.hypot(zeroEnd.x - zeroStart.x, zeroEnd.z - zeroStart.z) < 1e-8,
  'zero action must not translate',
)
assertChain(zeroSegments, 0.102, 2e-3)

const diagnosticSegments = createSegments()
const diagnosticPlant = createWormLocomotionPlant()
const diagnosticStart = center(diagnosticSegments)
for (let step = 0; step < 720; step += 1) {
  const action = diagnosticActuatorAction(step)
  stepArticulatedWorm(diagnosticPlant, diagnosticSegments, action, 1 / 60, field, world, anatomy)
  assertChain(diagnosticSegments, 0.102, 2.5e-3)
  assert.equal(center(diagnosticSegments).x, center(diagnosticSegments).x, 'center must stay finite')
}
const diagnosticEnd = center(diagnosticSegments)
const actuatedDisplacement = Math.hypot(
  diagnosticEnd.x - diagnosticStart.x,
  diagnosticEnd.z - diagnosticStart.z,
)
assert.ok(
  actuatedDisplacement > 0.015,
  `contact-driven actuator changes should move, got ${actuatedDisplacement}`,
)

const frictionlessField = {
  ...field,
  sample(x: number, z: number) {
    return { ...field.sample(x, z), friction: 0 }
  },
}
const frictionlessSegments = createSegments()
const frictionlessPlant = createWormLocomotionPlant()
const frictionlessStart = center(frictionlessSegments)
for (let step = 0; step < 720; step += 1) {
  stepArticulatedWorm(
    frictionlessPlant,
    frictionlessSegments,
    diagnosticActuatorAction(step),
    1 / 60,
    frictionlessField,
    world,
    anatomy,
  )
}
const frictionlessEnd = center(frictionlessSegments)
const frictionlessDisplacement = Math.hypot(
  frictionlessEnd.x - frictionlessStart.x,
  frictionlessEnd.z - frictionlessStart.z,
)
assert.ok(
  frictionlessDisplacement < Math.max(1e-5, actuatedDisplacement * 0.08),
  `internal forces must conserve center of mass without traction (${frictionlessDisplacement})`,
)

const artifact = parseLocomotionPolicy(
  JSON.parse(
    await readFile(new URL('../public/models/wurmkickflip_locomotion_policy.json', import.meta.url), 'utf8'),
  ),
)
const policy = new EvolvedLocomotionPolicy(artifact)
const neuralSegments = createSegments()
const neuralPlant = createWormLocomotionPlant()
for (let step = 0; step < 240; step += 1) {
  const action = policy.run(
    {
      targetForward: 1,
      targetRight: 0.2,
      targetDistance: 0.8,
      forwardSpeed: neuralPlant.forwardSpeed,
      angularSpeed: neuralPlant.angularSpeed,
      terrainFriction: 0.9,
      urgency: 0.8,
    },
    neuralPlant.joints,
    neuralPlant.jointVelocities,
  )
  const result = stepArticulatedWorm(neuralPlant, neuralSegments, action, 1 / 60, field, world, anatomy)
  assert.ok(
    Number.isFinite(result.root.x) && Number.isFinite(result.root.z),
    'neural rollout must remain finite',
  )
}

console.log(
  JSON.stringify({
    ok: true,
    actuatedDisplacement,
    frictionlessDisplacement,
    neuralDisplacement: Math.hypot(center(neuralSegments).x, center(neuralSegments).z),
    contactRatio: diagnosticPlant.contactRatio,
  }),
)

function createSegments(): SegmentSnapshot[] {
  return Array.from({ length: SEGMENT_COUNT }, (_, index) => ({
    x: (index - (SEGMENT_COUNT - 1) * 0.5) * 0.102,
    y: 0.105,
    z: 0,
    vx: 0,
    vy: 0,
    vz: 0,
    pitch: 0,
    yaw: 0,
  }))
}

// This hand-authored signal is a plant-only diagnostic proving that changing
// internal forces can exchange momentum with contact. It is never available to
// the browser controller or evolution path, which use the tracked neural model.
function diagnosticActuatorAction(step: number): PolicyAction {
  const action = new Float32Array(SEGMENT_COUNT * 2)
  for (let segment = 0; segment < SEGMENT_COUNT; segment += 1) {
    const phase = step * 0.14 - segment * 0.72
    const command = Math.sin(phase) * 0.92
    action[segment * 2] = command
    action[segment * 2 + 1] = -command
  }
  return action
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

function assertChain(segments: readonly SegmentSnapshot[], spacing: number, tolerance: number) {
  for (let index = 0; index < segments.length - 1; index += 1) {
    const distance = Math.hypot(
      segments[index + 1].x - segments[index].x,
      segments[index + 1].z - segments[index].z,
    )
    assert.ok(Math.abs(distance - spacing) <= tolerance, `joint ${index} length drifted to ${distance}`)
  }
}
