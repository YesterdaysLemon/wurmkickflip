import assert from 'node:assert/strict'
import { performance } from 'node:perf_hooks'
import { readFile } from 'node:fs/promises'
import { deriveWurmAnatomy } from '../src/creature/anatomy'
import { EvolvedLocomotionPolicy, parseLocomotionPolicy } from '../src/policy/locomotionPolicy'
import { SEGMENT_COUNT, type SegmentSnapshot } from '../src/policy/types'
import { createTerrainField } from '../src/scene/terrainField'
import { createTerrariumArenaBounds, prepareTerrariumCollisionWorld } from '../src/scene/terrariumCollisions'
import { stepArticulatedWorm } from '../src/scene/wormDynamics'
import { createWormLocomotionPlant } from '../src/scene/wormLocomotion'

const artifact = parseLocomotionPolicy(
  JSON.parse(
    await readFile(new URL('../public/models/wurmkickflip_locomotion_policy.json', import.meta.url), 'utf8'),
  ),
)
const field = createTerrainField(null)
const anatomy = deriveWurmAnatomy(null)
const world = prepareTerrariumCollisionWorld({
  bounds: createTerrariumArenaBounds(field.width, field.depth, 0.82),
  obstacles: [
    { id: 'bench-a', kind: 'rock', center: { x: 1.2, z: 0.5 }, radius: 0.28 },
    { id: 'bench-b', kind: 'tree', center: { x: -1.4, z: -0.8 }, radius: 0.2 },
    { id: 'bench-c', kind: 'bowl', center: { x: 0.2, z: 1.5 }, radius: 0.32 },
    { id: 'bench-d', kind: 'prop', center: { x: -0.4, z: -1.7 }, radius: 0.24 },
  ],
})

// Warm the JIT and allocation caches outside the measured region.
runRollout(180, 0)
const rolloutCount = 10
const stepsPerRollout = 600
const start = performance.now()
let checksum = 0
for (let rollout = 0; rollout < rolloutCount; rollout += 1) {
  checksum += runRollout(stepsPerRollout, rollout + 1)
}
const elapsedMs = performance.now() - start
const stepCount = rolloutCount * stepsPerRollout
const millisecondsPerStep = elapsedMs / stepCount

assert.ok(Number.isFinite(checksum), 'performance rollout checksum must remain finite')
assert.ok(
  millisecondsPerStep < 2,
  `articulated policy step averaged ${millisecondsPerStep.toFixed(3)} ms; 2 ms budget exceeded`,
)

console.log(
  JSON.stringify(
    {
      checksum: Number(checksum.toFixed(8)),
      elapsedMs: Number(elapsedMs.toFixed(3)),
      millisecondsPerStep: Number(millisecondsPerStep.toFixed(5)),
      rolloutCount,
      stepCount,
    },
    null,
    2,
  ),
)
console.log('Simulation performance verification passed.')

function runRollout(stepCount: number, seed: number) {
  const policy = new EvolvedLocomotionPolicy(artifact)
  const plant = createWormLocomotionPlant()
  const offsetX = ((seed % 3) - 1) * 0.25
  const offsetZ = ((seed % 5) - 2) * 0.18
  const segments: SegmentSnapshot[] = Array.from({ length: SEGMENT_COUNT }, (_, index) => ({
    x: offsetX + (index - (SEGMENT_COUNT - 1) * 0.5) * 0.102,
    y: field.sample(offsetX, offsetZ).height + 0.105,
    z: offsetZ,
    vx: 0,
    vy: 0,
    vz: 0,
    pitch: 0,
    yaw: 0,
  }))
  let checksum = 0
  for (let step = 0; step < stepCount; step += 1) {
    const targetRight = Math.sin(seed * 0.71 + step * 0.003) * 0.55
    const action = policy.run(
      {
        targetForward: Math.sqrt(Math.max(0, 1 - targetRight * targetRight)),
        targetRight,
        targetDistance: 0.72,
        forwardSpeed: plant.forwardSpeed,
        angularSpeed: plant.angularSpeed,
        terrainFriction: 0.9,
        urgency: 0.8,
        contactLoads: plant.contactLoads,
        slipSpeeds: plant.slipSpeeds,
        obstacleForward: plant.obstacleForward,
        obstacleRight: plant.obstacleRight,
      },
      plant.joints,
      plant.jointVelocities,
    )
    const result = stepArticulatedWorm(plant, segments, action, 1 / 60, field, world, anatomy)
    checksum += result.root.x * 0.31 + result.root.z * 0.17 + result.contactRatio * 0.07
  }
  return checksum
}
