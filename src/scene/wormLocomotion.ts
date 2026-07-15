import { MathUtils } from 'three'
import { LOCOMOTION_PLANT_CONTRACT } from '../policy/locomotionPolicy'
import type { PolicyAction } from '../policy/types'
import type { TerrainField } from './terrainField'

export const LOCOMOTION_PLANT_VERSION = 'joint-work-v1'
export const LOCOMOTION_SEGMENT_COUNT = 16

export type WormLocomotionPlant = {
  joints: number[]
  jointVelocities: number[]
  forwardSpeed: number
  angularSpeed: number
  waveWork: number
  activation: number
  contactRatio: number
}

export type WormLocomotionRoot = {
  x: number
  z: number
  heading: number
  vx: number
  vz: number
}

export type WormLocalPose = {
  forward: number
  lateral: number
  lift: number
  yaw: number
}

export type WormLocomotionStep = {
  distance: number
  friction: number
  hitBoundary: boolean
}

/**
 * A compact articulated locomotion plant shared by the browser and its causal
 * verifier. It has no target, clock, gait phase, or writable propulsion input:
 * translation can only emerge from work traveling between adjacent joints.
 */
export function createWormLocomotionPlant(): WormLocomotionPlant {
  return {
    joints: new Array(LOCOMOTION_SEGMENT_COUNT).fill(0),
    jointVelocities: new Array(LOCOMOTION_SEGMENT_COUNT).fill(0),
    forwardSpeed: 0,
    angularSpeed: 0,
    waveWork: 0,
    activation: 0,
    contactRatio: 1,
  }
}

export function resetWormLocomotionPlant(plant: WormLocomotionPlant) {
  plant.joints.fill(0)
  plant.jointVelocities.fill(0)
  plant.forwardSpeed = 0
  plant.angularSpeed = 0
  plant.waveWork = 0
  plant.activation = 0
  plant.contactRatio = 1
}

export function stepWormLocomotion(
  plant: WormLocomotionPlant,
  root: WormLocomotionRoot,
  action: PolicyAction,
  delta: number,
  field: TerrainField,
  boundaryMargin = 0.82,
): WormLocomotionStep {
  const dt = MathUtils.clamp(delta, 0, LOCOMOTION_PLANT_CONTRACT.timestep)
  let activation = 0

  for (let index = 0; index < LOCOMOTION_SEGMENT_COUNT; index += 1) {
    const command = muscleCommand(action, index)
    const joint = plant.joints[index] ?? 0
    let velocity = plant.jointVelocities[index] ?? 0
    velocity += (
      (command - joint) * LOCOMOTION_PLANT_CONTRACT.jointStiffness -
      velocity * LOCOMOTION_PLANT_CONTRACT.jointDamping
    ) * dt
    plant.jointVelocities[index] = velocity
    plant.joints[index] = MathUtils.clamp(
      joint + velocity * dt,
      -LOCOMOTION_PLANT_CONTRACT.jointLimit,
      LOCOMOTION_PLANT_CONTRACT.jointLimit,
    )
    activation += Math.abs(command)
  }

  let waveWork = 0
  for (let index = 0; index < LOCOMOTION_SEGMENT_COUNT - 1; index += 1) {
    const jointDifference = plant.joints[index + 1] - plant.joints[index]
    const sharedVelocity = (plant.jointVelocities[index + 1] + plant.jointVelocities[index]) * 0.5
    waveWork += jointDifference * sharedVelocity
  }
  waveWork /= LOCOMOTION_SEGMENT_COUNT - 1

  const terrain = field.sample(root.x, root.z)
  const traction = MathUtils.clamp(
    terrain.friction,
    LOCOMOTION_PLANT_CONTRACT.tractionMinimum,
    LOCOMOTION_PLANT_CONTRACT.tractionMaximum,
  )
  const forwardAcceleration = MathUtils.clamp(
    LOCOMOTION_PLANT_CONTRACT.waveGain * waveWork,
    -LOCOMOTION_PLANT_CONTRACT.maximumForwardAcceleration,
    LOCOMOTION_PLANT_CONTRACT.maximumForwardAcceleration,
  ) * traction
  plant.forwardSpeed += (
    forwardAcceleration -
    (
      LOCOMOTION_PLANT_CONTRACT.linearDrag +
      LOCOMOTION_PLANT_CONTRACT.inverseTractionDrag /
        Math.max(traction, LOCOMOTION_PLANT_CONTRACT.inverseTractionFloor)
    ) * plant.forwardSpeed
  ) * dt
  plant.forwardSpeed = MathUtils.clamp(
    plant.forwardSpeed,
    LOCOMOTION_PLANT_CONTRACT.minimumForwardSpeed,
    LOCOMOTION_PLANT_CONTRACT.maximumForwardSpeed,
  )

  let frontBias = 0
  for (let index = 0; index < LOCOMOTION_SEGMENT_COUNT; index += 1) {
    frontBias += plant.joints[index] * (1 - index / (LOCOMOTION_SEGMENT_COUNT - 1))
  }
  frontBias /= LOCOMOTION_SEGMENT_COUNT
  const angularAcceleration = frontBias * (
    LOCOMOTION_PLANT_CONTRACT.frontTurnBase +
    LOCOMOTION_PLANT_CONTRACT.frontTurnSpeedGain * Math.abs(plant.forwardSpeed)
  )
  plant.angularSpeed += (
    angularAcceleration - LOCOMOTION_PLANT_CONTRACT.angularDrag * plant.angularSpeed
  ) * dt
  plant.angularSpeed = MathUtils.clamp(
    plant.angularSpeed,
    LOCOMOTION_PLANT_CONTRACT.minimumAngularSpeed,
    LOCOMOTION_PLANT_CONTRACT.maximumAngularSpeed,
  )

  plant.waveWork = waveWork
  plant.activation = activation / LOCOMOTION_SEGMENT_COUNT
  plant.contactRatio = contactRatioFor(plant)

  root.heading = wrapAngle(root.heading + Math.atan(plant.angularSpeed * dt))
  root.vx = Math.cos(root.heading) * plant.forwardSpeed
  root.vz = Math.sin(root.heading) * plant.forwardSpeed
  const oldX = root.x
  const oldZ = root.z
  const halfWidth = Math.max(0.1, field.width * 0.5 - boundaryMargin)
  const halfDepth = Math.max(0.1, field.depth * 0.5 - boundaryMargin)
  root.x = MathUtils.clamp(root.x + root.vx * dt, -halfWidth, halfWidth)
  root.z = MathUtils.clamp(root.z + root.vz * dt, -halfDepth, halfDepth)
  const hitBoundary = root.x === -halfWidth || root.x === halfWidth || root.z === -halfDepth || root.z === halfDepth
  if (hitBoundary) {
    plant.forwardSpeed *= -0.08
    root.vx = Math.cos(root.heading) * plant.forwardSpeed
    root.vz = Math.sin(root.heading) * plant.forwardSpeed
  }

  return {
    distance: Math.hypot(root.x - oldX, root.z - oldZ),
    friction: terrain.friction,
    hitBoundary,
  }
}

/** Build a chain pose from learned joint states. Trigonometry here is only
 * forward kinematics; there is no time-authored wave or gait phase. */
export function deriveWormLocalPose(plant: WormLocomotionPlant, action: PolicyAction): WormLocalPose[] {
  const points: WormLocalPose[] = []
  let forward = 0
  let lateral = 0
  // The evolved plant numbers segments anterior-to-posterior, while the visual
  // rig is stored tail-to-head so its face remains at the final scene segment.
  let yaw = (plant.joints[LOCOMOTION_SEGMENT_COUNT - 1] ?? 0) * 0.055

  for (let sceneIndex = 0; sceneIndex < LOCOMOTION_SEGMENT_COUNT; sceneIndex += 1) {
    const plantIndex = LOCOMOTION_SEGMENT_COUNT - 1 - sceneIndex
    const command = muscleCommand(action, plantIndex)
    const joint = plant.joints[plantIndex] ?? 0
    const velocity = plant.jointVelocities[plantIndex] ?? 0
    const lift = MathUtils.clamp(Math.abs(velocity) * 0.004 + Math.abs(command - joint) * 0.008, 0, 0.055)
    points.push({ forward, lateral, lift, yaw })
    if (sceneIndex === LOCOMOTION_SEGMENT_COUNT - 1) break
    const nextJoint = plant.joints[plantIndex - 1] ?? joint
    yaw += (joint + nextJoint) * 0.043
    const spacing = 0.102 * (1 - Math.min(0.08, Math.abs(joint - nextJoint) * 0.035))
    forward += Math.cos(yaw) * spacing
    lateral += Math.sin(yaw) * spacing
  }

  const centerForward = points.reduce((sum, point) => sum + point.forward, 0) / points.length
  const centerLateral = points.reduce((sum, point) => sum + point.lateral, 0) / points.length
  return points.map((point) => ({
    ...point,
    forward: point.forward - centerForward,
    lateral: point.lateral - centerLateral,
  }))
}

export function muscleCommand(action: PolicyAction, segment: number) {
  const dorsal = action[segment * 2] ?? 0
  const ventral = action[segment * 2 + 1] ?? 0
  return MathUtils.clamp((dorsal - ventral) * 0.5, -1, 1)
}

function contactRatioFor(plant: WormLocomotionPlant) {
  const meanVelocity = plant.jointVelocities.reduce((sum, velocity) => sum + Math.abs(velocity), 0) / plant.jointVelocities.length
  return MathUtils.clamp(0.94 - meanVelocity * 0.12, 0.28, 0.96)
}

function wrapAngle(angle: number) {
  return Math.atan2(Math.sin(angle), Math.cos(angle))
}
