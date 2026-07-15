import { MathUtils } from 'three'
import { LOCOMOTION_PLANT_CONTRACT } from '../policy/locomotionPolicy'
import type { PolicyAction } from '../policy/types'

export const LOCOMOTION_PLANT_VERSION = LOCOMOTION_PLANT_CONTRACT.version
export const LOCOMOTION_SEGMENT_COUNT = 16

export type WormLocomotionPlant = {
  joints: number[]
  jointVelocities: number[]
  forwardSpeed: number
  angularSpeed: number
  activation: number
  contactRatio: number
  /** Per-segment support and slip feedback, stored anterior-to-posterior. */
  contactLoads: number[]
  slipSpeeds: number[]
  obstacleNormalX: number[]
  obstacleNormalZ: number[]
  obstacleForward: number[]
  obstacleRight: number[]
}

export type WormLocalPose = {
  forward: number
  lateral: number
  lift: number
  yaw: number
}

/**
 * Joint-servo state shared by the browser and its causal verifier. It has no
 * target, clock, gait phase, or writable propulsion input. Translation belongs
 * exclusively to the articulated body/contact solver in wormDynamics.ts.
 */
export function createWormLocomotionPlant(): WormLocomotionPlant {
  return {
    joints: new Array(LOCOMOTION_SEGMENT_COUNT).fill(0),
    jointVelocities: new Array(LOCOMOTION_SEGMENT_COUNT).fill(0),
    forwardSpeed: 0,
    angularSpeed: 0,
    activation: 0,
    contactRatio: 1,
    contactLoads: new Array(LOCOMOTION_SEGMENT_COUNT).fill(1),
    slipSpeeds: new Array(LOCOMOTION_SEGMENT_COUNT).fill(0),
    obstacleNormalX: new Array(LOCOMOTION_SEGMENT_COUNT).fill(0),
    obstacleNormalZ: new Array(LOCOMOTION_SEGMENT_COUNT).fill(0),
    obstacleForward: new Array(LOCOMOTION_SEGMENT_COUNT).fill(0),
    obstacleRight: new Array(LOCOMOTION_SEGMENT_COUNT).fill(0),
  }
}

export function resetWormLocomotionPlant(plant: WormLocomotionPlant) {
  plant.joints.fill(0)
  plant.jointVelocities.fill(0)
  plant.forwardSpeed = 0
  plant.angularSpeed = 0
  plant.activation = 0
  plant.contactRatio = 1
  plant.contactLoads.fill(1)
  plant.slipSpeeds.fill(0)
  plant.obstacleNormalX.fill(0)
  plant.obstacleNormalZ.fill(0)
  plant.obstacleForward.fill(0)
  plant.obstacleRight.fill(0)
}

/** Advance only the antagonistic joint servos. Translation belongs to the
 * articulated body/contact solver in wormDynamics.ts. */
export function stepWormActuators(plant: WormLocomotionPlant, action: PolicyAction, delta: number) {
  const dt = MathUtils.clamp(delta, 0, LOCOMOTION_PLANT_CONTRACT.timestep)
  let activation = 0

  for (let index = 0; index < LOCOMOTION_SEGMENT_COUNT; index += 1) {
    const command = muscleCommand(action, index)
    const joint = plant.joints[index] ?? 0
    let velocity = plant.jointVelocities[index] ?? 0
    velocity +=
      ((command - joint) * LOCOMOTION_PLANT_CONTRACT.jointStiffness -
        velocity * LOCOMOTION_PLANT_CONTRACT.jointDamping) *
      dt
    plant.jointVelocities[index] = velocity
    plant.joints[index] = MathUtils.clamp(
      joint + velocity * dt,
      -LOCOMOTION_PLANT_CONTRACT.jointLimit,
      LOCOMOTION_PLANT_CONTRACT.jointLimit,
    )
    activation += Math.abs(command)
  }
  plant.activation = activation / LOCOMOTION_SEGMENT_COUNT
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
  return points.map(point => ({
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
