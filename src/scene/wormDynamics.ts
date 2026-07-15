import type { WurmAnatomy } from '../creature/anatomy'
import { LOCOMOTION_CONTRACT } from '../policy/locomotionContract'
import type { PolicyAction, SegmentSnapshot } from '../policy/types'
import type { TerrainField } from './terrainField'
import { resolveSweptTerrariumMotion, type TerrariumCollisionWorld } from './terrariumCollisions'
import { LOCOMOTION_SEGMENT_COUNT, stepWormActuators, type WormLocomotionPlant } from './wormLocomotion'

export const ARTICULATED_WORM_CONTRACT = LOCOMOTION_CONTRACT.dynamics

export type ArticulatedWormRoot = {
  x: number
  y: number
  z: number
  vx: number
  vz: number
  heading: number
}

export type ArticulatedWormContact = {
  segment: number
  obstacleId: string
  normalX: number
  normalZ: number
}

export type ArticulatedWormStep = {
  root: ArticulatedWormRoot
  distance: number
  contactRatio: number
  collisionCount: number
  contacts: ArticulatedWormContact[]
}

type DynamicsScratch = {
  previousX: Float64Array
  previousY: Float64Array
  previousZ: Float64Array
  targetX: Float64Array
  targetZ: Float64Array
  accelerationX: Float64Array
  accelerationZ: Float64Array
  correctionX: Float64Array
  correctionZ: Float64Array
}

const scratchByPlant = new WeakMap<WormLocomotionPlant, DynamicsScratch>()

/**
 * Advance a free articulated chain. Muscle servos may change the chain's shape,
 * but they never write root translation. Net motion comes from those internal
 * shape forces interacting with per-segment anisotropic ground friction and
 * obstacle impulses. The returned root is measured from the resulting body.
 */
export function stepArticulatedWorm(
  plant: WormLocomotionPlant,
  segments: SegmentSnapshot[],
  action: PolicyAction,
  delta: number,
  field: TerrainField,
  collisionWorld: TerrariumCollisionWorld | null,
  anatomy: WurmAnatomy,
): ArticulatedWormStep {
  if (segments.length !== LOCOMOTION_SEGMENT_COUNT) {
    throw new Error(`articulated worm requires ${LOCOMOTION_SEGMENT_COUNT} segments`)
  }
  const dt = clamp(delta, 0, 1 / 60)
  if (dt <= 0) return measureStep(plant, segments, 0, 0, bodyHeading(segments), 1, [])

  stepWormActuators(plant, action, dt)
  const scratch = scratchFor(plant)
  const spacing = ARTICULATED_WORM_CONTRACT.spacing * anatomy.visualLengthScale
  const groundClearance = ARTICULATED_WORM_CONTRACT.baseGroundClearance * anatomy.verticalScale

  let centerX = 0
  let centerZ = 0
  let meanVx = 0
  let meanVz = 0
  for (let index = 0; index < segments.length; index += 1) {
    const segment = segments[index]
    scratch.previousX[index] = segment.x
    scratch.previousY[index] = segment.y
    scratch.previousZ[index] = segment.z
    centerX += segment.x
    centerZ += segment.z
    meanVx += segment.vx
    meanVz += segment.vz
    plant.obstacleNormalX[LOCOMOTION_SEGMENT_COUNT - 1 - index] = 0
    plant.obstacleNormalZ[LOCOMOTION_SEGMENT_COUNT - 1 - index] = 0
    plant.obstacleForward[LOCOMOTION_SEGMENT_COUNT - 1 - index] = 0
    plant.obstacleRight[LOCOMOTION_SEGMENT_COUNT - 1 - index] = 0
  }
  centerX /= segments.length
  centerZ /= segments.length
  meanVx /= segments.length
  meanVz /= segments.length

  const oldCenterX = centerX
  const oldCenterZ = centerZ
  const oldHeading = bodyHeading(segments)
  buildMuscleShape(plant, oldHeading, spacing, centerX, centerZ, scratch.targetX, scratch.targetZ)

  for (let index = 0; index < segments.length; index += 1) {
    const segment = segments[index]
    scratch.accelerationX[index] =
      (scratch.targetX[index] - segment.x) * ARTICULATED_WORM_CONTRACT.actuatorStiffness -
      (segment.vx - meanVx) * ARTICULATED_WORM_CONTRACT.actuatorDamping
    scratch.accelerationZ[index] =
      (scratch.targetZ[index] - segment.z) * ARTICULATED_WORM_CONTRACT.actuatorStiffness -
      (segment.vz - meanVz) * ARTICULATED_WORM_CONTRACT.actuatorDamping
  }

  // Internal muscle acceleration is explicitly mean-free. Without a contact
  // impulse this preserves center of mass, making the zero-friction ablation a
  // real physical invariant rather than a reward convention.
  const meanAx = mean(scratch.accelerationX)
  const meanAz = mean(scratch.accelerationZ)
  for (let index = 0; index < segments.length; index += 1) {
    const segment = segments[index]
    segment.vx += (scratch.accelerationX[index] - meanAx) * dt
    segment.vz += (scratch.accelerationZ[index] - meanAz) * dt

    const tangent = segmentTangent(segments, index, oldHeading)
    const rightX = -tangent.z
    const rightZ = tangent.x
    const forwardVelocity = segment.vx * tangent.x + segment.vz * tangent.z
    const lateralVelocity = segment.vx * rightX + segment.vz * rightZ
    const plantIndex = LOCOMOTION_SEGMENT_COUNT - 1 - index
    const jointSpeed = Math.abs(plant.jointVelocities[plantIndex] ?? 0)
    const terrain = field.sample(segment.x, segment.z)
    const quietLoad = clamp(1 - jointSpeed / 3.25, 0.08, 1)
    const support = clamp(terrain.normal[1] * terrain.friction * quietLoad, 0, 1)
    const longitudinalDamping =
      1 - Math.exp(-ARTICULATED_WORM_CONTRACT.longitudinalFriction * terrain.friction * dt)
    const lateralDamping = 1 - Math.exp(-ARTICULATED_WORM_CONTRACT.lateralFriction * support * dt)
    const dampedForward = forwardVelocity * (1 - longitudinalDamping)
    const dampedLateral = lateralVelocity * (1 - lateralDamping)
    segment.vx = tangent.x * dampedForward + rightX * dampedLateral
    segment.vz = tangent.z * dampedForward + rightZ * dampedLateral
    const drag = Math.exp(-ARTICULATED_WORM_CONTRACT.freeVelocityDrag * dt)
    segment.vx *= drag
    segment.vz *= drag
    plant.contactLoads[plantIndex] = support
    plant.slipSpeeds[plantIndex] = Math.abs(dampedLateral)
  }
  limitBodyVelocities(segments)
  for (const segment of segments) {
    segment.x += segment.vx * dt
    segment.z += segment.vz * dt
  }

  satisfyDistanceConstraints(segments, spacing, scratch)

  const contacts: ArticulatedWormContact[] = []
  if (collisionWorld) {
    const contactMask = new Uint8Array(LOCOMOTION_SEGMENT_COUNT)
    resolveArticulatedCollisions(
      plant,
      segments,
      field,
      collisionWorld,
      anatomy,
      oldHeading,
      contacts,
      scratch.previousX,
      scratch.previousZ,
      contactMask,
    )
    // Alternate body constraints and contact projection. A single final
    // constraint pass can pull several segments through a bowl and defer a
    // large correction until the next tick; these relaxation passes keep both
    // invariants local to the current fixed step.
    for (let pass = 0; pass < 3; pass += 1) {
      satisfyDistanceConstraints(segments, spacing, scratch, contactMask)
      resolveArticulatedCollisions(
        plant,
        segments,
        field,
        collisionWorld,
        anatomy,
        oldHeading,
        contacts,
        undefined,
        undefined,
        contactMask,
      )
    }
  }

  let contactTotal = 0
  for (let index = 0; index < segments.length; index += 1) {
    const segment = segments[index]
    const terrain = field.sample(segment.x, segment.z)
    const oldX = scratch.previousX[index]
    const oldY = scratch.previousY[index]
    const oldZ = scratch.previousZ[index]
    const targetY = terrain.height + groundClearance
    const maximumVerticalStep = ARTICULATED_WORM_CONTRACT.maximumVerticalSpeed * dt
    segment.y = oldY + clamp(targetY - oldY, -maximumVerticalStep, maximumVerticalStep)
    segment.vx = (segment.x - oldX) / dt
    segment.vy = (segment.y - oldY) / dt
    segment.vz = (segment.z - oldZ) / dt
    const tangent = segmentTangent(segments, index, oldHeading)
    segment.yaw = Math.atan2(tangent.z, tangent.x)
    const plantIndex = LOCOMOTION_SEGMENT_COUNT - 1 - index
    segment.pitch = -clamp(plant.jointVelocities[plantIndex] ?? 0, -3, 3) * 0.025
    contactTotal += plant.contactLoads[plantIndex] ?? 0
  }
  limitBodyVelocities(segments)
  plant.contactRatio = clamp(contactTotal / segments.length, 0, 1)
  const finalHeading = bodyHeading(segments)
  const forwardX = Math.cos(finalHeading)
  const forwardZ = Math.sin(finalHeading)
  const rightX = -forwardZ
  const rightZ = forwardX
  for (let index = 0; index < LOCOMOTION_SEGMENT_COUNT; index += 1) {
    const normalX = plant.obstacleNormalX[index] ?? 0
    const normalZ = plant.obstacleNormalZ[index] ?? 0
    plant.obstacleForward[index] = normalX * forwardX + normalZ * forwardZ
    plant.obstacleRight[index] = normalX * rightX + normalZ * rightZ
  }

  return measureStep(plant, segments, oldCenterX, oldCenterZ, oldHeading, dt, contacts)
}

function resolveArticulatedCollisions(
  plant: WormLocomotionPlant,
  segments: SegmentSnapshot[],
  field: TerrainField,
  collisionWorld: TerrariumCollisionWorld,
  anatomy: WurmAnatomy,
  fallbackHeading: number,
  contacts: ArticulatedWormContact[],
  sweepStartX?: ArrayLike<number>,
  sweepStartZ?: ArrayLike<number>,
  contactMask?: Uint8Array,
) {
  for (let index = 0; index < segments.length; index += 1) {
    const segment = segments[index]
    const startX = sweepStartX?.[index] ?? segment.x
    const startZ = sweepStartZ?.[index] ?? segment.z
    const terrain = field.sample(startX, startZ)
    const taper = 0.74 + Math.sin((index / (LOCOMOTION_SEGMENT_COUNT - 1)) * Math.PI) * 0.28
    const radius = ARTICULATED_WORM_CONTRACT.baseRadius * taper * anatomy.thicknessScale
    const tangent = segmentTangent(segments, index, fallbackHeading)
    const rightX = -tangent.z
    const rightZ = tangent.x
    const body = [
      {
        id: `segment-${index}`,
        offset: { x: 0, z: 0 },
        radius,
        vertical: {
          minY: segment.y - radius * anatomy.verticalScale,
          maxY: segment.y + radius * anatomy.verticalScale,
        },
      },
    ]
    for (const appendage of anatomy.appendages) {
      if (appendage.anchorIndex !== index) continue
      const lateral = appendage.side * radius * (1.1 + appendage.lengthScale * 0.7)
      body.push({
        id: `appendage-${appendage.id}`,
        offset: { x: rightX * lateral, z: rightZ * lateral },
        radius: radius * 0.62 * appendage.thicknessScale,
        vertical: {
          minY: segment.y - radius * 0.72,
          maxY: segment.y + radius * (0.7 + appendage.lengthScale * 0.55),
        },
      })
    }
    const collision = resolveSweptTerrariumMotion(collisionWorld, {
      position: { x: startX, z: startZ },
      displacement: { x: segment.x - startX, z: segment.z - startZ },
      velocity: { x: segment.vx, z: segment.vz },
      body,
      ground: {
        grounded: true,
        friction: terrain.friction,
        normalY: terrain.normal[1],
        contactRatio: plant.contactLoads[LOCOMOTION_SEGMENT_COUNT - 1 - index],
      },
    })
    segment.x = collision.position.x
    segment.z = collision.position.z
    segment.vx = collision.velocity.x
    segment.vz = collision.velocity.z
    for (const contact of collision.contacts) {
      contacts.push({
        segment: index,
        obstacleId: contact.id,
        normalX: contact.normal.x,
        normalZ: contact.normal.z,
      })
    }
    const firstContact = collision.contacts[0]
    if (firstContact) {
      if (contactMask) contactMask[index] = 1
      const plantIndex = LOCOMOTION_SEGMENT_COUNT - 1 - index
      plant.obstacleNormalX[plantIndex] = firstContact.normal.x
      plant.obstacleNormalZ[plantIndex] = firstContact.normal.z
    }
  }
}

function buildMuscleShape(
  plant: WormLocomotionPlant,
  heading: number,
  spacing: number,
  centerX: number,
  centerZ: number,
  targetX: Float64Array,
  targetZ: Float64Array,
) {
  targetX[0] = 0
  targetZ[0] = 0
  let yaw = heading
  for (let sceneIndex = 1; sceneIndex < LOCOMOTION_SEGMENT_COUNT; sceneIndex += 1) {
    const plantIndex = LOCOMOTION_SEGMENT_COUNT - sceneIndex
    const joint = plant.joints[plantIndex] ?? 0
    yaw += joint * ARTICULATED_WORM_CONTRACT.shapeBendScale
    targetX[sceneIndex] = targetX[sceneIndex - 1] + Math.cos(yaw) * spacing
    targetZ[sceneIndex] = targetZ[sceneIndex - 1] + Math.sin(yaw) * spacing
  }
  const localCenterX = mean(targetX)
  const localCenterZ = mean(targetZ)
  for (let index = 0; index < LOCOMOTION_SEGMENT_COUNT; index += 1) {
    targetX[index] += centerX - localCenterX
    targetZ[index] += centerZ - localCenterZ
  }
}

function satisfyDistanceConstraints(
  segments: SegmentSnapshot[],
  spacing: number,
  scratch: Pick<DynamicsScratch, 'correctionX' | 'correctionZ'>,
  contactMask?: Uint8Array,
) {
  const stiffness = 1 - ARTICULATED_WORM_CONTRACT.constraintCompliance
  for (let iteration = 0; iteration < ARTICULATED_WORM_CONTRACT.constraintIterations; iteration += 1) {
    scratch.correctionX.fill(0)
    scratch.correctionZ.fill(0)
    for (let index = 0; index < segments.length - 1; index += 1) {
      const first = segments[index]
      const second = segments[index + 1]
      const dx = second.x - first.x
      const dz = second.z - first.z
      const distance = Math.hypot(dx, dz)
      if (distance < 1e-9) continue
      const firstWeight = contactMask?.[index] ? 0 : 1
      const secondWeight = contactMask?.[index + 1] ? 0 : 1
      const weightTotal = firstWeight + secondWeight
      if (weightTotal <= 0) continue
      const correction = ((distance - spacing) / distance) * stiffness
      const correctionX = dx * correction
      const correctionZ = dz * correction
      scratch.correctionX[index] += correctionX * (firstWeight / weightTotal)
      scratch.correctionZ[index] += correctionZ * (firstWeight / weightTotal)
      scratch.correctionX[index + 1] -= correctionX * (secondWeight / weightTotal)
      scratch.correctionZ[index + 1] -= correctionZ * (secondWeight / weightTotal)
    }
    for (let index = 0; index < segments.length; index += 1) {
      segments[index].x += scratch.correctionX[index]
      segments[index].z += scratch.correctionZ[index]
    }
  }
}

function measureStep(
  plant: WormLocomotionPlant,
  segments: SegmentSnapshot[],
  oldCenterX: number,
  oldCenterZ: number,
  oldHeading: number,
  dt: number,
  contacts: ArticulatedWormContact[],
): ArticulatedWormStep {
  let x = 0
  let y = 0
  let z = 0
  let vx = 0
  let vz = 0
  for (const segment of segments) {
    x += segment.x
    y += segment.y
    z += segment.z
    vx += segment.vx
    vz += segment.vz
  }
  x /= segments.length
  y /= segments.length
  z /= segments.length
  vx /= segments.length
  vz /= segments.length
  const heading = bodyHeading(segments)
  plant.forwardSpeed = vx * Math.cos(heading) + vz * Math.sin(heading)
  plant.angularSpeed = clamp(wrapAngle(heading - oldHeading) / Math.max(dt, 1e-9), -3, 3)
  return {
    root: { x, y, z, vx, vz, heading },
    distance: Math.hypot(x - oldCenterX, z - oldCenterZ),
    contactRatio: plant.contactRatio,
    collisionCount: contacts.length,
    contacts,
  }
}

function bodyHeading(segments: readonly SegmentSnapshot[]) {
  const tail = segments[0]
  const head = segments[segments.length - 1]
  const dx = head.x - tail.x
  const dz = head.z - tail.z
  return Math.hypot(dx, dz) > 1e-8 ? Math.atan2(dz, dx) : head.yaw
}

function segmentTangent(segments: readonly SegmentSnapshot[], index: number, fallbackHeading: number) {
  const before = segments[Math.max(0, index - 1)]
  const after = segments[Math.min(segments.length - 1, index + 1)]
  const dx = after.x - before.x
  const dz = after.z - before.z
  const length = Math.hypot(dx, dz)
  return length > 1e-9
    ? { x: dx / length, z: dz / length }
    : { x: Math.cos(fallbackHeading), z: Math.sin(fallbackHeading) }
}

function limitBodyVelocities(segments: SegmentSnapshot[]) {
  let meanX = 0
  let meanZ = 0
  for (const segment of segments) {
    meanX += segment.vx / segments.length
    meanZ += segment.vz / segments.length
  }
  const meanSpeed = Math.hypot(meanX, meanZ)
  if (meanSpeed > ARTICULATED_WORM_CONTRACT.maximumSpeed) {
    const scale = ARTICULATED_WORM_CONTRACT.maximumSpeed / meanSpeed
    meanX *= scale
    meanZ *= scale
  }
  let clippedRelativeMeanX = 0
  let clippedRelativeMeanZ = 0
  for (const segment of segments) {
    let relativeX = segment.vx - meanX
    let relativeZ = segment.vz - meanZ
    const relativeSpeed = Math.hypot(relativeX, relativeZ)
    if (relativeSpeed > ARTICULATED_WORM_CONTRACT.maximumSpeed) {
      const scale = ARTICULATED_WORM_CONTRACT.maximumSpeed / relativeSpeed
      relativeX *= scale
      relativeZ *= scale
    }
    segment.vx = relativeX
    segment.vz = relativeZ
    clippedRelativeMeanX += relativeX / segments.length
    clippedRelativeMeanZ += relativeZ / segments.length
  }
  for (const segment of segments) {
    segment.vx += meanX - clippedRelativeMeanX
    segment.vz += meanZ - clippedRelativeMeanZ
  }
}

function scratchFor(plant: WormLocomotionPlant): DynamicsScratch {
  const existing = scratchByPlant.get(plant)
  if (existing) return existing
  const make = () => new Float64Array(LOCOMOTION_SEGMENT_COUNT)
  const scratch = {
    previousX: make(),
    previousY: make(),
    previousZ: make(),
    targetX: make(),
    targetZ: make(),
    accelerationX: make(),
    accelerationZ: make(),
    correctionX: make(),
    correctionZ: make(),
  }
  scratchByPlant.set(plant, scratch)
  return scratch
}

function mean(values: ArrayLike<number>) {
  let total = 0
  for (let index = 0; index < values.length; index += 1) total += values[index] ?? 0
  return total / Math.max(1, values.length)
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value))
}

function wrapAngle(value: number) {
  return Math.atan2(Math.sin(value), Math.cos(value))
}
