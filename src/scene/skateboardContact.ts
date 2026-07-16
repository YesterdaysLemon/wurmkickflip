import { MathUtils } from 'three'
import type { WurmAnatomy } from '../creature/anatomy'
import type { EnvironmentConfig } from '../creature/types'
import { LOCOMOTION_CONTRACT } from '../policy/locomotionContract'
import type { SegmentSnapshot } from '../policy/types'
import type { TerrainField } from './terrainField'

export type SkateboardDeckPose = {
  x: number
  y: number
  z: number
  vx: number
  vz: number
  heading: number
  pitch: number
  roll: number
}

export type SkateboardDeckContact = {
  ratio: number
  segmentCount: number
  headWeight: number
  midbodyWeight: number
  tailWeight: number
  centerOfSupportX: number
  centerOfSupportZ: number
  relativeSpeed: number
}

const BOARDING_EDGE_RAMP = 0.18
const BOARDING_FRICTION = 5.4

/**
 * Measure articulated deck contact and apply board-relative planar friction.
 *
 * This is an environmental contact response, not a mount animation: it never
 * writes position, heading, or a target body pose. Vertical support is exposed
 * to the plant through `withSkateboardDeckSupportField`, so its measured contact
 * loads and Y integration agree. Segments reach the deck only when their neural,
 * traction-mediated motion carries them into the oriented footprint.
 */
export function applySkateboardDeckSupport(
  segments: SegmentSnapshot[],
  board: SkateboardDeckPose,
  environmentConfig: EnvironmentConfig | null,
  anatomy: WurmAnatomy,
  delta: number,
): SkateboardDeckContact {
  const deck = skateboardDeckDimensions(environmentConfig)
  const groundClearance = LOCOMOTION_CONTRACT.dynamics.baseGroundClearance * anatomy.verticalScale
  const captureHeight =
    Math.max(deck.height + groundClearance * 2, environmentConfig?.skateboard.mountHeightTolerance ?? 0.38) +
    groundClearance * 0.6
  const velocityBlendScale = 1 - Math.exp(-BOARDING_FRICTION * Math.max(0, delta))
  const weights = new Array<number>(segments.length).fill(0)
  let weightTotal = 0
  let centerX = 0
  let centerZ = 0
  let relativeSpeedTotal = 0
  let segmentCount = 0

  for (let index = 0; index < segments.length; index += 1) {
    const segment = segments[index]
    const planarWeight = skateboardDeckSupportWeight(segment.x, segment.z, board, environmentConfig, anatomy)
    if (planarWeight <= 0) continue

    const supportedY =
      skateboardDeckSurfaceY(segment.x, segment.z, board, environmentConfig) + groundClearance
    const verticalGap = supportedY - segment.y
    // A low worm can climb the rounded deck edge, but a body already far above
    // the deck cannot be magnetically captured from mid-air.
    const reachable = verticalGap >= -groundClearance * 0.8 && verticalGap <= captureHeight
    if (!reachable) continue

    const contactHeightError = Math.abs(supportedY - segment.y)
    const heightWeight = 1 - MathUtils.clamp(contactHeightError / captureHeight, 0, 1)
    const weight = planarWeight * heightWeight
    if (weight <= 0) continue

    const velocityBlend = velocityBlendScale * weight
    segment.vx = MathUtils.lerp(segment.vx, board.vx, velocityBlend)
    segment.vz = MathUtils.lerp(segment.vz, board.vz, velocityBlend)
    weights[index] = weight
    weightTotal += weight
    centerX += segment.x * weight
    centerZ += segment.z * weight
    relativeSpeedTotal += Math.hypot(segment.vx - board.vx, segment.vz - board.vz) * weight
    if (weight >= 0.22) segmentCount += 1
  }

  const ratio = segments.length > 0 ? weightTotal / segments.length : 0
  const supportDivisor = Math.max(1e-9, weightTotal)
  return {
    ratio,
    segmentCount,
    tailWeight: regionWeight(weights, 0, Math.ceil(weights.length * 0.34)),
    midbodyWeight: regionWeight(weights, Math.floor(weights.length * 0.33), Math.ceil(weights.length * 0.67)),
    headWeight: regionWeight(weights, Math.floor(weights.length * 0.66), weights.length),
    centerOfSupportX: weightTotal > 0 ? centerX / supportDivisor : board.x,
    centerOfSupportZ: weightTotal > 0 ? centerZ / supportDivisor : board.z,
    relativeSpeed: weightTotal > 0 ? relativeSpeedTotal / supportDivisor : 0,
  }
}

/**
 * Present the oriented deck as a real support surface to the articulated plant.
 * This keeps vertical position, contact loads, and neural afferents on the same
 * surface instead of projecting to terrain and correcting upward afterward.
 */
export function withSkateboardDeckSupportField(
  field: TerrainField,
  board: SkateboardDeckPose,
  environmentConfig: EnvironmentConfig | null,
  anatomy: WurmAnatomy,
): TerrainField {
  const deckFriction = MathUtils.clamp(environmentConfig?.skateboard.wheelFriction ?? 0.82, 0.05, 1.5)
  return {
    ...field,
    sample(x, z) {
      const terrain = field.sample(x, z)
      const weight = skateboardDeckSupportWeight(x, z, board, environmentConfig, anatomy)
      const deckTop = skateboardDeckSurfaceY(x, z, board, environmentConfig)
      if (weight <= 0 || deckTop <= terrain.height) return terrain
      const boardNormal = skateboardDeckNormal(board)
      const normalX = MathUtils.lerp(terrain.normal[0], boardNormal[0], weight)
      const normalY = MathUtils.lerp(terrain.normal[1], boardNormal[1], weight)
      const normalZ = MathUtils.lerp(terrain.normal[2], boardNormal[2], weight)
      const normalLength = Math.max(1e-9, Math.hypot(normalX, normalY, normalZ))
      return {
        ...terrain,
        height: MathUtils.lerp(terrain.height, deckTop, weight),
        friction: MathUtils.lerp(terrain.friction, deckFriction, weight),
        normal: [normalX / normalLength, normalY / normalLength, normalZ / normalLength],
      }
    },
  }
}

/** Top face of the visible pitched/rolled deck at a world-space planar point. */
export function skateboardDeckSurfaceY(
  x: number,
  z: number,
  board: SkateboardDeckPose,
  environmentConfig: EnvironmentConfig | null,
) {
  const deck = skateboardDeckDimensions(environmentConfig)
  const { localForward, localRight } = skateboardLocalPlanar(x, z, board)
  return (
    board.y +
    deck.height * 0.5 +
    0.014 +
    Math.sin(board.pitch) * localForward -
    Math.sin(board.roll) * localRight
  )
}

/** Pure oriented-deck query used by lifecycle gates and deterministic verifiers. */
export function skateboardDeckSupportWeight(
  x: number,
  z: number,
  board: Pick<SkateboardDeckPose, 'x' | 'z' | 'heading'>,
  environmentConfig: EnvironmentConfig | null,
  anatomy: WurmAnatomy,
) {
  const deck = skateboardDeckDimensions(environmentConfig)
  const { localForward, localRight } = skateboardLocalPlanar(x, z, board)
  const segmentRadius = LOCOMOTION_CONTRACT.dynamics.baseRadius * anatomy.thicknessScale
  const forwardInset = deck.length * 0.5 + segmentRadius - Math.abs(localForward)
  const lateralInset = deck.width * 0.5 + segmentRadius - Math.abs(localRight)
  const shallowestInset = Math.min(forwardInset, lateralInset)
  if (shallowestInset <= 0) return 0
  return smoothStep(MathUtils.clamp(shallowestInset / BOARDING_EDGE_RAMP, 0, 1))
}

function skateboardLocalPlanar(x: number, z: number, board: Pick<SkateboardDeckPose, 'x' | 'z' | 'heading'>) {
  const dx = x - board.x
  const dz = z - board.z
  const forwardX = Math.cos(board.heading)
  const forwardZ = Math.sin(board.heading)
  return {
    localForward: dx * forwardX + dz * forwardZ,
    localRight: -dx * forwardZ + dz * forwardX,
  }
}

function skateboardDeckNormal(board: SkateboardDeckPose): [number, number, number] {
  const forwardX = Math.cos(board.heading)
  const forwardZ = Math.sin(board.heading)
  const rightX = -forwardZ
  const rightZ = forwardX
  const normalForward = -Math.sin(board.pitch)
  const normalRight = Math.sin(board.roll)
  const normalY = Math.max(0.1, Math.cos(board.pitch) * Math.cos(board.roll))
  const x = normalForward * forwardX + normalRight * rightX
  const z = normalForward * forwardZ + normalRight * rightZ
  const length = Math.max(1e-9, Math.hypot(x, normalY, z))
  return [x / length, normalY / length, z / length]
}

export function skateboardDeckDimensions(environmentConfig: EnvironmentConfig | null) {
  const configured = environmentConfig?.skateboard.deckSize ?? [1.9, 0.12, 0.58]
  return {
    length: MathUtils.clamp(configured[0], 1.65, 2.1),
    height: MathUtils.clamp(configured[1], 0.09, 0.15),
    width: MathUtils.clamp(configured[2], 0.52, 0.68),
  }
}

function regionWeight(weights: readonly number[], start: number, end: number) {
  const count = Math.max(1, end - start)
  let total = 0
  for (let index = start; index < end; index += 1) total += weights[index] ?? 0
  return total / count
}

function smoothStep(value: number) {
  const t = MathUtils.clamp(value, 0, 1)
  return t * t * (3 - 2 * t)
}
