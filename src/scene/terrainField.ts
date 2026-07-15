import type { EnvironmentConfig, Vec3 } from '../creature/types'

export type TerrainSurface = 'sand' | 'moss' | 'clay'

export type TerrainSample = {
  /** Absolute world-space Y coordinate of the terrain top. */
  height: number
  friction: number
  normal: Vec3
  surface: TerrainSurface
}

export type TerrainField = {
  width: number
  depth: number
  /** Minimum and maximum heights at the vertices of the rendered terrain grid. */
  minimumHeight: number
  maximumHeight: number
  gridResolution: typeof TERRAIN_GRID_RESOLUTION
  /** Exact vertex height used by the rendered triangle mesh. */
  heightAtGridVertex: (xIndex: number, zIndex: number) => number
  waypoints: Array<[number, number]>
  sample: (x: number, z: number) => TerrainSample
}

export const TERRAIN_GRID_RESOLUTION = 64

const DEFAULT_WIDTH = 11.5
const DEFAULT_DEPTH = 11.5
const TERRAIN_TOP_Y = 0.04
const MIN_DIMENSION = 8
const MAX_DIMENSION = 24

export function createTerrainField(environmentConfig: EnvironmentConfig | null): TerrainField {
  const width = sanitizeDimension(environmentConfig?.world.size[0], DEFAULT_WIDTH)
  const depth = sanitizeDimension(environmentConfig?.world.size[2], DEFAULT_DEPTH)
  const halfWidth = width * 0.5
  const halfDepth = depth * 0.5
  const seed = finiteOr(environmentConfig?.seed, 1337)
  const terrain = environmentConfig?.terrain
  const terrainKind = terrain?.kind ?? 'bumps'
  const roughness = Math.max(0, finiteOr(terrain?.roughness, 0.18))
  const baseFriction = Math.max(0.05, finiteOr(terrain?.baseFriction, 0.9))
  const slopeRadians = degreesToRadians(finiteOr(terrain?.slopeDegrees, 0))
  const targetDirection = planarDirection(environmentConfig?.task.targetDirection)
  const routeAngle = Math.atan2(targetDirection[1], targetDirection[0])
  const seedPhase = seededNoise(seed, 17) * Math.PI * 2

  const hillCenter: [number, number] = [
    -width * (0.075 + seededNoise(seed, 31) * 0.035),
    depth * (0.045 + seededNoise(seed, 37) * 0.04),
  ]
  const moundCenter: [number, number] = [
    width * (0.17 + seededNoise(seed, 43) * 0.045),
    -depth * (0.12 + seededNoise(seed, 47) * 0.05),
  ]
  const hillRadius = Math.min(width, depth) * (0.14 + seededNoise(seed, 53) * 0.018)
  const moundRadius = Math.min(width, depth) * (0.09 + seededNoise(seed, 59) * 0.014)
  const hillHeight = terrainHillHeight(terrainKind, roughness)
  const moundHeight = terrainKind === 'flat' ? 0.08 : 0.16 + Math.min(roughness, 0.7) * 0.18
  const reliefScale = terrainReliefScale(terrainKind)

  const frictionAngle = seededNoise(seed, 71) * Math.PI * 2
  const mossCenter = rotatePoint(width * 0.2, -depth * 0.16, frictionAngle)
  const clayCenter = rotatePoint(-width * 0.21, depth * 0.2, frictionAngle)
  const mossRotation = frictionAngle * 0.47 + 0.35
  const clayRotation = frictionAngle * -0.38 - 0.2

  const analyticHeightAt = (x: number, z: number) => {
    const alongTarget = x * targetDirection[0] + z * targetDirection[1]
    const slope = Math.tan(slopeRadians) * alongTarget
    const hill = gaussianHill(x, z, hillCenter[0], hillCenter[1], hillRadius, hillHeight)
    const mound = gaussianHill(x, z, moundCenter[0], moundCenter[1], moundRadius, moundHeight)
    const microRelief =
      roughness *
      reliefScale *
      0.12 *
      (Math.sin(x * 1.15 + seedPhase) * Math.cos(z * 0.91 - seedPhase) +
        Math.sin(x * 2.4 - z * 1.7 + seedPhase * 0.61) * 0.35)

    return TERRAIN_TOP_Y + slope + hill + mound + microRelief
  }

  const row = TERRAIN_GRID_RESOLUTION + 1
  const vertexHeights = new Float64Array(row * row)
  for (let zIndex = 0; zIndex <= TERRAIN_GRID_RESOLUTION; zIndex += 1) {
    const z = (zIndex / TERRAIN_GRID_RESOLUTION - 0.5) * depth
    for (let xIndex = 0; xIndex <= TERRAIN_GRID_RESOLUTION; xIndex += 1) {
      const x = (xIndex / TERRAIN_GRID_RESOLUTION - 0.5) * width
      vertexHeights[zIndex * row + xIndex] = analyticHeightAt(x, z)
    }
  }
  const heightAtGridVertex = (xIndex: number, zIndex: number) => {
    if (
      !Number.isInteger(xIndex) ||
      !Number.isInteger(zIndex) ||
      xIndex < 0 ||
      zIndex < 0 ||
      xIndex > TERRAIN_GRID_RESOLUTION ||
      zIndex > TERRAIN_GRID_RESOLUTION
    ) {
      throw new Error('terrain grid vertex indices must be integers inside the rendered grid')
    }
    return vertexHeights[zIndex * row + xIndex]
  }

  // Match the two triangles emitted by makeTerrainGeometry exactly. Physics,
  // resource placement, and rendering now sample the same piecewise-planar
  // surface instead of an analytic field hidden below a triangulated mesh.
  const renderedSurfaceAt = (worldX: number, worldZ: number) => {
    const gridX = clamp((worldX / width + 0.5) * TERRAIN_GRID_RESOLUTION, 0, TERRAIN_GRID_RESOLUTION)
    const gridZ = clamp((worldZ / depth + 0.5) * TERRAIN_GRID_RESOLUTION, 0, TERRAIN_GRID_RESOLUTION)
    const xIndex = Math.min(TERRAIN_GRID_RESOLUTION - 1, Math.floor(gridX))
    const zIndex = Math.min(TERRAIN_GRID_RESOLUTION - 1, Math.floor(gridZ))
    const localX = gridX - xIndex
    const localZ = gridZ - zIndex
    const topLeft = heightAtGridVertex(xIndex, zIndex)
    const topRight = heightAtGridVertex(xIndex + 1, zIndex)
    const bottomLeft = heightAtGridVertex(xIndex, zIndex + 1)
    const bottomRight = heightAtGridVertex(xIndex + 1, zIndex + 1)
    const cellWidth = width / TERRAIN_GRID_RESOLUTION
    const cellDepth = depth / TERRAIN_GRID_RESOLUTION
    if (localX + localZ <= 1) {
      return {
        height: topLeft + localX * (topRight - topLeft) + localZ * (bottomLeft - topLeft),
        slopeX: (topRight - topLeft) / cellWidth,
        slopeZ: (bottomLeft - topLeft) / cellDepth,
      }
    }
    return {
      height:
        bottomRight + (1 - localX) * (bottomLeft - bottomRight) + (1 - localZ) * (topRight - bottomRight),
      slopeX: (bottomRight - bottomLeft) / cellWidth,
      slopeZ: (bottomRight - topRight) / cellDepth,
    }
  }

  const sample = (x: number, z: number): TerrainSample => {
    const surfacePoint = renderedSurfaceAt(x, z)
    const height = surfacePoint.height
    const slopeX = surfacePoint.slopeX
    const slopeZ = surfacePoint.slopeZ
    const normalLength = Math.hypot(slopeX, 1, slopeZ)
    const normal: Vec3 = [-slopeX / normalLength, 1 / normalLength, -slopeZ / normalLength]

    const mossInfluence = ellipseInfluence(
      x,
      z,
      mossCenter[0],
      mossCenter[1],
      width * 0.145,
      depth * 0.105,
      mossRotation,
    )
    const clayInfluence = ellipseInfluence(
      x,
      z,
      clayCenter[0],
      clayCenter[1],
      width * 0.15,
      depth * 0.115,
      clayRotation,
    )
    const friction = clamp(baseFriction * (1 - mossInfluence * 0.52 + clayInfluence * 0.45), 0.05, 3)
    const surface: TerrainSurface =
      mossInfluence > clayInfluence && mossInfluence > 0.16 ? 'moss' : clayInfluence > 0.16 ? 'clay' : 'sand'

    return { height, friction, normal, surface }
  }

  const minimumHeight = Math.min(...vertexHeights)
  const maximumHeight = Math.max(...vertexHeights)

  return {
    width,
    depth,
    minimumHeight,
    maximumHeight,
    gridResolution: TERRAIN_GRID_RESOLUTION,
    heightAtGridVertex,
    waypoints: makeWaypoints(halfWidth, halfDepth, routeAngle),
    sample,
  }
}

function makeWaypoints(halfWidth: number, halfDepth: number, routeAngle: number): Array<[number, number]> {
  // Route centers must leave room for the full rotating skateboard footprint,
  // not merely its root point. Otherwise a turn near glass can create a deep
  // initial overlap and a visible one-frame projection.
  const inset = Math.min(1.7, Math.min(halfWidth, halfDepth) * 0.3)
  const xLimit = halfWidth - inset
  const zLimit = halfDepth - inset
  const normalizedRoute: Array<[number, number]> = [
    [-0.62, -0.46],
    [0.43, -0.64],
    [0.68, 0.18],
    [0.15, 0.67],
    [-0.66, 0.31],
  ]

  return normalizedRoute.map(([x, z]) => {
    const rotated = rotatePoint(x * halfWidth, z * halfDepth, routeAngle)
    return [clamp(rotated[0], -xLimit, xLimit), clamp(rotated[1], -zLimit, zLimit)]
  })
}

function terrainHillHeight(kind: EnvironmentConfig['terrain']['kind'], roughness: number) {
  if (kind === 'flat') return 0.12
  if (kind === 'slope') return 0.24 + Math.min(roughness, 0.7) * 0.15
  if (kind === 'obstacle_field') return 0.48 + Math.min(roughness, 0.7) * 0.35
  return 0.52 + Math.min(roughness, 0.7) * 0.3
}

function terrainReliefScale(kind: EnvironmentConfig['terrain']['kind']) {
  if (kind === 'flat') return 0.12
  if (kind === 'slope') return 0.35
  if (kind === 'obstacle_field') return 1.15
  return 1
}

function gaussianHill(
  x: number,
  z: number,
  centerX: number,
  centerZ: number,
  radius: number,
  height: number,
) {
  const dx = x - centerX
  const dz = z - centerZ
  return height * Math.exp(-(dx * dx + dz * dz) / (2 * radius * radius))
}

function ellipseInfluence(
  x: number,
  z: number,
  centerX: number,
  centerZ: number,
  radiusX: number,
  radiusZ: number,
  rotation: number,
) {
  const local = rotatePoint(x - centerX, z - centerZ, -rotation)
  const distanceSquared =
    (local[0] * local[0]) / (radiusX * radiusX) + (local[1] * local[1]) / (radiusZ * radiusZ)
  return 1 - smoothstep(0.62, 1, distanceSquared)
}

function planarDirection(direction: Vec3 | undefined): [number, number] {
  const x = finiteOr(direction?.[0], 1)
  const z = finiteOr(direction?.[2], 0)
  const length = Math.hypot(x, z)
  return length > 1e-6 ? [x / length, z / length] : [1, 0]
}

function rotatePoint(x: number, z: number, angle: number): [number, number] {
  const cosine = Math.cos(angle)
  const sine = Math.sin(angle)
  return [x * cosine - z * sine, x * sine + z * cosine]
}

function seededNoise(seed: number, value: number) {
  const noise = Math.sin(seed * 12.9898 + value * 78.233) * 43758.5453
  return noise - Math.floor(noise)
}

function smoothstep(edge0: number, edge1: number, value: number) {
  const amount = clamp((value - edge0) / (edge1 - edge0), 0, 1)
  return amount * amount * (3 - 2 * amount)
}

function sanitizeDimension(value: number | undefined, fallback: number) {
  return clamp(Math.abs(finiteOr(value, fallback)), MIN_DIMENSION, MAX_DIMENSION)
}

function finiteOr(value: number | undefined, fallback: number) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function degreesToRadians(value: number) {
  return (value * Math.PI) / 180
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value))
}
