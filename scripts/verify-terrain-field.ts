import { readFile } from 'node:fs/promises'
import type { EnvironmentConfig } from '../src/creature/types'
import { createTerrainField, type TerrainSurface } from '../src/scene/terrainField'

const environmentFiles = ['adaptive-skate-terrarium.json', 'ripple-yard.json', 'tilt-basin.json']

const environments = await Promise.all(
  environmentFiles.map(async file => {
    const url = new URL('../public/configs/environments/' + file, import.meta.url)
    return JSON.parse(await readFile(url, 'utf8')) as EnvironmentConfig
  }),
)

for (const environment of environments) {
  verifyEnvironmentTerrain(environment)
}

const baseline = environments[0]
const alternateSeed = createTerrainField({ ...baseline, seed: baseline.seed + 1 })
const baselineSample = createTerrainField(baseline).sample(1.234, -0.987)
const alternateSample = alternateSeed.sample(1.234, -0.987)
assert(
  Math.abs(baselineSample.height - alternateSample.height) > 1e-6 ||
    Math.abs(baselineSample.friction - alternateSample.friction) > 1e-6,
  'Changing the environment seed must change the terrain field.',
)

console.log('Terrain field verification passed.')

function verifyEnvironmentTerrain(environment: EnvironmentConfig) {
  const first = createTerrainField(environment)
  const second = createTerrainField(environment)
  assert(first.width === environment.world.size[0], environment.id + ' width must come from world.size[0].')
  assert(first.depth === environment.world.size[2], environment.id + ' depth must come from world.size[2].')
  assert(Number.isFinite(first.minimumHeight), environment.id + ' rendered minimum height must be finite.')
  assert(Number.isFinite(first.maximumHeight), environment.id + ' rendered maximum height must be finite.')
  assert(
    first.maximumHeight > first.minimumHeight,
    environment.id + ' rendered height range must be ordered.',
  )
  assert(Math.abs(first.width - first.depth) < 1e-9, environment.id + ' must configure a square arena.')
  assert(first.waypoints.length >= 5, environment.id + ' must provide a useful 2D waypoint loop.')
  assert(first.gridResolution === 64, environment.id + ' must expose the rendered grid contract.')

  for (const [xIndex, zIndex] of [
    [0, 0],
    [7, 19],
    [32, 32],
    [64, 64],
  ] as const) {
    const x = (xIndex / first.gridResolution - 0.5) * first.width
    const z = (zIndex / first.gridResolution - 0.5) * first.depth
    assert(
      Math.abs(first.sample(x, z).height - first.heightAtGridVertex(xIndex, zIndex)) < 1e-12,
      environment.id + ' physics height must equal the rendered vertex height.',
    )
  }
  verifyTriangleInterpolation(first, environment.id)

  const waypointXs = first.waypoints.map(waypoint => waypoint[0])
  const waypointZs = first.waypoints.map(waypoint => waypoint[1])
  for (const [x, z] of first.waypoints) {
    assert(Math.abs(x) < first.width * 0.5, environment.id + ' waypoint x must remain inside the arena.')
    assert(Math.abs(z) < first.depth * 0.5, environment.id + ' waypoint z must remain inside the arena.')
  }
  assert(range(waypointXs) > first.width * 0.5, environment.id + ' waypoints must traverse the x axis.')
  assert(range(waypointZs) > first.depth * 0.5, environment.id + ' waypoints must traverse the z axis.')

  const surfaces = new Set<TerrainSurface>()
  let minimumHeight = Number.POSITIVE_INFINITY
  let maximumHeight = Number.NEGATIVE_INFINITY
  let minimumFriction = Number.POSITIVE_INFINITY
  let maximumFriction = Number.NEGATIVE_INFINITY

  for (let xIndex = 0; xIndex <= 32; xIndex += 1) {
    const x = -first.width * 0.5 + (xIndex / 32) * first.width
    for (let zIndex = 0; zIndex <= 32; zIndex += 1) {
      const z = -first.depth * 0.5 + (zIndex / 32) * first.depth
      const sample = first.sample(x, z)
      const repeated = second.sample(x, z)
      assertSamplesEqual(sample, repeated, environment.id + ' must be deterministic.')
      assert(Number.isFinite(sample.height), environment.id + ' height must be finite.')
      assert(
        Number.isFinite(sample.friction) && sample.friction > 0,
        environment.id + ' friction must be positive.',
      )
      assert(sample.normal.every(Number.isFinite), environment.id + ' normal must be finite.')
      assert(
        Math.abs(Math.hypot(...sample.normal) - 1) < 1e-9,
        environment.id + ' normal must be normalized.',
      )
      minimumHeight = Math.min(minimumHeight, sample.height)
      maximumHeight = Math.max(maximumHeight, sample.height)
      minimumFriction = Math.min(minimumFriction, sample.friction)
      maximumFriction = Math.max(maximumFriction, sample.friction)
      surfaces.add(sample.surface)
    }
  }

  assert(maximumHeight - minimumHeight > 0.2, environment.id + ' must contain meaningful terrain relief.')
  assert(
    maximumFriction - minimumFriction > environment.terrain.baseFriction * 0.35,
    environment.id + ' must vary friction visibly.',
  )
  assert(surfaces.has('sand'), environment.id + ' must contain sand.')
  assert(surfaces.has('moss'), environment.id + ' must contain a low-friction moss region.')
  assert(surfaces.has('clay'), environment.id + ' must contain a high-friction clay region.')
}

function verifyTriangleInterpolation(field: ReturnType<typeof createTerrainField>, environmentId: string) {
  const xIndex = 11
  const zIndex = 23
  const localX = 0.27
  const localZ = 0.38
  const topLeft = field.heightAtGridVertex(xIndex, zIndex)
  const topRight = field.heightAtGridVertex(xIndex + 1, zIndex)
  const bottomLeft = field.heightAtGridVertex(xIndex, zIndex + 1)
  const expected = topLeft + localX * (topRight - topLeft) + localZ * (bottomLeft - topLeft)
  const x = ((xIndex + localX) / field.gridResolution - 0.5) * field.width
  const z = ((zIndex + localZ) / field.gridResolution - 0.5) * field.depth
  assert(
    Math.abs(field.sample(x, z).height - expected) < 1e-12,
    environmentId + ' physics must use the rendered triangle plane between vertices.',
  )
}

function assertSamplesEqual(
  first: ReturnType<ReturnType<typeof createTerrainField>['sample']>,
  second: ReturnType<ReturnType<typeof createTerrainField>['sample']>,
  message: string,
) {
  assert(first.height === second.height, message)
  assert(first.friction === second.friction, message)
  assert(first.surface === second.surface, message)
  assert(
    first.normal.every((value, index) => value === second.normal[index]),
    message,
  )
}

function range(values: number[]) {
  return Math.max(...values) - Math.min(...values)
}

function assert(condition: boolean, message: string): asserts condition {
  if (!condition) {
    throw new Error(message)
  }
}
