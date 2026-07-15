import {
  bodyCirclesFromWorldPoints,
  createTerrariumArenaBounds,
  resolveSweptTerrariumMotion,
  type PlanarPoint,
  type SweptBodyMotionResult,
  type TerrariumBodyCircle,
  type TerrariumCircleObstacle,
  type TerrariumCollisionWorld,
} from '../src/scene/terrariumCollisions'

const bounds = createTerrariumArenaBounds(10, 8, 0.2)
const obstacles: TerrariumCircleObstacle[] = [
  { id: 'tree-01', kind: 'tree', center: { x: 0, z: 0 }, radius: 0.46, friction: 0.48 },
  { id: 'rock-01', kind: 'rock', center: { x: 2.1, z: 1.4 }, radius: 0.34, friction: 0.62 },
]
const world: TerrariumCollisionWorld = { bounds, obstacles, boundaryFriction: 0.35 }
const rootBody: TerrariumBodyCircle[] = [{ id: 'root', offset: { x: 0, z: 0 }, radius: 0.2 }]

const tunnelling = move(world, {
  position: { x: -4, z: 0 },
  displacement: { x: 9, z: 0 },
  velocity: { x: 9, z: 0 },
  body: rootBody,
})
expect(tunnelling.hitObstacle, 'high-speed sweep missed the tree trunk')
expect(tunnelling.contacts.some((contact) => contact.id === 'tree-01'), 'tree contact was not reported')
expect(tunnelling.position.x <= -0.66, 'the worm tunnelled through the tree trunk')
expectSeparated(tunnelling, rootBody, obstacles)

const sliding = move(world, {
  position: { x: -2, z: -0.4 },
  displacement: { x: 3.8, z: 1.6 },
  velocity: { x: 3.8, z: 1.6 },
  body: rootBody,
})
expect(sliding.hitObstacle, 'angled motion never contacted the tree')
expect(sliding.position.z > 0.35, 'contact did not preserve useful tangent motion')
expect(Math.abs(sliding.velocity.z) > 0.05, 'sliding response erased all tangent velocity')
expect(sliding.support.slidingSpeed > 0, 'sliding contact telemetry was not populated')
expectSeparated(sliding, rootBody, obstacles)

const bodyPoints: PlanarPoint[] = [
  { x: -1.7, z: 0 },
  { x: -1.25, z: 0 },
  { x: -0.8, z: 0 },
]
const longBody = bodyCirclesFromWorldPoints({ x: -1.25, z: 0 }, bodyPoints, 0.16, 'worm')
const bodyAware = move(world, {
  position: { x: -1.25, z: 0 },
  displacement: { x: 1.4, z: 0 },
  velocity: { x: 1.4, z: 0 },
  body: longBody,
})
expect(bodyAware.hitObstacle, 'leading worm segment did not collide')
expect(bodyAware.contacts.some((contact) => contact.bodyId === 'worm-2'), 'solver only considered the root')
expect(bodyAware.position.x < -1.05, 'root advanced after the leading segment reached the trunk')
expectSeparated(bodyAware, longBody, obstacles)

const arenaSweep = move(world, {
  position: { x: 0, z: 2.8 },
  displacement: { x: 80, z: 40 },
  velocity: { x: 80, z: 40 },
  body: rootBody,
})
expect(arenaSweep.hitBoundary, 'large step missed the glass arena bounds')
expect(arenaSweep.position.x <= bounds.maxX - 0.2, 'body radius crossed the right wall')
expect(arenaSweep.position.z <= bounds.maxZ - 0.2, 'body radius crossed the far wall')
expect(arenaSweep.contacts.some((contact) => contact.id === 'wall-right'), 'right wall contact missing')
expect(arenaSweep.contacts.some((contact) => contact.id === 'wall-far'), 'corner slide did not reach the far wall')

const overlapped = move(world, {
  position: { x: 0.1, z: 0 },
  displacement: { x: 0, z: 0 },
  velocity: { x: -0.4, z: 0.2 },
  body: rootBody,
})
expect(overlapped.contacts.some((contact) => contact.penetration > 0), 'initial overlap was not reported')
expectSeparated(overlapped, rootBody, obstacles)

const wedgedObstacles: TerrariumCircleObstacle[] = [
  { id: 'close-rock-left', kind: 'rock', center: { x: -0.3, z: 0 }, radius: 0.3 },
  { id: 'close-rock-right', kind: 'rock', center: { x: 0.3, z: 0 }, radius: 0.3 },
]
const wedged = move({ bounds, obstacles: wedgedObstacles }, {
  position: { x: 0, z: 0 },
  displacement: { x: 0, z: 0 },
  velocity: { x: 0, z: 0 },
  body: rootBody,
})
expect(wedged.hitObstacle, 'wedged body did not report its prop contacts')
expectSeparated(wedged, rootBody, wedgedObstacles)
expect(Math.hypot(wedged.position.x, wedged.position.z) > 0.25, 'multi-prop overlap was not projected clear')

const grounded = move(world, {
  position: { x: -3.8, z: -2.4 },
  displacement: { x: 0.1, z: 0.05 },
  velocity: { x: 0.1, z: 0.05 },
  body: rootBody,
  ground: { grounded: true, friction: 0.82, normalY: 0.94, contactRatio: 0.76 },
})
close(grounded.support.traction, 0.82 * 0.94 * 0.76, 1e-12, 'ground traction telemetry')
expect(grounded.support.grounded, 'grounded support was lost')
close(grounded.support.contactRatio, 0.76, 1e-12, 'ground contact ratio')

const shuffledWorld: TerrariumCollisionWorld = { ...world, obstacles: [...obstacles].reverse() }
const deterministicInput = {
  position: { x: -3.4, z: -0.7 },
  displacement: { x: 7.2, z: 2.6 },
  velocity: { x: 7.2, z: 2.6 },
  body: longBody,
  ground: { grounded: true, friction: 0.71, normalY: 0.9, contactRatio: 0.84 },
}
const deterministicA = move(world, deterministicInput)
const deterministicB = move(world, deterministicInput)
const deterministicShuffled = move(shuffledWorld, deterministicInput)
expect(JSON.stringify(deterministicA) === JSON.stringify(deterministicB), 'identical sweeps are not deterministic')
expect(JSON.stringify(deterministicA) === JSON.stringify(deterministicShuffled), 'obstacle input order changed the result')
expectSeparated(deterministicA, longBody, obstacles)

console.log('Terrarium collision verification passed.')
console.log(JSON.stringify({
  tunnelStop: roundPoint(tunnelling.position),
  slideEnd: roundPoint(sliding.position),
  bodyStop: roundPoint(bodyAware.position),
  wedgedProjection: roundPoint(wedged.position),
  arenaEnd: roundPoint(arenaSweep.position),
  deterministicContacts: deterministicA.contacts.map((contact) => `${contact.bodyId}:${contact.id}`),
  support: {
    traction: round(grounded.support.traction),
    contactRatio: grounded.support.contactRatio,
  },
}, null, 2))

function move(
  collisionWorld: TerrariumCollisionWorld,
  motion: Parameters<typeof resolveSweptTerrariumMotion>[1],
) {
  return resolveSweptTerrariumMotion(collisionWorld, motion)
}

function expectSeparated(
  result: SweptBodyMotionResult,
  body: readonly TerrariumBodyCircle[],
  colliders: readonly TerrariumCircleObstacle[],
) {
  for (const sample of body) {
    const center = {
      x: result.position.x + sample.offset.x,
      z: result.position.z + sample.offset.z,
    }
    for (const obstacle of colliders) {
      const separation = Math.hypot(center.x - obstacle.center.x, center.z - obstacle.center.z)
      expect(
        separation + 1e-7 >= sample.radius + obstacle.radius,
        `${sample.id} remained inside ${obstacle.id}`,
      )
    }
  }
}

function roundPoint(point: PlanarPoint) {
  return { x: round(point.x), z: round(point.z) }
}

function round(value: number) {
  return Number(value.toFixed(6))
}

function close(actual: number, expected: number, tolerance: number, label: string) {
  expect(Math.abs(actual - expected) <= tolerance, `${label}: expected ${expected}, received ${actual}`)
}

function expect(condition: boolean, message: string): asserts condition {
  if (!condition) throw new Error(message)
}
