import {
  bodyCirclesFromWorldPoints,
  createTerrariumArenaBounds,
  prepareTerrariumCollisionWorld,
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

const preparedWorld = prepareTerrariumCollisionWorld(world)
expect(preparedWorld.prepared, 'prepared world marker is missing')
expect(Object.isFrozen(preparedWorld), 'prepared world is mutable')
expect(Object.isFrozen(preparedWorld.bounds), 'prepared bounds are mutable')
expect(Object.isFrozen(preparedWorld.obstacles), 'prepared obstacles are mutable')
expect(
  preparedWorld.obstacles.map(obstacle => obstacle.id).join(',') === 'rock-01,tree-01',
  'prepared obstacles were not deterministically sorted',
)
expect(
  prepareTerrariumCollisionWorld(preparedWorld) === preparedWorld,
  'preparing an already prepared world did not reuse it',
)

const mutableSourceObstacles: TerrariumCircleObstacle[] = [
  {
    id: 'snapshot-prop',
    kind: 'prop',
    center: { x: 0, z: 0 },
    radius: 0.4,
  },
]
const snapshotWorld = prepareTerrariumCollisionWorld({ bounds, obstacles: mutableSourceObstacles })
const snapshotInput = {
  position: { x: -2, z: 0 },
  displacement: { x: 4, z: 0 },
  velocity: { x: 4, z: 0 },
  body: rootBody,
}
const snapshotBeforeMutation = move(snapshotWorld, snapshotInput)
mutableSourceObstacles[0]!.center.x = 3
mutableSourceObstacles.push({
  id: 'late-prop',
  kind: 'prop',
  center: { x: -1, z: 0 },
  radius: 0.3,
})
const snapshotAfterMutation = move(snapshotWorld, snapshotInput)
expect(
  JSON.stringify(snapshotBeforeMutation) === JSON.stringify(snapshotAfterMutation),
  'prepared world retained mutable source references',
)

const tunnelling = move(world, {
  position: { x: -4, z: 0 },
  displacement: { x: 9, z: 0 },
  velocity: { x: 9, z: 0 },
  body: rootBody,
})
expect(tunnelling.hitObstacle, 'high-speed sweep missed the tree trunk')
expect(
  tunnelling.contacts.some(contact => contact.id === 'tree-01'),
  'tree contact was not reported',
)
expect(tunnelling.position.x <= -0.66, 'the worm tunnelled through the tree trunk')
expectSeparated(tunnelling, rootBody, obstacles)

const preparedTunnelling = move(preparedWorld, {
  position: { x: -4, z: 0 },
  displacement: { x: 9, z: 0 },
  velocity: { x: 9, z: 0 },
  body: rootBody,
})
expect(
  JSON.stringify(preparedTunnelling) === JSON.stringify(tunnelling),
  'prepared and legacy collision worlds disagree',
)

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
expect(
  bodyAware.contacts.some(contact => contact.bodyId === 'worm-2'),
  'solver only considered the root',
)
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
expect(
  arenaSweep.contacts.some(contact => contact.id === 'wall-right'),
  'right wall contact missing',
)
expect(
  arenaSweep.contacts.some(contact => contact.id === 'wall-far'),
  'corner slide did not reach the far wall',
)

const overlapped = move(world, {
  position: { x: 0.1, z: 0 },
  displacement: { x: 0, z: 0 },
  velocity: { x: -0.4, z: 0.2 },
  body: rootBody,
})
expect(
  overlapped.contacts.some(contact => contact.penetration > 0),
  'initial overlap was not reported',
)
expectSeparated(overlapped, rootBody, obstacles)

const wedgedObstacles: TerrariumCircleObstacle[] = [
  { id: 'close-rock-left', kind: 'rock', center: { x: -0.3, z: 0 }, radius: 0.3 },
  { id: 'close-rock-right', kind: 'rock', center: { x: 0.3, z: 0 }, radius: 0.3 },
]
const wedged = move(
  { bounds, obstacles: wedgedObstacles },
  {
    position: { x: 0, z: 0 },
    displacement: { x: 0, z: 0 },
    velocity: { x: 0, z: 0 },
    body: rootBody,
  },
)
expect(wedged.hitObstacle, 'wedged body did not report its prop contacts')
expectSeparated(wedged, rootBody, wedgedObstacles)
expect(Math.hypot(wedged.position.x, wedged.position.z) > 0.25, 'multi-prop overlap was not projected clear')

const lowObstacle: TerrariumCircleObstacle = {
  id: 'low-prop',
  kind: 'prop',
  center: { x: 0, z: 0 },
  radius: 0.5,
  vertical: { minY: 0, maxY: 0.5 },
}
const highObstacle: TerrariumCircleObstacle = {
  id: 'high-prop',
  kind: 'prop',
  center: { x: 0, z: 0 },
  radius: 0.5,
  vertical: { minY: 1, maxY: 2 },
}
const overBody: TerrariumBodyCircle[] = [
  {
    id: 'airborne',
    offset: { x: 0, z: 0 },
    radius: 0.2,
    vertical: { minY: 0.75, maxY: 0.95 },
  },
]
const underBody: TerrariumBodyCircle[] = [
  {
    id: 'grounded',
    offset: { x: 0, z: 0 },
    radius: 0.2,
    vertical: { minY: 0.1, maxY: 0.3 },
  },
]
const overlappingHeightBody: TerrariumBodyCircle[] = [
  {
    id: 'rising',
    offset: { x: 0, z: 0 },
    radius: 0.2,
    vertical: { minY: 0.4, maxY: 0.8 },
  },
]
const verticalMotion = {
  position: { x: -3, z: 0 },
  displacement: { x: 6, z: 0 },
  velocity: { x: 6, z: 0 },
}
const passedOver = move(prepareTerrariumCollisionWorld({ bounds, obstacles: [lowObstacle] }), {
  ...verticalMotion,
  body: overBody,
})
expect(!passedOver.hitObstacle, 'body above a bounded prop received a planar collision')
close(passedOver.position.x, 3, 1e-12, 'vertical overpass progress')

const passedUnder = move(prepareTerrariumCollisionWorld({ bounds, obstacles: [highObstacle] }), {
  ...verticalMotion,
  body: underBody,
})
expect(!passedUnder.hitObstacle, 'body below a bounded prop received a planar collision')
close(passedUnder.position.x, 3, 1e-12, 'vertical underpass progress')

const heightOverlap = move(
  { bounds, obstacles: [lowObstacle] },
  {
    ...verticalMotion,
    body: overlappingHeightBody,
  },
)
expect(heightOverlap.hitObstacle, 'overlapping vertical intervals skipped a planar collision')
expectSeparated(heightOverlap, overlappingHeightBody, [lowObstacle])

const legacyInfiniteHeight = move(
  { bounds, obstacles: [lowObstacle] },
  {
    ...verticalMotion,
    body: rootBody,
  },
)
expect(legacyInfiniteHeight.hitObstacle, 'legacy body circles stopped behaving as infinite-height samples')

expectThrows(
  () =>
    prepareTerrariumCollisionWorld({
      bounds,
      obstacles: [
        {
          id: 'inverted-height',
          kind: 'prop',
          center: { x: 0, z: 0 },
          radius: 0.2,
          vertical: { minY: 2, maxY: 1 },
        },
      ],
    }),
  'inverted vertical interval was accepted',
)

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
expect(
  JSON.stringify(deterministicA) === JSON.stringify(deterministicB),
  'identical sweeps are not deterministic',
)
expect(
  JSON.stringify(deterministicA) === JSON.stringify(deterministicShuffled),
  'obstacle input order changed the result',
)
expectSeparated(deterministicA, longBody, obstacles)

const nearTangent = move(
  {
    bounds,
    obstacles: [{ id: 'tangent', kind: 'rock', center: { x: 0, z: 0 }, radius: 0.5 }],
  },
  {
    position: { x: -4, z: 0.700099 },
    displacement: { x: 8, z: 0 },
    velocity: { x: 8, z: 0 },
    body: rootBody,
  },
)
expect(nearTangent.hitObstacle, 'near-tangent sweep was lost to discriminant precision')
expectSeparated(nearTangent, rootBody, [{ id: 'tangent', kind: 'rock', center: { x: 0, z: 0 }, radius: 0.5 }])

const extremeSweep = move(preparedWorld, {
  position: { x: -4, z: -3 },
  displacement: { x: 1e6, z: 3e5 },
  velocity: { x: 1e6, z: 3e5 },
  body: rootBody,
  maxIterations: 20,
})
expectFiniteResult(extremeSweep, 'extreme sweep')
expectWithinBounds(extremeSweep, rootBody, bounds)
expectSeparated(extremeSweep, rootBody, obstacles)

const randomizedCases = verifyRandomizedSweeps(240)

console.log('Terrarium collision verification passed.')
console.log(
  JSON.stringify(
    {
      tunnelStop: roundPoint(tunnelling.position),
      slideEnd: roundPoint(sliding.position),
      bodyStop: roundPoint(bodyAware.position),
      wedgedProjection: roundPoint(wedged.position),
      arenaEnd: roundPoint(arenaSweep.position),
      deterministicContacts: deterministicA.contacts.map(contact => `${contact.bodyId}:${contact.id}`),
      verticalFiltering: {
        over: roundPoint(passedOver.position),
        under: roundPoint(passedUnder.position),
        overlapStop: roundPoint(heightOverlap.position),
      },
      randomizedCases,
      support: {
        traction: round(grounded.support.traction),
        contactRatio: grounded.support.contactRatio,
      },
    },
    null,
    2,
  ),
)

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
      if (obstacle.enabled === false || !verticalIntervalsOverlap(sample.vertical, obstacle.vertical))
        continue
      const separation = Math.hypot(center.x - obstacle.center.x, center.z - obstacle.center.z)
      expect(
        separation + 1e-7 >= sample.radius + obstacle.radius,
        `${sample.id} remained inside ${obstacle.id}`,
      )
    }
  }
}

function expectWithinBounds(
  result: SweptBodyMotionResult,
  body: readonly TerrariumBodyCircle[],
  arena: ReturnType<typeof createTerrariumArenaBounds>,
) {
  for (const sample of body) {
    const center = {
      x: result.position.x + sample.offset.x,
      z: result.position.z + sample.offset.z,
    }
    expect(center.x - sample.radius >= arena.minX - 1e-7, `${sample.id} crossed the left bound`)
    expect(center.x + sample.radius <= arena.maxX + 1e-7, `${sample.id} crossed the right bound`)
    expect(center.z - sample.radius >= arena.minZ - 1e-7, `${sample.id} crossed the near bound`)
    expect(center.z + sample.radius <= arena.maxZ + 1e-7, `${sample.id} crossed the far bound`)
  }
}

function expectFiniteResult(result: SweptBodyMotionResult, label: string) {
  const values = [
    result.position.x,
    result.position.z,
    result.velocity.x,
    result.velocity.z,
    result.actualDisplacement.x,
    result.actualDisplacement.z,
    result.requestedDistance,
    result.traveledDistance,
    result.forwardProgressRatio,
    result.iterations,
    result.support.traction,
    result.support.obstacleContactRatio,
    result.support.obstacleFriction,
    result.support.slidingSpeed,
    result.support.blockedSpeed,
    ...result.contacts.flatMap(contact => [
      contact.normal.x,
      contact.normal.z,
      contact.point.x,
      contact.point.z,
      contact.time,
      contact.penetration,
      contact.friction,
    ]),
  ]
  expect(values.every(Number.isFinite), `${label} produced a non-finite number`)
}

function verifyRandomizedSweeps(caseCount: number) {
  const random = createRandom(0x57_55_52_4d)
  for (let caseIndex = 0; caseIndex < caseCount; caseIndex += 1) {
    const bodyVerticalChoice = caseIndex % 4
    const body: TerrariumBodyCircle[] = [
      {
        id: 'random-body',
        offset: { x: 0, z: 0 },
        radius: randomBetween(random, 0.06, 0.23),
        vertical:
          bodyVerticalChoice === 0
            ? { minY: 0.05, maxY: 0.45 }
            : bodyVerticalChoice === 1
              ? { minY: 0.8, maxY: 1.15 }
              : undefined,
      },
    ]
    const obstacleCount = 1 + Math.floor(random() * 6)
    const randomObstacles: TerrariumCircleObstacle[] = Array.from(
      { length: obstacleCount },
      (_, obstacleIndex) => {
        const verticalChoice = (caseIndex + obstacleIndex) % 4
        return {
          id: `random-${String(obstacleIndex).padStart(2, '0')}`,
          kind: obstacleIndex % 2 === 0 ? 'rock' : 'prop',
          center: {
            x: randomBetween(random, -2.7, 2.9),
            z: randomBetween(random, -2.9, 2.9),
          },
          radius: randomBetween(random, 0.1, 0.52),
          vertical:
            verticalChoice === 0
              ? { minY: 0, maxY: 0.55 }
              : verticalChoice === 1
                ? { minY: 0.7, maxY: 1.6 }
                : undefined,
          friction: randomBetween(random, 0, 1.2),
          restitution: randomBetween(random, 0, 0.35),
          enabled: (caseIndex + obstacleIndex) % 17 === 0 ? false : undefined,
        }
      },
    )
    const randomWorld: TerrariumCollisionWorld = {
      bounds,
      obstacles: randomObstacles,
      boundaryFriction: randomBetween(random, 0, 0.8),
      boundaryRestitution: randomBetween(random, 0, 0.25),
    }
    const displacement = {
      x: randomBetween(random, -2.5, 18),
      z: randomBetween(random, -9, 9),
    }
    const velocityScale = randomBetween(random, 0.2, 2.5)
    const input = {
      position: { x: -4.25, z: randomBetween(random, -2.8, 2.8) },
      displacement,
      velocity: {
        x: displacement.x * velocityScale,
        z: displacement.z * velocityScale,
      },
      body,
      maxIterations: 20,
    }

    const rawResult = move(randomWorld, input)
    const preparedResult = move(prepareTerrariumCollisionWorld(randomWorld), input)
    const shuffledResult = move({ ...randomWorld, obstacles: [...randomObstacles].reverse() }, input)
    const label = `random collision case ${caseIndex}`
    expectFiniteResult(rawResult, label)
    expectWithinBounds(rawResult, body, bounds)
    expectSeparated(rawResult, body, randomObstacles)
    expect(
      JSON.stringify(rawResult) === JSON.stringify(preparedResult),
      `${label} changed when its world was prepared`,
    )
    expect(
      JSON.stringify(rawResult) === JSON.stringify(shuffledResult),
      `${label} depended on obstacle input order`,
    )
  }
  return caseCount
}

function createRandom(seed: number) {
  let state = seed >>> 0
  return () => {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0
    return state / 0x1_0000_0000
  }
}

function randomBetween(random: () => number, minimum: number, maximum: number) {
  return minimum + (maximum - minimum) * random()
}

function verticalIntervalsOverlap(
  left: TerrariumBodyCircle['vertical'],
  right: TerrariumCircleObstacle['vertical'],
) {
  if (!left || !right) return true
  return left.maxY >= right.minY - 1e-8 && right.maxY >= left.minY - 1e-8
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

function expectThrows(action: () => unknown, message: string) {
  let threw = false
  try {
    action()
  } catch {
    threw = true
  }
  expect(threw, message)
}

function expect(condition: boolean, message: string): asserts condition {
  if (!condition) throw new Error(message)
}
