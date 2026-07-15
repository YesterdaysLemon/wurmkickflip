export type PlanarPoint = {
  x: number
  z: number
}

export type ArenaBounds = {
  minX: number
  maxX: number
  minZ: number
  maxZ: number
}

/** A vertical prop footprint in the terrarium's X/Z plane. */
export type TerrariumCircleObstacle = {
  id: string
  kind: 'tree' | 'rock' | 'bowl' | 'skateboard' | 'prop'
  center: PlanarPoint
  radius: number
  /** Coulomb friction coefficient used when motion is projected onto the tangent. */
  friction?: number
  restitution?: number
  enabled?: boolean
}

/**
 * A circle fixed to the moving root for the duration of one integration step.
 * Supplying the rendered segment centers here makes the whole worm collide,
 * rather than approximating its long body as a point at the root.
 */
export type TerrariumBodyCircle = {
  id: string
  offset: PlanarPoint
  radius: number
}

export type TerrariumCollisionWorld = {
  bounds: ArenaBounds
  obstacles: readonly TerrariumCircleObstacle[]
  boundaryFriction?: number
  boundaryRestitution?: number
}

export type GroundContactInput = {
  grounded: boolean
  friction: number
  normalY?: number
  contactRatio?: number
}

export type SweptBodyMotion = {
  position: PlanarPoint
  displacement: PlanarPoint
  velocity: PlanarPoint
  body: readonly TerrariumBodyCircle[]
  ground?: GroundContactInput
  /** Small separation shell. It prevents numerical re-entry without a visible gap. */
  skin?: number
  maxIterations?: number
}

export type TerrariumPlanarContact = {
  type: 'obstacle' | 'boundary'
  id: string
  obstacleKind?: TerrariumCircleObstacle['kind']
  bodyId: string
  normal: PlanarPoint
  point: PlanarPoint
  /** Normalized time along the requested step, where zero is the start. */
  time: number
  /** Positive only when an initially intersecting body had to be projected out. */
  penetration: number
  friction: number
}

export type TerrariumSupportContact = {
  grounded: boolean
  friction: number
  normalY: number
  contactRatio: number
  traction: number
  obstacleContactRatio: number
  obstacleFriction: number
  slidingSpeed: number
  blockedSpeed: number
}

export type SweptBodyMotionResult = {
  position: PlanarPoint
  velocity: PlanarPoint
  actualDisplacement: PlanarPoint
  requestedDistance: number
  traveledDistance: number
  forwardProgressRatio: number
  contacts: TerrariumPlanarContact[]
  hitObstacle: boolean
  hitBoundary: boolean
  iterations: number
  support: TerrariumSupportContact
}

type SweepCandidate = {
  time: number
  type: TerrariumPlanarContact['type']
  id: string
  obstacleKind?: TerrariumCircleObstacle['kind']
  body: TerrariumBodyCircle
  normal: PlanarPoint
  point: PlanarPoint
  friction: number
  restitution: number
  sortKey: string
}

type ProjectionCandidate = Omit<SweepCandidate, 'time'> & {
  penetration: number
}

type ConfigurationCircle = {
  center: PlanarPoint
  radius: number
  body: TerrariumBodyCircle
  obstacle: TerrariumCircleObstacle
  sortKey: string
}

const DEFAULT_SKIN = 1e-4
const DEFAULT_ITERATIONS = 10
const DEFAULT_OBSTACLE_FRICTION = 0.38
const DEFAULT_BOUNDARY_FRICTION = 0.3
const EPSILON = 1e-9
const HIT_EPSILON = 1e-8

/** Build centered glass-wall bounds with a configurable visual inset. */
export function createTerrariumArenaBounds(width: number, depth: number, inset = 0): ArenaBounds {
  requirePositive(width, 'arena width')
  requirePositive(depth, 'arena depth')
  requireNonNegative(inset, 'arena inset')
  const halfWidth = width * 0.5 - inset
  const halfDepth = depth * 0.5 - inset
  if (halfWidth <= 0 || halfDepth <= 0) throw new Error('arena inset leaves no navigable area')
  return { minX: -halfWidth, maxX: halfWidth, minZ: -halfDepth, maxZ: halfDepth }
}

/** Convert current world-space segment centers into root-relative collision probes. */
export function bodyCirclesFromWorldPoints(
  root: PlanarPoint,
  points: readonly PlanarPoint[],
  radius: number,
  idPrefix = 'segment',
): TerrariumBodyCircle[] {
  requirePoint(root, 'body root')
  requirePositive(radius, 'body radius')
  return points.map((point, index) => {
    requirePoint(point, `body point ${index}`)
    return {
      id: `${idPrefix}-${index}`,
      offset: { x: point.x - root.x, z: point.z - root.z },
      radius,
    }
  })
}

/**
 * Continuously sweeps a root with one or more attached circular body samples.
 * Earliest contacts are resolved before the remaining motion is projected onto
 * their tangents, preventing tunnelling while still allowing natural sliding.
 */
export function resolveSweptTerrariumMotion(
  world: TerrariumCollisionWorld,
  motion: SweptBodyMotion,
): SweptBodyMotionResult {
  validateWorld(world)
  validateMotion(motion)
  const skin = motion.skin ?? DEFAULT_SKIN
  requireNonNegative(skin, 'collision skin')
  const requestedIterations = motion.maxIterations ?? DEFAULT_ITERATIONS
  requirePositive(requestedIterations, 'collision iteration count')
  const maxIterations = Math.max(1, Math.floor(requestedIterations))
  const body = [...motion.body].sort((left, right) => compareText(left.id, right.id))
  validateBodyFitsBounds(body, world.bounds, skin)
  const obstacles = world.obstacles
    .filter((obstacle) => obstacle.enabled !== false)
    .slice()
    .sort((left, right) => compareText(left.id, right.id))
  const contactMap = new Map<string, TerrariumPlanarContact>()
  const start = copyPoint(motion.position)
  let position = copyPoint(motion.position)
  let velocity = copyPoint(motion.velocity)
  let traveledDistance = 0

  const projection = projectInitialIntersections(
    position,
    velocity,
    body,
    obstacles,
    world,
    skin,
    maxIterations * 2,
    contactMap,
  )
  position = projection.position
  velocity = projection.velocity
  traveledDistance += projection.distance

  let remaining = copyPoint(motion.displacement)
  let elapsed = 0
  let remainingTime = 1
  let iterations = 0

  while (lengthSquared(remaining) > EPSILON * EPSILON && iterations < maxIterations) {
    const hits = findEarliestSweepHits(position, remaining, body, obstacles, world, skin)
    if (hits.length === 0) {
      position = add(position, remaining)
      traveledDistance += length(remaining)
      remaining = { x: 0, z: 0 }
      break
    }

    const time = clamp(hits[0]?.time ?? 0, 0, 1)
    const advance = scale(remaining, time)
    position = add(position, advance)
    traveledDistance += length(advance)
    elapsed += remainingTime * time
    remainingTime *= 1 - time
    remaining = scale(remaining, 1 - time)

    for (const hit of hits) {
      addContact(contactMap, contactFromSweep(hit, elapsed))
      remaining = coulombSlide(remaining, hit.normal, hit.friction, 0)
      velocity = coulombSlide(velocity, hit.normal, hit.friction, hit.restitution)
    }

    iterations += 1
  }

  // A crowded corner can consume the iteration budget. Stopping the unresolved
  // remainder is preferable to tunnelling through a prop on that rare frame.
  if (lengthSquared(remaining) > EPSILON * EPSILON) remaining = { x: 0, z: 0 }

  const finalProjection = projectInitialIntersections(
    position,
    velocity,
    body,
    obstacles,
    world,
    skin,
    maxIterations * 2,
    contactMap,
  )
  position = finalProjection.position
  velocity = finalProjection.velocity
  traveledDistance += finalProjection.distance

  const contacts = [...contactMap.values()].sort(compareContacts)
  const requestedDistance = length(motion.displacement)
  const actualDisplacement = subtract(position, start)
  const requestDirection = requestedDistance > EPSILON
    ? scale(motion.displacement, 1 / requestedDistance)
    : { x: 0, z: 0 }
  const forwardProgress = Math.max(0, dot(actualDisplacement, requestDirection))
  const obstacleBodyIds = new Set(
    contacts.filter((contact) => contact.type === 'obstacle').map((contact) => contact.bodyId),
  )
  const obstacleContacts = contacts.filter((contact) => contact.type === 'obstacle')
  const ground = supportContactFor(motion.ground, obstacleContacts, obstacleBodyIds.size / body.length, velocity)
  const inputSpeed = length(motion.velocity)
  const outputSpeed = length(velocity)

  return {
    position,
    velocity,
    actualDisplacement,
    requestedDistance,
    traveledDistance,
    forwardProgressRatio: requestedDistance > EPSILON
      ? clamp(forwardProgress / requestedDistance, 0, 1)
      : 1,
    contacts,
    hitObstacle: obstacleContacts.length > 0,
    hitBoundary: contacts.some((contact) => contact.type === 'boundary'),
    iterations,
    support: {
      ...ground,
      blockedSpeed: Math.max(0, inputSpeed - outputSpeed),
    },
  }
}

function projectInitialIntersections(
  initialPosition: PlanarPoint,
  initialVelocity: PlanarPoint,
  body: readonly TerrariumBodyCircle[],
  obstacles: readonly TerrariumCircleObstacle[],
  world: TerrariumCollisionWorld,
  skin: number,
  maxIterations: number,
  contacts: Map<string, TerrariumPlanarContact>,
) {
  let position = copyPoint(initialPosition)
  let velocity = copyPoint(initialVelocity)
  let distance = 0

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    const candidate = deepestProjection(position, body, obstacles, world, skin)
    if (!candidate || candidate.penetration <= EPSILON) break
    const correction = scale(candidate.normal, candidate.penetration)
    position = add(position, correction)
    distance += length(correction)
    velocity = coulombSlide(velocity, candidate.normal, candidate.friction, candidate.restitution)
    addContact(contacts, contactFromProjection(candidate))
  }

  const unresolved = deepestProjection(position, body, obstacles, world, skin)
  if (unresolved && unresolved.penetration > EPSILON) {
    const clearPosition = nearestClearConfiguration(position, body, obstacles, world.bounds, skin)
    const correction = subtract(clearPosition, position)
    const correctionNormal = normalizeOr(correction, unresolved.normal)
    position = clearPosition
    distance += length(correction)
    velocity = coulombSlide(velocity, correctionNormal, unresolved.friction, unresolved.restitution)
    addContact(contacts, contactFromProjection(unresolved))
  }

  return { position, velocity, distance }
}

/**
 * Find the nearest valid root position on the boundary of the overlapping
 * configuration-space obstacle union. The iterative projection above handles
 * ordinary contact cheaply; this exact fallback handles spawn points wedged
 * between two touching props without leaving the body intersecting either.
 */
function nearestClearConfiguration(
  origin: PlanarPoint,
  body: readonly TerrariumBodyCircle[],
  obstacles: readonly TerrariumCircleObstacle[],
  bounds: ArenaBounds,
  skin: number,
) {
  const rootBounds = rootBoundsForBody(body, bounds, skin)
  const circles: ConfigurationCircle[] = []
  for (const sample of body) {
    for (const obstacle of obstacles) {
      circles.push({
        center: subtract(obstacle.center, sample.offset),
        radius: obstacle.radius + sample.radius + skin,
        body: sample,
        obstacle,
        sortKey: `${obstacle.id}:${sample.id}`,
      })
    }
  }
  circles.sort((left, right) => compareText(left.sortKey, right.sortKey))

  const component = overlappingCircleComponent(origin, circles)
  let best: PlanarPoint | null = null
  let bestDistanceSquared = Number.POSITIVE_INFINITY
  const consider = (candidate: PlanarPoint) => {
    if (!configurationPointIsClear(candidate, rootBounds, circles)) return
    const candidateDistanceSquared = lengthSquared(subtract(candidate, origin))
    if (
      candidateDistanceSquared < bestDistanceSquared - HIT_EPSILON ||
      (
        Math.abs(candidateDistanceSquared - bestDistanceSquared) <= HIT_EPSILON &&
        (!best || candidate.x < best.x - HIT_EPSILON ||
          (Math.abs(candidate.x - best.x) <= HIT_EPSILON && candidate.z < best.z))
      )
    ) {
      best = candidate
      bestDistanceSquared = candidateDistanceSquared
    }
  }

  consider({
    x: clamp(origin.x, rootBounds.minX, rootBounds.maxX),
    z: clamp(origin.z, rootBounds.minZ, rootBounds.maxZ),
  })
  for (const circle of component) {
    const direction = subtract(origin, circle.center)
    if (lengthSquared(direction) > EPSILON * EPSILON) {
      consider(add(circle.center, scale(normalizeOr(direction, { x: 1, z: 0 }), circle.radius)))
    } else {
      for (let index = 0; index < 32; index += 1) {
        const angle = index / 32 * Math.PI * 2
        consider(add(circle.center, { x: Math.cos(angle) * circle.radius, z: Math.sin(angle) * circle.radius }))
      }
    }
    considerCircleBoundsIntersections(circle, rootBounds, consider)
  }

  for (let leftIndex = 0; leftIndex < component.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < component.length; rightIndex += 1) {
      for (const intersection of circleIntersections(component[leftIndex], component[rightIndex])) {
        consider(intersection)
      }
    }
  }

  consider({ x: rootBounds.minX, z: clamp(origin.z, rootBounds.minZ, rootBounds.maxZ) })
  consider({ x: rootBounds.maxX, z: clamp(origin.z, rootBounds.minZ, rootBounds.maxZ) })
  consider({ x: clamp(origin.x, rootBounds.minX, rootBounds.maxX), z: rootBounds.minZ })
  consider({ x: clamp(origin.x, rootBounds.minX, rootBounds.maxX), z: rootBounds.maxZ })
  consider({ x: rootBounds.minX, z: rootBounds.minZ })
  consider({ x: rootBounds.minX, z: rootBounds.maxZ })
  consider({ x: rootBounds.maxX, z: rootBounds.minZ })
  consider({ x: rootBounds.maxX, z: rootBounds.maxZ })

  if (!best) throw new Error('body footprint is trapped by terrarium obstacles')
  return best
}

function overlappingCircleComponent(origin: PlanarPoint, circles: readonly ConfigurationCircle[]) {
  const included = new Set<number>()
  const queue: number[] = []
  circles.forEach((circle, index) => {
    if (lengthSquared(subtract(origin, circle.center)) < circle.radius * circle.radius - EPSILON) {
      included.add(index)
      queue.push(index)
    }
  })
  for (let queueIndex = 0; queueIndex < queue.length; queueIndex += 1) {
    const currentIndex = queue[queueIndex] ?? 0
    const current = circles[currentIndex]
    if (!current) continue
    circles.forEach((candidate, candidateIndex) => {
      if (included.has(candidateIndex)) return
      const combinedRadius = current.radius + candidate.radius
      if (lengthSquared(subtract(current.center, candidate.center)) <= combinedRadius * combinedRadius + HIT_EPSILON) {
        included.add(candidateIndex)
        queue.push(candidateIndex)
      }
    })
  }
  return [...included].sort((left, right) => left - right).map((index) => circles[index]).filter((circle) => circle !== undefined)
}

function configurationPointIsClear(
  point: PlanarPoint,
  bounds: ArenaBounds,
  circles: readonly ConfigurationCircle[],
) {
  if (
    point.x < bounds.minX - HIT_EPSILON || point.x > bounds.maxX + HIT_EPSILON ||
    point.z < bounds.minZ - HIT_EPSILON || point.z > bounds.maxZ + HIT_EPSILON
  ) return false
  return circles.every((circle) => {
    const radius = Math.max(0, circle.radius - HIT_EPSILON)
    return lengthSquared(subtract(point, circle.center)) >= radius * radius
  })
}

function circleIntersections(
  left: ConfigurationCircle | undefined,
  right: ConfigurationCircle | undefined,
) {
  if (!left || !right) return []
  const delta = subtract(right.center, left.center)
  const centerDistance = length(delta)
  if (
    centerDistance <= EPSILON ||
    centerDistance > left.radius + right.radius + HIT_EPSILON ||
    centerDistance < Math.abs(left.radius - right.radius) - HIT_EPSILON
  ) return []
  const along = (
    left.radius * left.radius - right.radius * right.radius + centerDistance * centerDistance
  ) / (2 * centerDistance)
  const heightSquared = Math.max(0, left.radius * left.radius - along * along)
  const direction = scale(delta, 1 / centerDistance)
  const base = add(left.center, scale(direction, along))
  const perpendicular = { x: -direction.z, z: direction.x }
  const height = Math.sqrt(heightSquared)
  if (height <= EPSILON) return [base]
  return [add(base, scale(perpendicular, height)), add(base, scale(perpendicular, -height))]
}

function considerCircleBoundsIntersections(
  circle: ConfigurationCircle,
  bounds: ArenaBounds,
  consider: (point: PlanarPoint) => void,
) {
  for (const x of [bounds.minX, bounds.maxX]) {
    const offset = x - circle.center.x
    if (Math.abs(offset) > circle.radius + HIT_EPSILON) continue
    const height = Math.sqrt(Math.max(0, circle.radius * circle.radius - offset * offset))
    consider({ x, z: circle.center.z - height })
    consider({ x, z: circle.center.z + height })
  }
  for (const z of [bounds.minZ, bounds.maxZ]) {
    const offset = z - circle.center.z
    if (Math.abs(offset) > circle.radius + HIT_EPSILON) continue
    const width = Math.sqrt(Math.max(0, circle.radius * circle.radius - offset * offset))
    consider({ x: circle.center.x - width, z })
    consider({ x: circle.center.x + width, z })
  }
}

function deepestProjection(
  position: PlanarPoint,
  body: readonly TerrariumBodyCircle[],
  obstacles: readonly TerrariumCircleObstacle[],
  world: TerrariumCollisionWorld,
  skin: number,
) {
  let deepest: ProjectionCandidate | null = null

  for (const sample of body) {
    const center = add(position, sample.offset)
    for (const obstacle of obstacles) {
      const difference = subtract(center, obstacle.center)
      const separation = length(difference)
      const required = sample.radius + obstacle.radius + skin
      const penetration = required - separation
      if (penetration <= EPSILON) continue
      const normal = separation > EPSILON ? scale(difference, 1 / separation) : { x: 1, z: 0 }
      const candidate: ProjectionCandidate = {
        type: 'obstacle',
        id: obstacle.id,
        obstacleKind: obstacle.kind,
        body: sample,
        normal,
        point: add(obstacle.center, scale(normal, obstacle.radius)),
        penetration,
        friction: coefficient(obstacle.friction, DEFAULT_OBSTACLE_FRICTION),
        restitution: unitCoefficient(obstacle.restitution, 0),
        sortKey: `obstacle:${obstacle.id}:${sample.id}`,
      }
      deepest = selectDeeper(deepest, candidate)
    }

    const boundaryCandidates = boundaryProjections(center, sample, world, skin)
    for (const candidate of boundaryCandidates) deepest = selectDeeper(deepest, candidate)
  }

  return deepest
}

function boundaryProjections(
  center: PlanarPoint,
  body: TerrariumBodyCircle,
  world: TerrariumCollisionWorld,
  skin: number,
): ProjectionCandidate[] {
  const minimumX = world.bounds.minX + body.radius + skin
  const maximumX = world.bounds.maxX - body.radius - skin
  const minimumZ = world.bounds.minZ + body.radius + skin
  const maximumZ = world.bounds.maxZ - body.radius - skin
  const friction = coefficient(world.boundaryFriction, DEFAULT_BOUNDARY_FRICTION)
  const restitution = unitCoefficient(world.boundaryRestitution, 0)
  const candidates: ProjectionCandidate[] = []
  if (center.x < minimumX) {
    candidates.push(boundaryProjection('wall-left', body, { x: 1, z: 0 }, { x: world.bounds.minX, z: center.z }, minimumX - center.x, friction, restitution))
  }
  if (center.x > maximumX) {
    candidates.push(boundaryProjection('wall-right', body, { x: -1, z: 0 }, { x: world.bounds.maxX, z: center.z }, center.x - maximumX, friction, restitution))
  }
  if (center.z < minimumZ) {
    candidates.push(boundaryProjection('wall-near', body, { x: 0, z: 1 }, { x: center.x, z: world.bounds.minZ }, minimumZ - center.z, friction, restitution))
  }
  if (center.z > maximumZ) {
    candidates.push(boundaryProjection('wall-far', body, { x: 0, z: -1 }, { x: center.x, z: world.bounds.maxZ }, center.z - maximumZ, friction, restitution))
  }
  return candidates
}

function boundaryProjection(
  id: string,
  body: TerrariumBodyCircle,
  normal: PlanarPoint,
  point: PlanarPoint,
  penetration: number,
  friction: number,
  restitution: number,
): ProjectionCandidate {
  return {
    type: 'boundary',
    id,
    body,
    normal,
    point,
    penetration,
    friction,
    restitution,
    sortKey: `boundary:${id}:${body.id}`,
  }
}

function findEarliestSweepHits(
  position: PlanarPoint,
  displacement: PlanarPoint,
  body: readonly TerrariumBodyCircle[],
  obstacles: readonly TerrariumCircleObstacle[],
  world: TerrariumCollisionWorld,
  skin: number,
) {
  const candidates: SweepCandidate[] = []
  for (const sample of body) {
    const center = add(position, sample.offset)
    for (const obstacle of obstacles) {
      const candidate = sweepCircleObstacle(center, displacement, sample, obstacle, skin)
      if (candidate) candidates.push(candidate)
    }
    candidates.push(...sweepArenaBounds(center, displacement, sample, world, skin))
  }
  if (candidates.length === 0) return []
  candidates.sort(compareCandidates)
  const earliest = candidates[0]?.time ?? 1
  return candidates.filter((candidate) => Math.abs(candidate.time - earliest) <= HIT_EPSILON)
}

function sweepCircleObstacle(
  center: PlanarPoint,
  displacement: PlanarPoint,
  body: TerrariumBodyCircle,
  obstacle: TerrariumCircleObstacle,
  skin: number,
): SweepCandidate | null {
  const relative = subtract(center, obstacle.center)
  const radius = body.radius + obstacle.radius + skin
  const quadraticA = lengthSquared(displacement)
  const quadraticB = 2 * dot(relative, displacement)
  const quadraticC = lengthSquared(relative) - radius * radius
  if (quadraticA <= EPSILON || quadraticB >= 0 || quadraticC < -HIT_EPSILON) return null
  const discriminant = quadraticB * quadraticB - 4 * quadraticA * quadraticC
  if (discriminant < 0) return null
  const time = (-quadraticB - Math.sqrt(Math.max(0, discriminant))) / (2 * quadraticA)
  if (time < -HIT_EPSILON || time > 1 + HIT_EPSILON) return null
  const impactCenter = add(center, scale(displacement, clamp(time, 0, 1)))
  const normal = normalizeOr(subtract(impactCenter, obstacle.center), scale(displacement, -1))
  if (dot(displacement, normal) >= -EPSILON) return null
  return {
    time: clamp(time, 0, 1),
    type: 'obstacle',
    id: obstacle.id,
    obstacleKind: obstacle.kind,
    body,
    normal,
    point: add(obstacle.center, scale(normal, obstacle.radius)),
    friction: coefficient(obstacle.friction, DEFAULT_OBSTACLE_FRICTION),
    restitution: unitCoefficient(obstacle.restitution, 0),
    sortKey: `obstacle:${obstacle.id}:${body.id}`,
  }
}

function sweepArenaBounds(
  center: PlanarPoint,
  displacement: PlanarPoint,
  body: TerrariumBodyCircle,
  world: TerrariumCollisionWorld,
  skin: number,
) {
  const candidates: SweepCandidate[] = []
  const minimumX = world.bounds.minX + body.radius + skin
  const maximumX = world.bounds.maxX - body.radius - skin
  const minimumZ = world.bounds.minZ + body.radius + skin
  const maximumZ = world.bounds.maxZ - body.radius - skin
  const friction = coefficient(world.boundaryFriction, DEFAULT_BOUNDARY_FRICTION)
  const restitution = unitCoefficient(world.boundaryRestitution, 0)
  const addBoundary = (
    id: string,
    time: number,
    normal: PlanarPoint,
    axis: 'x' | 'z',
    wallPosition: number,
  ) => {
    if (time < -HIT_EPSILON || time > 1 + HIT_EPSILON) return
    const clampedTime = clamp(time, 0, 1)
    const impactCenter = add(center, scale(displacement, clampedTime))
    const point = axis === 'x'
      ? { x: wallPosition, z: impactCenter.z }
      : { x: impactCenter.x, z: wallPosition }
    candidates.push({
      time: clampedTime,
      type: 'boundary',
      id,
      body,
      normal,
      point,
      friction,
      restitution,
      sortKey: `boundary:${id}:${body.id}`,
    })
  }
  if (displacement.x < -EPSILON) {
    addBoundary('wall-left', (minimumX - center.x) / displacement.x, { x: 1, z: 0 }, 'x', world.bounds.minX)
  } else if (displacement.x > EPSILON) {
    addBoundary('wall-right', (maximumX - center.x) / displacement.x, { x: -1, z: 0 }, 'x', world.bounds.maxX)
  }
  if (displacement.z < -EPSILON) {
    addBoundary('wall-near', (minimumZ - center.z) / displacement.z, { x: 0, z: 1 }, 'z', world.bounds.minZ)
  } else if (displacement.z > EPSILON) {
    addBoundary('wall-far', (maximumZ - center.z) / displacement.z, { x: 0, z: -1 }, 'z', world.bounds.maxZ)
  }
  return candidates.filter((candidate) => dot(displacement, candidate.normal) < -EPSILON)
}

function contactFromSweep(candidate: SweepCandidate, time: number): TerrariumPlanarContact {
  return {
    type: candidate.type,
    id: candidate.id,
    obstacleKind: candidate.obstacleKind,
    bodyId: candidate.body.id,
    normal: candidate.normal,
    point: candidate.point,
    time: clamp(time, 0, 1),
    penetration: 0,
    friction: candidate.friction,
  }
}

function contactFromProjection(candidate: ProjectionCandidate): TerrariumPlanarContact {
  return {
    type: candidate.type,
    id: candidate.id,
    obstacleKind: candidate.obstacleKind,
    bodyId: candidate.body.id,
    normal: candidate.normal,
    point: candidate.point,
    time: 0,
    penetration: candidate.penetration,
    friction: candidate.friction,
  }
}

function addContact(contacts: Map<string, TerrariumPlanarContact>, contact: TerrariumPlanarContact) {
  const key = `${contact.type}:${contact.id}:${contact.bodyId}`
  const existing = contacts.get(key)
  if (!existing) {
    contacts.set(key, contact)
    return
  }
  if (contact.time < existing.time) existing.time = contact.time
  if (contact.penetration > existing.penetration) existing.penetration = contact.penetration
}

function supportContactFor(
  input: GroundContactInput | undefined,
  obstacleContacts: readonly TerrariumPlanarContact[],
  obstacleContactRatio: number,
  velocity: PlanarPoint,
): Omit<TerrariumSupportContact, 'blockedSpeed'> {
  const grounded = input?.grounded ?? true
  const friction = coefficient(input?.friction, 1)
  const normalY = grounded ? clamp(finiteOr(input?.normalY, 1), 0, 1) : 0
  const contactRatio = grounded ? clamp(finiteOr(input?.contactRatio, 1), 0, 1) : 0
  const obstacleFriction = obstacleContacts.length > 0
    ? obstacleContacts.reduce((sum, contact) => sum + contact.friction, 0) / obstacleContacts.length
    : 0
  const slidingSpeed = obstacleContacts.reduce((maximum, contact) => {
    const tangentSpeed = Math.abs(-contact.normal.z * velocity.x + contact.normal.x * velocity.z)
    return Math.max(maximum, tangentSpeed)
  }, 0)
  return {
    grounded,
    friction,
    normalY,
    contactRatio,
    traction: grounded ? friction * normalY * contactRatio : 0,
    obstacleContactRatio: clamp(obstacleContactRatio, 0, 1),
    obstacleFriction,
    slidingSpeed,
  }
}

/** Remove inward normal motion and apply Coulomb friction to its tangent. */
function coulombSlide(
  vector: PlanarPoint,
  normal: PlanarPoint,
  friction: number,
  restitution: number,
): PlanarPoint {
  const inwardSpeed = dot(vector, normal)
  if (inwardSpeed >= -EPSILON) return vector
  const tangent = subtract(vector, scale(normal, inwardSpeed))
  const tangentSpeed = length(tangent)
  const frictionLoss = Math.min(tangentSpeed, friction * -inwardSpeed)
  const tangentScale = tangentSpeed > EPSILON ? (tangentSpeed - frictionLoss) / tangentSpeed : 0
  const bounce = scale(normal, -inwardSpeed * restitution)
  return add(scale(tangent, tangentScale), bounce)
}

function validateWorld(world: TerrariumCollisionWorld) {
  requireFinite(world.bounds.minX, 'bounds.minX')
  requireFinite(world.bounds.maxX, 'bounds.maxX')
  requireFinite(world.bounds.minZ, 'bounds.minZ')
  requireFinite(world.bounds.maxZ, 'bounds.maxZ')
  if (world.bounds.minX >= world.bounds.maxX || world.bounds.minZ >= world.bounds.maxZ) {
    throw new Error('terrarium collision bounds must have positive area')
  }
  const ids = new Set<string>()
  for (const obstacle of world.obstacles) {
    if (!obstacle.id) throw new Error('terrarium obstacles require stable ids')
    if (ids.has(obstacle.id)) throw new Error(`duplicate terrarium obstacle id: ${obstacle.id}`)
    ids.add(obstacle.id)
    requirePoint(obstacle.center, `obstacle ${obstacle.id} center`)
    requirePositive(obstacle.radius, `obstacle ${obstacle.id} radius`)
    coefficient(obstacle.friction, DEFAULT_OBSTACLE_FRICTION)
    unitCoefficient(obstacle.restitution, 0)
  }
  coefficient(world.boundaryFriction, DEFAULT_BOUNDARY_FRICTION)
  unitCoefficient(world.boundaryRestitution, 0)
}

function validateMotion(motion: SweptBodyMotion) {
  requirePoint(motion.position, 'motion position')
  requirePoint(motion.displacement, 'motion displacement')
  requirePoint(motion.velocity, 'motion velocity')
  if (motion.body.length === 0) throw new Error('swept terrarium motion requires at least one body circle')
  const ids = new Set<string>()
  for (const sample of motion.body) {
    if (!sample.id) throw new Error('body circles require stable ids')
    if (ids.has(sample.id)) throw new Error(`duplicate body circle id: ${sample.id}`)
    ids.add(sample.id)
    requirePoint(sample.offset, `body ${sample.id} offset`)
    requirePositive(sample.radius, `body ${sample.id} radius`)
  }
}

function validateBodyFitsBounds(
  body: readonly TerrariumBodyCircle[],
  bounds: ArenaBounds,
  skin: number,
) {
  rootBoundsForBody(body, bounds, skin)
}

function rootBoundsForBody(
  body: readonly TerrariumBodyCircle[],
  bounds: ArenaBounds,
  skin: number,
): ArenaBounds {
  let rootMinimumX = Number.NEGATIVE_INFINITY
  let rootMaximumX = Number.POSITIVE_INFINITY
  let rootMinimumZ = Number.NEGATIVE_INFINITY
  let rootMaximumZ = Number.POSITIVE_INFINITY
  for (const sample of body) {
    rootMinimumX = Math.max(rootMinimumX, bounds.minX + sample.radius + skin - sample.offset.x)
    rootMaximumX = Math.min(rootMaximumX, bounds.maxX - sample.radius - skin - sample.offset.x)
    rootMinimumZ = Math.max(rootMinimumZ, bounds.minZ + sample.radius + skin - sample.offset.z)
    rootMaximumZ = Math.min(rootMaximumZ, bounds.maxZ - sample.radius - skin - sample.offset.z)
  }
  if (rootMinimumX > rootMaximumX || rootMinimumZ > rootMaximumZ) {
    throw new Error('body footprint cannot fit inside the terrarium collision bounds')
  }
  return {
    minX: rootMinimumX,
    maxX: rootMaximumX,
    minZ: rootMinimumZ,
    maxZ: rootMaximumZ,
  }
}

function selectDeeper(current: ProjectionCandidate | null, candidate: ProjectionCandidate) {
  if (!current || candidate.penetration > current.penetration + HIT_EPSILON) return candidate
  if (Math.abs(candidate.penetration - current.penetration) <= HIT_EPSILON && candidate.sortKey < current.sortKey) {
    return candidate
  }
  return current
}

function compareCandidates(left: SweepCandidate, right: SweepCandidate) {
  if (Math.abs(left.time - right.time) > HIT_EPSILON) return left.time - right.time
  return compareText(left.sortKey, right.sortKey)
}

function compareContacts(left: TerrariumPlanarContact, right: TerrariumPlanarContact) {
  if (Math.abs(left.time - right.time) > HIT_EPSILON) return left.time - right.time
  return compareText(`${left.type}:${left.id}:${left.bodyId}`, `${right.type}:${right.id}:${right.bodyId}`)
}

function coefficient(value: number | undefined, fallback: number) {
  const result = finiteOr(value, fallback)
  if (result < 0) throw new Error('friction coefficients must be non-negative')
  return result
}

function unitCoefficient(value: number | undefined, fallback: number) {
  const result = finiteOr(value, fallback)
  if (result < 0 || result > 1) throw new Error('restitution must be between zero and one')
  return result
}

function requirePoint(point: PlanarPoint, label: string) {
  requireFinite(point.x, `${label}.x`)
  requireFinite(point.z, `${label}.z`)
}

function requirePositive(value: number, label: string) {
  requireFinite(value, label)
  if (value <= 0) throw new Error(`${label} must be positive`)
}

function requireNonNegative(value: number, label: string) {
  requireFinite(value, label)
  if (value < 0) throw new Error(`${label} must be non-negative`)
}

function requireFinite(value: number, label: string) {
  if (!Number.isFinite(value)) throw new Error(`${label} must be finite`)
}

function normalizeOr(vector: PlanarPoint, fallback: PlanarPoint): PlanarPoint {
  const vectorLength = length(vector)
  if (vectorLength > EPSILON) return scale(vector, 1 / vectorLength)
  const fallbackLength = length(fallback)
  return fallbackLength > EPSILON ? scale(fallback, 1 / fallbackLength) : { x: 1, z: 0 }
}

function copyPoint(point: PlanarPoint): PlanarPoint {
  return { x: point.x, z: point.z }
}

function add(left: PlanarPoint, right: PlanarPoint): PlanarPoint {
  return { x: left.x + right.x, z: left.z + right.z }
}

function subtract(left: PlanarPoint, right: PlanarPoint): PlanarPoint {
  return { x: left.x - right.x, z: left.z - right.z }
}

function scale(point: PlanarPoint, amount: number): PlanarPoint {
  return { x: point.x * amount, z: point.z * amount }
}

function dot(left: PlanarPoint, right: PlanarPoint) {
  return left.x * right.x + left.z * right.z
}

function length(point: PlanarPoint) {
  return Math.hypot(point.x, point.z)
}

function lengthSquared(point: PlanarPoint) {
  return point.x * point.x + point.z * point.z
}

function finiteOr(value: number | undefined, fallback: number) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value))
}

function compareText(left: string, right: string) {
  return left < right ? -1 : left > right ? 1 : 0
}
