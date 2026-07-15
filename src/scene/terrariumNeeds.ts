import type { Vec3 } from '../creature/types'
import type { TerrainField, TerrainSurface } from './terrainField'

export type NeedKind = 'hunger' | 'thirst' | 'wellbeing'
export type ResourcePresentation = 'bowl' | 'skateboard'

export type TerrariumResource = {
  id: 'food-bowl' | 'water-bowl' | 'skateboard'
  need: NeedKind
  label: string
  presentation: ResourcePresentation
  position: Vec3
  interactionRadius: number
  restorePerSecond: number
  requiresMount: boolean
  groundHeight: number
  terrainNormal: Vec3
  terrainFriction: number
  surface: TerrainSurface
  appearance: {
    vesselColor: string
    contentsColor: string
    radius: number
    height: number
  }
}

/** All need values are urgencies: 0 is satisfied and 1 is maximally urgent. */
export type NeedsState = {
  elapsedSeconds: number
  hunger: number
  thirst: number
  wellbeing: number
  targetResourceId: TerrariumResource['id'] | null
  lastInteraction: NeedKind | null
  fulfillment: Record<NeedKind, number>
}

export type NeedsStepContext = {
  resources: readonly TerrariumResource[]
  wormPosition: Vec3
  mounted: boolean
}

export type NeedsStepResult = {
  state: NeedsState
  target: TerrariumResource | null
  interaction: NeedKind | null
  restored: number
}

export type NeedsMetrics = {
  hunger: number
  thirst: number
  wellbeing: number
  activeNeed: NeedKind | null
  targetResourceId: TerrariumResource['id'] | null
  targetDistance: number
  interaction: NeedKind | null
  eating: boolean
  drinking: boolean
  ridingForWellbeing: boolean
}

export const NEED_ORDER: readonly NeedKind[] = ['hunger', 'thirst', 'wellbeing']
export const NEEDS_DIAGNOSTIC_OBSERVATION_SIZE = 15

const RESOURCE_ORDER: Record<NeedKind, number> = { hunger: 0, thirst: 1, wellbeing: 2 }
const NEED_GROWTH_PER_SECOND: Record<NeedKind, number> = {
  hunger: 0.010,
  thirst: 0.014,
  wellbeing: 0.008,
}
const ARENA_INSET = 0.72
const BOARD_CLEARANCE = 0.28
const MAX_NEEDS_STEP = 0.25

/**
 * Place the two bowls on deterministic, well-separated, reasonably level terrain.
 * The board resource can subsequently be synchronized to the live skateboard pose.
 */
export function createTerrariumResources(
  field: TerrainField,
  seed = 1337,
  skateboardPosition?: Vec3,
): TerrariumResource[] {
  const separation = Math.min(field.width, field.depth) * 0.34
  const foodPlanar = chooseBowlPosition(field, seed, 101, [], separation)
  const waterPlanar = chooseBowlPosition(field, seed, 211, [foodPlanar], separation)
  const initialBoard = skateboardPosition ?? defaultBoardPosition(field)

  return [
    makeResource(field, {
      id: 'food-bowl',
      need: 'hunger',
      label: 'Food',
      presentation: 'bowl',
      planar: foodPlanar,
      interactionRadius: 0.56,
      restorePerSecond: 0.62,
      requiresMount: false,
      vesselColor: '#68452d',
      contentsColor: '#d49a43',
      radius: 0.34,
      height: 0.14,
    }),
    makeResource(field, {
      id: 'water-bowl',
      need: 'thirst',
      label: 'Water',
      presentation: 'bowl',
      planar: waterPlanar,
      interactionRadius: 0.6,
      restorePerSecond: 0.78,
      requiresMount: false,
      vesselColor: '#53616d',
      contentsColor: '#59b9dc',
      radius: 0.38,
      height: 0.13,
    }),
    makeResource(field, {
      id: 'skateboard',
      need: 'wellbeing',
      label: 'Well-being',
      presentation: 'skateboard',
      planar: [initialBoard[0], initialBoard[2]],
      worldY: initialBoard[1],
      interactionRadius: 0.92,
      restorePerSecond: 0.2,
      requiresMount: true,
      vesselColor: '#432c22',
      contentsColor: '#e5c64d',
      radius: 0.72,
      height: 0.16,
    }),
  ]
}

/** Return a new resource list with the well-being resource following the live board. */
export function syncSkateboardResource(
  resources: readonly TerrariumResource[],
  field: TerrainField,
  skateboardPosition: Vec3,
): TerrariumResource[] {
  return resources.map((resource) => {
    if (resource.id !== 'skateboard') return resource
    const x = clamp(finiteOr(skateboardPosition[0], 0), -field.width * 0.5 + ARENA_INSET, field.width * 0.5 - ARENA_INSET)
    const z = clamp(finiteOr(skateboardPosition[2], 0), -field.depth * 0.5 + ARENA_INSET, field.depth * 0.5 - ARENA_INSET)
    const terrain = field.sample(x, z)
    const y = finiteOr(skateboardPosition[1], terrain.height + BOARD_CLEARANCE)
    return {
      ...resource,
      position: [x, y, z],
      groundHeight: terrain.height,
      terrainNormal: [...terrain.normal],
      terrainFriction: terrain.friction,
      surface: terrain.surface,
    }
  })
}

export function createNeedsState(seed = 1337): NeedsState {
  return {
    elapsedSeconds: 0,
    hunger: clamp01(0.38 + (seededUnit(seed, 307) - 0.5) * 0.12),
    thirst: clamp01(0.34 + (seededUnit(seed, 311) - 0.5) * 0.12),
    wellbeing: clamp01(0.42 + (seededUnit(seed, 313) - 0.5) * 0.12),
    targetResourceId: null,
    lastInteraction: null,
    fulfillment: { hunger: 0, thirst: 0, wellbeing: 0 },
  }
}

/**
 * Advance passive need accumulation and resource fulfillment without mutating inputs.
 * Food and water work by proximity; well-being is restored only while mounted on the board.
 */
export function advanceNeeds(
  previous: Readonly<NeedsState>,
  deltaSeconds: number,
  context: NeedsStepContext,
): NeedsStepResult {
  const dt = clamp(finiteOr(deltaSeconds, 0), 0, MAX_NEEDS_STEP)
  const next: NeedsState = {
    elapsedSeconds: previous.elapsedSeconds + dt,
    hunger: growNeed(previous.hunger, 'hunger', dt),
    thirst: growNeed(previous.thirst, 'thirst', dt),
    wellbeing: growNeed(previous.wellbeing, 'wellbeing', dt),
    targetResourceId: previous.targetResourceId,
    lastInteraction: null,
    fulfillment: { ...previous.fulfillment },
  }

  const resource = nearestUsableResource(context.resources, context.wormPosition, context.mounted)
  let restored = 0
  if (resource) {
    const before = readNeed(next, resource.need)
    const after = clamp01(before - resource.restorePerSecond * dt)
    restored = before - after
    writeNeed(next, resource.need, after)
    next.lastInteraction = restored > 0 ? resource.need : null
    next.fulfillment[resource.need] += restored
  }

  const target = selectNeedTarget(next, context.resources, context.wormPosition, previous.targetResourceId)
  next.targetResourceId = target?.id ?? null
  return { state: next, target, interaction: next.lastInteraction, restored }
}

/** Select the most urgent reachable resource, with mild target hysteresis to prevent flicker. */
export function selectNeedTarget(
  needs: Readonly<NeedsState>,
  resources: readonly TerrariumResource[],
  wormPosition: Vec3,
  currentTargetId: TerrariumResource['id'] | null = needs.targetResourceId,
): TerrariumResource | null {
  const ordered = [...resources].sort(
    (a, b) => RESOURCE_ORDER[a.need] - RESOURCE_ORDER[b.need] || a.id.localeCompare(b.id),
  )
  if (ordered.length === 0) return null
  const arenaScale = Math.max(1, ...ordered.map((resource) => distanceXZ(wormPosition, resource.position)))

  let best: TerrariumResource | null = null
  let bestScore = Number.NEGATIVE_INFINITY
  for (const resource of ordered) {
    const urgency = readNeed(needs, resource.need)
    const hysteresis = resource.id === currentTargetId && urgency > 0.08 ? 0.045 : 0
    const distancePenalty = (distanceXZ(wormPosition, resource.position) / arenaScale) * 0.025
    const score = urgency + hysteresis - distancePenalty
    if (score > bestScore) {
      best = resource
      bestScore = score
    }
  }
  return best
}

/**
 * Diagnostic/future homeostasis observation layout (15 floats). The live
 * locomotion brain consumes only its selected target through locomotionSensorsFor.
 * [three urgencies, local forward/right/distance for food, water, board, active-target one-hot].
 */
export function createNeedsDiagnosticObservation(
  needs: Readonly<NeedsState>,
  resources: readonly TerrariumResource[],
  wormPosition: Vec3,
  wormHeading: number,
  field: Pick<TerrainField, 'width' | 'depth'>,
): number[] {
  const heading = finiteOr(wormHeading, 0)
  const cosine = Math.cos(heading)
  const sine = Math.sin(heading)
  const scale = Math.max(1, Math.hypot(field.width, field.depth))
  const observation = [clamp01(needs.hunger), clamp01(needs.thirst), clamp01(needs.wellbeing)]

  for (const need of NEED_ORDER) {
    const resource = resources.find((candidate) => candidate.need === need)
    if (!resource) {
      observation.push(0, 0, 1)
      continue
    }
    const dx = resource.position[0] - wormPosition[0]
    const dz = resource.position[2] - wormPosition[2]
    const localForward = (dx * cosine + dz * sine) / scale
    const localRight = (-dx * sine + dz * cosine) / scale
    observation.push(clamp(localForward, -1, 1), clamp(localRight, -1, 1), clamp(Math.hypot(dx, dz) / scale, 0, 1))
  }

  const target = resources.find((resource) => resource.id === needs.targetResourceId)
  for (const need of NEED_ORDER) observation.push(target?.need === need ? 1 : 0)
  return observation
}

export function readNeedsMetrics(
  needs: Readonly<NeedsState>,
  resources: readonly TerrariumResource[],
  wormPosition: Vec3,
): NeedsMetrics {
  const target = resources.find((resource) => resource.id === needs.targetResourceId) ?? null
  return {
    hunger: needs.hunger,
    thirst: needs.thirst,
    wellbeing: needs.wellbeing,
    activeNeed: target?.need ?? null,
    targetResourceId: target?.id ?? null,
    targetDistance: target ? distanceXZ(wormPosition, target.position) : 0,
    interaction: needs.lastInteraction,
    eating: needs.lastInteraction === 'hunger',
    drinking: needs.lastInteraction === 'thirst',
    ridingForWellbeing: needs.lastInteraction === 'wellbeing',
  }
}

type ResourceSpec = {
  id: TerrariumResource['id']
  need: NeedKind
  label: string
  presentation: ResourcePresentation
  planar: [number, number]
  worldY?: number
  interactionRadius: number
  restorePerSecond: number
  requiresMount: boolean
  vesselColor: string
  contentsColor: string
  radius: number
  height: number
}

function makeResource(field: TerrainField, spec: ResourceSpec): TerrariumResource {
  const x = clamp(spec.planar[0], -field.width * 0.5 + ARENA_INSET, field.width * 0.5 - ARENA_INSET)
  const z = clamp(spec.planar[1], -field.depth * 0.5 + ARENA_INSET, field.depth * 0.5 - ARENA_INSET)
  const terrain = field.sample(x, z)
  const defaultY = terrain.height + (spec.presentation === 'bowl' ? spec.height * 0.5 : BOARD_CLEARANCE)
  return {
    id: spec.id,
    need: spec.need,
    label: spec.label,
    presentation: spec.presentation,
    position: [x, finiteOr(spec.worldY, defaultY), z],
    interactionRadius: spec.interactionRadius,
    restorePerSecond: spec.restorePerSecond,
    requiresMount: spec.requiresMount,
    groundHeight: terrain.height,
    terrainNormal: [...terrain.normal],
    terrainFriction: terrain.friction,
    surface: terrain.surface,
    appearance: {
      vesselColor: spec.vesselColor,
      contentsColor: spec.contentsColor,
      radius: spec.radius,
      height: spec.height,
    },
  }
}

function chooseBowlPosition(
  field: TerrainField,
  seed: number,
  salt: number,
  avoid: readonly [number, number][],
  desiredSeparation: number,
): [number, number] {
  // Keep bowls inside the camera's readable middle ring rather than hiding
  // them against the glass, while retaining enough separation to create trips.
  const xSpan = Math.max(0.1, (field.width - ARENA_INSET * 2) * 0.74)
  const zSpan = Math.max(0.1, (field.depth - ARENA_INSET * 2) * 0.74)
  let best: [number, number] = [0, 0]
  let bestScore = Number.NEGATIVE_INFINITY

  for (let index = 0; index < 96; index += 1) {
    const x = (seededUnit(seed, salt + index * 2) - 0.5) * xSpan
    const z = (seededUnit(seed, salt + index * 2 + 1) - 0.5) * zSpan
    const terrain = field.sample(x, z)
    const separation = avoid.length === 0 ? desiredSeparation : Math.min(...avoid.map((point) => Math.hypot(x - point[0], z - point[1])))
    const separationScore = Math.min(1, separation / desiredSeparation)
    const centerDistance = Math.hypot(x / field.width, z / field.depth)
    const score = terrain.normal[1] * 1.8 + separationScore * 1.35 - centerDistance * 0.08
    if (score > bestScore) {
      best = [x, z]
      bestScore = score
    }
  }
  return best
}

function nearestUsableResource(
  resources: readonly TerrariumResource[],
  wormPosition: Vec3,
  mounted: boolean,
): TerrariumResource | null {
  let nearest: TerrariumResource | null = null
  let nearestRatio = Number.POSITIVE_INFINITY
  for (const resource of resources) {
    if (resource.requiresMount && !mounted) continue
    const ratio = distanceXZ(wormPosition, resource.position) / resource.interactionRadius
    if (ratio <= 1 && ratio < nearestRatio) {
      nearest = resource
      nearestRatio = ratio
    }
  }
  return nearest
}

function defaultBoardPosition(field: TerrainField): Vec3 {
  const waypoint = field.waypoints[0] ?? [0, 0]
  const terrain = field.sample(waypoint[0], waypoint[1])
  return [waypoint[0], terrain.height + BOARD_CLEARANCE, waypoint[1]]
}

function growNeed(value: number, need: NeedKind, dt: number) {
  return clamp01(finiteOr(value, 0) + NEED_GROWTH_PER_SECOND[need] * dt)
}

function readNeed(state: Readonly<NeedsState>, need: NeedKind) {
  return state[need]
}

function writeNeed(state: NeedsState, need: NeedKind, value: number) {
  state[need] = value
}

function distanceXZ(first: Vec3, second: Vec3) {
  return Math.hypot(first[0] - second[0], first[2] - second[2])
}

function seededUnit(seed: number, value: number) {
  let state = (Math.trunc(finiteOr(seed, 1337)) ^ Math.imul(value + 1, 0x9e3779b1)) >>> 0
  state ^= state >>> 16
  state = Math.imul(state, 0x7feb352d) >>> 0
  state ^= state >>> 15
  state = Math.imul(state, 0x846ca68b) >>> 0
  state ^= state >>> 16
  return (state >>> 0) / 0x100000000
}

function finiteOr(value: number | undefined, fallback: number) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function clamp01(value: number) {
  return clamp(value, 0, 1)
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value))
}
