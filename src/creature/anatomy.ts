import type { CreatureGenome, PrimitiveShape } from './types'

export type GenomeAppendage = {
  id: string
  shape: PrimitiveShape
  color: string
  anchorIndex: number
  side: -1 | 0 | 1
  lengthScale: number
  thicknessScale: number
}

export type WurmAnatomy = {
  segmentShape: PrimitiveShape
  silhouette: string
  visualLengthScale: number
  thicknessScale: number
  verticalScale: number
  axialScale: number
  motionWidthScale: number
  connectorScale: number
  appendages: GenomeAppendage[]
  bodyPartCount: number
  totalMass: number
  stanceWidth: number
}

const fallbackAnatomy: WurmAnatomy = {
  segmentShape: 'capsule',
  silhouette: 'capsule chain',
  visualLengthScale: 1,
  thicknessScale: 1,
  verticalScale: 1,
  axialScale: 1,
  motionWidthScale: 1,
  connectorScale: 1,
  appendages: [],
  bodyPartCount: 0,
  totalMass: 0,
  stanceWidth: 0,
}

/**
 * Projects an open-ended genome onto the showcase's fixed 16-node control
 * lattice. This changes only the rendered anatomy: policy observations retain
 * their trained 16-segment positions and 174-float contract.
 */
export function deriveWurmAnatomy(creature: CreatureGenome | null): WurmAnatomy {
  const parts = creature?.morphology.bodyParts ?? []
  if (parts.length === 0) return fallbackAnatomy

  const root = parts.find((part) => part.parentId === null) ?? parts[0]
  const halfAxialSize = (part: (typeof parts)[number]) => Math.max(...part.size) * 0.5
  const xMin = Math.min(...parts.map((part) => part.position[0] - halfAxialSize(part)))
  const xMax = Math.max(...parts.map((part) => part.position[0] + halfAxialSize(part)))
  const zMin = Math.min(...parts.map((part) => part.position[2] - part.size[2] * 0.5))
  const zMax = Math.max(...parts.map((part) => part.position[2] + part.size[2] * 0.5))
  const xSpan = Math.max(0.2, xMax - xMin)
  const directChildren = parts.filter((part) => part.parentId === root.id)
  const branchedParts = directChildren.length > 1 ? directChildren.slice(0, 6) : []
  const rootWidth = (root.size[0] + root.size[2]) * 0.5
  const segmentShape = root.shape

  return {
    segmentShape,
    silhouette: silhouetteFor(segmentShape, branchedParts.length),
    visualLengthScale: clamp(xSpan / 1.7, 0.84, 1.12),
    thicknessScale: clamp(rootWidth / 0.19, 0.9, 1.35),
    verticalScale: clamp(root.size[1] / 0.32, 0.88, 1.18),
    axialScale: axialScaleFor(segmentShape),
    motionWidthScale: clamp(0.92 + (zMax - zMin) * 0.18, 0.92, 1.18),
    connectorScale: clamp(rootWidth / 0.21, 0.82, 1.24),
    appendages: branchedParts.map((part) => ({
      id: part.id,
      shape: part.shape,
      color: part.visual.color,
      anchorIndex: Math.round(clamp((part.position[0] - xMin) / xSpan, 0.08, 0.92) * 15),
      side: sideFor(part.position[2] - root.position[2]),
      lengthScale: clamp(Math.max(...part.size) / 0.5, 0.58, 1.18),
      thicknessScale: clamp(Math.min(...part.size) / 0.16, 0.65, 1.2),
    })),
    bodyPartCount: parts.length,
    totalMass: parts.reduce((sum, part) => sum + part.mass, 0),
    stanceWidth: zMax - zMin,
  }
}

function silhouetteFor(shape: PrimitiveShape, appendageCount: number) {
  const shapeLabel: Record<PrimitiveShape, string> = {
    capsule: 'capsule chain',
    sphere: 'orb chain',
    box: 'armored chain',
    cylinder: 'ringed chain',
  }
  return appendageCount > 0 ? `${shapeLabel[shape]} + ${appendageCount} limbs` : shapeLabel[shape]
}

function axialScaleFor(shape: PrimitiveShape) {
  if (shape === 'sphere') return 0.72
  if (shape === 'box') return 0.82
  if (shape === 'cylinder') return 0.9
  return 1
}

function sideFor(offset: number): -1 | 0 | 1 {
  if (offset < -0.04) return -1
  if (offset > 0.04) return 1
  return 0
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value))
}
