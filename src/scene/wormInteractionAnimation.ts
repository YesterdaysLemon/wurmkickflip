/**
 * Deterministic, scene-agnostic interaction choreography for the visual worm rig.
 *
 * Scene segments are ordered posterior-to-anterior: index 0 is the tail and the
 * final index is the face/head. The returned translations are local to the worm
 * frame (`forward`, `lateral`, `vertical`) and are additive pose overlays. Scale
 * values are multipliers. All pose overlays are neutral at one-shot boundaries,
 * which lets the lifecycle enter and leave an interaction without a visual pop.
 */

export const WORM_INTERACTION_ANIMATION_VERSION = 'contact-choreography-v1'

export type WormInteractionKind = 'mounting' | 'dismounting' | 'eating' | 'drinking'

export type WormInteractionPhase =
  | 'mount-reach'
  | 'mount-head-contact'
  | 'mount-haul'
  | 'mount-settle'
  | 'dismount-brace'
  | 'dismount-reach-down'
  | 'dismount-slide-off'
  | 'dismount-release'
  | 'eat-lower'
  | 'eat-bite'
  | 'eat-chew'
  | 'eat-lift'
  | 'drink-lower'
  | 'drink-sip'
  | 'drink-swallow'
  | 'drink-lift'

export type WormInteractionContactTarget = 'skateboard-deck' | 'food-bowl' | 'water-bowl'

export type WormInteractionAnimationInput = {
  kind: WormInteractionKind
  /** Time since this interaction began. Reset this when the interaction changes. */
  elapsedSeconds: number
  /** Defaults to the authored duration for the interaction kind. */
  durationSeconds?: number
  /** Defaults to the production rig's 16 segments. */
  segmentCount?: number
  /** Which side of the board the worm uses. Negative is left, positive is right. */
  side?: number
  /** Feeding can repeat seamlessly while the need system reports contact. */
  loop?: boolean
}

export type WormInteractionRootOffset = {
  forward: number
  lateral: number
  vertical: number
  yaw: number
}

export type WormInteractionSegmentOffset = WormInteractionRootOffset & {
  pitch: number
  /** Axial (head-to-tail) scale multiplier. */
  axialScale: number
  /** Cross-section scale multiplier. */
  radialScale: number
}

export type WormInteractionContactCue = {
  target: WormInteractionContactTarget
  /** Contact of the face/front segment with the primary target. */
  headWeight: number
  /** Contact of the central body with the primary target. */
  midbodyWeight: number
  /** Contact of the posterior segment with the primary target. */
  tailWeight: number
  /** A separate cue for the face bracing against the ground during transitions. */
  groundHeadWeight: number
  /** Stylized face cues in normalized [0, 1] units. */
  mouthOpen: number
  swallow: number
}

export type WormInteractionAnimationSample = {
  kind: WormInteractionKind
  phase: WormInteractionPhase
  durationSeconds: number
  /** Normalized progress through the current one-shot or loop cycle. */
  progress: number
  cycle: number
  complete: boolean
  side: -1 | 1
  headSegmentIndex: number
  /** 1 preserves the learned crawl pose; 0 lets the interaction fully hold it. */
  locomotionWeight: number
  root: WormInteractionRootOffset
  segments: WormInteractionSegmentOffset[]
  contact: WormInteractionContactCue
}

export const WORM_INTERACTION_DURATIONS: Readonly<Record<WormInteractionKind, number>> = {
  mounting: 1.8,
  dismounting: 1.2,
  eating: 1.45,
  drinking: 1.7,
}

export function sampleWormInteractionAnimation(
  input: Readonly<WormInteractionAnimationInput>,
): WormInteractionAnimationSample {
  const kind = input.kind
  const durationSeconds = positiveFiniteOr(input.durationSeconds, WORM_INTERACTION_DURATIONS[kind])
  const elapsedSeconds = Math.max(0, finiteOr(input.elapsedSeconds, 0))
  const rawProgress = elapsedSeconds / durationSeconds
  const loop = input.loop === true
  const cycle = loop ? Math.floor(rawProgress) : 0
  const progress = loop ? rawProgress - cycle : clamp01(rawProgress)
  const complete = !loop && rawProgress >= 1
  const segmentCount = clampInteger(input.segmentCount, 2, 64, 16)
  const side: -1 | 1 = finiteOr(input.side, 1) < 0 ? -1 : 1
  const channels = channelsFor(kind, progress, side)

  const segments = Array.from({ length: segmentCount }, (_, index) =>
    segmentOffsetFor(kind, index / (segmentCount - 1), side, channels),
  )

  return {
    kind,
    phase: phaseFor(kind, progress),
    durationSeconds,
    progress,
    cycle,
    complete,
    side,
    headSegmentIndex: segmentCount - 1,
    locomotionWeight: locomotionWeightFor(kind, progress, channels),
    root: rootOffsetFor(kind, side, channels),
    segments,
    contact: contactCueFor(kind, progress, channels),
  }
}

type AnimationChannels = {
  lead: number
  support: number
  release: number
  contact: number
  accent: number
  swallow: number
}

function channelsFor(kind: WormInteractionKind, progress: number, side: -1 | 1): AnimationChannels {
  if (kind === 'mounting') {
    return {
      lead: window(progress, 0.01, 0.18, 0.48, 0.66),
      support: window(progress, 0.24, 0.43, 0.82, 1),
      release: 0,
      contact: rise(progress, 0.12, 0.32),
      accent: window(progress, 0.34, 0.5, 0.7, 0.84),
      swallow: 0,
    }
  }

  if (kind === 'dismounting') {
    return {
      lead: window(progress, 0.01, 0.2, 0.46, 0.64),
      support: window(progress, 0.16, 0.36, 0.68, 0.88),
      release: window(progress, 0.48, 0.68, 0.86, 1),
      contact: window(progress, 0.18, 0.36, 0.78, 0.96),
      accent: window(progress, 0.3, 0.45, 0.6, 0.74),
      swallow: 0,
    }
  }

  if (kind === 'eating') {
    const firstBite = window(progress, 0.22, 0.34, 0.42, 0.54)
    const secondBite = window(progress, 0.44, 0.56, 0.66, 0.78)
    return {
      lead: window(progress, 0.01, 0.2, 0.78, 1),
      support: window(progress, 0.08, 0.28, 0.74, 0.94),
      release: 0,
      contact: window(progress, 0.14, 0.26, 0.74, 0.9),
      accent: clamp01(firstBite + secondBite),
      swallow: window(progress, 0.64, 0.76, 0.84, 0.96),
    }
  }

  const firstSip = window(progress, 0.26, 0.38, 0.43, 0.55)
  const secondSip = window(progress, 0.44, 0.56, 0.63, 0.75)
  return {
    lead: window(progress, 0.01, 0.22, 0.8, 1),
    support: window(progress, 0.1, 0.3, 0.76, 0.94),
    release: 0,
    contact: window(progress, 0.16, 0.28, 0.76, 0.9),
    accent: clamp01(firstSip + secondSip) * side,
    swallow: window(progress, 0.62, 0.74, 0.84, 0.96),
  }
}

function segmentOffsetFor(
  kind: WormInteractionKind,
  posteriorToAnterior: number,
  side: -1 | 1,
  channels: AnimationChannels,
): WormInteractionSegmentOffset {
  const anterior = smootherStep(clamp01((posteriorToAnterior - 0.34) / 0.66))
  const anteriorReach = Math.pow(anterior, 1.28)
  const posterior = 1 - posteriorToAnterior
  const midbody = 4 * posteriorToAnterior * (1 - posteriorToAnterior)

  if (kind === 'mounting') {
    return {
      forward: channels.lead * anteriorReach * 0.13 - channels.support * posterior * 0.035,
      lateral:
        side *
        (channels.lead * anteriorReach * 0.095 +
          channels.support * (0.25 + posteriorToAnterior * 0.75) * 0.055),
      vertical: channels.lead * (0.025 + anteriorReach * 0.17) + channels.support * (0.045 + midbody * 0.085),
      pitch: -channels.lead * anteriorReach * 0.5 + channels.support * (posteriorToAnterior - 0.5) * 0.16,
      yaw: side * (channels.lead * anteriorReach * 0.34 - channels.support * midbody * 0.1),
      axialScale: 1 - channels.accent * anteriorReach * 0.055,
      radialScale: 1 + channels.accent * anteriorReach * 0.075,
    }
  }

  if (kind === 'dismounting') {
    return {
      forward: channels.lead * anteriorReach * 0.09 + channels.release * posterior * 0.04,
      lateral:
        side *
        (channels.support * anteriorReach * 0.13 +
          channels.release * (0.3 + posteriorToAnterior * 0.7) * 0.085),
      vertical:
        channels.lead * anteriorReach * 0.055 -
        channels.support * anteriorReach * 0.145 +
        channels.release * posterior * 0.06,
      pitch:
        channels.lead * anteriorReach * 0.18 +
        channels.support * anteriorReach * 0.58 -
        channels.release * posterior * 0.12,
      yaw: side * (channels.support * anteriorReach * 0.4 + channels.release * midbody * 0.12),
      axialScale: 1 - channels.accent * anteriorReach * 0.045,
      radialScale: 1 + channels.accent * anteriorReach * 0.065,
    }
  }

  if (kind === 'eating') {
    return {
      forward: channels.lead * anteriorReach * 0.115,
      lateral: side * channels.accent * anteriorReach * 0.014,
      vertical: -channels.lead * anteriorReach * 0.18 + channels.accent * anteriorReach * 0.025,
      pitch: channels.lead * anteriorReach * 0.62 - channels.accent * anteriorReach * 0.12,
      yaw: side * channels.lead * anteriorReach * 0.035,
      axialScale: 1 - channels.accent * anteriorReach * 0.085,
      radialScale: 1 + channels.accent * anteriorReach * 0.11,
    }
  }

  return {
    forward: channels.lead * anteriorReach * 0.085,
    lateral: channels.accent * anteriorReach * 0.012,
    vertical: -channels.lead * anteriorReach * 0.225 - Math.abs(channels.accent) * anteriorReach * 0.012,
    pitch: channels.lead * anteriorReach * 0.76 + Math.abs(channels.accent) * anteriorReach * 0.045,
    yaw: channels.accent * anteriorReach * 0.045,
    axialScale: 1 - Math.abs(channels.accent) * anteriorReach * 0.035,
    radialScale: 1 + Math.abs(channels.accent) * anteriorReach * 0.045,
  }
}

function rootOffsetFor(
  kind: WormInteractionKind,
  side: -1 | 1,
  channels: AnimationChannels,
): WormInteractionRootOffset {
  if (kind === 'mounting') {
    return {
      forward: channels.lead * 0.045 - channels.support * 0.02,
      lateral: side * (channels.lead * 0.05 + channels.support * 0.035),
      vertical: channels.lead * 0.035 + channels.support * 0.055,
      yaw: side * (channels.lead * 0.1 + channels.support * 0.045),
    }
  }
  if (kind === 'dismounting') {
    return {
      forward: channels.lead * 0.025,
      lateral: side * (channels.support * 0.065 + channels.release * 0.05),
      vertical: channels.lead * 0.02 - channels.support * 0.025,
      yaw: side * (channels.support * 0.11 + channels.release * 0.065),
    }
  }
  if (kind === 'eating') {
    return {
      forward: channels.lead * 0.035,
      lateral: 0,
      vertical: -channels.lead * 0.012,
      yaw: side * channels.accent * 0.012,
    }
  }
  return {
    forward: channels.lead * 0.025,
    lateral: channels.accent * 0.006,
    vertical: -channels.lead * 0.016,
    yaw: channels.accent * 0.014,
  }
}

function contactCueFor(
  kind: WormInteractionKind,
  progress: number,
  channels: AnimationChannels,
): WormInteractionContactCue {
  if (kind === 'mounting') {
    return {
      target: 'skateboard-deck',
      headWeight: rise(progress, 0.12, 0.32),
      midbodyWeight: rise(progress, 0.42, 0.68),
      tailWeight: rise(progress, 0.68, 0.92),
      groundHeadWeight: 1 - rise(progress, 0.12, 0.35),
      mouthOpen: 0,
      swallow: 0,
    }
  }
  if (kind === 'dismounting') {
    return {
      target: 'skateboard-deck',
      headWeight: 1 - rise(progress, 0.1, 0.3),
      midbodyWeight: 1 - rise(progress, 0.36, 0.62),
      tailWeight: 1 - rise(progress, 0.62, 0.9),
      groundHeadWeight: channels.contact,
      mouthOpen: 0,
      swallow: 0,
    }
  }
  if (kind === 'eating') {
    return {
      target: 'food-bowl',
      headWeight: channels.contact,
      midbodyWeight: 0,
      tailWeight: 0,
      groundHeadWeight: 0,
      mouthOpen: channels.contact * (0.42 + (1 - channels.accent) * 0.5),
      swallow: channels.swallow,
    }
  }
  return {
    target: 'water-bowl',
    headWeight: channels.contact,
    midbodyWeight: 0,
    tailWeight: 0,
    groundHeadWeight: 0,
    mouthOpen: channels.contact * 0.2,
    swallow: Math.max(channels.swallow, Math.abs(channels.accent) * 0.48),
  }
}

function locomotionWeightFor(kind: WormInteractionKind, progress: number, channels: AnimationChannels) {
  if (kind === 'mounting') return 1 - rise(progress, 0.08, 0.82)
  if (kind === 'dismounting') return rise(progress, 0.18, 0.92)
  return 1 - channels.contact * 0.92
}

function phaseFor(kind: WormInteractionKind, progress: number): WormInteractionPhase {
  if (kind === 'mounting') {
    if (progress < 0.2) return 'mount-reach'
    if (progress < 0.43) return 'mount-head-contact'
    if (progress < 0.78) return 'mount-haul'
    return 'mount-settle'
  }
  if (kind === 'dismounting') {
    if (progress < 0.2) return 'dismount-brace'
    if (progress < 0.46) return 'dismount-reach-down'
    if (progress < 0.78) return 'dismount-slide-off'
    return 'dismount-release'
  }
  if (kind === 'eating') {
    if (progress < 0.24) return 'eat-lower'
    if (progress < 0.5) return 'eat-bite'
    if (progress < 0.78) return 'eat-chew'
    return 'eat-lift'
  }
  if (progress < 0.25) return 'drink-lower'
  if (progress < 0.62) return 'drink-sip'
  if (progress < 0.8) return 'drink-swallow'
  return 'drink-lift'
}

function window(value: number, inStart: number, inEnd: number, outStart: number, outEnd: number) {
  return rise(value, inStart, inEnd) * (1 - rise(value, outStart, outEnd))
}

function rise(value: number, start: number, end: number) {
  return smootherStep(clamp01((value - start) / Math.max(1e-9, end - start)))
}

function smootherStep(value: number) {
  return value * value * value * (value * (value * 6 - 15) + 10)
}

function positiveFiniteOr(value: number | undefined, fallback: number) {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : fallback
}

function finiteOr(value: number | undefined, fallback: number) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function clampInteger(value: number | undefined, minimum: number, maximum: number, fallback: number) {
  const finite = finiteOr(value, fallback)
  return Math.min(maximum, Math.max(minimum, Math.trunc(finite)))
}

function clamp01(value: number) {
  return Math.min(1, Math.max(0, value))
}
