import {
  WORM_INTERACTION_DURATIONS,
  sampleWormInteractionAnimation,
  type WormInteractionAnimationSample,
  type WormInteractionKind,
  type WormInteractionPhase,
} from '../src/scene/wormInteractionAnimation'

const kinds: WormInteractionKind[] = ['mounting', 'dismounting', 'eating', 'drinking']

for (const kind of kinds) verifyDeterminismAndBounds(kind)
verifyNeutralBoundaries()
verifyMountSequence()
verifyDismountSequence()
verifyFeedingSequence()
verifyNamedPhasesAndContinuity()
verifyLoopSeams()

const continuity = Object.fromEntries(kinds.map((kind) => [kind, maximumFrameDelta(kind, 240)]))
console.log(
  'Worm interaction animation verification passed.',
  JSON.stringify({
    convention: 'segment 0 tail, segment 15 face',
    durations: WORM_INTERACTION_DURATIONS,
    maximum240HzChannelDelta: continuity,
  }),
)

function verifyDeterminismAndBounds(kind: WormInteractionKind) {
  const duration = WORM_INTERACTION_DURATIONS[kind]
  for (let step = 0; step <= 180; step += 1) {
    const elapsedSeconds = (step / 180) * duration
    const input = { kind, elapsedSeconds, segmentCount: 16, side: -1 }
    const first = sampleWormInteractionAnimation(input)
    const repeated = sampleWormInteractionAnimation(input)
    expect(JSON.stringify(first) === JSON.stringify(repeated), `${kind} sampler must be deterministic`)
    expect(first.segments.length === 16, `${kind} must return one offset for every scene segment`)
    expect(first.headSegmentIndex === 15, `${kind} must identify scene segment 15 as the face`)
    expect(first.side === -1, `${kind} must preserve the requested board side`)
    expect(first.progress >= 0 && first.progress <= 1, `${kind} progress escaped [0, 1]`)
    expect(first.locomotionWeight >= 0 && first.locomotionWeight <= 1, `${kind} locomotion weight escaped [0, 1]`)
    expect(allNumbers(first).every(Number.isFinite), `${kind} emitted a non-finite channel`)
    expect(allContactWeights(first).every((value) => value >= 0 && value <= 1), `${kind} emitted an invalid contact cue`)
  }
}

function verifyNeutralBoundaries() {
  for (const kind of kinds) {
    const start = sampleWormInteractionAnimation({ kind, elapsedSeconds: 0 })
    const end = sampleWormInteractionAnimation({ kind, elapsedSeconds: WORM_INTERACTION_DURATIONS[kind] })
    expect(isNeutralPose(start), `${kind} must begin with a neutral additive pose`)
    expect(isNeutralPose(end), `${kind} must end with a neutral additive pose`)
    expect(end.complete, `${kind} one-shot must report completion at its duration`)
  }
}

function verifyMountSequence() {
  const early = at('mounting', 0.3)
  const middle = at('mounting', 0.55)
  const late = at('mounting', 0.82)
  const finished = at('mounting', 1)
  const head = early.segments[early.headSegmentIndex]
  const tail = early.segments[0]

  expect(early.phase === 'mount-head-contact', 'mount must name its head-contact phase')
  expect(head.vertical > tail.vertical + 0.1, 'the face must climb onto the board before the tail')
  expect(early.contact.headWeight > 0.8, 'the face must establish early deck contact')
  expect(early.contact.midbodyWeight < 0.02 && early.contact.tailWeight < 0.02, 'early mount must not teleport the body onto the deck')
  expect(middle.contact.midbodyWeight > 0.4, 'mount haul must transfer deck support into the midbody')
  expect(late.contact.tailWeight > 0.6, 'mount settle must bring the tail onto the deck')
  expect(finished.contact.headWeight === 1 && finished.contact.midbodyWeight === 1 && finished.contact.tailWeight === 1, 'completed mount must leave the whole worm supported by the deck')
  expect(finished.locomotionWeight === 0, 'completed mount must hand pose ownership away from the crawl plant')
}

function verifyDismountSequence() {
  const start = at('dismounting', 0)
  const headLed = at('dismounting', 0.35)
  const groundBrace = at('dismounting', 0.55)
  const finished = at('dismounting', 1)

  expect(start.contact.headWeight === 1 && start.contact.midbodyWeight === 1 && start.contact.tailWeight === 1, 'dismount must begin fully supported by the deck')
  expect(headLed.contact.headWeight < 0.02, 'the face must release the deck first when dismounting')
  expect(headLed.contact.midbodyWeight > 0.98 && headLed.contact.tailWeight > 0.98, 'midbody and tail must remain supported while the face reaches down')
  expect(groundBrace.contact.groundHeadWeight > 0.95, 'the face must visibly brace on terrain during the slide-off')
  expect(finished.contact.headWeight === 0 && finished.contact.midbodyWeight === 0 && finished.contact.tailWeight === 0, 'dismount must release all deck contacts')
  expect(finished.locomotionWeight === 1, 'completed dismount must return ownership to the learned crawl plant')
}

function verifyFeedingSequence() {
  const eating = at('eating', 0.45)
  const drinking = at('drinking', 0.45)
  const eatingHead = eating.segments[eating.headSegmentIndex]
  const drinkingHead = drinking.segments[drinking.headSegmentIndex]

  expect(eating.contact.target === 'food-bowl', 'eating must target the food bowl')
  expect(drinking.contact.target === 'water-bowl', 'drinking must target the water bowl')
  expect(eating.contact.headWeight > 0.98 && drinking.contact.headWeight > 0.98, 'feeding must establish face-to-bowl contact')
  expect(eating.contact.midbodyWeight === 0 && eating.contact.tailWeight === 0, 'the food bowl must contact the face, not the whole body')
  expect(drinking.contact.midbodyWeight === 0 && drinking.contact.tailWeight === 0, 'the water bowl must contact the face, not the whole body')
  expect(eatingHead.vertical < -0.12 && eatingHead.pitch > 0.35, 'eating must lower and pitch the face into the bowl')
  expect(drinkingHead.vertical < eatingHead.vertical - 0.025, 'drinking must use a distinct, deeper face dip')
  expect(Math.abs(eating.segments[0].vertical) < 1e-12, 'feeding must not drag the posterior segment into the bowl')
  expect(eating.contact.mouthOpen > drinking.contact.mouthOpen + 0.2, 'eating must read as a bite rather than a sip')
  expect(eating.locomotionWeight < 0.1 && drinking.locomotionWeight < 0.1, 'bowl contact must pause learned translation instead of skating through the vessel')
  expect(at('eating', 0.78).contact.swallow > 0.9, 'eating must include a swallow cue')
  expect(at('drinking', 0.76).contact.swallow > 0.9, 'drinking must include a swallow cue')
}

function verifyNamedPhasesAndContinuity() {
  const expected: Record<WormInteractionKind, WormInteractionPhase[]> = {
    mounting: ['mount-reach', 'mount-head-contact', 'mount-haul', 'mount-settle'],
    dismounting: ['dismount-brace', 'dismount-reach-down', 'dismount-slide-off', 'dismount-release'],
    eating: ['eat-lower', 'eat-bite', 'eat-chew', 'eat-lift'],
    drinking: ['drink-lower', 'drink-sip', 'drink-swallow', 'drink-lift'],
  }
  const boundaries: Record<WormInteractionKind, number[]> = {
    mounting: [0.2, 0.43, 0.78],
    dismounting: [0.2, 0.46, 0.78],
    eating: [0.24, 0.5, 0.78],
    drinking: [0.25, 0.62, 0.8],
  }

  for (const kind of kinds) {
    const seen = new Set<WormInteractionPhase>()
    for (let step = 0; step <= 200; step += 1) seen.add(at(kind, step / 200).phase)
    for (const phase of expected[kind]) expect(seen.has(phase), `${kind} never reported ${phase}`)

    for (const boundary of boundaries[kind]) {
      const before = at(kind, boundary - 1e-5)
      const after = at(kind, boundary + 1e-5)
      expect(maximumNumericDifference(before, after) < 0.001, `${kind} pose jumped at named phase boundary ${boundary}`)
    }
  }
}

function verifyLoopSeams() {
  for (const kind of ['eating', 'drinking'] as const) {
    const duration = WORM_INTERACTION_DURATIONS[kind]
    const before = sampleWormInteractionAnimation({ kind, elapsedSeconds: duration - 1e-5, loop: true })
    const seam = sampleWormInteractionAnimation({ kind, elapsedSeconds: duration, loop: true })
    const after = sampleWormInteractionAnimation({ kind, elapsedSeconds: duration + 1e-5, loop: true })
    expect(seam.cycle === 1 && seam.progress === 0, `${kind} loop must advance its cycle exactly at the seam`)
    expect(isNeutralPose(seam), `${kind} loop seam must be pose-neutral`)
    expect(maximumNumericDifference(before, after) < 0.001, `${kind} loop must cross its seam continuously`)
  }
}

function maximumFrameDelta(kind: WormInteractionKind, hertz: number) {
  const duration = WORM_INTERACTION_DURATIONS[kind]
  const frameCount = Math.ceil(duration * hertz)
  let previous = sampleWormInteractionAnimation({ kind, elapsedSeconds: 0 })
  let maximum = 0
  for (let frame = 1; frame <= frameCount; frame += 1) {
    const current = sampleWormInteractionAnimation({ kind, elapsedSeconds: Math.min(duration, frame / hertz) })
    maximum = Math.max(maximum, maximumNumericDifference(previous, current))
    previous = current
  }
  expect(maximum < 0.055, `${kind} changes too abruptly at ${hertz} Hz (${maximum.toFixed(6)})`)
  return Number(maximum.toFixed(6))
}

function at(kind: WormInteractionKind, progress: number) {
  return sampleWormInteractionAnimation({
    kind,
    elapsedSeconds: progress * WORM_INTERACTION_DURATIONS[kind],
    segmentCount: 16,
    side: 1,
  })
}

function isNeutralPose(sample: WormInteractionAnimationSample) {
  const rootNeutral = [sample.root.forward, sample.root.lateral, sample.root.vertical, sample.root.yaw].every((value) => Math.abs(value) < 1e-12)
  const segmentsNeutral = sample.segments.every(
    (segment) =>
      Math.abs(segment.forward) < 1e-12 &&
      Math.abs(segment.lateral) < 1e-12 &&
      Math.abs(segment.vertical) < 1e-12 &&
      Math.abs(segment.pitch) < 1e-12 &&
      Math.abs(segment.yaw) < 1e-12 &&
      Math.abs(segment.axialScale - 1) < 1e-12 &&
      Math.abs(segment.radialScale - 1) < 1e-12,
  )
  return rootNeutral && segmentsNeutral
}

function allNumbers(sample: WormInteractionAnimationSample) {
  return [
    sample.progress,
    sample.durationSeconds,
    sample.locomotionWeight,
    sample.root.forward,
    sample.root.lateral,
    sample.root.vertical,
    sample.root.yaw,
    ...sample.segments.flatMap((segment) => [
      segment.forward,
      segment.lateral,
      segment.vertical,
      segment.pitch,
      segment.yaw,
      segment.axialScale,
      segment.radialScale,
    ]),
    ...allContactWeights(sample),
  ]
}

function allContactWeights(sample: WormInteractionAnimationSample) {
  return [
    sample.contact.headWeight,
    sample.contact.midbodyWeight,
    sample.contact.tailWeight,
    sample.contact.groundHeadWeight,
    sample.contact.mouthOpen,
    sample.contact.swallow,
  ]
}

function maximumNumericDifference(first: WormInteractionAnimationSample, second: WormInteractionAnimationSample) {
  const firstNumbers = animationNumbers(first)
  const secondNumbers = animationNumbers(second)
  expect(firstNumbers.length === secondNumbers.length, 'sample channel count changed')
  return Math.max(...firstNumbers.map((value, index) => Math.abs(value - secondNumbers[index])))
}

function animationNumbers(sample: WormInteractionAnimationSample) {
  return [
    sample.locomotionWeight,
    sample.root.forward,
    sample.root.lateral,
    sample.root.vertical,
    sample.root.yaw,
    ...sample.segments.flatMap((segment) => [
      segment.forward,
      segment.lateral,
      segment.vertical,
      segment.pitch,
      segment.yaw,
      segment.axialScale,
      segment.radialScale,
    ]),
    ...allContactWeights(sample),
  ]
}

function expect(condition: boolean, message: string): asserts condition {
  if (!condition) throw new Error(message)
}
