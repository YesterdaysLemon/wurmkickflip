import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { deriveWurmAnatomy } from '../src/creature/anatomy'
import {
  EvolvedLocomotionPolicy,
  parseLocomotionPolicy,
  type LocomotionPolicyArtifact,
  type LocomotionPolicyTelemetry,
  type LocomotionSensors,
} from '../src/policy/locomotionPolicy'
import { SEGMENT_COUNT, type PolicyAction, type SegmentSnapshot } from '../src/policy/types'
import {
  advanceStunt,
  createStuntState,
  applyGaitLateralShove,
  gaitControllerOwnsBody,
  gaitExperimentLifecycleNotice,
  scriptedGaitMusclesAnteriorToPosterior,
} from '../src/scene/terrariumSimulation'
import { TERRAIN_GRID_RESOLUTION, type TerrainField } from '../src/scene/terrainField'
import { stepArticulatedWorm } from '../src/scene/wormDynamics'
import { createWormLocomotionPlant, type WormLocomotionPlant } from '../src/scene/wormLocomotion'

const TELEMETRY_VECTOR_KEYS = [
  'hidden',
  'drives',
  'commands',
  'requestedCommands',
  'sensedBends',
  'sensedBendVelocities',
  'sensedContactLoads',
  'sensedSlipSpeeds',
  'sensedObstacleForward',
  'sensedObstacleRight',
] as const

const artifactSource = process.env.WURMKICKFLIP_LOCOMOTION_ARTIFACT
  ? resolve(process.env.WURMKICKFLIP_LOCOMOTION_ARTIFACT)
  : new URL('../public/models/wurmkickflip_locomotion_policy.json', import.meta.url)
const artifact = parseLocomotionPolicy(JSON.parse(await readFile(artifactSource, 'utf8')))
const anatomy = deriveWurmAnatomy(null)
const field = flatField(0.9)
const timestep = artifact.plant.timestep
const diagnosticOnly = process.env.WURMKICKFLIP_GAIT_DIAGNOSTIC_ONLY === '1'
const performanceFailures: string[] = []

const telemetry = verifyDetachedFiniteTelemetry()
const numbNeuron = verifyNumbNeuronTicks()
const mirroredSensors = verifyMirroredSensorWiring()
const anatomicalOrdering = verifyScriptedAnatomicalOrdering()
const lifecycleActionOwnership = verifyLifecycleActionOwnership()
const causalCone = verifyLocalCausalCone()
const zeroTraction = verifyZeroTractionConservation()
const shove = verifyWholeBodyShove()
const recovery = verifyPairedRecovery()
const headLeading = verifyHeadLeadingPromotion(recovery)

console.log(
  JSON.stringify(
    {
      ok: performanceFailures.length === 0,
      diagnosticOnly,
      performanceFailures,
      modelVersion: artifact.modelVersion,
      telemetry,
      numbNeuron,
      mirroredSensors,
      anatomicalOrdering,
      lifecycleActionOwnership,
      causalCone,
      zeroTraction,
      shove,
      recovery: {
        baselineProgress: round(recovery.baseline.totalProgress),
        perturbedProgress: round(recovery.perturbed.totalProgress),
        progressRetention: round(recovery.progressRetention),
        baselineLateProgressSpeed: round(recovery.baseline.lateProgressSpeed),
        perturbedLateProgressSpeed: round(recovery.perturbed.lateProgressSpeed),
        lateSpeedRetention: round(recovery.lateSpeedRetention),
        maximumChainError: round(recovery.perturbed.maximumChainError),
        deterministicTraceHash: recovery.perturbed.traceHash,
      },
      gaitDiagnostics: {
        note:
          headLeading.required && headLeading.passed
            ? 'The v3 artifact cleared explicit target-facing, head-axis, and transverse-travel promotion gates.'
            : headLeading.required
              ? 'The v3 artifact did not clear every head-leading promotion gate.'
              : 'Body-axis alignment and transverse travel describe this legacy controller; they are not promoted as proof of a head-first gait.',
        promotionGate: headLeading,
        baseline: { ...recovery.baseline.diagnostics, diagnosticOnly: !headLeading.required },
        afterTransientPerturbation: {
          ...recovery.perturbed.diagnostics,
          diagnosticOnly: !headLeading.required,
        },
      },
    },
    null,
    2,
  ),
)
console.log(
  diagnosticOnly
    ? 'Gait microscope diagnostic report complete.'
    : 'Gait microscope causal and recovery verification passed.',
)

function verifyDetachedFiniteTelemetry() {
  const policy = new EvolvedLocomotionPolicy(artifact)
  const fixture = asymmetricFixture()
  policy.run(fixture.sensors, fixture.bends, fixture.bendVelocities)
  policy.applyPerturbation('numb-neuron', 7, timestep * 3)

  const snapshot = policy.getTelemetry()
  assertFiniteTelemetry(snapshot)
  const serializedSnapshot = JSON.stringify(snapshot)

  policy.run(fixture.sensors, fixture.bends, fixture.bendVelocities)
  assert.equal(
    JSON.stringify(snapshot),
    serializedSnapshot,
    'a telemetry snapshot must not change when the controller advances',
  )

  snapshot.hidden[0] = Number.NaN
  snapshot.commands[1] = Number.POSITIVE_INFINITY
  if (snapshot.activePerturbation) snapshot.activePerturbation.remainingSeconds = -99
  const fresh = policy.getTelemetry()
  assertFiniteTelemetry(fresh)
  assert.notEqual(fresh.hidden[0], snapshot.hidden[0], 'telemetry hidden state leaked a mutable reference')
  assert.notEqual(
    fresh.commands[1],
    snapshot.commands[1],
    'telemetry command state leaked a mutable reference',
  )
  assert.ok(
    (fresh.activePerturbation?.remainingSeconds ?? 0) >= 0,
    'telemetry perturbation metadata leaked a mutable reference',
  )

  return {
    vectorCount: TELEMETRY_VECTOR_KEYS.length,
    valuesPerVector: SEGMENT_COUNT,
    detachedFromControllerState: true,
  }
}

function verifyNumbNeuronTicks() {
  const segment = 7
  const durationTicks = 3
  const policy = new EvolvedLocomotionPolicy(artifact)
  const fixture = asymmetricFixture()
  policy.run(fixture.sensors, fixture.bends, fixture.bendVelocities)
  policy.applyPerturbation('numb-neuron', segment, durationTicks * timestep)

  const before = policy.getTelemetry()
  assert.equal(before.hidden[segment], 0, 'applying numb-neuron must immediately clear hidden state')
  assert.equal(before.commands[segment], 0, 'applying numb-neuron must immediately clear motor state')
  expectClose(
    before.activePerturbation?.remainingSeconds ?? -1,
    durationTicks * timestep,
    1e-12,
    'numb-neuron duration was not quantized to the requested ticks',
  )

  let requestedMagnitude = 0
  for (let tick = 0; tick < durationTicks; tick += 1) {
    const action = policy.run(fixture.sensors, fixture.bends, fixture.bendVelocities)
    const current = policy.getTelemetry()
    assert.ok(action[segment * 2] === 0, `numb neuron emitted a dorsal command on tick ${tick}`)
    assert.ok(action[segment * 2 + 1] === 0, `numb neuron emitted a ventral command on tick ${tick}`)
    assert.equal(current.hidden[segment], 0, `numb neuron retained hidden activation on tick ${tick}`)
    assert.equal(current.commands[segment], 0, `numb neuron retained motor activation on tick ${tick}`)
    requestedMagnitude = Math.max(requestedMagnitude, Math.abs(current.requestedCommands[segment]))
    const remainingTicks = durationTicks - tick - 1
    if (remainingTicks === 0) {
      assert.equal(
        current.activePerturbation?.remainingSeconds,
        0,
        'final affected tick must remain labeled until the next inference step',
      )
    } else {
      expectClose(
        current.activePerturbation?.remainingSeconds ?? -1,
        remainingTicks * timestep,
        1e-12,
        `numb-neuron remaining time drifted after tick ${tick}`,
      )
    }
  }
  assert.ok(requestedMagnitude > 1e-3, 'numb-neuron fixture did not suppress a real requested command')

  const recoveredAction = policy.run(fixture.sensors, fixture.bends, fixture.bendVelocities)
  const recovered = policy.getTelemetry()
  assert.equal(recovered.activePerturbation, null, 'expired perturbation survived into the recovery tick')
  assert.ok(Math.abs(recovered.hidden[segment]) > 1e-4, 'expired neuron did not recover hidden activation')
  assert.ok(Math.abs(recoveredAction[segment * 2] ?? 0) > 1e-4, 'expired neuron did not recover its motor')

  return {
    segment,
    durationTicks,
    requestedMagnitude: round(requestedMagnitude),
    recoveredHidden: round(recovered.hidden[segment]),
    recoveredCommand: round(recovered.commands[segment]),
  }
}

function verifyMirroredSensorWiring() {
  const fixture = asymmetricFixture()
  const reversedPolicy = new EvolvedLocomotionPolicy(artifact)
  const manualMirror = new EvolvedLocomotionPolicy(artifact)
  reversedPolicy.applyPerturbation('reverse-sensors', null, timestep * 2)

  const reversedAction = reversedPolicy.run(fixture.sensors, fixture.bends, fixture.bendVelocities)
  const manuallyMirroredSensors: LocomotionSensors = {
    ...fixture.sensors,
    contactLoads: [...fixture.contactLoads].reverse(),
    slipSpeeds: [...fixture.slipSpeeds].reverse(),
    obstacleForward: [...fixture.obstacleForward].reverse(),
    obstacleRight: [...fixture.obstacleRight].reverse(),
  }
  const manualAction = manualMirror.run(
    manuallyMirroredSensors,
    [...fixture.bends].reverse(),
    [...fixture.bendVelocities].reverse(),
  )
  assert.deepEqual(
    Array.from(reversedAction),
    Array.from(manualAction),
    'reverse-sensors must equal a manual anterior/posterior wiring mirror',
  )

  const telemetry = reversedPolicy.getTelemetry()
  assert.deepEqual(telemetry.sensedBends, [...fixture.bends].reverse(), 'bend wiring was not mirrored')
  assert.deepEqual(
    telemetry.sensedBendVelocities,
    [...fixture.bendVelocities].reverse(),
    'bend-velocity wiring was not mirrored',
  )
  assert.deepEqual(
    telemetry.sensedContactLoads,
    [...fixture.contactLoads].reverse(),
    'contact-load wiring was not mirrored',
  )
  assert.deepEqual(
    telemetry.sensedSlipSpeeds,
    [...fixture.slipSpeeds].reverse(),
    'slip-speed wiring was not mirrored',
  )
  assert.deepEqual(
    telemetry.sensedObstacleForward,
    [...fixture.obstacleForward].reverse(),
    'forward-obstacle wiring was not mirrored',
  )
  assert.deepEqual(
    telemetry.sensedObstacleRight,
    [...fixture.obstacleRight].reverse(),
    'right-obstacle wiring was not mirrored',
  )
  assert.equal(telemetry.activePerturbation?.segment, null, 'whole-chain sensor mirror acquired a segment')

  return {
    actionMatchesManualMirror: true,
    firstSensedContact: round(telemetry.sensedContactLoads[0]),
    lastSensedContact: round(telemetry.sensedContactLoads.at(-1) ?? 0),
  }
}

function verifyScriptedAnatomicalOrdering() {
  const sceneOrderedAction = new Float32Array(SEGMENT_COUNT * 2)
  for (let sceneSegment = 0; sceneSegment < SEGMENT_COUNT; sceneSegment += 1) {
    const command = (sceneSegment + 1) / SEGMENT_COUNT
    sceneOrderedAction[sceneSegment * 2] = command
    sceneOrderedAction[sceneSegment * 2 + 1] = -command
  }
  const headToTail = scriptedGaitMusclesAnteriorToPosterior(sceneOrderedAction)
  expectClose(headToTail[0], 1, 1e-7, 'scripted head command did not map to S01')
  expectClose(headToTail.at(-1) ?? 0, 1 / SEGMENT_COUNT, 1e-7, 'scripted tail command did not map to S16')
  assert.ok(
    headToTail.every((value, segment) => Math.abs(value - (SEGMENT_COUNT - segment) / SEGMENT_COUNT) <= 1e-7),
    'scripted scene-order action was not completely reversed into anatomy order',
  )
  return { inputOrder: 'tail-to-head', outputOrder: 'head-to-tail', head: 1, tail: 1 / SEGMENT_COUNT }
}

function verifyLifecycleActionOwnership() {
  const scriptedToNeural = createStuntState(field)
  scriptedToNeural.locomotionState = 'feeding'
  scriptedToNeural.locomotionTime = 99
  scriptedToNeural.feedingResourceId = 'food-bowl'
  assert.equal(
    gaitControllerOwnsBody(scriptedToNeural),
    false,
    'feeding unexpectedly owned the gait controller',
  )

  advanceStunt(
    scriptedToNeural,
    new Float32Array(SEGMENT_COUNT * 2),
    timestep,
    9.81,
    'freestyle',
    field,
    null,
  )
  assert.equal(
    scriptedToNeural.locomotionState,
    'crawling',
    'feeding handoff fixture did not return to crawl',
  )
  assert.equal(
    scriptedToNeural.previousActionOrder,
    'scene-tail-to-head',
    'handoff tick was relabeled using its post-step crawl state',
  )
  assert.equal(
    scriptedToNeural.previousActionApplication,
    'lifecycle-handoff',
    'scripted-to-neural transition was reported as an ordinary applied action',
  )
  assert.equal(gaitControllerOwnsBody(scriptedToNeural), true, 'crawl did not acquire the gait controller')

  const neuralAction = new Float32Array(SEGMENT_COUNT * 2)
  neuralAction[0] = 0.25
  neuralAction[1] = -0.25
  advanceStunt(scriptedToNeural, neuralAction, timestep, 9.81, 'freestyle', field, null)
  assert.equal(
    scriptedToNeural.previousActionOrder,
    'anterior-to-posterior',
    'first neural crawl tick retained the scripted action ordering',
  )
  assert.equal(
    scriptedToNeural.previousActionApplication,
    'neural',
    'first complete crawl tick was not neural',
  )

  const neuralToFeeding = createStuntState(field)
  neuralToFeeding.locomotionState = 'crawling'
  neuralToFeeding.mountBlend = 0
  const food = neuralToFeeding.resources.find(resource => resource.id === 'food-bowl')
  assert.ok(food, 'food bowl is missing from neural-to-feeding fixture')
  const head = neuralToFeeding.segments.at(-1)
  assert.ok(head, 'worm head is missing from neural-to-feeding fixture')
  const contentsY = food.groundHeight + food.appearance.height * 0.72
  head.x = food.position[0] - 0.055
  head.y = contentsY
  head.z = food.position[2]
  head.pitch = 0
  head.yaw = 0
  neuralToFeeding.needs.hunger = 1
  neuralToFeeding.needs.thirst = 0
  neuralToFeeding.needs.wellbeing = 0
  neuralToFeeding.needs.targetResourceId = 'food-bowl'
  advanceStunt(neuralToFeeding, neuralAction, timestep, 9.81, 'freestyle', field, null)
  assert.equal(
    neuralToFeeding.locomotionState,
    'feeding',
    'crawl-to-feeding handoff fixture did not enter feeding',
  )
  assert.equal(
    neuralToFeeding.previousActionOrder,
    'anterior-to-posterior',
    'crawl-to-feeding handoff lost the neural command ordering',
  )
  assert.equal(
    neuralToFeeding.previousActionApplication,
    'lifecycle-handoff',
    'ignored crawl command was reported as applied neural locomotion',
  )

  const neuralToMounting = createStuntState(field)
  neuralToMounting.locomotionState = 'seeking'
  neuralToMounting.needs.targetResourceId = 'skateboard'
  const mountingHead = neuralToMounting.segments.at(-1)
  assert.ok(mountingHead, 'worm head is missing from neural-to-mounting fixture')
  mountingHead.x = neuralToMounting.boardX
  mountingHead.z = neuralToMounting.boardZ
  advanceStunt(neuralToMounting, neuralAction, timestep, 9.81, 'kickflip', field, null)
  assert.equal(
    neuralToMounting.locomotionState,
    'mounting',
    'seeking-to-mounting handoff fixture did not enter mounting',
  )
  assert.equal(
    neuralToMounting.previousActionApplication,
    'lifecycle-handoff',
    'ignored seeking command was reported as applied neural locomotion',
  )

  return {
    handoffTickOrder: 'scene-tail-to-head',
    handoffTickApplication: 'lifecycle-handoff',
    followingTickOrder: 'anterior-to-posterior',
    followingTickApplication: 'neural',
    crawlToFeedingApplication: neuralToFeeding.previousActionApplication,
    seekingToMountingApplication: neuralToMounting.previousActionApplication,
  }
}

function verifyLocalCausalCone() {
  const pulseSegment = 7
  const baseline = new EvolvedLocomotionPolicy(artifact)
  const pulsed = new EvolvedLocomotionPolicy(artifact)
  const bends = new Float64Array(SEGMENT_COUNT)
  const velocities = new Float64Array(SEGMENT_COUNT)
  const deltas: number[][] = []

  for (let tick = 0; tick < 12; tick += 1) {
    const baselineSensors = baseSensors()
    const pulseSensors = baseSensors()
    if (tick === 0) {
      const contactLoads = pulseSensors.contactLoads as Float64Array
      const slipSpeeds = pulseSensors.slipSpeeds as Float64Array
      const obstacleForward = pulseSensors.obstacleForward as Float64Array
      const obstacleRight = pulseSensors.obstacleRight as Float64Array
      contactLoads[pulseSegment] = 0
      slipSpeeds[pulseSegment] = 1.7
      obstacleForward[pulseSegment] = 0.8
      obstacleRight[pulseSegment] = 0.8
    }
    const baselineAction = baseline.run(baselineSensors, bends, velocities)
    const pulsedAction = pulsed.run(pulseSensors, bends, velocities)
    const tickDelta = Array.from({ length: SEGMENT_COUNT }, (_, segment) =>
      Math.abs((pulsedAction[segment * 2] ?? 0) - (baselineAction[segment * 2] ?? 0)),
    )
    deltas.push(tickDelta)

    if (tick <= 5) {
      const maximumRadius = tick + 1
      for (let segment = 0; segment < SEGMENT_COUNT; segment += 1) {
        if (Math.abs(segment - pulseSegment) > maximumRadius) {
          assert.ok(
            tickDelta[segment] <= 1e-8,
            `local sensor pulse teleported to segment ${segment} on tick ${tick}`,
          )
        }
      }
    }
  }

  const initialLocalDelta = maximum(deltas[0].slice(pulseSegment - 1, pulseSegment + 2))
  const radiusTwoDelta = Math.max(deltas[1][pulseSegment - 2], deltas[1][pulseSegment + 2])
  const radiusThreeDelta = Math.max(deltas[2][pulseSegment - 3], deltas[2][pulseSegment + 3])
  const settledDelta = maximum(deltas[9])
  expectPerformance(
    initialLocalDelta > 0.1,
    'local sensor pulse did not materially affect its local neighborhood',
  )
  expectPerformance(radiusTwoDelta > 1e-3, 'local sensor effect did not propagate to graph radius two')
  expectPerformance(radiusThreeDelta > 1e-3, 'local sensor effect did not propagate to graph radius three')
  expectPerformance(settledDelta < 1e-5, `local sensor pulse failed to settle (${settledDelta})`)

  return {
    pulseSegment,
    initialLocalDelta: round(initialLocalDelta),
    radiusTwoDelta: round(radiusTwoDelta),
    radiusThreeDelta: round(radiusThreeDelta),
    settledDelta: round(settledDelta),
    causalRadiusPerTick: 1,
  }
}

function verifyZeroTractionConservation() {
  const policy = new EvolvedLocomotionPolicy(artifact)
  const plant = createWormLocomotionPlant()
  const segments = createSegments(artifact)
  const target = { x: 20, z: 0 }
  const start = center(segments)
  let maximumAction = 0
  let maximumChainError = 0

  for (let step = 0; step < 480; step += 1) {
    const action = policy.run(sensorsFor(segments, plant, target, 0), plant.joints, plant.jointVelocities)
    maximumAction = Math.max(maximumAction, ...Array.from(action, Math.abs))
    stepArticulatedWorm(plant, segments, action, timestep, field, null, anatomy, 0)
    maximumChainError = Math.max(maximumChainError, chainError(segments, artifact.plant.spacing))
  }

  const end = center(segments)
  const displacement = Math.hypot(end.x - start.x, end.z - start.z)
  expectPerformance(maximumAction > 0.5, 'zero-traction conservation did not exercise active neural motors')
  expectPerformance(
    displacement <= 1e-10,
    `zero traction translated horizontal center of mass (${displacement})`,
  )
  expectPerformance(
    maximumChainError < 0.003,
    `zero-traction chain escaped its constraints (${maximumChainError})`,
  )

  return {
    activeMotorMaximum: round(maximumAction),
    centerOfMassDisplacement: displacement,
    maximumChainError: round(maximumChainError),
  }
}

function verifyWholeBodyShove() {
  const authoredState = createStuntState(field)
  const authoredVelocity = authoredState.segments.map(segment => [segment.vx, segment.vz])
  assert.equal(
    gaitExperimentLifecycleNotice(authoredState),
    'Experiment skipped because authored motion took control. Return to Free crawl and try again.',
    'authored bodies must explain why a gait experiment was rejected',
  )
  assert.equal(applyGaitLateralShove(authoredState, 1.1), false, 'authored body accepted a gait shove')
  assert.deepEqual(
    authoredState.segments.map(segment => [segment.vx, segment.vz]),
    authoredVelocity,
    'a rejected gait shove mutated the authored body',
  )

  const first = runShoveTrial()
  const second = runShoveTrial()
  assert.equal(first.traceHash, second.traceHash, 'identical whole-body shoves must be deterministic')
  expectPerformance(
    first.maximumChainError < 0.005,
    `shoved chain escaped its constraints (${first.maximumChainError})`,
  )
  return first
}

function runShoveTrial() {
  const state = createStuntState(field)
  state.locomotionState = 'crawling'
  assert.equal(gaitExperimentLifecycleNotice(state), null, 'crawl body reported a false lifecycle race')
  state.wormHeading = 0.37
  state.segments = createSegments(artifact, state.wormX, state.wormZ, state.wormHeading)
  const before = state.segments.map(segment => ({
    x: segment.x,
    z: segment.z,
    vx: segment.vx,
    vz: segment.vz,
  }))
  const impulse = 1.1
  const rightX = -Math.sin(state.wormHeading)
  const rightZ = Math.cos(state.wormHeading)
  assert.equal(applyGaitLateralShove(state, impulse), true, 'crawl-state shove was rejected')

  for (let segment = 0; segment < SEGMENT_COUNT; segment += 1) {
    assert.equal(state.segments[segment].x, before[segment].x, 'shove changed a segment position')
    assert.equal(state.segments[segment].z, before[segment].z, 'shove changed a segment position')
    expectClose(
      state.segments[segment].vx - before[segment].vx,
      rightX * impulse,
      1e-12,
      `segment ${segment} did not receive the complete lateral x impulse`,
    )
    expectClose(
      state.segments[segment].vz - before[segment].vz,
      rightZ * impulse,
      1e-12,
      `segment ${segment} did not receive the complete lateral z impulse`,
    )
  }

  const policy = new EvolvedLocomotionPolicy(artifact)
  const target = {
    x: state.wormX + Math.cos(state.wormHeading) * 20,
    z: state.wormZ + Math.sin(state.wormHeading) * 20,
  }
  const hash = createHash('sha256')
  let maximumChainError = 0
  for (let step = 0; step < 240; step += 1) {
    const action = policy.run(
      sensorsFor(state.segments, state.locomotionPlant, target, 0.9),
      state.locomotionPlant.joints,
      state.locomotionPlant.jointVelocities,
    )
    const result = stepArticulatedWorm(
      state.locomotionPlant,
      state.segments,
      action,
      timestep,
      field,
      null,
      anatomy,
    )
    maximumChainError = Math.max(maximumChainError, chainError(state.segments, artifact.plant.spacing))
    hash.update(JSON.stringify(traceCheckpoint(step, result.root, state.segments, action)))
  }

  return {
    impulse,
    perSegmentVelocityDelta: [round(rightX * impulse), round(rightZ * impulse)],
    maximumChainError: round(maximumChainError),
    traceHash: hash.digest('hex'),
  }
}

function verifyPairedRecovery() {
  const baseline = runRecoveryRollout(false)
  const perturbed = runRecoveryRollout(true)
  const repeated = runRecoveryRollout(true)

  assert.equal(
    baseline.prefixHash,
    perturbed.prefixHash,
    'paired recovery rollouts diverged before the intervention',
  )
  assert.equal(perturbed.traceHash, repeated.traceHash, 'perturbed recovery rollout is not deterministic')
  assert.equal(perturbed.perturbationExpired, true, 'neural perturbation survived into the recovery window')
  expectPerformance(
    baseline.totalProgress > 8,
    `baseline recovery fixture made too little progress (${baseline.totalProgress})`,
  )

  const progressRetention = perturbed.totalProgress / baseline.totalProgress
  const lateSpeedRetention = perturbed.lateProgressSpeed / baseline.lateProgressSpeed
  expectPerformance(
    progressRetention >= 0.85,
    `transient perturbation retained only ${(progressRetention * 100).toFixed(1)}% of baseline progress`,
  )
  expectPerformance(
    lateSpeedRetention >= 0.8,
    `post-perturbation progress speed recovered to only ${(lateSpeedRetention * 100).toFixed(1)}%`,
  )
  expectPerformance(
    perturbed.maximumChainError < 0.005,
    `transient perturbation destabilized the chain (${perturbed.maximumChainError})`,
  )

  return { baseline, perturbed, progressRetention, lateSpeedRetention }
}

function verifyHeadLeadingPromotion(recovery: ReturnType<typeof verifyPairedRecovery>) {
  const required = artifact.training.objectiveVersion === 'articulated-head-leading-transient-recovery-v3'
  const baseline = recovery.baseline.diagnostics
  const perturbed = recovery.perturbed.diagnostics
  const progressRetention = recovery.perturbed.totalProgress / recovery.baseline.totalProgress
  const lateSpeedRetention = recovery.perturbed.lateProgressSpeed / recovery.baseline.lateProgressSpeed
  const criteria = {
    baselineProgress: recovery.baseline.totalProgress >= 8.5,
    progressRetention: progressRetention >= 0.88,
    lateSpeedRetention: lateSpeedRetention >= 0.83,
    baselineTargetAlignment: baseline.meanTargetAlignment >= 0.55,
    baselineHeadAxisVelocity: baseline.meanVelocityAlongBodyAxis >= 0.65,
    baselineTransverseFraction: baseline.meanAbsoluteTransverseFraction <= 0.45,
    recoveryTargetAlignment: perturbed.meanTargetAlignment >= 0.45,
    recoveryHeadAxisVelocity: perturbed.meanVelocityAlongBodyAxis >= 0.45,
    recoveryTransverseFraction: perturbed.meanAbsoluteTransverseFraction <= 0.55,
  }
  if (required) {
    expectPerformance(
      criteria.baselineProgress,
      `v3 baseline progress lacks publication margin (${recovery.baseline.totalProgress})`,
    )
    expectPerformance(
      criteria.progressRetention,
      `v3 progress retention lacks publication margin (${progressRetention})`,
    )
    expectPerformance(
      criteria.lateSpeedRetention,
      `v3 late-speed retention lacks publication margin (${lateSpeedRetention})`,
    )
    expectPerformance(
      criteria.baselineTargetAlignment,
      `v3 gait does not face its target while moving (${baseline.meanTargetAlignment})`,
    )
    expectPerformance(
      criteria.baselineHeadAxisVelocity,
      `v3 gait is not predominantly head-leading (${baseline.meanVelocityAlongBodyAxis})`,
    )
    expectPerformance(
      criteria.baselineTransverseFraction,
      `v3 gait remains too transverse (${baseline.meanAbsoluteTransverseFraction})`,
    )
    expectPerformance(
      criteria.recoveryTargetAlignment,
      `v3 gait lost target-facing recovery (${perturbed.meanTargetAlignment})`,
    )
    expectPerformance(
      criteria.recoveryHeadAxisVelocity,
      `v3 gait lost head-leading recovery (${perturbed.meanVelocityAlongBodyAxis})`,
    )
    expectPerformance(
      criteria.recoveryTransverseFraction,
      `v3 gait recovered as transverse travel (${perturbed.meanAbsoluteTransverseFraction})`,
    )
  }
  return {
    required,
    passed: !required || Object.values(criteria).every(Boolean),
    criteria,
    baselineMinimumProgress: 8.5,
    minimumProgressRetention: 0.88,
    minimumLateSpeedRetention: 0.83,
    baselineMinimumTargetAlignment: 0.55,
    baselineMinimumHeadAxisVelocity: 0.65,
    baselineMaximumTransverseFraction: 0.45,
    recoveryMinimumTargetAlignment: 0.45,
    recoveryMinimumHeadAxisVelocity: 0.45,
    recoveryMaximumTransverseFraction: 0.55,
  }
}

function runRecoveryRollout(intervene: boolean): RecoveryRollout {
  const policy = new EvolvedLocomotionPolicy(artifact)
  const plant = createWormLocomotionPlant()
  const segments = createSegments(artifact)
  const target = { x: 20, z: 0 }
  const perturbationStart = 240
  const perturbationEnd = 330
  const steps = 900
  const distances = [distanceToTarget(segments, target)]
  const prefixHash = createHash('sha256')
  const traceHash = createHash('sha256')
  let maximumChainError = 0
  let targetAlignmentTotal = 0
  let velocityAlongBodyTotal = 0
  let transverseFractionTotal = 0
  let diagnosticPathLength = 0

  for (let step = 0; step < steps; step += 1) {
    if (intervene && step === perturbationStart) {
      const heading = bodyHeading(segments)
      const rightX = -Math.sin(heading)
      const rightZ = Math.cos(heading)
      for (const segment of segments) {
        segment.vx += rightX * 1.1
        segment.vz += rightZ * 1.1
      }
      policy.applyPerturbation('numb-neuron', 7, (perturbationEnd - perturbationStart) * timestep)
    }
    const tractionScale = intervene && step >= perturbationStart && step < perturbationEnd ? 1 / 15 : 1
    const previousRoot = center(segments)
    const action = policy.run(
      sensorsFor(segments, plant, target, 0.9 * tractionScale),
      plant.joints,
      plant.jointVelocities,
    )
    const result = stepArticulatedWorm(plant, segments, action, timestep, field, null, anatomy, tractionScale)
    distances.push(Math.hypot(target.x - result.root.x, target.z - result.root.z))
    maximumChainError = Math.max(maximumChainError, chainError(segments, artifact.plant.spacing))
    const checkpoint = JSON.stringify(traceCheckpoint(step, result.root, segments, action))
    traceHash.update(checkpoint)
    if (step < perturbationStart) prefixHash.update(checkpoint)

    // Compare both paired rollouts over the same genuinely post-transient
    // window. Including the pre-event or lesioned path here would let strong
    // approach motion conceal a weak recovered gait.
    if (step >= perturbationEnd) {
      const dx = target.x - result.root.x
      const dz = target.z - result.root.z
      const targetDistance = Math.max(Math.hypot(dx, dz), 1e-9)
      const forwardX = Math.cos(result.root.heading)
      const forwardZ = Math.sin(result.root.heading)
      const rightX = -forwardZ
      const rightZ = forwardX
      const rootDx = result.root.x - previousRoot.x
      const rootDz = result.root.z - previousRoot.z
      const stepDistance = Math.hypot(rootDx, rootDz)
      if (stepDistance > 1e-12) {
        targetAlignmentTotal += ((dx * forwardX + dz * forwardZ) / targetDistance) * stepDistance
        velocityAlongBodyTotal += rootDx * forwardX + rootDz * forwardZ
        transverseFractionTotal += Math.abs(rootDx * rightX + rootDz * rightZ)
        diagnosticPathLength += stepDistance
      }
    }
  }

  return {
    totalProgress: distances[0] - distances.at(-1)!,
    lateProgressSpeed: windowProgressSpeed(distances, 450, 630),
    maximumChainError,
    perturbationExpired: policy.getTelemetry().activePerturbation === null,
    prefixHash: prefixHash.digest('hex'),
    traceHash: traceHash.digest('hex'),
    diagnostics: {
      meanTargetAlignment: round(targetAlignmentTotal / Math.max(1e-9, diagnosticPathLength)),
      meanVelocityAlongBodyAxis: round(velocityAlongBodyTotal / Math.max(1e-9, diagnosticPathLength)),
      meanAbsoluteTransverseFraction: round(transverseFractionTotal / Math.max(1e-9, diagnosticPathLength)),
      diagnosticOnly: true,
    },
  }
}

type RecoveryRollout = {
  totalProgress: number
  lateProgressSpeed: number
  maximumChainError: number
  perturbationExpired: boolean
  prefixHash: string
  traceHash: string
  diagnostics: {
    meanTargetAlignment: number
    meanVelocityAlongBodyAxis: number
    meanAbsoluteTransverseFraction: number
    diagnosticOnly: true
  }
}

function assertFiniteTelemetry(telemetry: LocomotionPolicyTelemetry) {
  for (const key of TELEMETRY_VECTOR_KEYS) {
    const values = telemetry[key]
    assert.equal(values.length, SEGMENT_COUNT, `telemetry.${key} length changed`)
    assert.ok(values.every(Number.isFinite), `telemetry.${key} contains a non-finite value`)
  }
  if (telemetry.activePerturbation) {
    assert.ok(
      Number.isFinite(telemetry.activePerturbation.remainingSeconds),
      'telemetry perturbation time is non-finite',
    )
  }
}

function asymmetricFixture() {
  const bends = Array.from({ length: SEGMENT_COUNT }, (_, segment) => (segment - 7.5) / 10)
  const bendVelocities = Array.from({ length: SEGMENT_COUNT }, (_, segment) => (7.5 - segment) / 8)
  const contactLoads = Array.from({ length: SEGMENT_COUNT }, (_, segment) => segment / 15)
  const slipSpeeds = Array.from({ length: SEGMENT_COUNT }, (_, segment) => segment / 10)
  const obstacleForward = Array.from({ length: SEGMENT_COUNT }, (_, segment) => (segment - 7.5) / 7.5)
  const obstacleRight = Array.from({ length: SEGMENT_COUNT }, (_, segment) => 1 - (segment * 2) / 15)
  return {
    bends,
    bendVelocities,
    contactLoads,
    slipSpeeds,
    obstacleForward,
    obstacleRight,
    sensors: {
      ...baseSensors(),
      contactLoads,
      slipSpeeds,
      obstacleForward,
      obstacleRight,
    } satisfies LocomotionSensors,
  }
}

function baseSensors(): LocomotionSensors {
  return {
    targetForward: 0.7,
    targetRight: -0.25,
    targetDistance: 0.8,
    forwardSpeed: 0,
    angularSpeed: 0,
    terrainFriction: 0.9,
    urgency: 0.8,
    contactLoads: new Float64Array(SEGMENT_COUNT).fill(1),
    slipSpeeds: new Float64Array(SEGMENT_COUNT),
    obstacleForward: new Float64Array(SEGMENT_COUNT),
    obstacleRight: new Float64Array(SEGMENT_COUNT),
  }
}

function sensorsFor(
  segments: readonly SegmentSnapshot[],
  plant: WormLocomotionPlant,
  target: { x: number; z: number },
  sensedFriction: number,
): LocomotionSensors {
  const root = center(segments)
  const heading = bodyHeading(segments)
  const forwardX = Math.cos(heading)
  const forwardZ = Math.sin(heading)
  const rightX = -forwardZ
  const rightZ = forwardX
  const dx = target.x - root.x
  const dz = target.z - root.z
  const distance = Math.max(Math.hypot(dx, dz), 1e-9)
  return {
    targetForward: (dx * forwardX + dz * forwardZ) / distance,
    targetRight: (dx * rightX + dz * rightZ) / distance,
    targetDistance: Math.min(distance / 5, 1.5),
    forwardSpeed: plant.forwardSpeed,
    angularSpeed: plant.angularSpeed,
    terrainFriction: sensedFriction,
    urgency: 0.85,
    contactLoads: plant.contactLoads,
    slipSpeeds: plant.slipSpeeds,
    obstacleForward: plant.obstacleForward,
    obstacleRight: plant.obstacleRight,
  }
}

function createSegments(
  model: LocomotionPolicyArtifact,
  centerX = 0,
  centerZ = 0,
  heading = 0,
): SegmentSnapshot[] {
  const forwardX = Math.cos(heading)
  const forwardZ = Math.sin(heading)
  return Array.from({ length: SEGMENT_COUNT }, (_, segment) => {
    const axial = (segment - (SEGMENT_COUNT - 1) * 0.5) * model.plant.spacing
    return {
      x: centerX + forwardX * axial,
      y: model.plant.baseGroundClearance,
      z: centerZ + forwardZ * axial,
      vx: 0,
      vy: 0,
      vz: 0,
      pitch: 0,
      yaw: heading,
    }
  })
}

function flatField(friction: number): TerrainField {
  return {
    width: 100,
    depth: 100,
    minimumHeight: 0,
    maximumHeight: 0,
    gridResolution: TERRAIN_GRID_RESOLUTION,
    heightAtGridVertex: () => 0,
    waypoints: [],
    sample: () => ({ height: 0, friction, normal: [0, 1, 0], surface: 'sand' }),
  }
}

function traceCheckpoint(
  step: number,
  root: { x: number; z: number; heading: number },
  segments: readonly SegmentSnapshot[],
  action: PolicyAction,
) {
  return {
    step,
    root: [round(root.x), round(root.z), round(root.heading)],
    segments: segments.map(segment => [round(segment.x), round(segment.z)]),
    commands: Array.from({ length: SEGMENT_COUNT }, (_, segment) => round(action[segment * 2] ?? 0)),
  }
}

function distanceToTarget(segments: readonly SegmentSnapshot[], target: { x: number; z: number }) {
  const root = center(segments)
  return Math.hypot(target.x - root.x, target.z - root.z)
}

function windowProgressSpeed(distances: readonly number[], start: number, end: number) {
  return ((distances[start] ?? 0) - (distances[end] ?? 0)) / ((end - start) * timestep)
}

function center(segments: readonly SegmentSnapshot[]) {
  return segments.reduce(
    (result, segment) => ({
      x: result.x + segment.x / segments.length,
      z: result.z + segment.z / segments.length,
    }),
    { x: 0, z: 0 },
  )
}

function bodyHeading(segments: readonly SegmentSnapshot[]) {
  const tail = segments[0]
  const head = segments[segments.length - 1]
  return Math.atan2(head.z - tail.z, head.x - tail.x)
}

function chainError(segments: readonly SegmentSnapshot[], spacing: number) {
  let maximumError = 0
  for (let segment = 0; segment < segments.length - 1; segment += 1) {
    const distance = Math.hypot(
      segments[segment + 1].x - segments[segment].x,
      segments[segment + 1].z - segments[segment].z,
    )
    maximumError = Math.max(maximumError, Math.abs(distance - spacing))
  }
  return maximumError
}

function maximum(values: readonly number[]) {
  return Math.max(...values)
}

function expectClose(actual: number, expected: number, tolerance: number, message: string) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${message}: ${actual} vs ${expected}`)
}

function expectPerformance(condition: boolean, message: string) {
  if (condition) return
  if (diagnosticOnly) {
    performanceFailures.push(message)
    return
  }
  assert.ok(condition, message)
}

function round(value: number) {
  return Math.round(value * 1e8) / 1e8
}
