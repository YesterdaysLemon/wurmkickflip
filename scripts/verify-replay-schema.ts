import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import {
  DOMAIN_SAMPLE_KIND,
  DOMAIN_SAMPLE_SCHEMA_VERSION,
  type DomainSample,
} from '../src/environment/seedForge'
import { computeReplayDigest } from '../src/replay/replayIntegrity'
import { LiveReplayCapture, replayEnvironmentSampleFor } from '../src/replay/liveReplayCapture'
import { replayPlayerFromJson, serializeReplayArtifact } from '../src/replay/replayBrowser'
import { cloneReplayEnvironmentSample } from '../src/replay/replayClone'
import { ReplayPlayer } from '../src/replay/replayPlayer'
import { ReplayRecorder, recordReplay } from '../src/replay/replayRecorder'
import {
  projectReplaySample,
  replayPairDrive,
  replayRecorderFrameForScene,
} from '../src/replay/replaySceneAdapter'
import {
  REPLAY_INTEGRITY_ALGORITHM,
  REPLAY_KIND,
  REPLAY_LEGACY_DOMAIN_DEFAULTS,
  REPLAY_MUSCLE_CHANNEL_COUNT,
  REPLAY_SCHEMA_VERSION,
} from '../src/replay/types'
import { validateReplayArtifact } from '../src/replay/replayValidation'

const root = resolve(import.meta.dirname, '..')
const fixturePath = resolve(root, 'fixtures/replay-artifact.json')

const fixture = JSON.parse(await readFile(fixturePath, 'utf8')) as unknown
const validation = validateReplayArtifact(fixture)
assert(
  validation.ok,
  `TypeScript replay validation failed: ${validation.ok ? '' : validation.errors.join(' ')}`,
)
const artifact = validation.value
const fixtureRecord = fixture as Record<string, unknown>
const fixtureIntegrity = fixtureRecord.integrity as { algorithm: string; digest: string }
assert.equal(artifact.source.policyBackend, 'neural-js', 'canonical fixture must use neural-js')
assert(artifact.playback, 'canonical fixture must exercise the recorder-core contract')
assert(artifact.integrity, 'canonical fixture must include verified integrity')
assert.equal(artifact.integrity.algorithm, REPLAY_INTEGRITY_ALGORITHM)
assert.equal(fixtureIntegrity.digest, computeReplayDigest(fixture))
assert.equal(
  artifact.integrity.digest,
  computeReplayDigest(artifact),
  'normalized historical artifacts must remain internally checksummed',
)
assert(validateReplayArtifact(artifact).ok, 'normalized historical artifacts must validate again')
assert(
  artifact.frames.every(frame => frame.muscleActivations?.length === REPLAY_MUSCLE_CHANNEL_COUNT),
  'recorder-core fixture must contain all 32 muscle channels in every frame',
)

assert.deepEqual(
  {
    actuatorStrength: artifact.environmentSample.actuatorStrength,
    actuatorLatencyMs: artifact.environmentSample.actuatorLatencyMs,
    sensorNoise: artifact.environmentSample.sensorNoise,
    spawnYawDegrees: artifact.environmentSample.spawnYawDegrees,
  },
  REPLAY_LEGACY_DOMAIN_DEFAULTS,
  'historical schema-v1 artifacts must receive honest nominal Seed Forge defaults',
)

// New recorder artifacts preserve every exact Seed Forge value through JSON.
const exactEnvironmentSample = {
  ...artifact.environmentSample,
  actuatorStrength: 1.375,
  actuatorLatencyMs: 37.5,
  sensorNoise: 0.0625,
  spawnYawDegrees: -17.25,
}
const exactDomainSample: DomainSample = {
  schemaVersion: DOMAIN_SAMPLE_SCHEMA_VERSION,
  kind: DOMAIN_SAMPLE_KIND,
  seed: exactEnvironmentSample.seed,
  gravityScale: exactEnvironmentSample.gravityScale,
  frictionScale: exactEnvironmentSample.frictionScale,
  dragScale: exactEnvironmentSample.dragScale,
  slopeDegrees: exactEnvironmentSample.slopeDegrees,
  roughness: exactEnvironmentSample.roughness,
  obstacleDensity: exactEnvironmentSample.obstacleDensity,
  skateboardSpawnX: exactEnvironmentSample.skateboardSpawn[0],
  skateboardSpawnZ: exactEnvironmentSample.skateboardSpawn[1],
  skateboardMass: exactEnvironmentSample.skateboardMass,
  wheelFriction: exactEnvironmentSample.wheelFriction,
  actuatorStrength: exactEnvironmentSample.actuatorStrength,
  actuatorLatencyMs: exactEnvironmentSample.actuatorLatencyMs,
  sensorNoise: exactEnvironmentSample.sensorNoise,
  spawnYawDegrees: exactEnvironmentSample.spawnYawDegrees,
}
assert.deepEqual(
  replayEnvironmentSampleFor({} as never, exactDomainSample),
  exactEnvironmentSample,
  'live capture must map the exact forged domain into replay provenance',
)
const recorded = recordReplay(
  {
    replayId: artifact.replayId,
    createdAt: artifact.createdAt,
    source: artifact.source,
    timestep: artifact.timestep,
    environmentSample: exactEnvironmentSample,
  },
  artifact.frames.map(frame => ({
    ...frame,
    muscleActivations: frame.muscleActivations!,
  })),
  artifact.taskMetrics,
)
assert.equal(
  Object.keys(recorded.environmentSample).length,
  14,
  'new replay samples must store all 14 named domain fields',
)
assert.deepEqual(
  recorded.environmentSample,
  exactEnvironmentSample,
  'recorder must preserve the exact Seed Forge sample',
)
const serializedRoundTrip = validateReplayArtifact(JSON.parse(JSON.stringify(recorded)) as unknown)
assert(serializedRoundTrip.ok, 'serialized recorder output must remain valid')
assert.deepEqual(serializedRoundTrip.value, recorded)

const callerOwnedEnvironmentSample = {
  ...exactEnvironmentSample,
  skateboardSpawn: [...exactEnvironmentSample.skateboardSpawn] as [number, number],
}
const clonedEnvironmentSample = cloneReplayEnvironmentSample(callerOwnedEnvironmentSample)
callerOwnedEnvironmentSample.skateboardSpawn[0] = 999
assert.notEqual(
  clonedEnvironmentSample.skateboardSpawn[0],
  999,
  'environment sample clones must not retain caller-owned tuples',
)
clonedEnvironmentSample.skateboardSpawn[1] = 999
assert.notEqual(
  callerOwnedEnvironmentSample.skateboardSpawn[1],
  999,
  'environment sample clones must not expose their owned tuple',
)

// Recorder copies caller-owned arrays and rejects invalid sequencing/actions.
const copiedChannels = [...artifact.frames[0].muscleActivations!]
const copyingRecorder = new ReplayRecorder({
  replayId: 'copy-safety',
  createdAt: artifact.createdAt,
  source: artifact.source,
  timestep: artifact.timestep,
  environmentSample: artifact.environmentSample,
})
copyingRecorder.record({ ...artifact.frames[0], muscleActivations: copiedChannels })
copiedChannels[0] = 0.99
const copySafeArtifact = copyingRecorder.finalize({
  ...artifact.taskMetrics,
  mountTime: artifact.timestep,
  survivalTime: artifact.timestep,
  maxRollingVelocity: 0,
})
assert.equal(copySafeArtifact.frames[0].muscleActivations[0], 0)
assert.throws(
  () =>
    copyingRecorder.record({
      ...artifact.frames[1],
      muscleActivations: artifact.frames[1].muscleActivations!,
    }),
  /already been finalized/,
)
assert.throws(
  () =>
    new ReplayRecorder({
      replayId: 'empty',
      createdAt: artifact.createdAt,
      source: artifact.source,
      timestep: artifact.timestep,
      environmentSample: artifact.environmentSample,
    }).finalize(artifact.taskMetrics),
  /without frames/,
)
const invalidChannelRecorder = new ReplayRecorder({
  replayId: 'invalid-channels',
  createdAt: artifact.createdAt,
  source: artifact.source,
  timestep: artifact.timestep,
  environmentSample: artifact.environmentSample,
})
assert.throws(
  () => invalidChannelRecorder.record({ ...artifact.frames[0], muscleActivations: [0, 0] }),
  /exactly 32 muscle channels/,
)
const invalidTimeRecorder = new ReplayRecorder({
  replayId: 'invalid-time',
  createdAt: artifact.createdAt,
  source: artifact.source,
  timestep: artifact.timestep,
  environmentSample: artifact.environmentSample,
})
invalidTimeRecorder.record({
  ...artifact.frames[0],
  time: artifact.timestep,
  muscleActivations: artifact.frames[0].muscleActivations!,
})
assert.throws(
  () =>
    invalidTimeRecorder.record({
      ...artifact.frames[1],
      time: artifact.timestep,
      muscleActivations: artifact.frames[1].muscleActivations!,
    }),
  /strictly greater/,
)

// Sampling is deterministic, interpolation is stable, and returned values are copies.
const player = new ReplayPlayer(fixture)
const normalizedPlayerArtifact = player.artifact()
assert.deepEqual(
  {
    actuatorStrength: normalizedPlayerArtifact.environmentSample.actuatorStrength,
    actuatorLatencyMs: normalizedPlayerArtifact.environmentSample.actuatorLatencyMs,
    sensorNoise: normalizedPlayerArtifact.environmentSample.sensorNoise,
    spawnYawDegrees: normalizedPlayerArtifact.environmentSample.spawnYawDegrees,
  },
  REPLAY_LEGACY_DOMAIN_DEFAULTS,
  'ReplayPlayer must expose normalized historical domain provenance',
)
assert.equal(
  normalizedPlayerArtifact.integrity?.digest,
  computeReplayDigest(normalizedPlayerArtifact),
  'ReplayPlayer must rechecksum normalized historical provenance',
)
assert(validateReplayArtifact(normalizedPlayerArtifact).ok)
const midpointTime = artifact.timestep * 0.5
const midpoint = player.sample(midpointTime)
assert.deepEqual(midpoint, player.sample(midpointTime), 'same replay time must produce the same sample')
assert.equal(midpoint.lowerFrameIndex, 0)
assert.equal(midpoint.upperFrameIndex, 1)
assert.equal(midpoint.interpolationAlpha, 0.5)
assert(Math.abs(midpoint.board.position[0] - -0.997) < 1e-12)
assert(Math.abs(midpoint.muscleActivations![0] - 0.06) < 1e-12)
midpoint.board.position[0] = 999
assert.notEqual(
  player.sample(midpointTime).board.position[0],
  999,
  'samples must not expose player-owned state',
)
assert.equal(player.seek(-10).time, 0)
assert.equal(player.advance(artifact.durationSeconds * 2).time, artifact.frames.at(-1)!.time)
assert(player.ended)
assert.equal(player.reset().time, 0)
assert.throws(() => player.sample(Number.NaN), /must be finite/)
assert.throws(() => player.frame(-1), /frame index/)

// The app-facing capture owns a zero-based grid and derives valid task metrics.
const liveCapture = new LiveReplayCapture({
  replayId: 'browser-capture',
  createdAt: artifact.createdAt,
  source: artifact.source,
  timestep: artifact.timestep,
  environmentSample: artifact.environmentSample,
})
for (const frame of artifact.frames) {
  liveCapture.record({ ...frame, time: 99, muscleActivations: frame.muscleActivations! })
}
const liveArtifact = liveCapture.finalize()
assert.deepEqual(
  liveArtifact.frames.map(frame => frame.time),
  [0, artifact.timestep, artifact.timestep * 2],
)
assert.equal(liveArtifact.frameCount, 3)
assert.equal(
  liveArtifact.frames.every(frame => frame.muscleActivations.length === 32),
  true,
)
assert.equal(liveArtifact.taskMetrics.survivalTime, liveArtifact.durationSeconds)
assert.equal(liveArtifact.taskMetrics.maxRollingVelocity, 0.64)
assert(validateReplayArtifact(liveArtifact).ok)
const delayedDiscoveryCapture = new LiveReplayCapture({
  replayId: 'delayed-discovery',
  createdAt: artifact.createdAt,
  source: artifact.source,
  timestep: artifact.timestep,
  environmentSample: artifact.environmentSample,
})
delayedDiscoveryCapture.record({
  ...artifact.frames[0],
  skateboardDiscovered: false,
  muscleActivations: artifact.frames[0].muscleActivations!,
})
delayedDiscoveryCapture.record({
  ...artifact.frames[1],
  skateboardDiscovered: true,
  muscleActivations: artifact.frames[1].muscleActivations!,
})
delayedDiscoveryCapture.record({
  ...artifact.frames[2],
  skateboardDiscovered: false,
  muscleActivations: artifact.frames[2].muscleActivations!,
})
const delayedDiscovery = delayedDiscoveryCapture.finalize()
assert.equal(delayedDiscovery.taskMetrics.skateboardDiscoveredAt, artifact.timestep)
assert.equal(delayedDiscovery.taskMetrics.firstContactAt, artifact.timestep)
assert.deepEqual(
  delayedDiscovery.frames.map(frame => frame.skateboardDiscovered),
  [false, true, true],
  'skateboard discovery must remain latched after the first observed event',
)
assert(validateReplayArtifact(delayedDiscovery).ok)
assert.throws(
  () =>
    new LiveReplayCapture({
      replayId: 'empty-browser-capture',
      createdAt: artifact.createdAt,
      source: artifact.source,
      timestep: artifact.timestep,
      environmentSample: artifact.environmentSample,
    }).finalize(),
  /Play the live terrarium/,
)

// Scene capture and playback projections cover board/root/contact/all actions.
const captureSnapshot = {
  boardX: 4,
  boardY: 0.5,
  boardZ: -2,
  boardVx: 0.6,
  boardVy: 0.1,
  boardVz: 0.8,
  boardRoll: 0.2,
  boardYaw: -0.4,
  boardPitch: 0.1,
  wormX: 2,
  wormY: 0.8,
  wormZ: -4,
  wormVx: 0.2,
  wormVz: 0.4,
  wormHeading: 0.1,
  mountBlend: 0.5,
  distanceToBoard: 0.4,
  contactRatio: 0.75,
  reward: 7,
  previousAction: artifact.frames[1].muscleActivations!,
}
const sceneFrame = replayRecorderFrameForScene(captureSnapshot, 1.35)
assert.deepEqual(sceneFrame.board.position, [4, 0.5, -2])
assert.deepEqual(sceneFrame.creatureRoot.position, [3, 0.74, -3])
assert.equal(sceneFrame.rollingVelocity, 1)
assert.equal(sceneFrame.skateboardDiscovered, true)
assert.equal(Array.from(sceneFrame.muscleActivations).length, 32)

const mountedFrame = replayRecorderFrameForScene({ ...captureSnapshot, mountBlend: 1 }, 1.35)
assert(
  Math.abs(mountedFrame.creatureRoot.position[1] - 0.68) < 1e-12,
  'mounted capture must use the visible board body center',
)
const mountedCapture = new LiveReplayCapture({
  replayId: 'mounted-browser-capture',
  createdAt: artifact.createdAt,
  source: artifact.source,
  timestep: artifact.timestep,
  environmentSample: artifact.environmentSample,
})
mountedCapture.record(mountedFrame)
const mountedPlayer = new ReplayPlayer(mountedCapture.finalize())
const mountedProjection = projectReplaySample(mountedPlayer.sample(0))
assert(
  Math.abs(mountedProjection.creatureRoot.y - 0.68) < 1e-12,
  'playback must preserve the mounted body center height',
)

const projected = projectReplaySample(player.sample(artifact.timestep))
assert.deepEqual([projected.board.x, projected.board.y, projected.board.z], artifact.frames[1].board.position)
assert.deepEqual(
  [projected.creatureRoot.x, projected.creatureRoot.y, projected.creatureRoot.z],
  artifact.frames[1].creatureRoot.position,
)
assert.equal(projected.contactRatio, artifact.frames[1].contactRatio)
Array.from(projected.action).forEach((activation, index) => {
  assert(Math.abs(activation - artifact.frames[1].muscleActivations![index]) < 1e-6)
})
assert.equal(projected.segments.length, 16)
assert.equal(replayPairDrive(player.sample(artifact.timestep)).length, 16)
assert.notEqual(
  projected.segments[0].z,
  projected.segments[1].z,
  'activations must affect reconstructed poses',
)

const stableJson = serializeReplayArtifact(fixture)
assert.equal(stableJson, serializeReplayArtifact(JSON.parse(stableJson) as unknown))
assert.deepEqual(replayPlayerFromJson(stableJson).artifact(), JSON.parse(stableJson))
assert.throws(() => replayPlayerFromJson('{broken'), /could not be parsed/)
assert.throws(
  () => replayPlayerFromJson(JSON.stringify(tamperReplay(artifact))),
  /integrity.digest does not match/,
)

// Legacy schema-v1 artifacts remain inspectable only through an explicit opt-out.
const legacy = structuredClone(artifact)
delete legacy.playback
delete legacy.integrity
assert(validateReplayArtifact(legacy).ok, 'legacy schema-v1 replay should remain structurally valid')
assert.throws(() => new ReplayPlayer(legacy), /requires a recorder-core artifact/)
assert.doesNotThrow(() => new ReplayPlayer(legacy, { requireIntegrity: false }))

// Integrity covers semantic fields and unknown extensions, independent of key order.
const reorderedSource = structuredClone(fixture) as Record<string, unknown>
const source = reorderedSource.source as Record<string, unknown>
reorderedSource.source = {
  modelVersion: source.modelVersion,
  policyBackend: source.policyBackend,
  environmentId: source.environmentId,
  creatureId: source.creatureId,
}
assert.equal(computeReplayDigest(reorderedSource), fixtureIntegrity.digest)

const tamperedReward = structuredClone(recorded)
tamperedReward.frames[1].reward += 1
expectValidationError(tamperedReward, 'replay.integrity.digest does not match the replay payload.')

for (const field of ['actuatorStrength', 'actuatorLatencyMs', 'sensorNoise', 'spawnYawDegrees'] as const) {
  const tamperedDomain = structuredClone(recorded)
  tamperedDomain.environmentSample[field] += 0.125
  expectValidationError(tamperedDomain, 'replay.integrity.digest does not match the replay payload.')
}

const partialSeedForgeSample = structuredClone(recorded) as unknown as Record<string, unknown>
const partialEnvironment = partialSeedForgeSample.environmentSample as Record<string, unknown>
delete partialEnvironment.sensorNoise
;(partialSeedForgeSample.integrity as { digest: string }).digest = computeReplayDigest(partialSeedForgeSample)
expectValidationError(
  partialSeedForgeSample,
  'replay.environmentSample must include all Seed Forge actuator/noise fields or omit all four for a historical schema-v1 artifact.',
)

for (const [field, invalidValue, expected] of [
  ['seed', -1, 'replay.environmentSample.seed must be an unsigned 32-bit integer.'],
  ['gravityScale', -1, 'replay.environmentSample.gravityScale must be at least 0.'],
  ['frictionScale', -1, 'replay.environmentSample.frictionScale must be at least 0.'],
  ['dragScale', -1, 'replay.environmentSample.dragScale must be at least 0.'],
  ['obstacleDensity', 1.01, 'replay.environmentSample.obstacleDensity must be between 0 and 1.'],
  ['actuatorStrength', 0, 'replay.environmentSample.actuatorStrength must be greater than 0.'],
  ['actuatorLatencyMs', -1, 'replay.environmentSample.actuatorLatencyMs must be at least 0.'],
  ['sensorNoise', 1.01, 'replay.environmentSample.sensorNoise must be between 0 and 1.'],
  ['spawnYawDegrees', 180.01, 'replay.environmentSample.spawnYawDegrees must be between -180 and 180.'],
] as const) {
  const invalidDomain = structuredClone(recorded) as unknown as Record<string, unknown>
  ;(invalidDomain.environmentSample as Record<string, unknown>)[field] = invalidValue
  ;(invalidDomain.integrity as { digest: string }).digest = computeReplayDigest(invalidDomain)
  expectValidationError(invalidDomain, expected)
}

const tamperedExtension = structuredClone(recorded) as unknown as Record<string, unknown>
tamperedExtension.unrecognizedExtension = { value: 1 }
expectValidationError(tamperedExtension, 'replay.integrity.digest does not match the replay payload.')

const compatibleExtension = structuredClone(recorded) as unknown as Record<string, unknown>
compatibleExtension.futureAnnotation = { label: 'preserved' }
;(compatibleExtension.integrity as { digest: string }).digest = computeReplayDigest(compatibleExtension)
const extendedPlayer = new ReplayPlayer(compatibleExtension)
const extendedCopy = extendedPlayer.artifact() as unknown as Record<string, unknown>
assert.deepEqual(extendedCopy.futureAnnotation, { label: 'preserved' })
assert(
  validateReplayArtifact(extendedCopy).ok,
  'player must preserve checksummed forward-compatible extensions',
)

const malformedDigest = structuredClone(artifact)
malformedDigest.integrity!.digest = 'NOT-A-DIGEST'
expectValidationError(malformedDigest, 'replay.integrity.digest must be 16 lowercase hexadecimal characters.')

const invalid = structuredClone(artifact)
invalid.frameCount = invalid.frames.length + 1
expectValidationError(invalid, `replay.frameCount must equal replay.frames length ${artifact.frames.length}.`)

const invalidBackend = structuredClone(artifact) as unknown as { source: { policyBackend: string } }
invalidBackend.source.policyBackend = 'unregistered-backend'
expectValidationErrorPrefix(invalidBackend, 'replay.source.policyBackend must be one of:')

const invalidDuration = structuredClone(artifact)
invalidDuration.durationSeconds += artifact.timestep
expectValidationErrorPrefix(
  invalidDuration,
  'replay.durationSeconds must equal the final frame time plus one timestep',
)

const pythonConstants = await readFile(resolve(root, 'training/wurmkickflip_rl/replay_schema.py'), 'utf8')
assert(
  pythonConstants.includes(`REPLAY_SCHEMA_VERSION = ${REPLAY_SCHEMA_VERSION}`),
  'Python schema version drifted',
)
assert(pythonConstants.includes(`REPLAY_KIND = "${REPLAY_KIND}"`), 'Python replay kind drifted')
assert(pythonConstants.includes('"neural-js"'), 'Python backend allowlist is missing neural-js')

const python = spawnSync(
  'uv',
  ['run', '--locked', 'python', '-m', 'wurmkickflip_rl.replay_schema', '../fixtures/replay-artifact.json'],
  {
    cwd: resolve(root, 'training'),
    encoding: 'utf8',
  },
)
assert.equal(python.status, 0, `${python.stdout}\n${python.stderr}`)

const pythonSeedForgeReplay = spawnSync(
  'uv',
  [
    'run',
    '--locked',
    'python',
    '-c',
    [
      'import json, sys',
      'from wurmkickflip_rl.replay_schema import validate_replay_artifact',
      'payload = json.load(sys.stdin)',
      'valid_errors = validate_replay_artifact(payload["valid"])',
      'partial_errors = validate_replay_artifact(payload["partial"])',
      'unsafe_errors = validate_replay_artifact(payload["unsafe"])',
      'expected = "replay.environmentSample must include all Seed Forge actuator/noise fields or omit all four for a historical schema-v1 artifact."',
      'unsafe_expected = "replay.environmentSample.actuatorStrength must be greater than 0."',
      'raise SystemExit(0 if not valid_errors and expected in partial_errors and unsafe_expected in unsafe_errors else 1)',
    ].join('; '),
  ],
  {
    cwd: resolve(root, 'training'),
    encoding: 'utf8',
    input: JSON.stringify({
      valid: recorded,
      partial: partialSeedForgeSample,
      unsafe: {
        ...recorded,
        environmentSample: { ...recorded.environmentSample, actuatorStrength: 0 },
      },
    }),
  },
)
assert.equal(
  pythonSeedForgeReplay.status,
  0,
  `Python replay validation drifted from the Seed Forge field contract.\n${pythonSeedForgeReplay.stdout}\n${pythonSeedForgeReplay.stderr}`,
)

const pythonRejectsUnknownBackend = spawnSync(
  'uv',
  [
    'run',
    '--locked',
    'python',
    '-c',
    [
      'import json',
      'from pathlib import Path',
      'from wurmkickflip_rl.replay_schema import validate_replay_artifact',
      'artifact = json.loads(Path("../fixtures/replay-artifact.json").read_text(encoding="utf-8"))',
      'artifact["source"]["policyBackend"] = "unregistered-backend"',
      'errors = validate_replay_artifact(artifact)',
      'raise SystemExit(0 if any(error.startswith("replay.source.policyBackend must be one of:") for error in errors) else 1)',
    ].join('; '),
  ],
  { cwd: resolve(root, 'training'), encoding: 'utf8' },
)
assert.equal(
  pythonRejectsUnknownBackend.status,
  0,
  `Python replay validation accepted an unregistered backend.\n${pythonRejectsUnknownBackend.stdout}\n${pythonRejectsUnknownBackend.stderr}`,
)

console.log('Replay schema, recorder, deterministic player, and integrity verification passed.')

function expectValidationError(value: unknown, expected: string) {
  const result = validateReplayArtifact(value)
  assert(!result.ok, `Expected replay validation to fail with: ${expected}`)
  assert(
    result.errors.includes(expected),
    `Missing validation error: ${expected}\n${result.errors.join('\n')}`,
  )
}

function expectValidationErrorPrefix(value: unknown, expectedPrefix: string) {
  const result = validateReplayArtifact(value)
  assert(!result.ok, `Expected replay validation to fail with prefix: ${expectedPrefix}`)
  assert(
    result.errors.some(error => error.startsWith(expectedPrefix)),
    `Missing validation prefix: ${expectedPrefix}\n${result.errors.join('\n')}`,
  )
}

function tamperReplay(value: typeof artifact) {
  const tampered = structuredClone(value)
  tampered.frames[0].reward += 1
  return tampered
}
