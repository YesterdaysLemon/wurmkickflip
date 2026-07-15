# Replay Artifacts

Replay artifacts are browser- or training-generated JSON records for inspecting skateboard task behavior. They are not model checkpoints and should usually be written under ignored local output folders such as `training/runs/`.

## Source Files

Keep these aligned:

- `src/replay/types.ts`
- `src/replay/index.ts`
- `src/replay/replayValidation.ts`
- `src/replay/replayRecorder.ts`
- `src/replay/replayPlayer.ts`
- `src/replay/replayIntegrity.ts`
- `training/wurmkickflip_rl/replay_schema.py`
- `fixtures/replay-artifact.json`

Run `npm run verify:replay` after changing the replay schema, fixture, recorder, player, or checksum rules. It verifies the TypeScript/Python base contract, recorder round trips, deterministic interpolation, defensive copies, legacy opt-in, and corruption/tamper rejection.

## Versioning And Identity

The cross-language base identity remains:

- `schemaVersion = 1`
- `kind = "wurmkickflip.replay"`

Required identity fields are `replayId`, canonical ISO-8601 `createdAt`, and the creature, environment, policy backend, and model version under `source`.

`source.policyBackend` is a closed enum: `scripted`, `neural-js`, `onnx-webgpu`, `onnx-wasm`, or `training`. Unknown backend names are rejected by both validators.

The deterministic TypeScript recorder adds a paired extension:

- `playback.coreVersion = 1`
- `playback.interpolation = "linear-v1"`
- `playback.actuatorLayout`
- `integrity.algorithm = "fnv1a-64-v1"`
- `integrity.digest`

Early schema-v1 artifacts may omit both `playback` and `integrity`. Structural validation still accepts them, but `ReplayPlayer` requires a checksummed recorder-core artifact by default. A caller must explicitly pass `{ requireIntegrity: false }` to inspect legacy data. Supplying only one member of the extension pair is invalid.

This extension version lets playback rules evolve without pretending that the older Python base fields changed. A breaking base-field change still requires `schemaVersion` to advance in TypeScript, Python, fixtures, and docs together.

## Frames And Timing

Every frame records:

- time
- board pose and velocity
- creature-root pose and velocity
- contact ratio
- skateboard discovery state
- rolling velocity
- reward
- optional muscle activations for legacy compatibility

Recorder-core frames require `muscleActivations` and use end-exclusive timing:

- the first frame is at time `0`
- each frame lies on the replay timestep grid
- recorded times are strictly increasing, though dropped grid frames are allowed
- `durationSeconds` equals the final frame time plus one timestep

The current controller lattice has **16 antagonistic actuator pairs**, one pair per anatomical segment. Each pair has dorsal and ventral scalar activations, so a complete policy action and recorder frame contains **32 muscle channels**, ordered `[segment 0 dorsal, segment 0 ventral, segment 1 dorsal, segment 1 ventral, ...]`. Each channel is finite and normalized to `[-1, 1]`. “16 segments,” “16 actuator pairs,” and “32 muscle channels” are related but not interchangeable counts.

## Recorder API

`ReplayRecorder` is a deterministic, single-use builder. The caller supplies `createdAt` instead of letting the recorder read wall-clock time, and the recorder defensively copies poses, environment values, and muscle channels.

```ts
const recorder = new ReplayRecorder({
  replayId: 'run-0042',
  createdAt: '2026-07-15T18:00:00.000Z',
  source,
  timestep: 1 / 60,
  environmentSample,
})

recorder.record({
  board,
  creatureRoot,
  contactRatio,
  skateboardDiscovered,
  rollingVelocity,
  reward,
  muscleActivations: currentPolicyAction,
})

const artifact = recorder.finalize(taskMetrics)
```

Omitting a frame time assigns `frameIndex * timestep`. Explicit times support dropped frames but must remain increasing and on-grid. Finalization rejects empty recordings or inconsistent metrics, attaches the playback contract, computes integrity, validates the complete artifact, and prevents later recording.

The convenience `recordReplay(options, frames, metrics)` function performs the same operation for an existing frame sequence.

## Player API

`ReplayPlayer` validates once and owns defensive copies. It provides exact frame access plus cursor-based `seek`, `advance`, and `reset`, while `sample(time)` is pure and deterministic.

`linear-v1` behavior is fixed:

- positions, velocities, contact ratio, rolling velocity, reward, and muscle channels interpolate linearly
- rotations take the shortest angular path
- discrete discovery state is left-continuous and changes at a recorded frame
- times outside the recorded sample interval hold the first or final frame
- repeated sampling at the same time produces deeply equal values

Returned frames, samples, and artifacts are copies; mutating them cannot change later playback. Checksummed forward-compatible JSON extension fields are preserved when an artifact is returned from the player.

## Integrity

The digest is FNV-1a 64 over a canonical JSON representation of the complete top-level payload except `integrity` itself. Object keys are sorted, array order is retained, and unknown extension fields are covered. Any changed reward, pose, metric, muscle activation, identity field, or extension invalidates the digest.

This checksum detects accidental corruption and ordinary tampering. It is not a cryptographic signature and does not establish who created a replay. Untrusted artifacts still require authenticated transport or a separately signed manifest.

## Browser Integration

`LiveReplayCapture` starts its own frame-zero clock even when capture begins in the middle of a live rollout. It latches skateboard discovery and derives first contact, mount duration, rolling distance/velocity, contact average, and energy only from recorded fixed-step frames; contact events cannot precede discovery.

The Replay lab in `App.tsx` can start/finish a live capture, export stable validated JSON, import and verify JSON, play/pause/restart it, and return to live simulation. A rejected or tampered import never replaces the active live/replay state. Scene playback uses recorded board/root poses, contacts, reward, and all 32 channels. Schema v1 does not store every segment pose, lifecycle/homeostasis state, or bowl inventory, so the visual adapter deterministically reconstructs the body from the recorded root and muscle channels and labels that limitation in the UI.

## Task Metrics

Each replay summarizes skateboard behavior with:

- `skateboardDiscoveredAt`
- `firstContactAt`
- `mountTime`
- `rollingDistance`
- `maxRollingVelocity`
- `averageContactRatio`
- `energy`
- `survivalTime`
- `fallReason`

Recorder-core validation ensures event timestamps and accumulated durations do not exceed replay duration, discovery is not recorded after first contact, and `maxRollingVelocity` covers every recorded frame. These metrics bridge offline training, browser replay inspection, and future generation-comparison views.
