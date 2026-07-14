# Replay Artifacts

Replay artifacts are browser- or training-generated JSON records for inspecting skateboard task behavior. They are not model checkpoints and should usually be written under ignored local output folders such as `training/runs/`.

## Source Files

Keep these aligned:

- `src/replay/types.ts`
- `src/replay/replayValidation.ts`
- `training/wurmkickflip_rl/replay_schema.py`
- `fixtures/replay-artifact.json`

Run `npm run verify:replay` after changing the replay schema or fixture.

## Artifact Identity

Current values:

- `schemaVersion = 1`
- `kind = "wurmkickflip.replay"`

Required identity fields:

- `replayId`
- `createdAt`
- `source.creatureId`
- `source.environmentId`
- `source.policyBackend`
- `source.modelVersion`

`source.policyBackend` is a closed enum: `scripted`, `neural-js`, `onnx-webgpu`, `onnx-wasm`, or `training`. The tracked browser policy records `neural-js`; unknown backend names are rejected by both validators.

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

These metrics are the bridge between offline training, browser replay inspection, and future generation comparison views.

## Frames

Each frame records:

- `time`
- board pose and velocity
- creature root pose and velocity
- `contactRatio`
- `skateboardDiscovered`
- `rollingVelocity`
- `reward`

Future recorders can add optional detail fields, but they should preserve these base fields so old replay viewers and validators remain useful.
