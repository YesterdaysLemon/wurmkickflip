# Policy Contract

This document describes the browser/training interface that future agents must preserve.

## Source Files

Keep these files aligned:

- `src/policy/types.ts`
- `training/wurmkickflip_rl/contracts.py`
- `public/models/wurmkickflip_policy.meta.json`

Any change to segment count, observation size, action size, or timestep must update all three files and the code that constructs observations/actions.

## Constants

Current values:

- `SEGMENT_COUNT = 16`
- `MUSCLE_COUNT = 32`
- `OBSERVATION_SIZE = 118`
- `ACTION_SIZE = 32`
- `POLICY_TIMESTEP = 1 / 60`

The action size is two activations per segment:

- even index: dorsal activation
- odd index: ventral activation

Each activation is normalized to `[-1, 1]`.

## Observation Layout

The browser builds observations in `snapshotToObservation`.

Current order:

1. `time`
2. board position: `x`, `y`, `z`
3. board velocity: `vx`, `vy`, `vz`
4. board orientation: `pitch`, `roll`, `yaw`
5. `contactRatio`
6. target direction: `x`, `y`, `z`
7. 16 segment records, each with:
   - relative position to board: `x`, `y`, `z`
   - velocity: `vx`, `vy`, `vz`
   - `pitch`
   - `yaw`
8. previous action, 32 floats
9. zero padding to length 118 if needed

The Python environment must emit the same semantic layout before any trained policy is considered browser-compatible.

## Action Semantics

The policy outputs a 32-float `Float32Array`.

For segment `i`:

- `action[i * 2]` is dorsal activation.
- `action[i * 2 + 1]` is ventral activation.
- Bend is currently interpreted as `(dorsal - ventral) * 0.5`, clamped to `[-1, 1]`.

Future physics may convert these activations into Rapier joint motors, soft-body approximations, or MuJoCo actuator controls. Do not change the exported action shape unless there is a deliberate contract migration.

## Metadata Contract

`public/models/wurmkickflip_policy.meta.json` must include:

- `modelVersion`
- `modelPath`
- `observationSize`
- `actionSize`
- `timestep`
- `trainingReward`
- `observationMean`
- `observationStd`

The runtime validates `observationSize` and `actionSize` before loading ONNX. Add stricter validation only if it keeps scripted fallback behavior intact.

## Compatibility Rules

- Do not silently change observation order.
- Do not train against a Python layout that differs from the browser layout.
- Do not export ONNX with input/output names that future browser code cannot discover or map.
- Keep scripted fallback working when no ONNX model exists.
- Add parity tests before any contract expansion.
