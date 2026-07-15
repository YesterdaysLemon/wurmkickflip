# Policy Contract

This document describes the 174-input mounted stunt interface that future agents must preserve. Detached crawling has a deliberately separate recurrent contract described below and in `training/LOCOMOTION_POLICY.md`; do not feed a gait clock into it or conflate it with this stunt observation.

## Source Files

Keep these files aligned:

- `src/policy/types.ts`
- `src/policy/simulationAdapter.ts`
- `src/policy/neuralPolicy.ts`
- `training/wurmkickflip_rl/contracts.py`
- `training/wurmkickflip_rl/train_stunt_policy.py`
- `training/wurmkickflip_rl/validate_stunt_policy.py`
- `public/models/wurmkickflip_stunt_policy.json`
- `public/models/wurmkickflip_policy.meta.json`

Any change to segment count, observation size, action size, or timestep must update every applicable contract, artifact, validator, and observation/action implementation in the same change.

## Constants

Current values:

- `SEGMENT_COUNT = 16`
- `MUSCLE_COUNT = 32`
- `OBSERVATION_SIZE = 174`
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
9. zero-fill any missing segment record or previous-action value in its fixed slot

The size is exact: 14 board/task values + (16 segments x 8 values) + 32 previous-action values = 174. Segment 16 occupies indices 134-141, and the complete previous action occupies indices 142-173. Records never shift when one is missing.

The Python environment must emit the same semantic layout before any trained policy is considered browser-compatible.

### Distilled Teacher Feature Mask

The 174-float observation remains the public runtime and training interface. The tracked `stunt-distilled-v2` showcase policy intentionally learns from only 37 teacher-supported entries inside that interface: cycle time; board roll, yaw, and contact ratio; target `x`; and pitch/yaw for each of the 16 segments. Its artifact records the exact observation indices in `training.teacherFeatureIndices`.

Training must force every first-layer weight outside that mask to exactly zero. This prevents unsupported absolute position, velocity, and previous-action values from producing feedback jitter while leaving the complete observation contract available to future policies. `npm run verify:stunt-policy` perturbs ignored features by large values and requires the output to remain unchanged.

## Action Semantics

The policy outputs a 32-float `Float32Array`.

For segment `i`:

- `action[i * 2]` is dorsal activation.
- `action[i * 2 + 1]` is ventral activation.
- Bend is currently interpreted as `(dorsal - ventral) * 0.5`, clamped to `[-1, 1]`.
- Co-contraction is interpreted as `(dorsal + ventral) * 0.5`.
- The browser stunt plant combines per-segment bend, co-contraction, and left/right asymmetry into mounted cruise and pose cues. The current kickflip launch and board rotation are scripted rather than caused by those outputs.

Future physics may convert these activations into Rapier joint motors, soft-body approximations, or MuJoCo actuator controls. Do not change the exported action shape unless there is a deliberate contract migration.

## Default Learned JSON Contract

`public/models/wurmkickflip_stunt_policy.json` is the tracked mounted exhibition policy. It must include:

- `schemaVersion = 1`
- `kind = "wurmkickflip.stuntPolicy"`
- non-empty `modelVersion`
- `inputSize = 174`
- positive `hiddenSize`
- `outputSize = 32`
- `activation = "tanh"`
- finite `hiddenWeights[hiddenSize][174]` and `hiddenBias[hiddenSize]`
- finite `outputWeights[32][hiddenSize]` and `outputBias[32]`
- training provenance: `seed`, `samples`, `epochs`, `validationMse`, `teacherAgreement`, and the exact `teacherFeatureIndices` mask

The JavaScript runtime validates this shape before inference. `npm run verify:stunt-policy` also runs held-out canonical behavior checks for coil, release, kick, airborne tuck, traveling waves, roll feedback, exact-zero ignored input columns, and ignored-feature perturbation invariance. The model is behavior-distilled imitation from a deterministic state-aware teacher; it is not PPO/RL and its passing signals are not proof of transfer in a high-fidelity physics simulator.

## Optional ONNX Metadata Contract

`public/models/wurmkickflip_policy.meta.json` must include:

- `modelVersion`
- `modelPath`
- `observationSize`
- `actionSize`
- `timestep`
- `trainingReward`
- `observationMean`
- `observationStd`

The runtime validates `observationSize` and `actionSize` before loading ONNX. ONNX is an explicit optional backend, not the default. Old local `ppo-smoke-v1` files with 118-float inputs are stale and must be retrained and re-exported for the 174-float contract.

## Evolved Locomotion Contract

`public/models/wurmkickflip_locomotion_policy.json` is the tracked detached-crawl controller. Keep it aligned with:

- `src/policy/locomotionPolicy.ts`
- `src/policy/locomotionRunner.ts`
- `src/scene/wormLocomotion.ts`
- `training/wurmkickflip_rl/evolve_locomotion_policy.py`
- `scripts/verify-locomotion-policy.ts`

The artifact has 16 recurrent `tanh` neurons, one per segment, and emits the same 32-value dorsal/ventral action shape. Seven global sensors are `targetForward`, `targetRight`, `targetDistance`, `forwardSpeed`, `angularSpeed`, `terrainFriction`, and `urgency`. Each neuron also receives its anatomical position, bend, bend velocity, previous command, recurrent state, and immediate anterior/posterior neighbor state.

The 39-number evolved genome consists of 16 initial recurrent states, 13 shared input weights, three shared recurrent weights, and seven shared output weights. It must not gain time, clock, cycle, gait-phase, or trigonometric teacher inputs without an explicit redesign. Its trainer must not contain a disguised sinusoidal CPG recipe.

The locomotion plant is part of the causal contract: target data can affect neural outputs but cannot directly alter root pose. Root thrust comes from adjacent-joint traveling work and terrain traction; steering comes from front-weighted joint shape. The artifact enumerates joint gains/limits, traction cap and inverse-traction floor, propulsion/drag gains, steering gains, and linear/angular speed clamps; the parser rejects any drift from the shared browser contract. `npm run verify:locomotion` requires the evolved controller to beat zero-action, frozen-pose, and deterministic segment-shuffle ablations, requires zero-friction displacement from rest to remain zero, and executes a Python-versus-browser checkpoint trace across positions, heading, speeds, joint states, and segment commands.

The kickflip is outside this locomotion contract. The browser scripts its pop, board rotation, landing window, and lifecycle.

### Runtime Ownership Boundary

The evolved artifact owns only the recurrent segment commands produced during `crawling` and `seeking`. The deterministic needs selector supplies a goal; it is not a learned planner. Downstream joint integration, adjacent-joint-work propulsion, shape steering, and terrain traction are derived plant dynamics. `terrariumCollisions.ts` then derives swept wall/prop contact and frictional sliding, while per-segment contact anchors derive stick-slip grip from joint speed, terrain friction, and strain. None of those downstream rules should be presented as additional neural outputs or evidence that evolution learned contact physics.

The scene authors target-aware tangent reorientation after blocked contact and all resource/stunt transitions. `wormInteractionAnimation.ts` scripts head-to-tail mounting, head-first dismounting, and face-only eating/drinking; the scene also scripts board routing, pop, aerial rotation, and landing. A food or water approach can therefore be neural while the visible contact cycle is scripted. Run `npm run verify:collisions` for the derived contact layer, `npm run verify:interactions` for authored contact choreography, and `npm run verify:motion` for their integration with the evolved controller and resource lifecycle.

## Compatibility Rules

- Do not silently change observation order.
- Do not train against a Python layout that differs from the browser layout.
- Do not accept a learned JSON artifact whose matrix dimensions or behavior checks fail.
- Keep the distilled teacher mask, exported `teacherFeatureIndices`, zeroed unused weights, and validators aligned when retraining the tracked JSON artifact.
- Do not export ONNX with input/output names that future browser code cannot discover or map.
- Keep scripted fallback working when a requested mounted-stunt JSON or ONNX artifact is absent or invalid. A missing crawl artifact must instead hold detached actuators at zero and report the unavailable brain.
- Keep the locomotion artifact, browser recurrence, Python evolution equations, causal joint-work plant, and ablation thresholds aligned.
- Do not add authored oscillator time/phase or sine/cosine gait outputs to the evolved locomotion path.
- Do not describe collision solving, stick-slip anchors, need selection, feeding/mounting choreography, board routing, or the kickflip as learned by the detached locomotion artifact.
- Add parity tests before any contract expansion.
