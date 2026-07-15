# Architecture

## Direction

The project is pivoting from a worm-specific skateboard demo into an evolutionary creature lab for skateboarding. The current worm/skateboard scene is the first scenario and should remain useful, but new code should favor generic creature, environment, controller, skateboard-task, and replay concepts.

## Browser App

The browser app is a Vite + React + TypeScript application.

Primary responsibilities:

- Mount the React UI in `src/main.tsx`.
- Own application-level state and controls in `src/App.tsx`.
- Render the current scenario in `src/scene/WurmkickflipScene.tsx`.
- Load creature/environment config artifacts from `public/configs/`.
- Display policy status, rollout metrics, environment parameters, and creature/genome metadata in the viewer.
- Let the user switch between the current creature pool and procedural environment presets.

The frontend is not intended to be the high-throughput trainer. It is the inspection surface for current best genomes, replay artifacts, and policy/model state.

## Creature And Environment Configs

The new config layer separates generated artifacts from hard-coded visuals:

- Creature genome JSON describes body parts, joints, controller hints, mutation ranges, and visual defaults.
- Environment JSON describes scenario type, world dimensions, physics constants, terrain generation, obstacles, reward weights, and randomization ranges.
- Skateboard parameters are part of the environment/task config, including spawn distribution, deck size, mass, wheel friction, discovery radius, and reward weights.
- The frontend can render a config-derived creature even before training uses the full schema.
- Python evolution/training code should emit artifacts that conform to the same config shape.

## Dynamic Environment Generation

Environment generation should follow a seed plus parameter-ranges model:

1. Scenario config defines physics constants and randomization ranges.
2. Training samples seeds and environment parameters for each rollout.
3. Training tasks ask creatures to locomote, find the skateboard, make useful contact, and roll in a target direction.
4. Evaluation uses held-out seeds to measure generalization.
5. Frontend displays the concrete environment parameters used for a replay.

Important dynamic parameters include gravity, friction, drag, restitution, slope, terrain roughness, obstacle density, skateboard spawn, skateboard mass, wheel friction, actuator latency, sensor noise, spawn pose, and reward weights.

The current showcase materializes those terrain inputs through `src/scene/terrainField.ts`. A seeded square heightfield provides a terrain-top height, normalized surface normal, friction, and sand/moss/clay surface label for every horizontal coordinate. The same samples build the vertex-colored render mesh and drive board height, pitch, traction, worm clearance, and crawl speed, so visible hills and friction regions agree with the authored dynamics.

## Current Scenario

The current browser simulation composes two deliberately separate 60 Hz controllers in `src/scene/WurmkickflipScene.tsx`:

- Detached locomotion uses `locomotion-segmental-es-quality-robust-v1`, a tracked 39-parameter recurrent neural controller. Sixteen locally coupled `tanh` neurons each own one anatomical segment and emit its dorsal/ventral actuator pair. Sensors describe target direction/distance, root speed, angular speed, terrain friction, need urgency, and local joint feedback. There is no time, clock, cycle, gait phase, or trigonometric gait teacher.
- Mounted exhibition uses the existing distilled stunt action prior for segment pose. The browser scripts the pop, aerial board rotation, landing window, and ride lifecycle. A successful kickflip is therefore not claimed as learned physics.

`src/scene/wormLocomotion.ts` is the causal crawl plant. Antagonistic activations drive damped joint angles and velocities; forward acceleration is derived from non-reciprocal work across adjacent joints and scaled by terrain traction. Steering torque comes from the front-weighted joint shape. The target never enters the plant and cannot directly translate or rotate the root. Zero actions stay stationary, frozen-pose and segment-shuffle ablations lose progress, and zero friction produces zero displacement from rest. Segment and render transforms are damped at fixed rates to keep the resulting neural articulation stable across render frame rates.

The board follows smooth waypoints across both horizontal axes. Arena-margin steering and bounded coordinates replace the earlier one-axis wraparound behavior, so neither the board nor worm teleports at a route boundary. Food and water bowls occupy deterministic terrain positions; the moving skateboard is the well-being resource. Hunger, thirst, and well-being urgencies accumulate over time. `src/scene/terrariumNeeds.ts` selects the most urgent resource with mild hysteresis, restores food/water by proximity, and restores well-being only while mounted. Selection produces a target for the recurrent controller; it does not author a gait or move the root. After a scripted stunt the worm can dismount, neurally crawl to the selected resource, eat or drink, seek the skateboard when well-being becomes urgent, and blend continuously back into a mounted pose. Free crawl excludes the skateboard target but retains food and water.

The plant is physically inspired but authored for repeatable, legible exhibition behavior. It is not Rapier rigid-body dynamics, a high-fidelity skateboard simulator, or proof of policy transfer from the Python surrogate. Creature genomes are projected onto the fixed controller lattice: root primitive and dimensions change the 16 rendered nodes' silhouette and proportions, while root-level branches become visible appendages and the source mass/stance are exposed in the UI. This projection intentionally leaves the policy's 16 segment snapshots and 174-float observation unchanged, so switching genome anatomy does not invalidate the learned stunt. Environment configs influence terrain size, shape, friction, the board, and terrarium, while deterministic fixed stepping keeps the showcase reproducible. A future MuJoCo/Rapier training plant should preserve the policy and artifact contracts while replacing these simplified dynamics.

## Policy Runtime

`src/policy/policyRunner.ts` owns runtime selection.

Load order:

1. Load and strictly parse `public/models/wurmkickflip_locomotion_policy.json` through `src/policy/locomotionRunner.ts` for every detached crawl.
2. Run its dependency-free segmental recurrent network directly in JavaScript and report `locomotion-segmental-es-quality-robust-v1` as the autonomous brain.
3. Separately default the mounted stunt action path to `neural` unless a `policyBackend` query parameter requests another backend.
4. Fetch and strictly parse `public/models/wurmkickflip_stunt_policy.json` for mounted exhibition poses.
5. If `?policyBackend=webgpu` or `?policyBackend=wasm` is present, validate the legacy ONNX metadata and load a current 174-input ONNX model with that provider.
6. Use `?policyBackend=scripted` for the deterministic diagnostic action wave.
7. Fall back to the scripted mounted action wave if the stunt artifact fails; if the locomotion artifact fails, emit zero detached actuator commands and report it unavailable rather than inventing a gait.

The tracked locomotion model has a separate compact contract documented in `training/LOCOMOTION_POLICY.md`; it is not a second consumer of the 174-float stunt observation. The tracked `stunt-distilled-v2` JSON model is behavior-distilled imitation, not PPO/RL. Its teacher uses 37 documented features within the 174-float interface, and training forces every unused first-layer column to exactly zero. This makes unsupported absolute position, velocity, and previous-action channels unable to feed high-frequency scene motion back into inference while preserving the mounted browser/training contract. The optional ignored `ppo-smoke-v1` ONNX artifact in some local workspaces predates the 174-float contract and is stale; retrain and re-export it before using `?policyBackend=wasm` or `?policyBackend=webgpu`.

## Training Scaffold

The offline training workspace lives under `training/`.

Current files:

- `training/wurmkickflip_rl/contracts.py` mirrors browser policy constants.
- `training/wurmkickflip_rl/env.py` defines the current Gymnasium surrogate environment and derives dynamics from browser-readable creature/environment JSON.
- `training/wurmkickflip_rl/train.py` trains a Stable Baselines3 PPO policy.
- `training/wurmkickflip_rl/export_policy.py` exports the policy to ONNX and writes browser metadata.
- `training/wurmkickflip_rl/evolve_locomotion_policy.py` evolves the 39-parameter clock-free segmental recurrent controller used by the browser crawl plant. The published refinement uses seed `20260719`, 80 generations, population 128, 18 elites, 420-step episodes, a risk-sensitive objective, and eight approach/friction scenarios. It warm-starts from the preserved 110-generation base artifact under `training/seeds/`.
- `training/wurmkickflip_rl/evolve.py` is the older experimental path that evolves sinusoidal CPG parameters plus morphology scales. It can still write generated creature configs, but its CPG is not the browser's tracked crawl brain.
- `training/wurmkickflip_rl/train_stunt_policy.py` behavior-distills the state-aware stunt teacher into the tracked JSON network.
- `training/wurmkickflip_rl/validate_stunt_policy.py` validates artifact shape, finite inference, phase signals, traveling waves, roll feedback, the exact teacher feature mask, zeroed unused weights, and ignored-feature invariance on held-out canonical states.

The locomotion evolution uses the same joint-work equations and controller update order as the browser crawl plant. `npm run verify:locomotion` checks byte-identical same-seed smoke exports, a Python-versus-browser trace across controller/plant checkpoints, target steering, visible body articulation, and zero/frozen/shuffled/no-friction causal ablations. It is still a compact planar plant, not contact-rich articulated physics.

The legacy Gymnasium surrogate environment returns episode metrics in `info`: fall reason, current and average contact ratio, energy use, distance, and survival time. Legacy CPG evolution summaries consume those fields so future replay writers and trainers share metric names. This surrogate is separate from the evolved browser crawl plant, the distilled stunt teacher, and the browser stunt plant.

The surrogate models morphology through body size, mass, material friction, body spread, joint stiffness, damping, and motor strength, and it samples terrain/skateboard/randomization fields from the environment config. It is not a Rapier clone; transfer quality should be improved by calibrating Python rollouts against browser behavior.

## Data Flow

Current detached-locomotion data flow:

1. The homeostasis layer updates hunger, thirst, and well-being and selects a food, water, or skateboard resource by urgency.
2. Resource direction, distance, urgency, terrain friction, root motion, joint state, and neighbor recurrent state feed the 16-neuron locomotion policy.
3. Each neuron produces one segment command, expanded into a dorsal/ventral antagonistic pair.
4. The joint-work plant advances segment angles and velocities, derives traction-scaled propulsion and shape-based steering, and updates the independent worm root.
5. Resource proximity or a completed mount restores the contacted need; the selector then chooses the next goal from the updated urgencies.

Mounted stunt data flow:

1. Scene state is converted into a controller-local canonical `SimulationSnapshot` so arena travel and terrain elevation do not introduce out-of-distribution absolute coordinates.
2. `snapshotToObservation` converts the snapshot into a `Float32Array` of length 174.
3. `PolicyRunner.run` returns a `Float32Array` of length 32.
4. The action is smoothed and interpreted as dorsal/ventral activations for 16 mounted segments.
5. The scene scripts pop, aerial board rotation, landing, and the transition back into the autonomous needs loop.

Future evolution data flow:

1. Python generates creature genome candidates and sampled environments.
2. Python evaluates each candidate on locomotion, skateboard discovery, skateboard contact, and rolling distance.
3. Python or the browser writes replay artifacts using the schema in `docs/replay-artifacts.md`.
4. Browser loads selected genome/environment/replay artifacts.
5. Browser visualizes behavior and allows parameter inspection.

## Artifact Boundaries

Tracked source:

- TypeScript app and policy runtime.
- Config schemas and sample configs.
- Python training/evolution scaffold.
- `policy.meta.json` bootstrap metadata.
- Markdown docs.

Generated or local-only artifacts:

- `node_modules/`
- `dist/`
- Python `__pycache__/`
- Python `.venv/`
- training run outputs under `training/runs/`
- evolved population outputs
- trained ONNX files unless explicitly requested by the user.

Tracked learned artifact:

- `public/models/wurmkickflip_locomotion_policy.json`, the evolved recurrent crawl controller.
- `public/models/wurmkickflip_stunt_policy.json`, the reproducible distilled showcase policy.
