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

The current browser simulation is a deterministic, fixed-step stunt plant in `src/scene/WurmkickflipScene.tsx`. It advances skateboard and 16-segment worm state at 60 Hz, converts a controller-local canonical snapshot into the shared 174-float policy observation, and decodes dorsal/ventral outputs into bend, co-contraction, and asymmetric kick signals. Those signals drive cruise, coil, pop, airborne rotation, landing evaluation, recovery, and crawl waves. Policy actions, simulation poses, rotations, and rendered segment transforms are damped at fixed rates to keep motion stable across render frame rates.

The board follows smooth waypoints across both horizontal axes. Arena-margin steering and bounded coordinates replace the earlier one-axis wraparound behavior, so neither the board nor worm teleports at a route boundary. After a successful landing the worm can dismount into an independent root pose, crawl across terrain, seek the board, and blend continuously back into a mounted pose before another stunt. Free crawl keeps the worm detached and independently mobile while the board remains optional.

The plant is physically inspired but authored for repeatable, legible exhibition behavior. It is not Rapier rigid-body dynamics, a high-fidelity skateboard simulator, or proof of policy transfer from the Python surrogate. Creature genomes are projected onto the fixed controller lattice: root primitive and dimensions change the 16 rendered nodes' silhouette and proportions, while root-level branches become visible appendages and the source mass/stance are exposed in the UI. This projection intentionally leaves the policy's 16 segment snapshots and 174-float observation unchanged, so switching genome anatomy does not invalidate the learned stunt. Environment configs influence terrain size, shape, friction, the board, and terrarium, while deterministic fixed stepping keeps the showcase reproducible. A future MuJoCo/Rapier training plant should preserve the policy and artifact contracts while replacing these simplified dynamics.

## Policy Runtime

`src/policy/policyRunner.ts` owns runtime selection.

Load order:

1. Default to `neural` unless a `policyBackend` query parameter requests another backend.
2. Fetch and strictly parse `public/models/wurmkickflip_stunt_policy.json`.
3. Run its 174-to-hidden-to-32 tanh network directly in JavaScript as `neural-js`.
4. If `?policyBackend=webgpu` or `?policyBackend=wasm` is present, validate the legacy ONNX metadata and load a current 174-input ONNX model with that provider.
5. Use `?policyBackend=scripted` for the deterministic diagnostic wave.
6. Fall back to scripted control if a requested learned artifact cannot load or produces an invalid action.

The tracked `stunt-distilled-v2` JSON model is behavior-distilled imitation, not PPO/RL. Its teacher uses 37 documented features within the 174-float interface, and training forces every unused first-layer column to exactly zero. This makes unsupported absolute position, velocity, and previous-action channels unable to feed high-frequency scene motion back into inference while preserving the full browser/training contract. The optional ignored `ppo-smoke-v1` ONNX artifact in some local workspaces predates the 174-float contract and is stale; retrain and re-export it before using `?policyBackend=wasm` or `?policyBackend=webgpu`.

## Training Scaffold

The offline training workspace lives under `training/`.

Current files:

- `training/wurmkickflip_rl/contracts.py` mirrors browser policy constants.
- `training/wurmkickflip_rl/env.py` defines the current Gymnasium surrogate environment and derives dynamics from browser-readable creature/environment JSON.
- `training/wurmkickflip_rl/train.py` trains a Stable Baselines3 PPO policy.
- `training/wurmkickflip_rl/export_policy.py` exports the policy to ONNX and writes browser metadata.
- `training/wurmkickflip_rl/evolve.py` evolves CPG controller parameters plus morphology scales from the browser creature/environment configs and writes generation summary artifacts or browser-loadable generated creatures.
- `training/wurmkickflip_rl/train_stunt_policy.py` behavior-distills the state-aware stunt teacher into the tracked JSON network.
- `training/wurmkickflip_rl/validate_stunt_policy.py` validates artifact shape, finite inference, phase signals, traveling waves, roll feedback, the exact teacher feature mask, zeroed unused weights, and ignored-feature invariance on held-out canonical states.

The Gymnasium surrogate environment returns episode metrics in `info`: fall reason, current and average contact ratio, energy use, distance, and survival time. Evolution summaries consume those fields so future replay writers and trainers share metric names. This surrogate is separate from both the distilled stunt teacher and the browser stunt plant.

The surrogate models morphology through body size, mass, material friction, body spread, joint stiffness, damping, and motor strength, and it samples terrain/skateboard/randomization fields from the environment config. It is not a Rapier clone; transfer quality should be improved by calibrating Python rollouts against browser behavior.

## Data Flow

Current policy data flow:

1. Scene state is converted into a controller-local canonical `SimulationSnapshot` so arena travel and terrain elevation do not introduce out-of-distribution absolute coordinates.
2. `snapshotToObservation` converts the snapshot into a `Float32Array` of length 174.
3. `PolicyRunner.run` returns a `Float32Array` of length 32.
4. The action is smoothed and interpreted as dorsal/ventral activations for 16 segments.
5. The scene updates board, independent worm root, segment poses, locomotion lifecycle, reward, contact, and activity readouts.

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

- `public/models/wurmkickflip_stunt_policy.json`, the reproducible distilled showcase policy.
