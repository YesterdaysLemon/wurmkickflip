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

## Current Scenario

The current browser simulation is a Rapier-backed inspection sandbox. It creates dynamic rigid bodies for the skateboard and selected creature body parts, connects creature parts with joint constraints, drives the bodies with scripted/ONNX-compatible motor impulses, and exposes state for policy inference and UI metrics.

Terrain is procedurally generated from the selected environment seed and parameters. Visual tiles are lightweight, while the physical world uses a base collider, generated obstacle colliders, terrarium walls, and dynamic board/creature bodies. Future work should improve Rapier/MuJoCo parity without breaking the policy contract.

## Policy Runtime

`src/policy/policyRunner.ts` owns runtime selection.

Load order:

1. Fetch `public/models/wurmkickflip_policy.meta.json`.
2. Validate metadata shapes against TypeScript constants.
3. Check for `public/models/wurmkickflip_policy.onnx`.
4. Use ONNX Runtime WebGPU when `navigator.gpu` exists.
5. Use ONNX Runtime WASM when WebGPU is unavailable.
6. Fall back to scripted muscle waves if metadata/model load or inference setup fails.

For validation, append `?policyBackend=wasm` or `?policyBackend=webgpu` to the local app URL to force a specific ONNX execution provider when the model exists.

## Training Scaffold

The offline training workspace lives under `training/`.

Current files:

- `training/wurmkickflip_rl/contracts.py` mirrors browser policy constants.
- `training/wurmkickflip_rl/env.py` defines the current Gymnasium surrogate environment.
- `training/wurmkickflip_rl/train.py` trains a Stable Baselines3 PPO policy.
- `training/wurmkickflip_rl/export_policy.py` exports the policy to ONNX and writes browser metadata.
- `training/wurmkickflip_rl/evolve.py` runs a first CPG-controller evolution scaffold from the browser creature/environment configs and writes generation summary artifacts.

The surrogate environment returns episode metrics in `info`: fall reason, current and average contact ratio, energy use, distance, and survival time. Evolution summaries consume those fields so future replay writers and trainers share metric names.

Future training should expand the evolution layer to emit creature genomes, environment samples, and replay files.

## Data Flow

Current policy data flow:

1. Scene state is converted into a `SimulationSnapshot`.
2. `snapshotToObservation` converts the snapshot into a `Float32Array` of length 118.
3. `PolicyRunner.run` returns a `Float32Array` of length 32.
4. The action is interpreted as dorsal/ventral activations for 16 segments.
5. The scene updates board, creature, reward, contact, and activity readouts.

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
