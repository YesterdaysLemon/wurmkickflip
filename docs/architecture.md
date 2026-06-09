# Architecture

## Browser App

The browser app is a Vite + React + TypeScript application.

Primary responsibilities:

- Mount the React UI in `src/main.tsx`.
- Own application-level state and controls in `src/App.tsx`.
- Render the terrarium, skateboard, worm visual rig, and simulation loop in `src/scene/WurmkickflipScene.tsx`.
- Display policy status, rollout metrics, and muscle activity in the right-side training viewer.

The current scene is procedural. It uses code-native geometry for the terrarium, deck, trucks, wheels, and worm segments. OpenWorm/VirtualWorm visual import is a future enhancement, not a dependency of the current render path.

## Simulation Shape

The current browser simulation is intentionally lightweight. It is not a full rigid-body training environment yet. It advances a local state model that keeps the board moving, computes worm segment poses from muscle activations, and exposes enough state for policy inference and UI metrics.

Rapier is already present in the app to support future physical bodies, colliders, contacts, and constraints. Future work should move toward richer Rapier/MuJoCo parity without breaking the policy contract.

## Policy Runtime

`src/policy/policyRunner.ts` owns runtime selection.

Load order:

1. Fetch `public/models/wurmkickflip_policy.meta.json`.
2. Validate metadata shapes against TypeScript constants.
3. Check for `public/models/wurmkickflip_policy.onnx`.
4. Use ONNX Runtime WebGPU when `navigator.gpu` exists.
5. Use ONNX Runtime WASM when WebGPU is unavailable.
6. Fall back to scripted muscle waves if metadata/model load or inference setup fails.

The scripted fallback is implemented in `src/policy/scriptedPolicy.ts`. It exists so the scene is always inspectable before training succeeds.

## Training Scaffold

The offline training workspace lives under `training/`.

Primary files:

- `training/wurmkickflip_rl/contracts.py` mirrors browser policy constants.
- `training/wurmkickflip_rl/env.py` defines the current Gymnasium surrogate environment.
- `training/wurmkickflip_rl/train.py` trains a Stable Baselines3 PPO policy.
- `training/wurmkickflip_rl/export_policy.py` exports the policy to ONNX and writes browser metadata.

The current Python environment is a surrogate trainer. The longer-term direction is to replace or augment it with a MuJoCo model while preserving the same observation/action wire shape.

## Data Flow

At each browser simulation tick:

1. Scene state is converted into a `SimulationSnapshot`.
2. `snapshotToObservation` converts the snapshot into a `Float32Array` of length 118.
3. `PolicyRunner.run` returns a `Float32Array` of length 32.
4. The action is interpreted as dorsal/ventral activations for 16 worm segments.
5. The scene updates board, worm, reward, contact, and muscle activity readouts.

## Artifact Boundaries

Tracked source:

- TypeScript app and policy runtime.
- Python training scaffold.
- `policy.meta.json` bootstrap metadata.
- Markdown docs.

Generated or local-only artifacts:

- `node_modules/`
- `dist/`
- Python `__pycache__/`
- Python `.venv/`
- training run outputs under `training/runs/`
- trained ONNX files unless explicitly requested by the user.
