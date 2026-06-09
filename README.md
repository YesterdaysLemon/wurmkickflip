# Wurmkickflip

A React, Three.js, Rapier, and WebGPU-ready prototype for training an OpenWorm-inspired worm controller to ride a skateboard in a small terrarium.

The current app boots with deterministic muscle-wave control so the physics lab stays responsive before RL training completes. When `public/models/wurmkickflip_policy.onnx` exists and matches `public/models/wurmkickflip_policy.meta.json`, ONNX Runtime can be tested explicitly with `?policyBackend=webgpu` or `?policyBackend=wasm`.

The viewer includes a selectable pool of primitive physics creatures and procedural terrain presets. The skateboard and creature parts are Rapier rigid bodies; current controllers are intentionally rough and may flop while the training loop matures.

## Web App

```powershell
npm install
npm run dev
```

Build verification:

```powershell
npm run build
```

Full verification:

```powershell
npm run check
```

## Training Scaffold

The Python workspace is under `training/` and is intended to be run with Python 3.11 through `uv`.

```powershell
cd training
uv python install 3.11
uv sync
uv run python -m wurmkickflip_rl.train --timesteps 200000
uv run python -m wurmkickflip_rl.export_policy --checkpoint runs\ppo_wurmkickflip.zip
uv run python -m wurmkickflip_rl.validate_onnx
uv run python -m wurmkickflip_rl.evolve --generations 4 --population-size 12 --out runs\evolution\summary.json
```

The exported policy writes:

- `public/models/wurmkickflip_policy.onnx`
- `public/models/wurmkickflip_policy.meta.json`

The ONNX file and training checkpoints are local generated artifacts and are ignored by Git.

The evolution scaffold reads the browser creature/environment configs and writes a local generation summary JSON under `training/runs/`.

Smoke-test workflow:

```powershell
cd training
uv run python -m wurmkickflip_rl.train --timesteps 4096 --out runs\ppo_smoke.zip
uv run python -m wurmkickflip_rl.export_policy --checkpoint runs\ppo_smoke.zip --version ppo-smoke-v1
uv run python -m wurmkickflip_rl.validate_onnx
```

When a local ONNX artifact exists, force browser providers with:

- `http://127.0.0.1:5173/?policyBackend=webgpu`
- `http://127.0.0.1:5173/?policyBackend=wasm`

Runtime modes:

- `scripted` is the default deterministic muscle-wave controller. It is fastest to boot and best for debugging scene physics, but it is not learned behavior.
- `ONNX + wasm` runs the exported neural policy on the CPU through WebAssembly. It is the most broadly compatible ONNX path, but can use more CPU and run slower.
- `ONNX + webgpu` runs the exported neural policy on the GPU. It should scale better for larger models, but browser/GPU support and shader warm-up can make it less predictable during development.

## Architecture

- Browser scene: procedural terrarium, skateboard, trucks, wheels, and worm visual wrapper.
- Policy contract: 118 float observations to 32 dorsal/ventral muscle activations at 60 Hz.
- Runtime: scripted control by default, with ONNX Runtime Web available through explicit WebGPU or WASM query parameters.
- Training: Gymnasium + Stable Baselines3 PPO surrogate environment, ready to be replaced by a MuJoCo model behind the same observation/action contract.

## Documentation

Agent-facing requirements, architecture notes, policy contract details, and the prioritized backlog live in [`docs/README.md`](docs/README.md).
