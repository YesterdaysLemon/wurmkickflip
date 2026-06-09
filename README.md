# Wurmkickflip

A React, Three.js, Rapier, and WebGPU-ready prototype for training an OpenWorm-inspired worm controller to ride a skateboard in a small terrarium.

The current app boots with a deterministic muscle-wave policy so the simulation is usable before RL training completes. When `public/models/wurmkickflip_policy.onnx` exists and matches `public/models/wurmkickflip_policy.meta.json`, the browser runtime attempts ONNX Runtime WebGPU first and falls back to WASM/scripted control.

## Web App

```powershell
npm install
npm run dev
```

Build verification:

```powershell
npm run build
```

## Training Scaffold

The Python workspace is under `training/` and is intended to be run with Python 3.11 through `uv`.

```powershell
cd training
uv python install 3.11
uv sync
uv run python -m wurmkickflip_rl.train --timesteps 200000
uv run python -m wurmkickflip_rl.export_policy --checkpoint runs\ppo_wurmkickflip.zip
```

The exported policy writes:

- `public/models/wurmkickflip_policy.onnx`
- `public/models/wurmkickflip_policy.meta.json`

## Architecture

- Browser scene: procedural terrarium, skateboard, trucks, wheels, and worm visual wrapper.
- Policy contract: 118 float observations to 32 dorsal/ventral muscle activations at 60 Hz.
- Runtime: `onnxruntime-web/webgpu` when available, with WASM/scripted fallback.
- Training: Gymnasium + Stable Baselines3 PPO surrogate environment, ready to be replaced by a MuJoCo model behind the same observation/action contract.
