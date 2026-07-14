# !!! DEEPLY UNSERIOUS REPO MOSTLY CODEX SLOP EXPERIMENTATION !!!

# Wurmkickflip

A React and Three.js virtual terrarium where a segmented worm uses a tracked neural controller to cruise, kickflip, land, dismount, crawl, find its board, and mount again. ONNX remains available as an experimental policy path; a trainable rigid-body plant is future work rather than an unused browser dependency.

The app boots with the `stunt-distilled-v2` artifact in `public/models/wurmkickflip_stunt_policy.json` through the `neural-js` backend. It is a 174-input, one-hidden-layer tanh network behavior-distilled from a deterministic state-aware stunt teacher. Its explicit 37-feature teacher mask and zeroed unused input weights prevent unsupported position and velocity channels from creating feedback jitter. It is learned imitation behavior, not PPO/RL and not evidence of transfer from a high-fidelity physics trainer.

The viewer includes creature and terrain presets, a repeatable kickflip exhibition, Free crawl mode, live neural activity, phase telemetry, landing quality, and terrarium camera controls. The browser plant moves the board along smooth, bounded two-dimensional routes across a larger square seeded terrain with hills, mounds, microrelief, and sand/moss/clay friction regions; it steers back from the walls instead of wrapping or teleporting. Fixed-step action, pose, rotation, and render smoothing keep the learned muscle waves legible while the worm transitions through riding, dismounting, crawling, seeking, and mounting. It is intentionally a responsive showcase rather than rigid-body transfer physics.

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

The full check includes deterministic terrain sampling (`verify:terrain`) and motion/lifecycle regression coverage (`verify:motion`) in addition to policy, runtime, training, replay, and build verification.

## Training Scaffold

The Python workspace is under `training/` and is intended to be run with Python 3.11 through `uv`.

Reproduce and validate the tracked browser policy:

```powershell
cd training
uv sync
uv run python -m wurmkickflip_rl.train_stunt_policy
uv run python -m wurmkickflip_rl.validate_stunt_policy
```

The generated `public/models/wurmkickflip_stunt_policy.json` is intentionally tracked and is the default browser brain. Training folds input normalization into the exported weights and records the teacher feature mask in the artifact. `npm run verify:stunt-policy` checks its schema, exact 174-to-hidden-to-32 matrix dimensions, finite parameters, held-out stunt signals, zeroed ignored-feature columns, and invariance to large ignored-feature perturbations. See [`training/STUNT_POLICY.md`](training/STUNT_POLICY.md) for the imitation-learning and physics limitations.

The older PPO surrogate, ONNX export, and CPG evolution experiments remain available:

```powershell
cd training
uv python install 3.11
uv sync
uv run python -m wurmkickflip_rl.train --timesteps 200000
uv run python -m wurmkickflip_rl.export_policy --checkpoint runs\ppo_wurmkickflip.zip
uv run python -m wurmkickflip_rl.validate_onnx
uv run python -m wurmkickflip_rl.evolve --generations 4 --population-size 12 --out runs\evolution\summary.json
```

The optional PPO export writes:

- `public/models/wurmkickflip_policy.onnx`
- `public/models/wurmkickflip_policy.meta.json`

The ONNX file and training checkpoints are local generated artifacts and are ignored by Git. Any local `ppo-smoke-v1` ONNX file created before the 174-observation contract repair has a stale 118-float input and must be retrained and re-exported before use.

The evolution scaffold reads the browser creature/environment configs and writes a local generation summary JSON under `training/runs/`. The Python surrogate consumes creature morphology, joint, material, skateboard, terrain, and randomization fields so evolved controller/body scales affect rollout fitness before export.
To publish the current best evolved controller back into the local browser exhibit, export a generated creature config and manifest:

```powershell
cd training
uv run python -m wurmkickflip_rl.evolve --generations 8 --population-size 18 --out runs\evolution\summary.json --export-creature ..\public\configs\evolved\best-creature.json --export-manifest ..\public\configs\evolved\manifest.json
```

The browser loads `public/configs/evolved/manifest.json` when it exists and appends those generated creatures to the built-in selector. Generated evolved configs are ignored by Git until one is intentionally promoted.

Smoke-test workflow:

```powershell
cd training
uv run python -m wurmkickflip_rl.train --timesteps 4096 --out runs\ppo_smoke.zip
uv run python -m wurmkickflip_rl.export_policy --checkpoint runs\ppo_smoke.zip --version ppo-smoke-v1
uv run python -m wurmkickflip_rl.validate_onnx
```

When a newly exported 174-input ONNX artifact exists, force the optional providers with:

- `http://127.0.0.1:5173/?policyBackend=webgpu`
- `http://127.0.0.1:5173/?policyBackend=wasm`

Runtime modes:

- `neural-js` is the default tracked distilled stunt brain. Use `?policyBackend=neural` to request it explicitly.
- `scripted` is the deterministic diagnostic wave selected with `?policyBackend=scripted` or used as a safe fallback when a requested learned artifact fails.
- `ONNX + wasm` and `ONNX + webgpu` are optional legacy/retraining paths selected explicitly with query parameters. They require a current 174-input export.

## Architecture

- Browser scene: seeded square heightfield terrarium plus a deterministic fixed-step, two-dimensional stunt/crawl plant and worm/skateboard visual rig.
- Policy contract: 174 float observations to 32 dorsal/ventral muscle activations at 60 Hz.
- Runtime: tracked JSON neural policy by default, scripted diagnostic fallback, and explicit optional ONNX providers.
- Training: behavior-distilled JSON showcase policy plus separate Gymnasium/PPO and CPG-evolution surrogate experiments. None is currently a high-fidelity transfer-physics trainer.

## Documentation

Agent-facing requirements, architecture notes, policy contract details, and the prioritized backlog live in [`docs/README.md`](docs/README.md).
