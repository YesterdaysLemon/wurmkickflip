# Backlog

## P0 - Policy Integration And Contract Safety

- Add shared observation layout tests that compare browser `snapshotToObservation` against Python `_observe` for the same fixture.
- Add metadata validation tests for `public/models/wurmkickflip_policy.meta.json`.
- Train a short PPO smoke-test policy and export `wurmkickflip_policy.onnx`.
- Validate ONNX inference in the browser with both WebGPU and WASM paths.
- Add a policy output sanity check: length 32, finite values, clipped to `[-1, 1]`.
- Record a deterministic scripted rollout fixture for regression checks.
- Fix any mismatch between Python observation semantics and browser observation semantics before longer training runs.

## P1 - Training Quality And Simulation Fidelity

- Improve reward shaping for true board balance and forward progress.
- Add episode metrics: fall reason, average contact ratio, energy use, distance, and survival time.
- Add a replay recorder that can save browser rollout JSON for later inspection.
- Calibrate Python surrogate dynamics against the browser scene.
- Evaluate adding MuJoCo for offline training while preserving the same policy contract.
- Move more board and worm motion into Rapier bodies and constraints.
- Add wheel/truck tuning controls for friction, stiffness, mass, and deck dimensions.
- Add a manual-control/debug mode for muscle activations.
- Add a training summary panel for model version, reward, and export timestamp.

## P2 - Visual Fidelity And OpenWorm Alignment

- Investigate VirtualWorm/OpenWorm anatomy asset import path.
- Add a procedural visual wrapper that better resembles C. elegans while still driven by the segment rig.
- Add optional neuron/muscle activity overlays.
- Add terrarium visual details such as substrate variation, scale references, and better glass edge highlights.
- Code-split ONNX Runtime so initial browser bundle size is smaller before a model is loaded.
- Add WebGPU capability diagnostics to the policy status panel.
- Add mobile layout verification for the training viewer.
- Add documentation links to OpenWorm/c302/Sibernetic references in a future research notes file.

## Done

- Greenfield React + TypeScript + Vite app.
- Three/R3F/Drei/Rapier scene shell.
- Procedural terrarium, skateboard, trucks, wheels, and worm visual rig.
- Scripted muscle-wave fallback controller.
- ONNX Runtime WebGPU/WASM policy loader path.
- Python 3.11 `uv` training scaffold with Gymnasium + Stable Baselines3.
- Initial local Git commit.
