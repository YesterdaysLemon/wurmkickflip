# Backlog

## P0 - Policy Integration And Contract Safety

No open P0 tasks. Keep contract and runtime verification green before longer training runs.

- Add typed creature genome and dynamic skateboard environment config layer. Done for first sample configs.
- Validate creature and environment config JSON before runtime use. Done in `npm run verify:configs`.
- Add automated policy contract verification for TypeScript, Python, metadata, and docs. Done in `npm run verify:contract`.
- Repair the observation contract to the complete 174-float layout and prove the tail segment plus all 32 previous actions survive. Done in `npm run verify:parity`.
- Wire scene observations through `PolicyRunner.run` and fail verification if the live scene silently bypasses the selected policy. Done in `npm run verify:runtime`.
- Track a reproducible 174-to-hidden-to-32 distilled JSON policy and validate held-out stunt signals. Done in `npm run verify:stunt-policy`.
- Train a short PPO smoke-test policy and export `wurmkickflip_policy.onnx`. Historical 118-input artifacts exist locally but are stale after the contract repair.
- Retrain and revalidate optional ONNX inference with both WebGPU and WASM against the 174-float contract.
- Add a policy output sanity check: length 32, finite values, clipped to `[-1, 1]`. Done in `npm run verify:runtime`.
- Record a deterministic scripted rollout fixture for regression checks. Done in `npm run verify:runtime`.

## P1 - Training Quality And Simulation Fidelity

- Add Python evolution scaffold that reads creature/environment configs and emits generation summaries. Done in `npm run verify:evolution`.
- Export best evolved CPG controller as a browser-loadable generated creature config. Done in `npm run verify:evolution`.
- Make Python surrogate dynamics consume creature morphology and environment config fields. Done in `npm run verify:env`.
- Add replay artifact schema for skateboard discovery/contact/rolling metrics. Done in `npm run verify:replay`.
- Improve reward shaping for true board balance and forward progress.
- Replace or supplement the authored browser stunt plant with a physically faithful skateboard environment that can train and evaluate genuine kickflip transfer.
- Add episode metrics: fall reason, average contact ratio, energy use, distance, and survival time. Done in `npm run verify:env`.
- Add a replay recorder that can save browser rollout JSON for later inspection.
- Extend evolution beyond CPG controller parameters into calibrated morphology mutation. Done for first body-scale/material/joint-scale pass.
- Calibrate or replace the Python surrogate against the intended high-fidelity physics plant; do not calibrate training claims solely to the authored showcase.
- Evaluate adding MuJoCo for offline training while preserving the same policy contract.
- Evaluate Rapier, MuJoCo, or another rigid-body backend for trainable board/worm contacts and articulated constraints.
- Add wheel/truck tuning controls for friction, stiffness, mass, and deck dimensions.
- Add a manual-control/debug mode for muscle activations.
- Add a training summary panel for model version, reward, and export timestamp.

## P2 - Visual Fidelity And OpenWorm Alignment

- Add branching morphology rendering for non-worm creatures. Done for first primitive creature pool.
- Add skateboard discovery visual markers and contact/rolling overlays.
- Add optional neuron/muscle activity overlays. Done for the default neural showcase.
- Add terrarium visual details such as substrate variation, scale references, and better glass edge highlights.
- Code-split ONNX Runtime so initial browser bundle size is smaller before a model is loaded.
- Add WebGPU capability diagnostics to the policy status panel.
- Add mobile layout verification for the training viewer.
- Add documentation links to OpenWorm/c302/Sibernetic references in a future research notes file.

## Done

- Greenfield React + TypeScript + Vite app.
- Three/R3F/Drei terrarium and stunt scene shell.
- Procedural terrarium, skateboard, trucks, wheels, and worm visual rig.
- First creature genome and adaptive skateboard environment JSON configs.
- Scripted muscle-wave fallback controller.
- Tracked behavior-distilled `neural-js` stunt policy with reproducible training and behavioral validation.
- Deterministic 60 Hz browser stunt plant with kickflip, landing, recovery, and free-flop modes.
- ONNX Runtime WebGPU/WASM policy loader path.
- Python 3.11 `uv` training scaffold with Gymnasium + Stable Baselines3.
- Initial local Git commit.

## Guardrails

- Fix any mismatch between Python observation semantics and browser observation semantics before longer training runs.
