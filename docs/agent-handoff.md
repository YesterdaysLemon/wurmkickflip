# Agent Handoff

## Operating Rules

- Read `/docs/README.md` and `/docs/policy-contract.md` before changing policy, training, or simulation code.
- Read `/docs/evolution-lab.md` before changing creature, environment, or skateboard task logic.
- Preserve TypeScript types and Python constants together. A policy contract change is not complete until browser code, training code, metadata, and docs agree.
- Preserve the skateboard objective. Creature morphology can generalize, but the task family is finding and riding a skateboard.
- Prefer small, committed checkpoints after successful verification.
- Do not commit generated caches, local virtualenvs, build output, or training run directories.
- Do not commit `public/models/wurmkickflip_policy.onnx` unless the user explicitly asks to version a trained model.
- GitHub CLI was unavailable when the repo was initialized. Local Git commits are acceptable unless `gh` is installed later.

## Required Verification

For docs-only changes:

- Inspect changed docs for consistency with current constants.
- Run `npm run verify:contract` after docs that mention policy constants.
- Run `npm run verify:parity` after docs that describe observation layout.
- Run `npm run verify:needs` after docs that describe homeostasis/resource semantics.
- Run `npm run verify:locomotion` after docs that describe the recurrent controller, evolved parameters, or joint-work plant.
- Run `npm run build` when docs reference source paths or when any code changed.
- Run `git status --short` before committing.

For app/runtime changes:

- Run `npm run verify:contract` after policy contract changes.
- Run `npm run verify:parity` after changing observation construction.
- Run `npm run verify:terrain` after changing the terrain field, environment dimensions, terrain rendering, or terrain-dependent motion.
- Run `npm run verify:needs` after changing food/water/board resources, need growth/restoration, target selection, or homeostasis observations.
- Run `npm run verify:locomotion` after changing the segmental recurrent artifact/runtime, neural sensors, joint-work plant, or locomotion evolution.
- Run `npm run verify:motion` after changing board travel, worm-root locomotion, resource visits, mount transitions, action smoothing, or segment pose updates.
- Run `npm run build`.
- Start or reuse the dev server.
- Verify the app renders a nonblank canvas.
- Verify scripted fallback works without an ONNX file.
- Verify ONNX WebGPU and WASM paths with `?policyBackend=webgpu` and `?policyBackend=wasm` when a local model artifact exists.
- Check browser console errors.

For training changes:

- Run Python syntax checks at minimum.
- Prefer `uv run python -m ...` from `training/`.
- Training progress bars are opt-in with `--progress-bar` and require Stable Baselines3 extra progress dependencies.
- Keep `training/uv.lock` updated when dependency requirements change.
- Run `npm run verify:contract` after changing training constants or export metadata.
- Run `npm run verify:parity` after changing `_observe`, `snapshotToObservation`, or fixture observation code.
- Validate ONNX export shape with `uv run python -m wurmkickflip_rl.validate_onnx` before calling a policy browser-compatible.

## Current Known Limits

- The browser simulation is a deterministic, physically inspired stunt showcase, not a physically faithful RL environment.
- The current board/worm plants are authored fixed-step state dynamics, not Rapier rigid-body or soft-body physics.
- Detached locomotion uses a genuinely evolved segmental recurrent controller, but only in the compact joint-work plant; this is not evidence of transfer to physical worm locomotion.
- The kickflip pop, aerial board rotation, landing window, and lifecycle are scripted. The separate stunt JSON is behavior-distilled imitation, not PPO/RL.
- The tracked version 2 JSON policy deliberately masks unsupported observation channels; do not restore nonzero ignored-feature weights without a teacher and verifier that use those features.
- The older sinusoidal CPG/morphology evolution and Gymnasium environment are legacy experiments, not the browser's recurrent crawl brain. They should eventually be replaced or calibrated against high-fidelity trainable physics, not treated as validated transfer from the showcase.
- The project is no longer worm-only; the existing segmented body is a starter morphology.
- The optional local ONNX smoke artifact predates the 174-input contract and must be regenerated before WebGPU/WASM claims are renewed.

## Recommended Commit Practice

Use concise commit messages that describe the outcome:

- `Document agent handoff requirements`
- `Validate ONNX policy contract`
- `Add policy parity tests`
- `Improve skateboard physics calibration`

Before every commit, confirm the tree contains only intentional changes.
