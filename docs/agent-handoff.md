# Agent Handoff

## Operating Rules

- Read `/docs/README.md` and `/docs/policy-contract.md` before changing policy, training, or simulation code.
- Preserve TypeScript types and Python constants together. A policy contract change is not complete until browser code, training code, metadata, and docs agree.
- Prefer small, committed checkpoints after successful verification.
- Do not commit generated caches, local virtualenvs, build output, or training run directories.
- Do not commit `public/models/wurmkickflip_policy.onnx` unless the user explicitly asks to version a trained model.
- GitHub CLI was unavailable when the repo was initialized. Local Git commits are acceptable unless `gh` is installed later.

## Required Verification

For docs-only changes:

- Inspect changed docs for consistency with current constants.
- Run `npm run verify:contract` after docs that mention policy constants.
- Run `npm run verify:parity` after docs that describe observation layout.
- Run `npm run build` when docs reference source paths or when any code changed.
- Run `git status --short` before committing.

For app/runtime changes:

- Run `npm run verify:contract` after policy contract changes.
- Run `npm run verify:parity` after changing observation construction.
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

- The browser simulation is a visual/control prototype, not a physically faithful RL environment.
- Rapier is present but not yet the source of all board/worm motion.
- The Python environment is a surrogate model and should eventually be calibrated against browser physics or replaced with a MuJoCo environment.
- OpenWorm/VirtualWorm assets are not imported yet.
- WebGPU ONNX has not been validated with a real trained policy in this repo.

## Recommended Commit Practice

Use concise commit messages that describe the outcome:

- `Document agent handoff requirements`
- `Validate ONNX policy contract`
- `Add policy parity tests`
- `Improve skateboard physics calibration`

Before every commit, confirm the tree contains only intentional changes.
