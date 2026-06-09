# Wurmkickflip Docs

This folder is the handoff point for future AI agents working on Wurmkickflip. Read these files before changing code, training policies, or altering the browser/runtime contract.

## Reading Order

1. [`requirements.md`](requirements.md) - product goal, MVP behavior, acceptance criteria, and non-goals.
2. [`architecture.md`](architecture.md) - current app, scene, policy runtime, and training scaffold.
3. [`policy-contract.md`](policy-contract.md) - source-of-truth observation/action contract that must stay aligned across TypeScript, Python, and exported metadata.
4. [`agent-handoff.md`](agent-handoff.md) - rules and verification checklist for future agents.
5. [`backlog.md`](backlog.md) - prioritized task board for next implementation passes.

## Current Project Status

The repository contains a working TypeScript prototype. The browser app renders a procedural terrarium with a skateboard and worm visual rig, then drives the worm with a scripted dorsal/ventral muscle-wave controller until a trained ONNX model exists.

The training scaffold under `training/` is a surrogate Gymnasium + Stable Baselines3 PPO environment. It is designed to produce an ONNX policy compatible with the browser runtime, but the current repo does not include a trained `wurmkickflip_policy.onnx` artifact.

## Start Here For Agents

- Preserve the current policy contract unless you update every aligned source of truth in one change.
- Keep generated training outputs out of Git unless the user explicitly asks to version a model artifact.
- Run `npm run build` after code or docs that touch referenced source paths.
- Use local Git commits for completed checkpoints. GitHub CLI was not installed when this project was initialized.
