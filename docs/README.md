# Wurmkickflip Docs

This folder is the handoff point for future AI agents working on Wurmkickflip. Read these files before changing code, training policies, or altering the browser/runtime contract.

## Reading Order

1. [`requirements.md`](requirements.md) - product goal, MVP behavior, acceptance criteria, and non-goals.
2. [`architecture.md`](architecture.md) - current app, scene, policy runtime, and training scaffold.
3. [`evolution-lab.md`](evolution-lab.md) - creature evolution, dynamic environments, and skateboarding objective.
4. [`policy-contract.md`](policy-contract.md) - source-of-truth observation/action contract that must stay aligned across TypeScript, Python, and exported metadata.
5. [`replay-artifacts.md`](replay-artifacts.md) - source-of-truth replay artifact shape for skateboard task inspection.
6. [`asset-roadmap.md`](asset-roadmap.md) - current primitive assets and future authored asset needs.
7. [`agent-handoff.md`](agent-handoff.md) - rules and verification checklist for future agents.
8. [`backlog.md`](backlog.md) - prioritized task board for next implementation passes.

## Current Project Status

The repository contains a working TypeScript stunt showcase. The browser renders a procedural terrarium, skateboard, and segmented worm driven at 60 Hz by a tracked 174-input distilled neural policy. A deterministic fixed-step plant turns its bend, co-contraction, and kick signals into repeatable cruise, kickflip, landing, victory-wiggle, and free-flop behavior.

The default `neural-js` artifact is behavior-distilled imitation from a state-aware stunt teacher; it is not PPO/RL and the browser plant is not high-fidelity transfer physics. The older Gymnasium/PPO, ONNX, and CPG-evolution scaffolds remain experimental paths. ONNX binaries stay untracked, and old local 118-input smoke exports are incompatible with the current 174-float contract.

## Start Here For Agents

- Preserve the current policy contract unless you update every aligned source of truth in one change.
- Preserve skateboarding as the central task family while making creature morphology more general.
- Keep generated training outputs out of Git unless the user explicitly asks to version a model artifact.
- Run `npm run build` after code or docs that touch referenced source paths.
- Use local Git commits for completed checkpoints. GitHub CLI was not installed when this project was initialized.
