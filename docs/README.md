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

The repository contains a working TypeScript autonomous terrarium and scripted stunt showcase. The browser renders a large square seeded terrain, food and water bowls, a skateboard, and a 16-segment worm. Hunger, thirst, and well-being accumulate over time; an urgency selector targets food, water, or the board, and the viewer reports resource interactions. Board and worm travel is bounded in both horizontal axes with no wraparound teleport.

Detached movement uses the tracked `locomotion-segmental-es-quality-robust-v1` controller: 16 locally coupled recurrent neurons, one per segment, with 39 parameters refined for 80 risk-sensitive generations at population 128 from a preserved 110-generation base model. It has no clock, gait phase, or trigonometric gait teacher. Its activations drive damped joints, and only traveling inter-segment work creates traction-scaled root propulsion. Zero, frozen, shuffled, and no-friction ablations protect that causal boundary.

The kickflip remains scripted. The separate `stunt-distilled-v2` artifact is behavior-distilled imitation used for mounted exhibition poses, not the autonomous crawl brain or proof of learned stunt physics. The older Gymnasium/PPO, ONNX, and sinusoidal CPG/morphology scaffolds remain legacy experiments. ONNX binaries stay untracked, and old local 118-input smoke exports are incompatible with the current 174-float stunt contract.

## Start Here For Agents

- Preserve the current policy contract unless you update every aligned source of truth in one change.
- Preserve skateboarding as the central task family while making creature morphology more general.
- Keep generated training outputs out of Git unless the user explicitly asks to version a model artifact.
- Run `npm run build` after code or docs that touch referenced source paths.
- Run `npm run verify:needs` after changing resources/homeostasis, `npm run verify:locomotion` after changing the evolved controller or joint-work plant, and `npm run verify:motion` after changing the integrated board/worm lifecycle.
- Use local Git commits for completed checkpoints. GitHub CLI was not installed when this project was initialized.
