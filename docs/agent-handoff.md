# Agent Handoff

## Operating Rules

- Read [`README.md`](README.md), [`policy-contract.md`](policy-contract.md), and [`evolution-lab.md`](evolution-lab.md) before changing policies, training, or simulation ownership.
- Treat `contracts/locomotion-v2.json` as canonical. A contract change must update TypeScript, Python, artifacts, validators, and docs together.
- Preserve skateboarding as the task family, even if morphology becomes more general.
- Keep generated caches, virtual environments, build output, and training runs untracked. Promote a model only with reproducible provenance.
- Never add a hidden root-motion channel, gait clock, phase input, or trigonometric gait recipe to detached locomotion.
- Keep claims explicit: the network owns segment commands; the plant owns constraints/friction/collisions; the scene owns goals and contact/stunt choreography.

## Verification Routing

- Contracts or policy constants: `npm run verify:contract` and `npm run verify:parity`.
- Terrain mesh/physics sampling: `npm run verify:terrain`.
- Bounds, props, bowls, board, or swept contacts: `npm run verify:collisions`.
- Articulated forces, constraints, friction, or body invariants: `npm run verify:dynamics` and `npm run verify:locomotion`.
- Food, water, inventories, mouth contact, refill, or goal selection: `npm run verify:needs`.
- Mount/dismount/feed poses and ownership handoffs: `npm run verify:interactions`.
- Integrated board/worm/resource lifecycle: `npm run verify:motion`.
- Replay schema/recorder/player: `npm run verify:replay`.
- Config loading/runtime adapter: `npm run verify:configs`.
- UI or accessibility behavior: `npm run check:browser` plus an interactive browser pass.
- Any implementation change: finish with `npm run check` and `git diff --check`.
- Published locomotion artifact or recipe: also run the intentionally long `npm run check:repro`.

Python changes additionally require `npm run python:check`. Run Python commands through `uv` from `training/` and keep `training/uv.lock` aligned with dependency changes.

## Current Boundaries

- The browser simulation is deterministic and physically inspired, not a faithful biological or skateboard transfer environment.
- Detached locomotion is genuinely evolved inside a compact free-particle contact plant. Mean-free muscle forces and constraints cannot translate the center of mass without friction or obstacle impulses.
- Homeostasis chooses targets but is not a learned planner. Food/water restoration requires live 3D mouth contact with finite contents; well-being requires mounting.
- Feeding, mounting, dismounting, route selection, post-contact recovery, pop, aerial board rotation, landing, and lifecycle timing are scripted.
- The mounted stunt JSON is behavior-distilled imitation, not reinforcement-learned kickflip physics.
- Browser ONNX Runtime is retired. PPO/ONNX and sinusoidal CPG/morphology tools are offline legacy experiments only.
- Creature genomes currently project appearance onto a fixed 16-segment, 32-channel runtime lattice; their declared legacy controller metadata is not executed.

Before handing off, report the exact checks run, any intentionally skipped long lane, and any remaining ownership or fidelity limitation.
