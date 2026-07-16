# Agent Handoff

## Operating Rules

- Read [`README.md`](README.md), [`policy-contract.md`](policy-contract.md), and [`evolution-lab.md`](evolution-lab.md) before changing policies, training, or simulation ownership.
- Treat `contracts/locomotion-v2.json` as canonical. A contract change must update TypeScript, Python, artifacts, validators, and docs together.
- Preserve skateboarding as the task family, even if morphology becomes more general.
- Keep generated caches, virtual environments, build output, and training runs untracked. Promote a model only with reproducible provenance.
- Treat tracked locomotion seed bytes as provenance: they are excluded from formatting, and changing one requires a reviewed artifact promotion plus the exact reproduction lane.
- Keep every implicit trainer output under `training/runs/`; writing `public/models/` must be an explicit reviewed promotion.
- Never add a hidden root-motion channel, gait clock, phase input, or trigonometric gait recipe to detached locomotion.
- Keep claims explicit: the network owns planar travel and segment commands through live mounting; the plant owns constraints, terrain/deck support, friction, collisions, and boarding qualification; the scene owns goals and authored feed/dismount/stunt choreography.

## Verification Routing

- Contracts or policy constants: `npm run verify:contract` and `npm run verify:parity`.
- Terrain mesh/physics sampling: `npm run verify:terrain`.
- Bounds, props, bowls, board, or swept contacts: `npm run verify:collisions`.
- Articulated forces, constraints, friction, body invariants, neural perturbations, or gait telemetry: `npm run verify:dynamics`, `npm run verify:locomotion`, and `npm run verify:gait` as applicable.
- Food, water, inventories, mouth contact, refill, or goal selection: `npm run verify:needs`.
- Interaction reference samples, dismount/feed poses, and ownership handoffs: `npm run verify:interactions`.
- Integrated board/worm/resource lifecycle: `npm run verify:motion`.
- Replay schema/recorder/player: `npm run verify:replay`.
- Config loading/runtime adapter: `npm run verify:configs`.
- UI or accessibility behavior: `npm run check:browser` plus an interactive browser pass.
- Any implementation change: finish with `npm run check` and `git diff --check`.
- Published locomotion artifact or recipe: also run the intentionally long `npm run check:repro`.

Python changes additionally require `npm run python:check`. Run Python commands through `uv` with its `--locked` flag from `training/`; update `training/uv.lock` explicitly only when dependencies change.

## Current Boundaries

- The browser simulation is deterministic and physically inspired, not a faithful biological or skateboard transfer environment.
- Detached locomotion is genuinely evolved inside a compact free-particle contact plant. Mean-free muscle forces and constraints cannot translate the center of mass without friction or obstacle impulses.
- Homeostasis chooses targets but is not a learned planner. Food/water restoration requires live 3D mouth contact with finite contents; well-being requires mounting.
- Neural segment commands drive planar skateboard approach and live mounting, but the unchanged locomotion artifact was not evolved against a skateboard objective. Deck support/friction and the stable-contact gate are deterministic plant behavior.
- Feeding, dismounting, route selection, post-contact recovery, pop, aerial board rotation, landing, and stunt lifecycle timing are scripted.
- The mounted stunt JSON is behavior-distilled imitation, not reinforcement-learned kickflip physics.
- The scene graph must contain one worm root, 16 segments, 15 connectors, and one face; preserve current/high-water browser telemetry across keyed resets. Keep worm body cast shadows disabled unless a replacement cannot be mistaken for a duplicate articulated rig.
- Browser ONNX Runtime is retired. PPO/ONNX and sinusoidal CPG/morphology tools are offline legacy experiments only.
- Creature genomes currently project appearance onto a fixed 16-segment, 32-channel runtime lattice; their declared legacy controller metadata is not executed.

Before handing off, report the exact checks run, any intentionally skipped long lane, and any remaining ownership or fidelity limitation.
