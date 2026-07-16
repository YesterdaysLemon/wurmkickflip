# Wurmkickflip Docs

This folder is the technical handoff for the virtual terrarium, its evolved locomotion controller, its contact-driven boarding layer, and the deliberately scripted stunt layer.

## Reading Order

1. [`requirements.md`](requirements.md) — product goal, acceptance criteria, and non-goals.
2. [`architecture.md`](architecture.md) — browser, simulator, runtime, and training boundaries.
3. [`evolution-lab.md`](evolution-lab.md) — current evolution strategy and future contact-rich research.
4. [`policy-contract.md`](policy-contract.md) — canonical mounted-stunt and detached-locomotion contracts.
5. [`replay-artifacts.md`](replay-artifacts.md) — checksummed replay schema and playback rules.
6. [`asset-roadmap.md`](asset-roadmap.md) — primitive assets and future art needs.
7. [`agent-handoff.md`](agent-handoff.md) — change rules and verification checklist.
8. [`backlog.md`](backlog.md) — remaining work and completed milestones.

## Current Status

The app renders a seeded square heightfield, a skateboard, finite food and water bowls, solid decor, and a 16-segment worm. A clock-free recurrent network owns all 32 dorsal/ventral crawl channels. Muscle forces reshape a free segment chain; anisotropic terrain friction and obstacle impulses produce motion, while the root is measured from the body rather than translated by a hidden propulsion scalar.

Hunger, thirst, and well-being select food, water, or the skateboard as the current goal. Food and water restoration requires continued 3D contact between the live mouth point and visible finite contents; well-being restores only while mounted. The recurrent controller owns planar approach and segment commands through live boarding. Deterministic deck geometry owns vertical support, friction, and the stable-contact gate; feeding poses, dismounting, board routing, and the kickflip remain scripted. The controller weights were not changed or newly promoted for this browser behavior.

The browser runtime uses tracked JSON policies only. Browser ONNX Runtime and its WASM payloads have been retired; older PPO/ONNX and sinusoidal CPG tooling remains offline legacy research. Deterministic replay capture/playback, granular config loading, reduced-motion behavior, collision properties, performance budgets, autonomous boarding interventions, and browser flows are covered by automated checks. Scene-graph telemetry records current and high-water worm-part counts so a stale/new rig overlap cannot hide during reset or stunt transitions; worm body cast shadows are disabled because the offset articulated shadow could look like a duplicate worm while riding.

## Rules For Changes

- Treat [`contracts/locomotion-v2.json`](../contracts/locomotion-v2.json) as the canonical articulated locomotion contract and update every TypeScript/Python consumer in the same change.
- Keep ownership claims precise: recurrent segment commands drive planar travel and boarding; constraints, terrain/deck support, friction, collisions, homeostasis, contact gates, board routing, and the kickflip are deterministic plant or authored behavior.
- Never add a clock, phase, sine/cosine gait recipe, or direct root-motion channel to the evolved crawl path without an explicit redesign.
- Keep generated checkpoints local unless intentionally promoting a reproducible model artifact.
- Run `npm run check` for implementation changes, `npm run check:browser` for UI changes, and `npm run check:repro` when changing the published locomotion recipe or artifact.
