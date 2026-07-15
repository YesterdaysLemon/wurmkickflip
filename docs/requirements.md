# Requirements

## Product Goal

Wurmkickflip is a browser-based evolutionary creature lab for skateboarding behavior. Python trains or evolves creatures, and the React/Three.js frontend visualizes current best runs, replay state, environment parameters, and controller/model status.

The morphology is open-ended, but skateboarding is non-negotiable. Creatures should first learn to move through a 3D environment, then find a skateboard, mount or contact it, and roll around on it.

## Core Training Model

- Python owns training, evolution, evaluation, and artifact generation.
- The browser owns visualization, interactive inspection, replay, and parameter exploration.
- Creature definitions are genome/config artifacts, not hard-coded mesh assumptions.
- Environment definitions are configurable and dynamically generated for training.
- Skateboard discovery, mounting/contact, and rolling are first-class task requirements.
- Physics constants must be first-class training parameters, similar to drone simulation domain randomization.

## Dynamic Environment Requirements

Training environments must support configurable ranges for:

- gravity vector and magnitude
- air or fluid drag
- ground friction and restitution
- terrain shape, slope, bumps, and obstacle density
- platform, ramp, or skateboard parameters when a scenario uses them
- skateboard spawn location, mass, wheel friction, and deck dimensions
- initial pose, spawn distribution, and target direction
- sensor noise and actuator latency
- reward weights and failure conditions

The system should be able to generate many environment variants from a scenario seed so evolved creatures do not overfit one static setup.

## Creature Genome Requirements

A creature genome must eventually describe:

- body parts with primitive shapes, dimensions, mass, material, and visual style
- joints between body parts with axis, limits, stiffness, damping, and motor capabilities
- sensors such as body pose, velocity, contacts, target direction, and environment probes
- controller parameters or policy metadata
- mutation ranges and constraints so generated creatures remain simulatable

The first implementation can use a simple segmented creature. Future implementations should support branching morphologies, but all viable task variants should preserve the skateboard objective.

## MVP Behavior

- Render a nonblank 3D environment in a modern browser.
- Load a creature genome JSON and render the creature from config.
- Make creature selection visibly change anatomy (primitive silhouette, proportions, and configured branching appendages), not only palette.
- Load an environment JSON and show key environment parameters in the viewer.
- Keep the existing worm/skateboard policy demo working as the first skateboard scenario.
- Keep board and creature travel continuous, bounded, and genuinely two-dimensional in the current square terrarium; do not wrap or teleport at arena edges.
- Let the current worm detach, crawl independently over terrain, seek the skateboard, and remount through a continuous transition.
- Give the terrarium deterministic food and water bowls, treat the skateboard as the well-being resource, and expose hunger, thirst, well-being, current target, and fulfillment in the viewer.
- Select resource targets by need urgency with enough hysteresis to prevent rapid target flicker.
- Drive ordinary detached movement through the tracked evolved segmental recurrent controller. Its segment actuators must cause root movement through the locomotion plant; resource coordinates must never directly translate the worm.
- Keep time, clock, cycle, authored phase, and trigonometric gait generators out of the evolved crawl-controller inputs and trainer.
- Keep the kickflip scripted and label the distilled stunt JSON accurately as an imitation-learned mounted pose prior rather than a learned physics stunt.
- Run safely without the locomotion artifact by holding detached actuator commands at zero and reporting the unavailable brain.
- Run without the mounted stunt artifact by falling back to deterministic scripted stunt control.
- Load ONNX policy artifacts only when explicitly requested and compatible with the current metadata contract.
- Degrade requested mounted-stunt JSON or ONNX policies safely to scripted stunt control; never substitute an authored crawl gait for a missing locomotion brain.

## Success Criteria

The completed browser/config milestone is:

- A typed creature genome and environment config exist in shared browser-readable JSON.
- The frontend renders a creature and environment from those configs.
- Environment config includes skateboard task parameters and randomization ranges.
- The docs describe Python evolution/training as the source of generated artifacts.
- The existing smoke policy path and verification commands still pass.

The current refinement baseline additionally includes a shared seeded heightfield with surface-dependent friction, bounded two-dimensional travel, and a three-resource homeostasis loop. Detached crawling is controlled by a 39-parameter, 16-neuron recurrent policy refined for 80 risk-sensitive generations at population 128 from a preserved 110-generation base model. Causal joint-work propulsion and zero/frozen/shuffled/no-friction ablations establish that its segment activations matter in the compact plant. The kickflip remains scripted. The next training-fidelity milestone is contact-rich articulated physics rather than treating either authored plant as transfer evidence.

Longer-term success:

- Python evolves a population of generated creatures across randomized environments.
- The best creatures can locomote, locate the skateboard, and roll on it across held-out environment seeds.
- The browser can inspect the best genome, skateboard interaction replay, and generation comparisons.
- Fitness improves across generations on held-out environment seeds.

## Non-Goals

- Do not make OpenWorm/c302/NEURON tooling a browser runtime dependency.
- Do not hard-code future work around worm-only morphology.
- Do not remove skateboarding as the central task family.
- Do not treat the browser as the high-throughput training engine.
- Do not replace typed contracts with ad hoc unvalidated JSON.
- Do not commit generated checkpoints or model binaries unless the user explicitly asks.

## Runtime Requirements

- Web app: Vite, React, TypeScript, Three.js, React Three Fiber, Drei, ONNX Runtime Web.
- Browser: Chrome or Edge recommended for WebGPU; fallback must keep the app usable elsewhere.
- Training: Python 3.11 managed with `uv`, Gymnasium, Stable Baselines3, PyTorch, ONNX.
- Future training candidates: MuJoCo, Brax, PufferLib, or custom vectorized simulators once the genome/environment contract stabilizes.

## Acceptance Checks

- `npm run check` completes.
- The local app loads at the dev server URL and shows a visible canvas plus training viewer.
- Creature and environment config files load without TypeScript errors.
- Built-in creature selections render distinct anatomy while the stunt policy keeps its fixed 16-segment action lattice.
- With the tracked locomotion JSON present, detached crawling reports the evolved model and uses segment-local recurrence without a gait clock.
- With the tracked stunt JSON present, mounted pose inference uses the 174-float observation; kickflip launch and board rotation remain scripted.
- `npm run verify:terrain` confirms deterministic height, normal, surface, friction, and waypoint sampling.
- `npm run verify:needs` confirms deterministic bowls/resources, urgency selection, restoration semantics, stable observation order, and all three need interactions.
- `npm run verify:locomotion` confirms evolved artifact provenance, deterministic reproduction, steering, segment-owned antagonistic outputs, browser-plant parity, and causal ablations.
- `npm run verify:motion` confirms deterministic two-axis travel, bounded positions, scripted kickflip landings, neural detached crawling, food/water visits across all three terrarium presets, board seeking, remounting, integrated zero/frozen/shuffled/no-traction interventions, and segment/root motion budgets.
- With the locomotion JSON absent or invalid, detached actions stay at zero and status clearly reports the unavailable brain.
- With the stunt JSON absent or invalid, the mounted action request falls back to `scripted` with a clear message.
- With an explicitly requested ONNX model present, metadata shape validation passes before inference starts.
