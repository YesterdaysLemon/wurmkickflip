# Architecture

## System Boundary

Wurmkickflip is a Vite/React/TypeScript inspection app backed by deterministic browser simulation and an offline Python evolution workspace. The browser is responsible for interactive visualization, fixed-step playback, config inspection, and small diagnostic rollouts. Python owns population search and generated artifacts.

The present creature is a fixed 16-segment worm, but configs and UI use creature/environment concepts so future bodies need not be worm-shaped. Skateboard discovery, mounting/contact, and rolling remain the task family.

## Browser Layers

### App and configuration

`src/App.tsx` owns controls, selected creature/environment, policy status, reduced-motion behavior, replay controls, and viewer telemetry.

`src/creature/` provides strict config validation and resource-granular loading. Every built-in creature, environment, optional manifest, and generated creature reports `loaded`, `error`, or `optional-missing`; one bad sibling does not discard valid resources. The UI exposes failing paths and retry.

The current exhibit validates `fixed-wurm-articulated-v1`, which binds:

- plant `articulated-contact-v2`;
- 16 segments and 16 antagonistic actuator pairs;
- 32 scalar dorsal/ventral channels;
- the tracked recurrent JSON artifact.

Creature genomes project appearance only onto this fixed lattice. Their declared legacy `cpg`, `onnx_policy`, or `hybrid` controller metadata is displayed as provenance and is not executed.

### Pure simulation and rendering

`src/scene/terrariumSimulation.ts` owns deterministic state creation and advancement, homeostasis/lifecycle orchestration, board routing, policy sensors, snapshot conversion, decor/collider construction, and fixed-step helper math. It has no React, Fiber, or JSX dependency.

`src/scene/WurmkickflipScene.tsx` owns React Three Fiber integration, meshes/materials/lights, live frame accumulation, policy calls, replay instrumentation, and metric delivery. Keeping the state machine outside the view lets headless verifiers execute the same simulation without mounting React.

`src/scene/terrainField.ts` precomputes a seeded grid. Rendering and physics sample the exact same piecewise triangles, so visible surface height, normals, board pitch, segment ground height, and friction cannot drift apart.

### Detached locomotion

`src/policy/locomotionPolicy.ts` runs a clock-free 16-neuron recurrent network. Every neuron owns one anatomical segment and produces its dorsal/ventral pair. The 45 evolved values comprise 16 initial states, 17 input weights, three neighbor-recurrent weights, and nine output weights. Global goal/body sensors are combined with local bend, prior command, support, slip, and obstacle feedback. There is no time, gait phase, trigonometric teacher, or direct root-motion output.

`src/scene/wormLocomotion.ts` advances only damped antagonistic joint servos. `src/scene/wormDynamics.ts` maps those joints into mean-free internal shape forces on a free 16-particle chain, applies equal-and-opposite spacing constraints, per-segment anisotropic friction, and swept obstacle projection, then measures root position/velocity/heading from the resulting body. With no friction or obstacles, active muscles conserve horizontal center of mass to numerical precision.

`src/scene/terrariumCollisions.ts` prepares deterministic height-aware collision worlds. It handles glass planes and sorted circular footprints for tree/rock trunks, annular bowl rims, board deck probes, and every tapered body segment. Continuous sweeps, overlap recovery, tangent projection, and contact normals prevent tunneling while preserving useful sliding.

### Homeostasis and interactions

`src/scene/terrariumNeeds.ts` owns hunger, thirst, well-being, target hysteresis, finite inventories, deterministic refill, and 3D mouth-to-contents tests. The selector supplies a goal but never a gait. Food/water restore only during continued live-mouth contact with nonempty visible contents; well-being restores only while mounted.

`src/scene/wormInteractionAnimation.ts` provides additive head-to-tail mounting, head-first dismounting, and face-focused eating/drinking poses. A narrow approach-aligned aperture opens only in the active nearby bowl rim; all other rim samples and props stay solid. Feeding, mounting, dismounting, target recovery, board routing, pop, aerial rotation, landing, and lifecycle timing are authored behavior, not neural outputs.

### Mounted stunt

Mounted pose inference uses the tracked `stunt-distilled-v2` JSON with the stable 174-float observation and 32-channel action. Its teacher-supported input mask is enforced by zero weights and perturbation tests. The browser still scripts the kickflip launch and board trajectory, so the artifact is correctly described as an imitation-learned pose prior rather than learned physics.

`src/policy/policyRunner.ts` loads tracked JSON or the explicit `?policyBackend=scripted` mounted diagnostic. A missing stunt JSON falls back safely; a missing crawl JSON holds detached muscle channels at zero and reports unavailable. Browser ONNX Runtime and its WASM assets are retired. Historical replay backend labels and offline Python ONNX exporters remain for provenance only.

### Replay

`src/replay/` contains a versioned recorder/player independent of React. Recorder-core artifacts use strict 60 Hz frame timing, all 32 muscle channels, deterministic interpolation, defensive copies, semantic metric validation, and a canonical FNV-1a integrity digest. The digest detects ordinary corruption, not malicious forgery. App controls can finalize/export a live capture, validate/import JSON, play/pause/seek it, and return cleanly to live simulation.

## Offline Evolution

`contracts/locomotion-v2.json` is the canonical TypeScript/Python plant and sensor contract. `training/wurmkickflip_rl/articulated_locomotion.py` vectorizes the causal body model; `evolve_locomotion_policy.py` evaluates populations across 12 obstacle/friction/body/target/contact scenarios. The public controller is reproduced by a deterministic 40-generation base stage and 10-generation refinement. Warm starts are identified by SHA-256 and model version rather than filesystem path.

The older Gymnasium/PPO/ONNX and morphology-plus-sinusoidal-CPG paths remain offline experiments. They are not browser runtime dependencies or provenance for the tracked recurrent model.

## Data Flow

Detached tick:

1. Needs update finite resources and select the most urgent available goal.
2. Goal/body sensors plus segment-local proprioception/contact feed the recurrent policy.
3. Sixteen neurons emit 32 antagonistic muscle channels.
4. Joint servos and the free articulated solver reshape the body and resolve terrain/obstacle contacts.
5. The body center and local contact signals become the next observation.
6. Real mouth/deck contact may hand control to an authored feed or mount lifecycle.

Mounted tick:

1. Simulation state becomes a canonical 174-float stunt observation.
2. The mounted JSON prior or scripted diagnostic returns 32 pose channels.
3. The authored board/stunt plant advances route, pop, rotation, landing, and riding state.
4. The same state can be rendered, recorded, or sampled headlessly.

## Verification Shape

Static checks cover TypeScript, ESLint/Prettier, Ruff, Pyright, dependency audit, and bundle budgets. Focused verifiers cover configs, contracts, parity, shared terrain, collision properties, articulated invariants, finite resources, interaction continuity, replay integrity, policy evolution, and performance. The integrated motion rollout exercises repeated food/water/board cycles, collisions, remounting, scripted kickflips, and neural interventions. Playwright covers recovery/error UI, reduced motion, and replay flows; the long reproduction lane reruns both published evolution stages in an isolated workspace.

## Honest Limits

- This is a deterministic exhibition plant, not biological worm or real skateboard transfer physics.
- Full resource planning, contact manipulation, mounting, and stunts are not yet inside the evolution objective.
- Genome morphology is currently appearance-only on a fixed control lattice.
- The integrity checksum is not an authenticated signature.
- A future MuJoCo, Brax, or custom contact-rich trainer should preserve explicit versioned contracts while replacing simplified plant dynamics.
