# Backlog

## P0 — Contract And Regression Safety

No open P0 implementation task. Keep these invariants green before promoting another model:

- one canonical `articulated-contact-v2` contract across TypeScript and Python;
- exactly 16 segment owners and 32 dorsal/ventral channels;
- no gait clock/phase/trigonometric teacher or direct root-motion action;
- obstacle-free zero-friction horizontal center-of-mass conservation;
- reproducible public model bytes from an isolated two-stage run;
- real mouth contact and nonempty inventory required for food/water restoration;
- scripted kickflip and lifecycle layers labeled honestly.

## P1 — Learning And Physics Fidelity

- Move resource discovery, mouth contact, finite consumption, board approach, mounting, and rolling outcomes into a contact-rich training environment. Today the approach is neural but transitions are authored.
- Train a genuine skateboard-contact objective before claiming that mounting, balance, rolling, or kickflips were learned.
- Compare MuJoCo, Brax, and a custom vectorized contact plant using held-out seeds, throughput, determinism, and browser-contract parity.
- Co-evolve morphology and control without silently projecting every genome onto the fixed 16-segment lattice. Version the actuator/sensor topology when it changes.
- Add actuator delay/noise, partial sensor loss, board mass/wheel friction, restitution, and stronger obstacle-layout randomization to evaluation.
- Calibrate compact plant distributions against the chosen higher-fidelity backend rather than against authored browser choreography.
- Add generation comparison and held-out evaluation views using the existing replay/metric contracts.
- Add an explicit manual muscle debugger with per-segment commands, contacts, slip, and obstacle normals.
- Move expensive simulation or lightweight evolution into a worker only if interactive workloads outgrow the current measured frame budget.

## P2 — Presentation And Inspection

- Add optional segment force/contact vectors and neuron/recurrent-state overlays.
- Add skateboard discovery, mouth-contact, inventory/refill, and rolling markers to replay inspection.
- Add a scrubber with frame/time labels and side-by-side replay comparison.
- Improve bowl eating/drinking art, board trucks/wheels, glass highlights, scale references, and terrain material transitions.
- Add explicit mobile and keyboard-navigation visual regression coverage.
- Expand genuinely branching morphology rendering once the runtime supports non-fixed control topologies.
- Add research notes comparing the compact plant with OpenWorm/c302/Sibernetic and modern articulated simulation backends without implying biological equivalence.

## Completed Foundation

- Vite/React/TypeScript/Three.js terrarium with config-derived creatures and seeded environments.
- Resource-granular config loading, strict validation, failure preservation, retry UI, and a fixed-runtime compatibility adapter.
- Shared render/physics triangle heightfield with deterministic normals, surfaces, and friction.
- Pure headless terrarium simulation separated from React/Three rendering.
- Free 16-particle articulated body with mean-free muscle forces, equal/opposite constraints, anisotropic ground friction, height-aware swept contacts, and measured root state.
- Clock-free 45-value recurrent controller: 16 segment owners, local contact/slip/obstacle feedback, 32 antagonistic channels, deterministic two-stage evolution, and exact reproduction lane.
- Zero, frozen, segment-deranged, steering/locality, deterministic-trace, and zero-friction COM checks.
- Finite food/water inventories, deterministic refill, exact 3D mouth-to-contents restoration, skateboard well-being, urgency/hysteresis, and repeated autonomous resource cycles.
- Solid glass, trees, rocks, annular bowl rims, skateboard probes, and tapered body segments with anti-tunneling, overlap recovery, tangent motion, and collision property tests.
- Scripted head-to-tail mounting, head-first dismounting, feeding poses, board routing, kickflip/pop/landing, and an honestly labeled distilled mounted pose prior.
- Versioned checksummed replay recorder/player with strict timing, all muscle channels, deterministic interpolation, import/export UI, and tamper rejection.
- Reduced-motion simulation pause with explicit user override.
- Static analysis, Python checks, bundle budget and retired-ONNX assertion, collision/dynamics/performance verifiers, long integrated lifecycle rollout, and Playwright flows.
- Browser ONNX Runtime/WASM payloads removed; older PPO/ONNX and sinusoidal CPG/morphology paths retained only as offline legacy experiments.

## Guardrails

- Fix browser/Python semantic drift before longer training.
- Treat checksums as corruption detection, not authentication.
- Never describe deterministic homeostasis, constraints, friction, collision response, contact choreography, route planning, or stunt timing as network outputs.
- Promote generated artifacts only with stable provenance and reproduction instructions.
