# !!! DEEPLY UNSERIOUS REPO, SERIOUSLY TESTED WORM !!!

# Wurmkickflip

Wurmkickflip is a React/Three.js virtual terrarium containing a 16-segment neural worm, finite food and water bowls, and a skateboard that satisfies well-being. The worm autonomously chooses the most urgent resource and uses its evolved recurrent controller to crawl there. That same controller now owns the planar approach and segment actuation while boarding the skateboard; eating/drinking contact poses, dismounting, board routing, and the aerial kickflip remain scripted.

The boundary is intentional: ordinary travel and boarding are neural/contact-driven, while an environmental deck response supplies vertical support, friction, and a measured stability gate. The aerial kickflip is authored because this compact plant is not a high-fidelity skateboard simulator. No new locomotion artifact was promoted for boarding: the tracked controller's existing target-directed recurrence is being exercised in this browser-native contact challenge.

## How The Worm Moves

The tracked controller is [`public/models/wurmkickflip_locomotion_policy.json`](public/models/wurmkickflip_locomotion_policy.json). It is a clock-free chain of 16 locally coupled recurrent `tanh` neurons. Every neuron owns one anatomical segment and emits that segment's dorsal/ventral muscle pair, for 32 action channels at 60 Hz.

Its 45 evolved values are 16 initial recurrent states, 17 shared sensor weights, three recurrent-neighbor weights, and nine output weights. Inputs include resource direction and distance, measured body motion, urgency, joint state, prior command, and segment-local support, slip, and obstacle normals. There is no time, cycle, phase, sine, cosine, demonstration gait, or writable root-thrust channel.

Muscle servos reshape a free particle chain. Internal accelerations are explicitly mean-free, distance constraints are equal-and-opposite, and the root pose is measured from the resulting segment center of mass. Translation emerges from that changing shape interacting with anisotropic terrain friction and obstacle impulses. In an obstacle-free zero-friction world, active muscles conserve horizontal center of mass to numerical precision.

The live gait microscope shows every segment's recurrent activation, effective muscle command, bend, support proxy, and lateral slip in anatomical head-to-tail order. It also exposes body-forward versus sideways speed and goal-facing alignment. Fixed-tick experiments can numb one segment, mirror anterior/posterior sensory wiring, change real plant traction, or deliver a whole-body side shove; requested and applied commands remain distinct in telemetry.

Verification compares the live controller with zero action, frozen action, and a fixed segment shuffle; checks steering and local contact sensitivity; enforces the zero-friction invariant; and measures recovery after neural, traction, and body perturbations. This is a compact deterministic articulated plant, not evidence of transfer to a real soft body. The displayed support value is the plant's friction/contact proxy, not a simulated biological normal force.

## Terrarium Life

- Hunger, thirst, and well-being decay continuously. A deterministic selector with hysteresis chooses the most urgent available resource; it chooses a goal but never prescribes a gait.
- Food and water have finite deterministic inventories. Restoration requires continued 3D contact between the live head-derived mouth point and the visible bowl contents. Empty bowls stop restoring and refill deterministically.
- The skateboard is the well-being resource. The worm begins off-board, approaches through recurrent segment commands, and reaches riding only after sustained measured deck contact. Well-being restores only while mounted.
- Trees, rocks, glass bounds, annular bowl rims, and every body segment participate in deterministic swept collision handling. The oriented skateboard deck separately projects contacting segments onto its top surface and applies deck friction without writing planar position, heading, or a target body pose.
- Eating/drinking poses, dismounting, board routing, pop, aerial rotation, and landing are scripted and are reported as such in the UI.

The separate `stunt-distilled-v2` artifact shapes mounted exhibition poses. It is behavior-distilled imitation and is not the source of detached locomotion or proof that a kickflip was learned.

## Run It

Requires Node 22.

```powershell
npm install
npm run dev
```

Useful checks:

```powershell
npm run check          # static checks, fast verifiers, build, integration rollout
npm run check:browser  # Playwright UI coverage
npm run check:repro    # exact long-form published locomotion reproduction
npm run verify:gait    # neural locality, perturbations, traction, and recovery
```

The fast suite covers the shared render/physics heightfield, articulated dynamics, swept contacts, mouth resources, interaction poses, policy contracts, replay integrity, and bundle budgets. The integration suite runs deterministic long terrarium lifecycles, autonomous boarding gates, and locomotion interventions. Browser checks also enforce one live worm rig (16 segments, 15 connectors, and one face), including high-water counts across resets and stunt transitions. The worm body no longer casts the separated articulated shadow that could resemble a second worm while elevated over the skateboard.

## Evolve The Crawl Brain

The Python 3.11 workspace is under `training/` and uses `uv`:

```powershell
cd training
uv sync --locked --group dev
cd ..
npm run verify:locomotion
```

The published head-leading controller is a retained, hash-identified result from deterministic v3 evolution plus three bounded coordinate refinements. A four-member confirmation generation challenges that retained genome with mutants and returns it exactly. The trainer evaluates 12 randomized contact scenarios plus causal and transient-robustness domains; the stricter 900-tick TypeScript gait exam remains the promotion authority. `npm run check:repro` stages the retained seed in a fresh temporary directory, reruns the confirmation recipe, and requires byte-for-byte equality with the tracked artifact. See [`training/LOCOMOTION_POLICY.md`](training/LOCOMOTION_POLICY.md) for the exact recipe, genome layout, objective, provenance, and ablations.

The trainer and browser both consume [`contracts/locomotion-v2.json`](contracts/locomotion-v2.json), which is the canonical 16-segment `articulated-contact-v2` contract.

## Runtime And Legacy Experiments

The current browser ships only two mounted-policy modes:

- `neural-js` loads the tracked recurrent crawl brain and tracked mounted stunt JSON.
- `scripted`, selected with `?policyBackend=scripted`, keeps neural ground travel/boarding but replaces the mounted pose prior with a deterministic diagnostic fallback.

The old browser ONNX Runtime dependency and WASM payloads were retired. The Python PPO/ONNX exporter and sinusoidal morphology/CPG evolution remain offline legacy experiments for research, but their output is not loaded by the exhibit and must not be described as provenance for the tracked crawl model.

## Architecture At A Glance

- `src/scene/terrariumSimulation.ts`: pure fixed-step terrarium state machine and simulation orchestration.
- `src/scene/wormDynamics.ts`: free articulated chain, muscle forces, constraints, friction, and segment collisions.
- `src/scene/skateboardContact.ts`: oriented deck support, friction, and measured boarding contact.
- `src/scene/terrariumNeeds.ts`: goal selection, finite resource inventories, mouth-contact restoration, and refill.
- `src/scene/WurmkickflipScene.tsx`: Three.js rendering, browser integration, and scene-graph integrity telemetry.
- `src/policy/locomotionPolicy.ts`: dependency-free recurrent inference, exact trace snapshots, perturbations, and schema migration.
- `src/replay/`: checksummed deterministic recorder/player core.
- `training/wurmkickflip_rl/`: vectorized evolution plus older experimental trainers.
- `scripts/`: contract, physics, lifecycle, performance, reproducibility, and artifact verifiers.

More detail lives in [`docs/README.md`](docs/README.md).
