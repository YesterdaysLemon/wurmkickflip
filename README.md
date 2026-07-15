# !!! DEEPLY UNSERIOUS REPO, SERIOUSLY TESTED WORM !!!

# Wurmkickflip

Wurmkickflip is a React/Three.js virtual terrarium containing a 16-segment neural worm, finite food and water bowls, and a skateboard that satisfies well-being. The worm autonomously chooses the most urgent resource, crawls there with an evolved recurrent controller, and uses scripted contact choreography to eat, drink, mount, dismount, and kickflip.

The boundary is intentional: ordinary travel is neural and contact-driven; the aerial kickflip is authored because this compact plant is not a high-fidelity skateboard simulator.

## How The Worm Moves

The tracked controller is [`public/models/wurmkickflip_locomotion_policy.json`](public/models/wurmkickflip_locomotion_policy.json). It is a clock-free chain of 16 locally coupled recurrent `tanh` neurons. Every neuron owns one anatomical segment and emits that segment's dorsal/ventral muscle pair, for 32 action channels at 60 Hz.

Its 45 evolved values are 16 initial recurrent states, 17 shared sensor weights, three recurrent-neighbor weights, and nine output weights. Inputs include resource direction and distance, measured body motion, urgency, joint state, prior command, and segment-local support, slip, and obstacle normals. There is no time, cycle, phase, sine, cosine, demonstration gait, or writable root-thrust channel.

Muscle servos reshape a free particle chain. Internal accelerations are explicitly mean-free, distance constraints are equal-and-opposite, and the root pose is measured from the resulting segment center of mass. Translation emerges from that changing shape interacting with anisotropic terrain friction and obstacle impulses. In an obstacle-free zero-friction world, active muscles conserve horizontal center of mass to numerical precision.

Verification compares the live controller with zero, frozen, and segment-deranged actions; checks steering and local contact sensitivity; and enforces the zero-friction invariant. This is a compact deterministic articulated plant, not evidence of transfer to a real soft body.

## Terrarium Life

- Hunger, thirst, and well-being decay continuously. A deterministic selector with hysteresis chooses the most urgent available resource; it chooses a goal but never prescribes a gait.
- Food and water have finite deterministic inventories. Restoration requires continued 3D contact between the live head-derived mouth point and the visible bowl contents. Empty bowls stop restoring and refill deterministically.
- The skateboard is the well-being resource. Well-being restores only while mounted.
- Trees, rocks, glass bounds, annular bowl rims, the skateboard deck, and every body segment participate in deterministic swept collision handling.
- Eating/drinking poses, mount/dismount transitions, board routing, pop, aerial rotation, and landing are scripted and are reported as such in the UI.

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
```

The fast suite covers the shared render/physics heightfield, articulated dynamics, swept contacts, mouth resources, interaction poses, policy contracts, replay integrity, and bundle budgets. The integration suite runs deterministic long terrarium lifecycles and locomotion interventions.

## Evolve The Crawl Brain

The Python 3.11 workspace is under `training/` and uses `uv`:

```powershell
cd training
uv sync --group dev
cd ..
npm run verify:locomotion
```

The published controller uses a deterministic two-stage evolution recipe across 12 randomized contact scenarios. `npm run check:repro` stages warm starts in fresh temporary directories, reruns both stages, and requires canonical output equality with the tracked artifact. See [`training/LOCOMOTION_POLICY.md`](training/LOCOMOTION_POLICY.md) for the exact commands, genome layout, objective, provenance, and ablations.

The trainer and browser both consume [`contracts/locomotion-v2.json`](contracts/locomotion-v2.json), which is the canonical 16-segment `articulated-contact-v2` contract.

## Runtime And Legacy Experiments

The current browser ships only two mounted-policy modes:

- `neural-js` loads the tracked recurrent crawl brain and tracked mounted stunt JSON.
- `scripted`, selected with `?policyBackend=scripted`, keeps neural crawling but replaces the mounted pose prior with a deterministic diagnostic fallback.

The old browser ONNX Runtime dependency and WASM payloads were retired. The Python PPO/ONNX exporter and sinusoidal morphology/CPG evolution remain offline legacy experiments for research, but their output is not loaded by the exhibit and must not be described as provenance for the tracked crawl model.

## Architecture At A Glance

- `src/scene/terrariumSimulation.ts`: pure fixed-step terrarium state machine and simulation orchestration.
- `src/scene/wormDynamics.ts`: free articulated chain, muscle forces, constraints, friction, and segment collisions.
- `src/scene/terrariumNeeds.ts`: goal selection, finite resource inventories, mouth-contact restoration, and refill.
- `src/scene/WurmkickflipScene.tsx`: Three.js rendering and browser integration.
- `src/policy/locomotionPolicy.ts`: dependency-free recurrent inference and schema migration.
- `src/replay/`: checksummed deterministic recorder/player core.
- `training/wurmkickflip_rl/`: vectorized evolution plus older experimental trainers.
- `scripts/`: contract, physics, lifecycle, performance, reproducibility, and artifact verifiers.

More detail lives in [`docs/README.md`](docs/README.md).
