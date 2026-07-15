# !!! DEEPLY UNSERIOUS REPO MOSTLY CODEX SLOP EXPERIMENTATION !!!

# Wurmkickflip

A React and Three.js virtual terrarium where a segmented worm develops hunger, thirst, and well-being, then travels between food, water, and its skateboard. Detached movement comes from a tracked, evolved recurrent neural controller whose 16 neurons each own one body segment. The kickflip remains deliberately scripted because reliable aerial board physics is outside the current plant.

The ordinary crawl brain is `locomotion-segmental-es-quality-robust-v1` in `public/models/wurmkickflip_locomotion_policy.json`. It is a clock-free chain of 16 locally coupled recurrent `tanh` neurons. The published brain is an 80-generation, population-128 risk-sensitive refinement warm-started from the preserved 110-generation base model. Its 39 evolved parameters contain recurrent state and shared sensor, neighbor, and output weights; it receives body feedback, target direction, terrain friction, and need urgency, but no time, clock, cycle, phase, sine, cosine, or demonstration gait. Each neuron emits the antagonistic actuator pair for its own segment.

The fixed-step locomotion plant turns joint commands into segment bend and joint velocity, then derives root thrust only from non-reciprocal work traveling across adjacent joints. Target coordinates never directly move the root. Causal verification shows that zero actions stay still, a frozen pose cannot substitute for traveling joint work, shuffling segment activations damages progress, and zero terrain friction prevents displacement from rest. This is still a compact authored plant, not rigid-body or soft-body transfer physics.

The terrarium includes deterministic food and water bowls plus the live skateboard as a third resource. Need urgencies accumulate over time; a deterministic urgency selector with target hysteresis chooses where to go. A substantial head-to-bowl contact starts an authored eating or drinking cycle, while well-being is restored only after the worm reaches and mounts the skateboard. The active bowl is released from anterior collision only during the contact pose, then a short release cooldown lets the worm clear it without snapping back or immediately retriggering. The neural crawl controller steers toward the selected resource; the selector and lifecycle do not prescribe a gait or translate the creature.

The older `stunt-distilled-v2` artifact remains the mounted pose prior inside the exhibition path. The browser scripts the pop, aerial rotation, landing window, and ride lifecycle. Its imitation-learning signals are not the source of detached crawling and are not evidence that a kickflip was learned in physics.

## Behavior Ownership

- **Evolved neural output:** during `crawling` and `seeking`, the 16-neuron recurrent controller turns goal sensors and proprioception into 32 segment muscle activations. The needs selector chooses the goal; it is deterministic homeostasis, not another learned policy.
- **Derived scene physics:** damped joints turn those activations into poses, adjacent-joint work creates root thrust and steering, terrain friction supplies traction, and quiet ground segments establish stick-slip anchors that release under joint motion or strain. Deterministic swept contacts constrain the worm root, every rendered segment, and board deck probes against glass bounds, trees, rocks, bowls, and the skateboard, preserving frictional tangent motion without tunneling.
- **Scripted choreography:** board routing, collision tangent reorientation, staged head-to-tail mounting, head-first dismounting, eating/drinking poses and bowl effects, and the pop/aerial flip/landing sequence are authored. These layers make resource contact and the kickflip readable; they are not outputs learned by the crawl model.

## Web App

```powershell
npm install
npm run dev
```

Build verification:

```powershell
npm run build
```

Full verification:

```powershell
npm run check
```

The full check includes deterministic terrain and homeostasis sampling (`verify:terrain`, `verify:needs`), swept collision/traction checks (`verify:collisions`), mount/feed pose continuity (`verify:interactions`), evolved-controller contract and causal ablations (`verify:locomotion`), and complete integrated motion/lifecycle coverage (`verify:motion`) in addition to policy, runtime, training, replay, and build verification.

## Evolved Locomotion

The Python workspace is under `training/` and is intended to be run with Python 3.11 through `uv`.

Reproduce the tracked crawl controller:

```powershell
cd training
uv sync
uv run python -m wurmkickflip_rl.evolve_locomotion_policy --seed 20260719 --generations 80 --population-size 128 --elite-count 18 --episode-steps 420 --model-version locomotion-segmental-es-quality-robust-v1 --warm-start seeds/wurmkickflip_locomotion_warm_start_v1.json --out ../public/models/wurmkickflip_locomotion_policy.json --summary runs/locomotion_evolution/latest-summary.json
cd ..
npm run verify:locomotion
```

`npm run verify:locomotion:published` reruns the full 80-generation refinement
and requires the tracked artifact to match its canonical JSON exactly. It is kept
out of the fast `npm run check` path because it takes about a minute on this machine.

The evolutionary search scores target progress and steering across eight approach/friction scenarios. It has no trigonometric gait teacher or CPG recipe. See [`training/LOCOMOTION_POLICY.md`](training/LOCOMOTION_POLICY.md) for the neuron, genome, plant, reproduction, and ablation details.

## Stunt And Legacy Training Scaffold

Reproduce and validate the separate mounted stunt prior:

```powershell
cd training
uv sync
uv run python -m wurmkickflip_rl.train_stunt_policy
uv run python -m wurmkickflip_rl.validate_stunt_policy
```

The generated `public/models/wurmkickflip_stunt_policy.json` is intentionally tracked as the mounted exhibition prior, not the autonomous crawl brain. Training folds input normalization into the exported weights and records the teacher feature mask in the artifact. `npm run verify:stunt-policy` checks its schema, exact 174-to-hidden-to-32 matrix dimensions, finite parameters, held-out stunt signals, zeroed ignored-feature columns, and invariance to large ignored-feature perturbations. The browser still scripts the actual pop and aerial kickflip. See [`training/STUNT_POLICY.md`](training/STUNT_POLICY.md) for its imitation-learning and physics limitations.

The older PPO surrogate, ONNX export, and morphology-plus-CPG evolution experiments remain available as legacy experiments:

```powershell
cd training
uv python install 3.11
uv sync
uv run python -m wurmkickflip_rl.train --timesteps 200000
uv run python -m wurmkickflip_rl.export_policy --checkpoint runs\ppo_wurmkickflip.zip
uv run python -m wurmkickflip_rl.validate_onnx
uv run python -m wurmkickflip_rl.evolve --generations 4 --population-size 12 --out runs\evolution\summary.json
```

The optional PPO export writes:

- `public/models/wurmkickflip_policy.onnx`
- `public/models/wurmkickflip_policy.meta.json`

The ONNX file and training checkpoints are local generated artifacts and are ignored by Git. Any local `ppo-smoke-v1` ONNX file created before the 174-observation contract repair has a stale 118-float input and must be retrained and re-exported before use.

That legacy evolution scaffold reads the browser creature/environment configs and writes a local generation summary JSON under `training/runs/`. The Python surrogate consumes creature morphology, joint, material, skateboard, terrain, and randomization fields so evolved controller/body scales affect rollout fitness before export. Its sinusoidal CPG is not the tracked recurrent locomotion model used for browser crawling.
To publish the current best evolved controller back into the local browser exhibit, export a generated creature config and manifest:

```powershell
cd training
uv run python -m wurmkickflip_rl.evolve --generations 8 --population-size 18 --out runs\evolution\summary.json --export-creature ..\public\configs\evolved\best-creature.json --export-manifest ..\public\configs\evolved\manifest.json
```

The browser loads `public/configs/evolved/manifest.json` when it exists and appends those generated creatures to the built-in selector. Generated evolved configs are ignored by Git until one is intentionally promoted.

Smoke-test workflow:

```powershell
cd training
uv run python -m wurmkickflip_rl.train --timesteps 4096 --out runs\ppo_smoke.zip
uv run python -m wurmkickflip_rl.export_policy --checkpoint runs\ppo_smoke.zip --version ppo-smoke-v1
uv run python -m wurmkickflip_rl.validate_onnx
```

When a newly exported 174-input ONNX artifact exists, force the optional providers with:

- `http://127.0.0.1:5173/?policyBackend=webgpu`
- `http://127.0.0.1:5173/?policyBackend=wasm`

Runtime modes:

- `neural-js` reports the tracked evolved locomotion brain used whenever the worm is detached. Mounted exhibition poses may also use the separate distilled stunt prior.
- `scripted` is the mounted diagnostic wave selected with `?policyBackend=scripted` or used as a safe stunt fallback when a requested stunt artifact fails. A missing locomotion artifact instead yields zero detached actuator commands and a clear unavailable status.
- `ONNX + wasm` and `ONNX + webgpu` are optional legacy/retraining paths selected explicitly with query parameters. They require a current 174-input export.

## Architecture

- Browser scene: seeded square heightfield terrarium, solid swept-contact props and glass bounds, three-resource homeostasis loop, causal joint-work crawl plant with segment stick-slip grip, scripted resource/mount choreography, scripted stunt plant, and worm/skateboard visual rig.
- Locomotion contract: seven global sensors plus per-segment joint feedback feed 16 locally coupled recurrent neurons and 32 antagonistic activations at 60 Hz.
- Stunt contract: a separate 174-float observation to 32-activation mounted pose prior; pop and aerial board motion remain scripted.
- Training: tracked 39-parameter evolved locomotion model, behavior-distilled stunt prior, and separate legacy Gymnasium/PPO/CPG experiments. None is currently a high-fidelity transfer-physics trainer.

## Documentation

Agent-facing requirements, architecture notes, policy contract details, and the prioritized backlog live in [`docs/README.md`](docs/README.md).
