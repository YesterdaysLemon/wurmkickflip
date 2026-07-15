# Evolution Lab Concept

## Mental Model

The project should behave like a robotics/drone simulation lab with a skateboarding objective:

- Define a task family.
- Randomize simulation constants and environment parameters.
- Generate or mutate candidate bodies/controllers.
- Evaluate them in Python.
- Save best genomes and replays.
- Inspect the results in a rich 3D browser viewer.

The task family must keep skateboarding central. Creatures may change shape, but they should learn to move through the world, find a skateboard, interact with it, and roll around.

## Why Dynamic Environments Matter

Static environments produce brittle creatures. The training loop should vary environment parameters so creatures learn robust movement rather than memorizing one terrain or physics setup.

Useful randomization dimensions:

- gravity magnitude and direction
- terrain slope and roughness
- friction and restitution
- drag and damping
- obstacle layout
- skateboard spawn position, deck mass, and wheel friction
- spawn pose and velocity
- target location or heading
- actuator strength, delay, and noise
- sensor noise and dropped readings

## Evolution Strategy

Start with a simple genetic algorithm before adding complex RL:

1. Generate a population of genomes.
2. Evaluate each genome across multiple environment seeds.
3. Score locomotion distance, skateboard discovery, useful skateboard contact, rolling velocity, stability, energy use, survival, and task completion.
4. Select top genomes.
5. Mutate dimensions, joints, masses, and controller parameters.
6. Repeat and save generation summaries.

The live browser crawl controller has moved beyond the original CPG proposal. It is a clock-free, segmental recurrent neural controller evolved directly against a causal joint-work locomotion plant. Its 16 neurons own the 16 body segments; a shared 39-parameter genome covers initial recurrent state plus sensor, neighbor, and output weights. No gait clock, authored phase, sine/cosine teacher, or demonstration trajectory is available to evolution.

The published `locomotion-segmental-es-quality-robust-v1` refinement uses seed `20260719`, 80 generations, population 128, 18 elites, 420-step episodes, and eight approach/friction scenarios spanning traction values from 0.33 to 1.12. It warm-starts from the preserved 110-generation base artifact. Its risk-sensitive objective emphasizes the bottom two scenarios plus approach, close, and reached thresholds, and plant-bound actions are quantized like the browser. Verification compares the full controller with zero-action, frozen-pose, deterministic segment-shuffle, and zero-friction ablations. These tests show causal actuator work in the compact plant; they do not establish transfer to real soft-body physics.

## Current Recurrent Locomotion Path

`training/wurmkickflip_rl/evolve_locomotion_policy.py` writes `public/models/wurmkickflip_locomotion_policy.json`. The dependency-free browser runtime consumes the same update equations and plant constants. Food, water, and skateboard well-being are not gait teachers: the urgency selector chooses a resource, and its local direction/distance plus current urgency become sensors. The evolved network must turn those sensors and body feedback into segment work that moves and steers the worm.

The live exhibit composes that evolved output with deterministic browser-only layers that were not part of evolution. Joint integration and traveling-work propulsion are derived plant physics; per-segment stick-slip anchors and swept wall/tree/rock/resource contacts are derived contact handling. Target-aware collision reorientation, timed eating/drinking after bowl contact, staged mounting/dismounting, board routing, and bowl/mouth effects are scripted. These layers make the terrarium lifecycle coherent, but they are not evidence that the 39-parameter genome learned collision physics, resource manipulation, or board contact.

The kickflip is a separate authored exhibition. Pop, aerial rotation, landing timing, and the ride lifecycle are scripted; the older distilled stunt model may shape mounted segments but does not control detached locomotion. A future contact-rich training backend should move collision, grip, resource contact, and mounting outcomes into the evaluated plant before any of those behaviors are claimed as evolved.

## Legacy CPG Export Path

The older `wurmkickflip_rl.evolve` genetic algorithm mutates sinusoidal CPG controller parameters plus morphology scales for part size, mass, material friction, body spread, joint stiffness, joint damping, and motor strength. The surrogate environment derives rollout dynamics from browser creature and environment JSON fields, so morphology is part of that experiment's fitness rather than only display metadata.

When run with `--export-creature` and `--export-manifest`, the GA writes the best controller and mutated morphology back into a browser-readable creature genome. The browser treats `public/configs/evolved/manifest.json` as optional and appends generated creatures to the exhibit selector when the file exists.

This legacy CPG/morphology path remains useful experimentation, but it is not the new browser crawl brain. Its generated creature configs and fitness summaries must not be presented as provenance for `locomotion-segmental-es-quality-robust-v1`.

## Frontend Role

The frontend should not run massive training loops. It should:

- show the current best creature
- replay saved trajectories
- display environment parameters used for a run
- display skateboard task parameters and contact/rolling metrics
- compare generation fitness
- expose manual debug controls
- optionally run lightweight interactive simulations for inspection

## Future Backend Candidates

- Gymnasium + Stable Baselines3: good for current simple RL and ONNX export.
- MuJoCo: good for articulated 3D body training and contact-rich simulation.
- Brax: good for accelerated differentiable/vectorized simulation.
- PufferLib: interesting later for high-throughput training once the environment is stable, especially if we write a C environment.
