# Evolution Lab

## Mental Model

Wurmkickflip is a small evolutionary robotics lab with an intentionally silly skateboard objective:

1. Define a seeded task family.
2. Randomize terrain, friction, obstacles, bodies, targets, and contact loss.
3. Evaluate a population of compact controllers.
4. Select and mutate the best candidates.
5. Save a reproducible artifact and replay its behavior in the browser.

The browser is the inspection surface, not the high-throughput trainer.

## Current Crawl Evolution

`training/wurmkickflip_rl/evolve_locomotion_policy.py` evolves the tracked detached-crawl policy. The dependency-free TypeScript runtime consumes the same canonical contract and recurrence.

The controller has 16 locally coupled recurrent `tanh` neurons, one per segment. Its 45-value genome contains:

- 16 initial hidden states;
- 17 shared input weights;
- three shared recurrent weights (`self`, anterior neighbor, posterior neighbor);
- nine shared output weights.

Each neuron emits one dorsal/ventral actuator pair. Inputs include target direction/distance, measured root motion, terrain friction, urgency, bend and bend velocity, prior command, and segment-local support, slip, and obstacle-normal feedback. There is no time, gait phase, trigonometric teacher, demonstration trajectory, or direct root-motion action.

The published model is a deterministic two-stage run:

1. seed `20260720`: 40 generations, population 96, 20 elites, 360-step episodes, warm-started from the preserved schema-v1 legacy seed;
2. seed `20260721`: 10 generations, population 64, 12 elites, 480-step episodes, warm-started from stage one.

Both stages evaluate 12 scenarios with obstacles, spatial friction, body-scale variation, target changes, and intermittent contact loss. Artifact identity records warm-start SHA-256 and model version, never a machine-specific path. `npm run check:repro` repeats both stages in a fresh temporary directory and requires canonical equality with the published JSON.

## Articulated Plant

Evolution and the browser share the `articulated-contact-v2` model from [`../contracts/locomotion-v2.json`](../contracts/locomotion-v2.json). Antagonistic commands drive damped joints; joint states create mean-free internal forces on a 16-particle chain; equal-and-opposite constraints preserve spacing. Ground motion comes from per-segment anisotropic friction, and obstacles contribute resolved contact impulses. The reported root is the measured segment center of mass.

This gives useful causal tests:

- zero commands do not travel;
- frozen commands cannot substitute for recurrent changing actions;
- assigning commands to the wrong segments damages progress;
- active muscles conserve horizontal center of mass in an obstacle-free zero-friction world;
- local contact/surface observations change only the owning neuron's response before recurrence propagates them.

These checks establish causality inside this compact model. They do not establish transfer to biological worms or real skateboards.

## Terrarium Composition

The trained model owns segment commands only during `crawling` and `seeking`. The live exhibit adds deterministic systems that were not learned:

- homeostasis chooses food, water, or skateboard well-being as a goal;
- swept scene contacts handle glass, props, annular bowl rims, and the deck;
- finite bowl inventories restore only during 3D live-mouth contact and refill deterministically;
- feeding poses, mounting, dismounting, route planning, pop, aerial rotation, and landing are scripted.

The kickflip is therefore an authored exhibition. The mounted `stunt-distilled-v2` network is an imitation-learned pose prior, not learned contact physics.

## Future Training Work

The most valuable next research step is to bring more of the lifecycle inside the evaluated plant: finite resources, mouth contacts, skateboard discovery, useful deck contact, stable mounting, and rolling. Domain randomization should include gravity, roughness, friction, restitution, body scale, actuator delay/noise, sensor loss, obstacle layout, board mass, and wheel friction. Held-out seeds should measure generalization rather than one memorized terrarium.

MuJoCo, Brax, or a custom vectorized contact model are plausible future backends. Any replacement must preserve explicit observation/action/artifact versions and continue separating learned behavior from authored presentation.

## Legacy Paths

`wurmkickflip_rl.evolve` still explores sinusoidal CPG parameters and morphology, while the Gymnasium/PPO/ONNX tools remain offline experiments. Generated creature configs can be loaded as appearance projections, but these legacy paths are not provenance for the tracked recurrent crawl controller and the browser no longer ships an ONNX runtime.
