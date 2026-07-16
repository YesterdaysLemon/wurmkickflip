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

The published model uses a deterministic v3 retention check: seed `20260737`, one generation, population 4, one elite, and 480-step episodes, warm-started from the retained `wurmkickflip_locomotion_terrarium_causal_candidate_warm_start_v3.json` checkpoint in `training/seeds/` (SHA-256 `0c46518c424528337f3a2ad5a96912b7812dda473688d468556023be095551ba`). That checkpoint embeds a historical 45-value discovery genome and records three bounded coordinate refinements (`terrainFriction +0.10`, `targetRight +0.05`, and `obstacleRight -0.01`). The contract verifier reconstructs this embedded-genome composition exactly and checks its hash. Historical ignored-run paths, source artifact hashes, and parent-summary hashes are retained provenance labels, not independently verified ancestry. The four-member publication generation evaluates the retained worm against deterministic mutants and returns the retained 45-value genome exactly; it is a reproduction/retention recipe, not a claim that one tiny generation rediscovered the lineage.

The base objective evaluates 12 scenarios with obstacles, spatial friction, body-scale variation, target changes, and intermittent contact loss. V3 adds head-leading geometry, deterministic transient domains, and a local causal probe. Its bounded 480-step combined-recovery domain is an arena-limited training proxy; publication additionally requires the authoritative 900-tick TypeScript rollout with a shove, segment lesion, and 1/15 traction. Artifact identity records warm-start SHA-256 and model version, never a machine-specific path. `npm run check:repro` stages the retained seed in a fresh temporary directory, repeats the exact refinement, and requires byte-for-byte equality with the published JSON.

For host-independent selection, v3 uses stable ordering on four-decimal canonical fitness. Selection is lexicographic: candidates inside every guard band rank ahead of candidates that merely pass the gates, and both rank ahead of infeasible candidates. Within every tier, guard-band count and then nominal gate count precede canonical fitness. Gate diagnostics retain eight decimal places, and ablation gates inspect the unrounded rollout values. During exploratory searches, deterministic micro-mutations and coordinate probes use available non-elite population slots to repair near-feasible lineages.

The policy artifact contains the eight-decimal genome, locked recipe identity, thresholds, and individual/aggregate selection, ablation, and guard-band booleans. Host-sensitive metric magnitudes, recovery values, ablation magnitudes, and the diagnostic trace hash live only in the evolution summary. `canonicalGenomeRecipeHash` binds the model version, genome precision and values, full recipe, trainer, articulated plant, Python contract loader, canonical JSON contract, `pyproject.toml`, and `uv.lock`. Paired Windows/Linux reproduction is an external release gate; the trainer never self-attests that comparison from command-line arguments.

## Articulated Plant

Evolution and the browser share the `articulated-contact-v2` model from [`../contracts/locomotion-v2.json`](../contracts/locomotion-v2.json). Antagonistic commands drive damped joints; joint states create mean-free internal forces on a 16-particle chain; equal-and-opposite constraints preserve spacing. Every tick performs one initial constraint solve plus the contract's three post-integration relaxation solves. Ground motion comes from per-segment anisotropic friction, and the reported root is the measured segment center of mass. The obstacle-free base transition is numerically parity-tested. Training uses vectorized circular-obstacle projection as a contact surrogate; the browser's swept, height-aware collision layer alternates projection with relaxation and pins contacted particles, so contact-rich behavior is promoted by browser-native collision and integrated rollout gates.

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
