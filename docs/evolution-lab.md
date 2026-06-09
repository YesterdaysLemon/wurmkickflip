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

The first controller can be a central pattern generator with sinusoidal motor outputs. Later versions can train neural controllers per morphology or co-evolve policy weights.

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
