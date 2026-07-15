# Evolved locomotion controller

The detached worm uses a clock-free segmental recurrent neural controller. It is
separate from the mounted exhibition path because the two jobs have different
contracts: the kickflip is scripted, while ordinary locomotion must emerge from
segment actuator work.

## Neural architecture

There are 16 recurrent `tanh` neurons, one per body segment. A neuron receives the
current target direction, target distance, root speed, angular speed, terrain
friction, need urgency, its own joint state and command, local contact load, slip,
obstacle direction, and the previous states of its immediate anterior and posterior
neighbors. It emits one bend command which is expanded into the antagonistic pair
`[command, -command]`.

Controller/plant indices run anterior-to-posterior. The Three.js rig is stored
tail-to-face, so the articulated adapter reverses that indexing when it maps joint
and contact arrays onto visible segments; controller segment 0 therefore owns the
rendered head.

There is deliberately no time, clock, cycle, gait phase, sine, or cosine input. The
only recurrence is neural state and neighbor coupling. The controller can therefore
settle, reverse, steer, or form an emergent limit cycle in response to its body and
environment rather than reading an authored animation phase.

The tracked browser artifact is
`public/models/wurmkickflip_locomotion_policy.json`, and its dependency-free browser
runtime is `src/policy/locomotionPolicy.ts`.

This controller is the live detached-crawl brain. It is not the older sinusoidal CPG
in `wurmkickflip_rl.evolve`, and it is not the 174-input distilled stunt prior. The
mounted pop and aerial kickflip remain scripted.

## Actuator plant

For segment `i`, the two policy outputs become a joint command:

```text
command[i] = clamp((dorsal[i] - ventral[i]) / 2, -1, 1)
jointVelocity[i] += ((command[i] - joint[i]) * 22 - jointVelocity[i] * 7) * dt
joint[i] = clamp(joint[i] + jointVelocity[i] * dt, -1.15, 1.15)
```

`stepArticulatedWorm` applies those servos to a free 16-segment planar chain. The
joint shape defines a centered target curve, and its actuator acceleration is made
mean-free before integration. Pairwise length constraints apply equal and opposite
corrections, so neither the neural network nor the servo layer can write root
translation.

```text
shapeAcceleration[i] = (centeredTarget[i] - position[i]) * actuatorStiffness
shapeAcceleration -= mean(shapeAcceleration)
position += velocity * dt
repeat distance constraints with equal/opposite endpoint corrections
```

Per-segment longitudinal and lateral ground damping are anisotropic. A traveling
shape change can therefore exchange momentum with contact and move the measured
center of mass. The plant reports anterior-to-posterior contact load, slip speed,
and obstacle normal components for the next neural step. Target coordinates never
enter the plant; they affect motion only by changing neural actuator outputs. Zero
commands remain stationary. With no obstacles and zero friction, internal forces
conserve planar center of mass to floating-point precision even while the neural
commands and joint servos remain active.

All names and constants are owned by `contracts/locomotion-v2.json`. The browser
runtime and Python evolution plant both load that `articulated-contact-v2` contract.

## Runtime composition and ownership

The browser composes deterministic interaction handling around the evolved
articulated plant:

1. The recurrent network owns only the 32 segment activations during `crawling`
   and `seeking`.
2. Joint servos, mean-free shape forces, distance constraints, anisotropic ground
   contact, and obstacle impulses are plant dynamics. They produce measured motion
   and segment-local feedback; there is no scalar propulsion channel.
3. Swept segment contacts constrain motion against glass, trees, rocks, and
   inactive resources. Frictional sliding is derived, while any target-aware
   recovery outside the policy remains authored scene logic.
4. Eating/drinking, mount/dismount contact stages, board routing, and the aerial
   kickflip are scripted choreography. They were not present in locomotion
   evolution and must not be credited to this artifact.

## Reproduce the published evolution

The current public model has two deterministic stages:

1. Base: legacy schema-v1 warm start, seed `20260720`, 40 generations,
   population 96, 20 elites, and 360-step episodes. It emits
   `locomotion-articulated-contact-es-v2`.
2. Refinement: that base artifact as warm start, seed `20260721`, 10 generations,
   population 64, 12 elites, and 480-step episodes. It emits
   `locomotion-articulated-contact-es-v2.1-candidate`.

Run `npm run verify:locomotion:published` from the repository root to execute both
stages and require canonical artifact equality. The script copies the legacy seed
and writes both stages into a unique temporary directory, then removes it. It never
writes `training/runs/` or the tracked public model. This is a long reproducibility
lane and is intentionally separate from normal checks.

One third of each warm-started population is the prior genome plus local mutations.
Warm provenance records only canonical SHA-256 and model version, never a relative
or absolute filesystem path, so artifact equality is location-independent. The
objective evaluates 12 contact/obstacle/friction/body-scale scenarios and mutates a
45-number genome: 16 initial recurrent states, 17 shared input weights, three shared
recurrent weights, and nine shared output weights. Only the plant-bound action is
quantized to Float32, matching the browser while recurrent state remains Float64.
No demonstration gait or trigonometric teacher is used.

## Homeostasis composition

The browser places a food bowl, a water bowl, and the moving skateboard in the
terrarium. Hunger, thirst, and well-being are urgency values in `[0, 1]`. A
deterministic urgency selector with mild target hysteresis chooses one resource;
food and water restore only while the live head-derived mouth point overlaps the
visible finite contents in 3D; empty bowls stop restoring and refill on a
deterministic schedule. The skateboard restores well-being only while mounted.
The selected target is converted to local
forward/right/distance sensors for the recurrent controller. Need selection does
not choose segment poses and never writes root position, so the evolved actuators
must still produce the approach. Confirmed mouth restoration enters an
authored feeding state: learned translation pauses, the face performs the contact
cycle, and a narrow approach aperture plus release cooldown lets the anterior body
enter and clear the otherwise solid annular rim.
Mounting and dismounting likewise use authored head/midbody/tail contact weights;
the neural controller owns the approach and regains control after dismount.

Run `npm run verify:needs` to validate deterministic placement, finite
inventories/refills, exact mouth-contact restoration, empty-resource behavior,
immutability, the diagnostic homeostasis vector, and a complete three-resource loop.
Run `npm run verify:collisions` to validate continuous anti-tunneling sweeps,
long-body contacts, wall/corner response, frictional sliding, overlap recovery,
support telemetry, and deterministic obstacle ordering. Run `npm run
verify:interactions` to validate phase-continuous mount, dismount, eat, and drink
poses plus their locomotion handoffs.
Run `npm run verify:locomotion` to validate schema-v1 migration, malformed v1/v2
rejection, contract fields, local contact and steering sensors, deterministic
articulated traces, location-independent same-seed export, exact zero-friction COM
conservation, and zero/frozen/shuffled causal ablations. `npm run
verify:motion` exercises the integrated needs, crawl, mount, and scripted-kickflip
lifecycle across the three tracked terrarium presets, including feeding,
tree/rock contacts, resource release, remounting, and full-scene
zero/frozen/shuffled/no-traction interventions.
The current six-scenario TypeScript rollout records about 4.17 m mean target
progress, versus 0 m for zero action, about 0.02 m for a frozen output, and about
1.44 m after deterministic segment shuffling. Its active zero-friction COM drift is
below `1e-12` m. The artifact's 12-scenario Python training metadata records its own
closed-loop ablations; those numbers answer a different evaluation question and
should not be compared directly.
