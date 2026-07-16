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
translation. Every tick performs one initial constraint solve plus the contract's
three post-integration relaxation solves. When collision geometry is present,
contact projection alternates with those additional solves.

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
Their obstacle-free articulated transition is checked numerically at tight tolerance.
Contact-rich evolution uses a vectorized circular-obstacle projection surrogate,
while the browser adds swept 3D-height-aware contacts and pins contacted particles
during later relaxation solves. Contact behavior is therefore certified by the
independent browser locomotion, gait, collision, and integrated-motion gates rather
than claimed as step-for-step Python parity.

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

The public `locomotion-articulated-head-leading-es-v3` model is one deterministic
retention refinement from the non-public checkpoint at
`training/seeds/wurmkickflip_locomotion_terrarium_causal_candidate_warm_start_v3.json`.
The checkpoint's canonical SHA-256 is
`0c46518c424528337f3a2ad5a96912b7812dda473688d468556023be095551ba`. It embeds a
historical 45-value discovery genome and reconstructs the retained genome by applying
three named bounded coordinate deltas: `terrainFriction +0.10`, `targetRight +0.05`,
and `obstacleRight -0.01`. The contract verifier independently recomputes that
embedded-genome composition and its derivation hash. Historical run paths, source
artifact hashes, and parent-summary hashes remain provenance labels from ignored search
runs, not independently verified ancestry claims. The exact retained genome passes all
17 selection gates and guard bands plus all five ablation gates and guard bands on the
recorded Windows host.

The public retention recipe uses objective v3, the `bounded-480` combined-recovery
selector, random seed `20260737`, one generation, population 4, one elite, and 480-step
episodes. The four-member generation evaluates the retained worm alongside deterministic
mutants and returns the retained 45-value genome exactly; it proves deterministic
retention and artifact construction, not rediscovery of the exploratory lineage.

Run `npm run verify:locomotion:published` from the repository root to copy that seed
into a unique temporary directory, execute the exact recipe, and require
**byte-for-byte** equality with the tracked public artifact. Both compact and pretty
trainer outputs are written as UTF-8 with explicit LF endings, so equality does not
depend on the operating system. The script removes its temporary directory and never
writes `training/runs/` or the tracked public model. This is a long reproducibility
lane and is intentionally separate from normal checks.

During broader exploratory searches, one third of each warm-started population is the
prior genome plus local mutations. After elite carryover, v3 uses available population
slots for deterministic near-feasible repair: half are small whole-genome mutations and
half are paired coordinate probes. The four-member publication retention recipe uses the
same declared algorithm without claiming to repeat the broader discovery search.
Warm provenance records only canonical SHA-256 and model version, never a relative
or absolute filesystem path, so artifact equality is location-independent. The
objective evaluates 12 contact/obstacle/friction/body-scale scenarios and mutates a
45-number genome: 16 initial recurrent states, 17 shared input weights, three shared
recurrent weights, and nine shared output weights. Only the plant-bound action is
quantized to Float32, matching the browser while recurrent state remains Float64.
No demonstration gait or trigonometric teacher is used.

## Published head-leading objective v3

The published artifact uses objective v3. When `--out` and `--summary` are omitted,
new v3 experiments still write under `training/runs/locomotion_head_leading_v3/`
rather than overwriting the browser model. The controller genome, sensors, actuator
command, Float32 plant boundary, and `articulated-contact-v2` contract are unchanged.
Legacy v2 experiments also default under `training/runs/`; no implicit trainer run
writes into `public/models/`.

V3 selection applies the browser's exact controller-input ranges before neural
inference: target bearings `[-1, 1]`, normalized distance `[0, 1.5]`, forward speed
`[-2, 2]`, angular speed `[-3, 3]`, terrain friction `[0, 1.2]`, urgency/contact
`[0, 1]`, slip `[0, 2]`, and obstacle components `[-1, 1]`. These non-mutating
views do not clamp the plant state or the raw slip values used by physics and
penalties. The published artifact records the ranges in `controllerSensorClamps`.

V3 measures COM displacement against the live tail-to-head body axis, whose
positive direction points toward the rendered head. It records signed and positive
head-leading fractions, lateral motion fraction, head-leading speed, motion-weighted
target-facing alignment, and late target-facing alignment. World heading is never
rewarded directly. Target-switch scenarios provide turns, with a short scoring grace
period immediately after the switch; no time, gait phase, teacher trajectory, or
authored wave is exposed to the network.

Three deterministic transient domains shape robustness:

1. a low-friction window that ends before the midpoint target switch;
2. an alternating lateral/backward whole-body shove after the target switch; and
3. a paired proxy modeled on the browser gait audit: a 1.1 m/s lateral shove,
   segment-7 neural lesion, and 1/15 traction from step 128 through 176 of its
   480-step episode, compared with an otherwise identical control rollout.

The publication recipe uses `--combined-recovery-mode bounded-480`. This is the
arena-limited shaping domain that produced the browser-certified lineage; its metrics
must not be described as a 900-step runtime result. An optional
`--combined-recovery-mode recentered-900` diagnostic accumulates a virtual root and
recenters the obstacle-free physical chain after every plant step. That frame change
does not modify the plant contract, but experiments found controllers that scored well
there and collapsed in the browser, so it is not a publication selector. In either
mode, a candidate must pass the authoritative 900-step TypeScript gait audit,
whose analogous intervention runs from step 240 through 330, including absolute
progress, recovery, head-axis, transverse-motion, causal, and constraint gates.

The bounded selector requires at least 4.36 m control progress, a 0.5 broad
correct-segment versus shuffled-segment score gap, and a 0.05 m full-versus-shuffled
target-progress gap before its feasibility bonuses. These margins keep absolute pace
and anatomical action ownership from being traded away for prettier body alignment.
V3 ranks candidates lexicographically: every guard-band-safe candidate precedes
gate-safe candidates, which precede infeasible candidates. Within every tier,
guard-band count and then nominal gate count precede four-decimal canonical fitness;
stable first-tie ordering is used throughout.
Gate diagnostics retain eight decimals, while ablation gates consume the unrounded
rollout measurements.

Host-sensitive metric magnitudes, perturbation recovery values, ablation magnitudes,
and the deterministic trace hash are written only under `hostDiagnostics` in the
evolution summary. The policy artifact contains the eight-decimal 45-value genome,
the thresholds, and every individual and aggregate selection/ablation/guard boolean.
Its `canonicalGenomeRecipeHash` binds the model version, genome precision and values,
recipe, trainer, articulated plant, Python contract loader, canonical JSON contract,
`pyproject.toml`, and `uv.lock`. The public model version is reserved bidirectionally:
the exact public recipe must use it, and no other recipe may claim it. The accepted
recipe must reproduce the same genome and artifact bytes on Windows and Linux; this
is an external release check, never a status self-attested by the trainer.
The current release reproduced byte-for-byte on Windows and Linux with artifact
SHA-256 `558d78d6fbe9bf4915d7bba71b332723101a8337597e87e33e8a561c64183117`.

V3 also vectorizes the browser microscope's exact 12-tick local causal probe. A
one-tick contact/slip/obstacle pulse at segment 7 must affect its local neighborhood,
propagate to graph radii two and three on successive ticks, and settle by tick 9.
Returned actions are compared at Float32 precision while recurrent command state stays
Float64, matching the browser. Normalized feasibility hinges use conservative margins
above the runtime gates, so a fast or attractive gait cannot outrank a controller that
has lost segment-local sensor responsiveness. Nominal metadata additionally reports
hidden-state saturation and mean `tanh` derivative.

Run the focused deterministic checks with:

```powershell
cd training
uv run --locked pytest tests/test_locomotion_objective_v3.py -q
```

A new training run must name a candidate path explicitly when it is intended for
review. Do not copy it into `public/models/` until the full browser gait verifier,
zero-traction conservation, causal ablations, and ordinary repository checks pass.

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
conservation, and causal ablations that use zero action, frozen action, and a fixed
segment shuffle. Each intervention
replaces both the Float32 plant action and the Float64 previous-command feedback
committed for the next recurrent step. `npm run
verify:motion` exercises the integrated needs, crawl, mount, and scripted-kickflip
lifecycle across the three tracked terrarium presets, including feeding,
tree/rock contacts, resource release, remounting, and full-scene
zero/frozen/shuffled/no-traction interventions. Its fixed food challenge requires
the full controller to restore hunger and reach the neural-to-authored feeding
handoff within 16 s, gain at least 1 m more cumulative need-target progress than
both frozen and fixed-shuffle controls, and reach that feeding handoff at least 3 s
earlier (or while the intervention never reaches it). This
avoids treating the nearly quantized restoration amount of two eventual feeding
cycles as a locomotion-speed measurement.
The published v3 artifact's six-scenario TypeScript rollout records about 3.9737 m
mean target progress, versus 0 m for zero action, -0.0138 m for a frozen output, and
3.5290 m after deterministic segment shuffling. Its motion is 0.9166 head-axis,
0.3296 transverse, and 0.6703 aligned toward the target. Active zero-friction COM
drift is about `9.1e-14` m. The evolution summary's 12-scenario Python
`hostDiagnostics` records its own closed-loop ablation magnitudes; those numbers
answer a different evaluation question and should not be compared directly.
