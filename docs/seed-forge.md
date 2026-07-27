# Seed Forge

Seed Forge is the browser's deterministic domain-randomization laboratory. It changes the challenge around the tracked policies; it does not retrain them or turn scripted lifecycle and stunt behavior into learned behavior.

## Startup And Controls

Each built-in environment starts at its authored nominal domain, using that environment's seed. Startup is not a random draw. The nominal sample uses:

- gravity, ground friction, air drag, and actuator strength scales of `1`;
- authored terrain slope, roughness, and obstacle density;
- zero actuator latency, sensor noise, and spawn yaw;
- authored skateboard X/Z spawn, mass, and wheel friction.

Selecting another environment restores that environment's nominal sample. **Restore authored baseline** also restores its original seed and clears every lock.

The seed field accepts unsigned 32-bit integers from `0` through `4,294,967,295`. Applying a seed deterministically resamples every unlocked channel. **Reroll** advances the seed by `0x9e3779b9` modulo `2^32`, then performs the same sample operation. Programmatic seed helpers truncate and wrap values to uint32, but reject non-finite inputs or a truncated value outside JavaScript's exact safe-integer range. Authored environment seeds must already be safe integers.

Locks preserve the current value across both an applied seed and a reroll; preserved values are still clamped to the selected environment's declared range. Applying a challenge preset replaces the channel values while retaining the lock switches for later seeded sampling. The four presets are:

- **Nominal**: the authored baseline;
- **Ice rink**: minimum ground and wheel grip with a clean, light course;
- **Moon dirt**: a low-gravity, rougher range-relative challenge;
- **Cinderblock deck**: every channel at its configured maximum, including board mass and actuator latency.

Every applied seed, reroll, preset, or baseline restore restarts the live simulation with the resulting sample.

## Domain Contract

A domain sample has `schemaVersion = 1`, `kind = "wurmkickflip.domainSample"`, a uint32 seed, and these 14 finite channels:

| Group          | Channel             | Fixed-step effect                                                                                                      |
| -------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| World + ground | `gravityScale`      | Multiplies the environment gravity vector.                                                                             |
| World + ground | `frictionScale`     | Multiplies authored terrain base friction before terrain sampling.                                                     |
| World + ground | `dragScale`         | Multiplies authored air drag used by skateboard damping.                                                               |
| World + ground | `slopeDegrees`      | Rebuilds the shared render/physics heightfield at the sampled slope.                                                   |
| World + ground | `roughness`         | Rebuilds the shared heightfield at the sampled roughness.                                                              |
| World + ground | `obstacleDensity`   | Rebuilds deterministic decor/collision density for the sampled seed.                                                   |
| Wurm signals   | `actuatorStrength`  | Scales the delayed 32-channel command and clamps each result to `[-1, 1]`.                                             |
| Wurm signals   | `actuatorLatencyMs` | Becomes a whole-number 60 Hz delay using nearest-tick rounding; the queue emits zero until filled.                     |
| Wurm signals   | `sensorNoise`       | Adds deterministic signed noise to crawl inputs and mounted-policy observations by seed, simulation step, and channel. |
| Wurm signals   | `spawnYawDegrees`   | Sets the initial skateboard heading and the opposing worm approach orientation.                                        |
| Skateboard     | `skateboardSpawnX`  | Sets the initial board X coordinate within arena bounds.                                                               |
| Skateboard     | `skateboardSpawnZ`  | Sets the initial board Z coordinate within arena bounds.                                                               |
| Skateboard     | `skateboardMass`    | Changes board traction/response calculations.                                                                          |
| Skateboard     | `wheelFriction`     | Changes deck contact friction and board traction.                                                                      |

Every channel is sampled within the selected environment's `randomization` range and quantized to six decimal places. Declared range endpoints, authored nominal values, and persisted samples must sit on that same six-decimal lattice and remain small enough for exact scaled-integer checks. Quantization occurs before a final range clamp, and interpolation uses a bounded convex form. Those rules make every validator-approved domain closed under sampling, including fixed ranges and wide signed ranges.

TypeScript and Python use the same Mulberry32 sequence, uint32 arithmetic, channel order, quantization, nominal values, and range-relative presets. Python accepts integral JSON numbers such as `1.0` wherever JavaScript necessarily sees the same number as an integer, while rejecting numeric strings and unsafe integer magnitudes. A canonical seed-1337 fingerprint and adversarial boundary vectors protect the algorithm from silent drift.

The perturbation layer remains fixed-step. Sensor noise is a stateless integer hash rather than `Math.random()`, and actuator latency is a policy-tick queue, so their values and queue behavior are stable at a given simulation step. Mounted JSON inference is now synchronous inside the canonical `TerrariumEpisode` step, so live and headless policy actions share the same 60 Hz ordering instead of depending on render cadence.

## Replay Provenance

New browser recordings store the seed and all 14 forged channel values in `environmentSample`. Skateboard X/Z are represented as the two-element `skateboardSpawn` tuple; the other values retain their Seed Forge names. When the matching base environment is available, import reconstructs and range-validates the domain before using it for scene setup.

Historical schema-v1 artifacts predate four fields. When all four are absent, validation supplies these nominal in-memory defaults:

- `actuatorStrength = 1`
- `actuatorLatencyMs = 0`
- `sensorNoise = 0`
- `spawnYawDegrees = 0`

A partial quartet is invalid. For a checksummed historical artifact, TypeScript validation verifies the original payload before adding defaults, then gives the normalized in-memory artifact a matching checksum. Newly recorded samples require all four fields, preserve exact values through JSON, and remain covered by the replay integrity digest. See [`replay-artifacts.md`](replay-artifacts.md) for timing, playback, and checksum rules.

## Verification

Run the focused checks from the repository root:

```powershell
npm run verify:seed-forge     # determinism, locks, ranges, presets, and TS/Python parity
npm run verify:domain-runtime # fixed-tick latency, strength, noise, spawn, and drag effects
npm run verify:replay         # exact provenance, historical defaults, cloning, and tamper rejection
npm run check:browser         # desktop/mobile Seed Forge controls and live reset behavior
```

`npm run check` includes the static, Seed Forge, domain-runtime, replay, build, and integration lanes. Browser coverage remains a separate command.
