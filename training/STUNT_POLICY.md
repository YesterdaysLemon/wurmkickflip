# Distilled Stunt Policy

`wurmkickflip_rl.train_stunt_policy` trains a small PyTorch behavior-cloning model and exports it as browser-readable JSON. This is supervised behavior distillation from a deterministic state-aware expert; it is not PPO and it does not claim that a physics rollout has learned a kickflip. In the current browser it is a mounted exhibition pose prior. Board routing, pop, aerial board rotation, landing window, and ride lifecycle are scripted explicitly. Staged head-to-tail mounting and head-first dismounting come from the separate deterministic `src/scene/wormInteractionAnimation.ts` sampler, not this model.

The teacher runs a 7.2-second loop with a traveling bend wave, a positive co-contraction coil from roughly 2.2-2.65 seconds, a negative co-contraction release from roughly 2.65-2.85 seconds, a left/right kick signal during release and early air, a moderate air tuck, and a damped landing. Board roll and segment state perturbations teach corrective feedback instead of a time-only open-loop sequence. This cycle is not used for detached locomotion: ordinary crawling uses the separate clock-free recurrent controller described in [`LOCOMOTION_POLICY.md`](LOCOMOTION_POLICY.md).

Train and export reproducibly from `training/`:

```powershell
uv run python -m wurmkickflip_rl.train_stunt_policy
uv run python -m wurmkickflip_rl.validate_stunt_policy
```

The tracked output is `public/models/wurmkickflip_stunt_policy.json`. It contains a raw-input `174 -> hidden -> 32` tanh MLP; input normalization used during training is folded into the first layer. The artifact records its seed, sample count, epoch count, validation MSE, and teacher-agreement score. Teacher agreement is the fraction of held-out action coordinates within `0.12` of the expert target.

The expert's input mask is deliberately narrower than the shared observation contract. It uses cycle time, board roll/yaw, contact ratio, forward target, and every segment's pitch/yaw (37 inputs total). Board/segment positions, linear velocities, lateral target components, and previous actions are ignored. Training applies this mask before the first layer, export records the 37 feature indices, and every ignored first-layer column is exactly zero. This prevents out-of-distribution velocity spikes from becoming an accidental high-gain muscle feedback loop; validation perturbs all ignored inputs aggressively and requires identical actions.

The action semantics are two values per segment. For segment `i`, `bend = (dorsal - ventral) / 2` and `coContraction = (dorsal + ventral) / 2`. The kick signal is the mean of `bend * sideWeight`, where segments 0-7 have weight -1 and segments 8-15 have weight +1.

This model is an integration-ready stunt prior, not the worm's autonomous locomotion brain. A learned kickflip still requires a simulator with real articulated contacts, airborne rotation and landing metrics, and held-out rollout evaluation.

Run `npm run verify:interactions` from the repository root to validate the authored
mount/dismount contact sequence independently of this policy, and `npm run
verify:motion` to validate its integration with the scripted kickflip, resource
lifecycle, swept contacts, and evolved detached crawl.
