# Forge Trials / Wurm Olympics

Forge Trials turns Seed Forge from a domain editor into a deterministic evaluation meet. The browser runs the tracked recurrent controller across held-out seeds, repeats every domain with causal controls, and retains a recorder-core replay for the neural lane.

This is evaluation inside the compact Wurmkickflip plant. It is not training, biological evidence, or real-world skateboard transfer.

Here, “held-out” means excluded from the active live attempt. It is not a claim that the seeds were absent from every historical trainer domain.

## Canonical Episode Engine

`src/scene/terrariumEpisode.ts` is the only production fixed-step episode authority. It owns:

- recurrent and mounted policy observation/inference;
- deterministic sensor noise;
- actuator latency and strength;
- policy-to-authored lifecycle handoffs;
- the exact `advanceStunt` plant call;
- recorder-core frame emission.

`WurmkickflipScene.tsx` accumulates render time and asks this engine for 60 Hz steps. `src/forgeTrials/forgeTrials.ts` creates the same engine headlessly. The live exhibit and batch evaluation therefore cannot silently diverge through separate action-selection loops.

Mounted JSON inference is synchronous and dependency-free inside the episode step. Rendering can display several completed fixed steps after a slow frame, but render cadence no longer chooses which mounted action reaches those steps.

## Meet Schedule

The UI supports 8, 16, 24, or 32 heats. The active live seed is the base but is never itself evaluated. Every heat uses the next uint32 Seed Forge value in the golden-ratio schedule already used by reroll:

```text
next = (previous + 0x9e3779b9) mod 2^32
```

Each seed materializes all 14 domain channels before the episode begins. All three lanes within a heat use the exact same sample.

## Control Lanes

Every heat runs for eight simulated seconds:

- **Neural:** the tracked clock-free recurrent controller owns crawling, seeking, and mounting.
- **Zero:** the recurrent network still runs for diagnostic activity, but all applied muscle commands and recurrent command feedback are zero.
- **Frozen:** the controller runs normally for 24 locomotion ticks, then its applied action and command feedback remain fixed.

The mounted pose prior and authored lifecycle are unchanged. A transition into `riding` still requires the deterministic multi-region contact gate.

## Verdicts

- **Causal win:** neural reaches riding; zero and frozen do not.
- **Contested:** either control reaches riding. This is reported as a control result, not hidden or relabeled.
- **Neural miss:** neural does not reach riding and neither control does.

The report also shows minimum board distance, maximum deck-contact ratio, actuator latency, wheel grip, neural/controller mount counts, control mount counts, and median neural ride time.

These finite results answer whether the tracked controller succeeds in the sampled compact domains. They do not prove general competence beyond those seeds.

## Replay Handoff

The neural lane retains a checksummed schema-v1 recorder-core artifact for every heat. “Inspect replay” loads that artifact into the existing Replay lab with:

- Seed Forge provenance locked to the evaluated sample;
- timeline scrubbing;
- discovery, first-contact, and ride-contact jumps when present;
- 0.5×, 1×, and 2× playback.

Schema v1 preserves timing, board/root poses, contact, reward, and all 32 muscle channels. It does not preserve individual segment poses or the complete lifecycle/homeostasis state, so the viewer reconstructs segment presentation and labels that limitation.

## Verification

```powershell
npm run verify:forge-trials
npm run verify:runtime
npm run verify:replay
npm run check:browser
```

The focused verifier checks uint32 seed scheduling, the canonical neural/zero/frozen challenge, replay integrity/provenance, repeated-meet trajectory determinism, and live-scene ownership of the shared episode engine.
