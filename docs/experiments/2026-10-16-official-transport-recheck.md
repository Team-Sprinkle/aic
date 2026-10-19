> **October 16 protocol correction:** This batch uses the post-qualification released `eval_config.yaml`; its stress cases inherit those layouts too. The entire batch is **evaluation-only, ineligible for route/data/threshold/checkpoint tuning**. Preserve the frozen job and results. Do not execute this document's earlier recommendations to repair components based on these scenes. [The revised protocol](../EVALUATION_PROTOCOL.md) governs subsequent development using pre-release defaults/specifications. Artifact `development_use_restriction.json` records the boundary.

# Public official-setting transport recheck

## Why the recent numbers looked better

The latest transport controller is a **rule-based controller**, not the newly trained insertion actor or the old diffusion transport actor. SC follows Low Lateral Bypass; SFP follows its CheatCode-style staged route. Measured TCP progress advances stages. Camera-based requested-port localization can correct the acquisition waypoint, and a native-resolution landmark refiner plus multi-camera fit must approve the requested opening before handoff.

The improved SC crop/landmark models and requested-port checks repaired real acquisition failures. However, the recent best screen also used a familiar development board placement. The corrected mixed test was only ten SC and ten SFP cases; the repaired SC screen was twenty cases at one/five-card extremes. These are not matched evidence that transport became generally reliable. The earlier broad test achieved 44/50 on its primary placement but **0/10 on the alternate published board placement**. No wrong acceptance was observed; safe rejection is not a successful handoff.

The official published SC board is at yaw −1.8 rad; the common development template is near +3.1 rad. Official SFP examples also move the board between y=−0.21 m and +0.05 m. A fixed corridor/fixture-normal prior cannot be assumed to generalize across those changes.

## Upstream source and exactness

On October 16 the current upstream file was downloaded again:

- [Official evaluation configuration](https://github.com/intrinsic-dev/aic/blob/main/aic_engine/config/eval_config.yaml).
- [Release PR #558](https://github.com/intrinsic-dev/aic/pull/558), described by the organizers as the qualification evaluation configuration.
- SHA-256 remains `9f987c07174481474eec2c4fa21faf7d84dae4743a037b809ed7945efb2e5ef5`.

It contains three scenes, not every card count and port combination:

| Public trial | Task | NIC cards | Requested target |
| --- | --- | --- | --- |
| 1 | SFP to NIC | One, rail 2 | Card 2 / SFP port 0 |
| 2 | SFP to NIC | Two, rails 3 and 4 | Card 4 / SFP port 1 |
| 3 | SC to SC, reversed cable | Three, rails 0–2 | SC port 1 |

Each scene is isolated into its own evaluation launch while retaining its original trial/task keys, board/components/cable/grasp, task time limit, global limits and robot home. First-launch robot joints are explicitly set to the official values in the robot xacro. Passive cameras, contact recorders and ground-truth scoring recorders are additional diagnostic instrumentation; no collisions are disabled and privileged streams do not guide the policy. **This is an official-scene transport test, not an unchanged three-trial sequential cloud submission or a full insertion evaluation.** Later hidden/physical evaluation coverage cannot be inferred from this file.

## Frozen experiment

Artifact root: `/home/nas/chmin/aic_archives/2026-10-16_official_transport_recheck_v1`.

1. **15 official-scene repeats:** five fresh simulator launches of each of the three public trials with the latest candidate family bundle.
2. **Five matched SC baseline repeats:** the same public SC setting with the previous SC native refiner, preserving the route/coarse/settings. This isolates the candidate refiner effect on the official SC scene.
3. **100 published-range stress episodes:** SC/SFP × 1–5 NIC cards × two requested ports × five replicates. SFP requested card cycles through rails 0–4 and is always present. Requested rails span the released limits; other rails/card subsets vary. Board/home/grasp remain those of the chosen official source trial. These derivatives are explicitly not called exact official evaluation.

All models are copied and hashed before evaluation. Source hashes are checked between episodes. There is no tuning, retraining, insertion actor, or backtracking in this transport-only comparison. Functional failures remain in the matrix; an infrastructure/scoring failure stops it for repair without being called a policy failure. Existing final scenes remain sealed.

Report separately: correct requested-port acquisition; strict qualification including opening estimate ≤1 mm, actual stand-off 130–180 mm, actual lateral ≤25 mm, actual orientation ≤15°, no scored harmful contact, and active inference p95 <300 ms; rejected/no-progress episodes; wrong accepts; contact/force; and terminal geometry. Geometry is evaluated only after the episode at the retained pre-return timestamp. Five repeats have wide statistical uncertainty and do not establish production reliability.

Budget: **120 sequential launches**, approximately 6–12 hours including startup and scoring; existing per-policy wall cap 300 s and transport simulation cap 115 s. Render GPU3 and perception GPU4 use cooperative locks. Legacy GPU0 remains untouched. No training GPU is allocated.

```bash
.pixi/envs/default/bin/python scripts/prepare_official_transport_recheck.py \
  --output /home/nas/chmin/aic_archives/2026-10-16_official_transport_recheck_v1 \
  --prior-root /home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1
.pixi/envs/default/bin/python -u scripts/run_official_transport_recheck.py \
  --output /home/nas/chmin/aic_archives/2026-10-16_official_transport_recheck_v1 \
  --prior-root /home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1
```

`plan.json` contains scenes, source hashes and checkpoints. `current_job.json`, `progress.json`, `progress.md` and `runner.log` describe live state. Final `summary.json` exists only when the complete matrix finishes. Each run retains commands/effective container settings, raw scoring MCAP, trace, camera frames, geometry curves, official scores and `transport_metrics.json`.

## What was trained in the October 15–16 continuation

| Component | Training performed | Stopping/result |
| --- | --- | --- |
| SC requested-port coarse locator | Original SC sources plus 29 expanded-rail recordings; 105 train / 28 validation episodes | Current-image continuation plateaued at 9,750 updates, best 7,750. History variant was tried but not selected. |
| SFP requested-port coarse locator | Full synchronized corpus plus acquisition coverage extension; 350 train / 63 validation source episodes | Current-image continuation plateaued at 5,000 updates, best 3,000. History was not better in the measured comparison. |
| Original-image point refiner | Substantial SC+SFP corpus: 72,857 native RGB triplets / 497 source episodes / 422 reset groups before extensions | Continued to a validation plateau: 18,750 cumulative updates. |
| Five-landmark native pose refiner | Same broad two-family corpus, synchronized camera data | Base fit reached a validation plateau; subsequent repair retained original data. |
| SFP native-pose repair | Added acquisition views from 20 episodes while retaining both families | 16,000 cumulative updates, validation plateau. Current SFP checkpoint. |
| SC native-pose rail repair | Added 4,526 crops from the existing 29 SC recordings, retaining both-family data | 4,000 additional updates to a validation plateau. Correct acquisition 20/20, strict pass 18/20 in its bounded SC screen; overall component acceptance had a small regression. Candidate, not full-system promotion. |
| SC crop association/identity checker | Original plus expanded-rail data; wrong-port/absent queries | Trained and evaluated separately; component weaknesses remain documented. |
| SFP held-plug geometry helper | 6,115 native examples | 15,000 updates, validation plateau. Translation support has a separately bounded free-approach approval; full grasp orientation is not approved. |
| Shared full-approach/insertion BC actor | 67,039 examples / 500 usable episodes; 421 train / 79 validation; SC and SFP | Two matched 30,000-update fits: ImageNet initialization versus transport visual-weight initialization with fresh action heads. No meaningful late improvement justified extra fitting. Full insertion comparison stopped at the acquisition prerequisite. |

The transport route itself was not learned in these new acquisition tests. No new diffusion transport actor, SAC/SERL, reward model, world model or imagination training was performed in this round. Hierarchical recovery was implemented but has not yet been validated with the completed insertion actors.

## Decision rule

Do not claim official-setting robustness from the recent primary-board screens. Finish this frozen matrix. If public-board failures dominate, repair the observation-based board/route acquisition or explicitly limit the demonstrated operating envelope; more insertion BC cannot fix a transport stage that never reaches acquisition. If transport passes but latency fails, profile/repair runtime without relabeling physical success. Only after a valid acquisition configuration passes its gate should the completed insertion comparison resume.

## First recorded official-scene pass — preliminary, not the repeated result

At 14:12 UTC, the first three candidate runs finished; the remaining repeats and stress matrix are running unchanged:

| Scene | Strict handoff | Observation |
| --- | --- | --- |
| Trial 1, SFP / one card / card 2 port 0 | 1/1 | Correct requested-port acquisition and strict gate pass. |
| Trial 2, SFP / two cards / card 4 port 1 | 0/1 | Physical stand-off 177.29 mm, lateral 2.28 mm, orientation 0.78°. Useful geometric arrival, but native multi-view fit reprojection ≈3.28 px exceeds frozen 2 px limit. No coherent pose accepted; guarded no-progress stop. Do not call this simply a transport-route miss or relax the gate mid-test. |
| Trial 3, SC / three cards / SC port 1 | 0/1 | Rule stage no-progress; terminal stand-off 154.15 mm, lateral 204.17 mm, orientation 77.95°. Did not reach the correct acquisition corridor. Native geometry rejected. |

No scored harmful contact in these three runs. These are initial single repetitions, not final rates. They already invalidate extrapolation of familiar-board success to all public scenes.

### Correction to the previous acquisition screen bookkeeping

The prior `corrected_intended_perception_live20_v2` summary required `wrong_accepted_port is False` even on the safely rejected SFP episode, where no accepted port exists and the correct value is `None`. That silently required all 20 acquisitions despite a declared ≥90% per family/no wrong accept gate. A separate `corrected_intended_perception_gate_bookkeeping_audit_v2.json` preserves the original failed summary and records that 9/10 SC plus 9/10 SFP meet that numerical screen. The individual SFP failure and SC latency miss remain unchanged. This audit correction does not establish broad coverage, promote any model or alter the current frozen experiment. Repair the original summary code after this source-frozen batch, preserving both versions.
