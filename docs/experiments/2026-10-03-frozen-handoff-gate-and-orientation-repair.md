# Frozen handoff gate and bounded SC orientation repair

**Date:** 2026-10-03. **Decision:** the supervised transport-to-insertion handoff gate **fails**. The automatic switch, insertion-actor attachment, and RL remain disabled. Reserved final scenes were not opened. No active job was interrupted; all work here used saved development observations, followed by one bounded orientation-head continuation on GPU 1.

## Fixed gate on existing saved observations

The requested-port identity, SC position/reprojection rule, SFP cascade, SFP candidate verifier, and 0.5 identity thresholds were frozen before this review. Scored geometry and teacher plans were used **after inference** to mark known unsafe or valid stops; they were not actor, crop, or model inputs. `scripts/summarize_selected_port_frozen_review.py` aggregates the exact [machine decision](../../outputs/experiments/2026-10-02_two_family_perception/selected_port_frozen_gate_20261003.json).

| Frozen subtest | Result | Meaning |
| --- | ---: | --- |
| SC valid visual stops, selected-port identity accepted | **5/5** | Clear valid arrivals are retained in these five saved runs. |
| SC known wrong-posture/unsafe-route stop, selected-port identity accepted | **1/1** | Identity cannot authorize insertion. The previous visual position residual also passed at this stop. |
| SFP saved wrong-card cascade proposals rejected by V5 | **4/4** | Small two-reset replay improvement. |
| SFP saved correct cascade proposals retained by V5 | **69/69**, plus **21/21** on two new resets | The 69 share one reset; the new two resets produced no accepted wrong proposal. |
| SC provisional hidden triplets accepted by old keypoint plus identity conjunction | **0/20** | Single-reviewer, repeated-scene pilot; not certified hard-occlusion performance. |

The SC unsafe stop is decisive for the zero-false-accepted-proposal gate. It was **6.7°** from the scored teacher orientation while the running relative-orientation probe reported only about **0.1°** left. The request identity head accepted all 10 nearby observations around that stop. Four additional valid-stop replays were saved under `sc_spatial_identity_probe_v2/*_stop_replay_v1.json`; along with the previous valid and false replays they cover five valid and one false complete stop. A shadow pause/inspect/resume controller was **not** attached.

The three original grouped-validation false point accepts for SFP V5 were audited without retraining. [All three marked low-resolution images and scores](/home/nas/chmin/aic_archives/2026-10-03_sfp_offscreen_request_v1/correspondence_v5_balanced_native/grouped_point_error_audit_v1/audit.json) come from **one held-out `sfp_139` reset**, request `sfp_card_4_port_0`, candidate `sfp_card_4_port_1`. Their projected openings are 22–24 px apart at 288×256. Green/red markers in that folder show requested/neighboring labels. This is a same-card two-port confusion, unlike the four saved different-card/offscreen proposals; it does not establish an all-three-camera false handoff, since those three validation images were sampled at different times. Original-resolution card-pair evidence is a concrete future SFP ablation, not a reason to loosen the frozen gate.

## Can SC orientation be checked independently?

Three fixed alternatives were audited before any new fit:

1. **Training-mean TCP orientation:** a 3° guard rejects the known bad stop but also rejects **2/5 valid** stops. [Per-stop replay](../../outputs/experiments/2026-10-01_transport_reliability/sc_orientation_prior_stop_replay_20261003.json).
2. **Direction between the two SC openings:** even with *scored, exact* pair centers, a train-only pair-axis rule has **5.56° p95** orientation error on 21 held-out successful episodes; a train-mean orientation has 5.27° p95. This deliberately privileged [upper-bound diagnostic](../../outputs/experiments/2026-10-02_two_family_perception/sc_oracle_pair_orientation_ceiling_20261003.json) shows that one axis does not determine the full mating attitude. It cannot serve as a deployable shortcut.
3. **Existing frozen RGB port-orientation head:** combined with the train-only nominal TCP-in-port orientation, it would reject the known false stop (6.86° predicted mismatch) and retain **4/5 valid** stops at 3°. The fifth valid stop is 3.12° by that head. More decisively, on four new one-port development resets, synchronized near-port orientation p95 is **7.80°**. The original 2.56° SC p95 came from the older checkpoint-selection validation set and did not generalize to these one-port scenes. [Six-stop replay](../../outputs/experiments/2026-10-02_two_family_perception/sc_visual_orientation_stop_replay_20261003.json), [new-reset baseline](/home/nas/chmin/aic_archives/2026-10-03_sfp_offscreen_request_v1/sc_oneport_orientation_adaptation_v1/new_reset_synchronized_baseline_v1.json).

I visually inspected the known false and valid SC stop center images and example new one-port images. The requested blue connector is visible; the main error is orientation/posture judgment, not an absent target. These inspections do not certify every camera or occlusion case.

## Bounded one-port orientation continuation

**Hypothesis:** the orientation head had seen too few one-port SC scenes and could improve if it received explicit one-port examples while replaying original SC and SFP groups. **Bounded test:** 600 updates; pass only if new reset-grouped near-port orientation p95 ≤**3°** and a frozen 3° stop replay rejects the one known bad stop while retaining all five known good stops. I also checked that original SC/SFP validation did not degrade materially (reference tolerances 3.06°/4.20° p95). The tolerances and six-stop rule are development choices, not a calibrated final safety standard.

`scripts/train_sc_oneport_orientation_adaptation.py` initializes from the frozen RGB orientation head and fits that head on **196** native one-port camera triplets plus the original grouped training replay. The original locator/visual encoder is frozen. Eight one-port training reset configs were requested, but only **seven supplied usable synchronized, projected-in-frame triplets**; `train_p1_d` supplied none. All four new development groups were excluded from fitting. The original 319-episode source and the new one-port resets have disjoint scene hashes; no transition-random split. The original SC/SFP validation data were never fitted. One-port labels are scored port orientation, supplied for supervision only. The model continues to receive RGB-derived features, task code, and measured TCP at inference.

| Orientation result | Frozen head | One-port continuation V2 |
| --- | ---: | ---: |
| Original SC validation p95 (2,468 visible rows) | **2.564°** | **2.562°** |
| Original SFP validation p95 (1,774 visible rows) | **3.698°** | **3.749°** |
| Four new one-port resets, near 100 mm, synchronized p95 (149 triplets) | **7.803°** | **4.648°** |
| Six saved SC stops, valid retained at 3° | 4/5 | **5/5** |
| Six saved SC stops, known false rejected at 3° | 1/1 | **1/1** |

The continuation clearly helps, but it **fails its independent 3° accuracy gate**. The four-reset p95 is dominated by `development_p0_b` (all-frame p95 **4.91°**) and `development_p1_b` (**3.71°**); the other two groups are 2.18° and 1.55°. The six-stop pass is only a small, threshold-selected development diagnostic, so it cannot override the larger independent error result.

The first continuation V1 used the center camera's measured TCP for all three camera poses on rows whose saved images could be up to 150 ms apart. I fixed this construction bug to use each camera's causal measured TCP and repeated **the same 600-update run** as V2. The metrics barely changed (synchronized near p95 4.648° → 4.648°), so that bug is documented but does not explain the remaining gap. Separately, the original per-camera 1 s label subsampling caused apparent 150 ms cross-camera pairs on two development resets. I rebuilt labels from the **existing scored streams** at all timestamps and compared both checkpoints on the same 227 truly synchronized projected-inframe triplets. No new simulation was collected. [V1 summary](/home/nas/chmin/aic_archives/2026-10-03_sfp_offscreen_request_v1/sc_oneport_orientation_adaptation_v1/summary.json), [V2 summary and checkpoint](/home/nas/chmin/aic_archives/2026-10-03_sfp_offscreen_request_v1/sc_oneport_orientation_adaptation_v2_camera_pose_correct/summary.json), [fixed four-reset evaluation](/home/nas/chmin/aic_archives/2026-10-03_sfp_offscreen_request_v1/sc_oneport_orientation_adaptation_v2_camera_pose_correct/new_reset_synchronized_eval_v1.json), [fixed six-stop replay](/home/nas/chmin/aic_archives/2026-10-03_sfp_offscreen_request_v1/sc_oneport_orientation_adaptation_v2_camera_pose_correct/frozen_stop_replay_v1.json), and [machine decision](../../outputs/experiments/2026-10-02_two_family_perception/selected_port_orientation_repair_decision_20261003.json).

### Reproduction

Run from the repository root with GPU 1; the V2 `summary.json` records the exact checkpoint and counts:

```bash
.pixi/envs/default/bin/python scripts/train_sc_oneport_orientation_adaptation.py \
  --labels outputs/experiments/2026-10-02_two_family_perception/projection_labels/rows.jsonl \
  --sc-predictions outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/sc_predictions.jsonl \
  --sfp-predictions outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/sfp_predictions.jsonl \
  --visual-features outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/visual_features.npy \
  --oneport-train-glob '/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_v1/train_*/projection_state_v2/rows.jsonl' \
  --oneport-development-glob '/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_v1/development_*/projection_state_v2/rows.jsonl' \
  --locator outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/best.pt \
  --initial outputs/experiments/2026-10-02_two_family_perception/orientation_visual_v2/best.pt \
  --calibration configs/hierarchical_recovery/sc_visual_standoff_20261001.json \
  --output /home/nas/chmin/aic_archives/2026-10-03_sfp_offscreen_request_v1/sc_oneport_orientation_adaptation_v2_camera_pose_correct \
  --updates 600 --device cuda:1
```

The two additional exact-label runs used `scripts/build_sc_original_rgb_projection_audit.py <scored_stream.json> --stride-s 0 --output <scene>/projection_all_exact_20261003`. `scripts/evaluate_sc_visual_orientation_new_resets.py` scored the fixed four-scene list at at most 80 evenly spaced in-frame triplets per group. `scripts/replay_sc_visual_orientation_on_stops.py` produced the matched six-stop comparison. The rootless Gazebo scene captures were from the previous SC identity experiment; this continuation ran no Gazebo or Isaac training job. Script compilation, machine JSON validation and `git diff --check` passed.

## Decision and next bounded repair

Keep the handoff gate **open**. The V2 head is a useful diagnostic checkpoint, not a deployed posture verifier. Do not attach the insertion actor, start RL, or open final scenes from its six-stop result. The SC false-stop problem is now narrowed: request identity and 3D position can be correct while orientation is wrong; a one-port-aware RGB head improves but misses 3° on some new reset configurations. The SFP residual errors are adjacent ports on the same card, not the earlier different-card case.

Next test a **requested-port original-resolution crop orientation head** as a separate, fixed-budget ablation. Use an RGB-only coarse proposal to crop the *original 1152×1024 image before downsizing*; keep global three-view context, train orientation against scored port frames on complete reset groups with both one-port SC identities and SFP neighboring ports, and reject occluded/ambiguous crops rather than assigning a precise pose. No scored port position may choose an evaluation crop. Budget at most **12 new targeted SC one-port training resets plus two development resets, four GPU-hours of fitting/replay, and one day of image review**. A collection is justified only if an audit confirms the existing seven usable training groups do not span the needed board poses. Stop if the independent new-reset SC near-port p95 stays above **3°**, any known wrong stop is accepted, clear-stop retention drops below **95%**, SFP neighbor confusion persists, or complete live p95 latency exceeds **300 ms**. Compare against frozen V2 on the same groups. Do not collect general episodes or repeat the same low-resolution head sweep without a new hypothesis.
