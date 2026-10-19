# Spatial transport prerequisite on saved autonomous RGB (October 5)

The next proposed transport experiment was a request-conditioned spatial visual actor, compared with a matched frozen-feature control. Its declared order was to check selected-target identity first, then early action error, then at most three new nonfinal autonomous starts. I ran the identity prerequisite on the **three already saved, independent one/two-card candidate routes** from the October 5 paired-data actor comparison. No model was trained or deployed in this audit, no simulator scene was launched, and final scenes remained sealed.

`scripts/build_sfp_autonomous_native_labels.py` sampled the saved measured TCP state and synchronized original 1152×1024 center/left/right JPEGs at 1 Hz from simulation seconds 4–40. It found 110 triplets across three reset groups; one triplet was excluded for camera age. The requested port and all other port projections came from the saved scene configuration and fixed calibration **only for posthoc scoring**. `scripts/audit_sfp_spatial_actor_prerequisite.py` then ran two frozen RGB models on those triplets: a request-conditioned selected-port locator and a structured all-slot locator whose public task code selects one output. Each used image, camera identity, measured TCP for triangulation, and the requested card/port code. Three-view in-frame scores ≥0.5 and reprojection ≤5 px determined acceptance; actual port geometry was never an inference input.

| Frozen visual model | Accepted / 110 | Accepted wrong port | Early correct target (≤3 s from first sample) | Accepted 3D p95 |
| --- | ---: | ---: | ---: | ---: |
| Selected-port locator | 38 | 0 | 3/3 episodes | 3.00 mm |
| Structured slot locator | 64 | 0 | 3/3 episodes | 4.23 mm |

The selected locator first accepted at 0–2 s after the first sampled frame, depending on the episode. This is encouraging **only for these sparse-card routes**. The matching transport actor still reached **0/3** safe requested-port buffered arrivals. Thus the missed arrivals here cannot be explained simply as the requested port being impossible for these frozen vision models to find. The replay does not establish that the actor's own pooled features encode or use the correct target, nor that its route would improve if given one of these estimates. Its p95 image-to-port error is also far too large for insertion; this is a coarse transport cue.

The full identity prerequisite **does not pass**. The prior ten-scene, one-per-request five-card/sparse diagnostic had only 3/10 early anchors for the selected-only head despite zero accepted wrong identities; the direct original-resolution row detector reached 4/10 early there. The direct/rail crop pipeline on three earlier five-card autonomous routes found a usable selected-port crop in 2/3 and accepted only 1/3. The current three new starts cover card 0/port 0, card 1/port 0 and card 3/port 0, with only one or two cards installed. They do not cover both ports, five-card occlusion, sparse slot ordinal ambiguity, or the prior wrong-neighbor cases. A zero-wrong finding on 38 or 64 accepted observations in three correlated episodes is not a safety bound. **Do not start the spatial actor fit or live gate under the declared experiment until the full selected-target acquisition prerequisite is repaired and checked on episode-grouped, harder development scenes.** Learned handoff, insertion and RL remain off.

I also inspected unmarked original [center](</home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/live_card4_port1_cards4_frozen_history/frames/1791050326448383521_6000000000_center.jpg>) and [right](</home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/live_card4_port1_cards4_frozen_history/frames/1791050326479595082_6000000000_right.jpg>) views at 6 s from the earlier four-card/card-4-port-1 actor failure. The right image clips part of the card row, and the center image does not show an unambiguous slot-zero reference. This particular visual inspection supports the slot-reference ambiguity hypothesis; it does **not** establish that the target was physically hidden or that a human could never identify it from a longer sequence. The posthoc projected target locations were not overlaid during this review.

I then scored the previously frozen ten-request original-image replay at each episode's first seven samples (0–3 s). Both card-0 requests acquired the requested port early; **all six card-1/2/3 requests acquired zero early anchors**, even though their requested openings projected inside all three camera frames in **7/7** early samples apiece. Card-4/port-0 projected into all views in only 3/7 samples and acquired none; card-4/port-1 acquired four. Geometric projection does not prove unoccluded visibility. However, the unmarked [center](</home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/teacher_card2_port0/frames/1791083983680671923_2700000000_center.jpg>) and [right](</home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/teacher_card2_port0/frames/1791083983707170903_2700000000_right.jpg>) images for the card-2/port-0 first sample show a full five-card row in center and much of it in right. At that instant the selected-head three camera in-frame scores were approximately 1.0, yet its triangulated point was 26 mm from the requested opening and failed the three-view acceptance rule. This is a concrete visible-row identity/geometry failure, rather than simply no port entering the cameras. [Per-request machine audit](../../outputs/experiments/2026-10-05_spatial_transport_prerequisite_v1/hard_scene_early_slot_audit.json).

The specific next repair is an observation-only full-board/slot identity representation on existing original RGB plus distance-sampled history, with explicit abstention when the board origin or requested slot is not visible. Review original unmarked autonomous early images before training to separate (1) visually clear target with wrong slot selection, (2) target present but ordinally ambiguous because board reference is clipped, and (3) target absent or occluded. Use simulator geometry only for training labels and posthoc scoring. A single bounded candidate versus the frozen selected-port locator should be selected on complete training/reset groups, then checked on **ten new nonfinal, one-per-request development starts**. Budget ≤2,000 updates on one GPU and at most ten new starts for the final grouped identity check; stop at any accepted wrong port or fewer than 9/10 early correct acquisitions. Do not open reserved final scenes or attach the actor if that gate fails. If visibility itself is absent in early views, revise the observation posture/route rather than fitting another crop head to the same images.

## Commands and artifacts

Both commands ran from the repository root in `.pixi/envs/default/bin/python`; the replay used GPU 1. The exact three `--run` paths are preserved in [`actor_labels/summary.json`](../../outputs/experiments/2026-10-05_spatial_transport_prerequisite_v1/actor_labels/summary.json). The label command was:

```bash
.pixi/envs/default/bin/python scripts/build_sfp_autonomous_native_labels.py \
  --run /home/nas/chmin/aic_archives/2026-10-05_sfp_paired_development_actor_v1/cards1_episode250_to_card0_port0_candidate \
  --run /home/nas/chmin/aic_archives/2026-10-05_sfp_paired_development_actor_v1/cards2_episode261_to_card1_port0_candidate \
  --run /home/nas/chmin/aic_archives/2026-10-05_sfp_paired_development_actor_v1/cards2_episode106_to_card3_port0_candidate \
  --camera-calibration configs/hierarchical_recovery/sc_visual_standoff_20261001.json \
  --sfp-calibration configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json \
  --output outputs/experiments/2026-10-05_spatial_transport_prerequisite_v1/actor_labels \
  --first-second 4 --last-second 40 --stride-second 1
```

```bash
CUDA_VISIBLE_DEVICES=1 .pixi/envs/default/bin/python scripts/audit_sfp_spatial_actor_prerequisite.py \
  --labels outputs/experiments/2026-10-05_spatial_transport_prerequisite_v1/actor_labels \
  --selected /home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/selected_anchor_native576_v1/best.pt \
  --structured /home/nas/chmin/aic_archives/2026-10-04_sfp_structured_slot_v1/best.pt \
  --output outputs/experiments/2026-10-05_spatial_transport_prerequisite_v1/frozen_rgb_identity_v2
```

The [machine summary](../../outputs/experiments/2026-10-05_spatial_transport_prerequisite_v1/frozen_rgb_identity_v2/summary.json) and per-frame selected/structured predictions are saved there. The first run of the audit is preserved as `frozen_rgb_identity/`; its scores match, while its boolean gate name overstated the three-scene result. The V2 summary explicitly distinguishes the limited sparse-scene subtest from the failed full ten-request prerequisite. Existing 256×288 cached training images cannot reproduce the original-resolution detector input; any later actor fit using these detector outputs must either retain corresponding original JPEGs or state the resulting modality mismatch.

The frozen harder-case audit used:

```bash
.pixi/envs/default/bin/python scripts/audit_sfp_early_slot_failures.py \
  --labels /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/labels_all_ten \
  --predictions /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/all_ten_model_scores_all_channels_v2/selected.jsonl \
  --output outputs/experiments/2026-10-05_spatial_transport_prerequisite_v1/hard_scene_early_slot_audit.json
```
