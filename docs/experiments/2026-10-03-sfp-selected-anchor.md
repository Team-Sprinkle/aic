# SFP selected-port spatial anchor: bounded development result

**October 3, 2026. Decision: do not attach to transport.** This experiment tested whether a requested-card/port image keypoint could supply the missing coarse SFP route anchor. The earlier route-stage probe had 48.2 mm p95 waypoint error against a 20 mm gate. The new model takes one original 1152×1024 wrist RGB image (resized to 576×512 inside the network), camera identity, family bit, and requested card/port code. It predicts one opening pixel and a projected-in-frame score. Three independently predicted pixels are triangulated using measured TCP and fixed camera calibration. Simulator port geometry is an **offline label and scoring reference only**; no true opening pose or true crop center is an inference input.

## Fixed data and bounded fit

The manifest samples every ~1 s from **50 scored-successful balanced SFP-to-NIC CheatCode train resets and ten separate scored-successful development resets**. It contains 692 training and 136 development three-camera observations across all ten card/port requests. Alternate present ports were sampled as counterfactual training requests. Scene/reset hashes do not cross the split. This is SFP only; it does not update SC perception or any actor. The old October 2 task-conditioned ResNet18 checkpoint initialized the model; it predates these October 3 resets. Training used one GPU (GPU 2), batch eight, at most **2,000 updates**, and selected the development checkpoint at update 2,000. The active older GPU-0 Isaac job was untouched. Final scenes stayed sealed.

The fixed acceptance rule was all three per-camera in-frame scores ≥0.5 and triangulation reprojection disagreement ≤5 px. The originally stated numeric gate was zero *accepted* wrong identities and accepted 3D position p95 ≤20 mm. On 136 development observations, the new model accepted **51 (37.5%)**, with **0 accepted wrong identities** and **2.65 mm accepted-position p95**. Thus that narrow numeric gate passes. It is **not an operational pass**: only **4/10 complete development episodes** ever acquired an anchor, all for outer cards 0 and 4. The middle-card episodes 1, 2 and 3 had **0/6** acquisitions. Only those same four episodes acquired an anchor within the first three seconds. One fixed requested-port mean baseline had **69.71 mm p95** across all 136 rows, so the successful accepted positions use real visual information, but they cover the wrong subset of the task.

| Requested card | Accepted / development rows | Per-camera pixel error p95 |
| --- | ---: | ---: |
| 0 | 25/27 | 3.4 px |
| 1 | 0/27 | 165.8 px |
| 2 | 0/26 | 168.8 px |
| 3 | 0/28 | 284.1 px |
| 4 | 26/28 | 4.8 px |

The frozen earlier **all-port** locator, evaluated with the *same images, task requests, TCP and three-view rule*, accepted **53/136**, also with zero accepted wrong identities and 4.69 mm accepted-position p95. It acquired an anchor in **8/10** episodes, including both card-1 requests, but almost never for cards 2–3 and inconsistently for card 4. This comparison suggests the new single-output head lost useful card structure; simply increasing global RGB resolution did not solve middle-card correspondence. It is not a proof that multiport prediction is the correct final architecture. The old model uses a 288×256 resized view and had zero additional training updates here.

A final **no-training** card-row diagnostic ran the frozen multiport model on every development triplet and saved all ten candidate projections. Its three-view-consistent detections remain sparse for cards 2–3: port 0/1 yielded **6/5 card-2** and **2/2 card-3** detections across 136 frames, versus 57/58 for card 0. Fitting a straight row from at least two reliable same-port candidates did not rescue any additional episode: 54/136 estimates, zero nearest-port errors, and eight episodes with an estimate. A one-card fixed-offset extrapolation is also weak in these layouts: on the **training scenes themselves**, the residual p95 of a mean card-1→card-2 offset is **28.7 mm**, and card-1→card-3 is **38.5 mm**; the lateral rail shifts vary by ~17–19 mm standard deviation. These are optimistic in-sample diagnostics, not held-out success figures. They rule out treating a constant card-row spacing as a sufficient 20 mm coarse anchor. The target itself, or a richer observation-derived slot geometry, must be identified.

One inspected card-2 start has multiple visually similar silver card faces in the wide views. Its selected projection is geometrically inside each image, but this flag does **not** certify that the opening is visually identifiable: another card or the gripper can cover it. The training loss currently treats every projected-in-frame opening as a positive keypoint, including potentially hidden ones. That label mismatch is a plausible contributor; we have **not** measured a certified occlusion rate, so it is not established as the sole cause. The large cross-camera errors and failed triangulation are measured, not inferred.

## Stopping decision and next targeted repair

The 2,000-update cap is exhausted. **No actor continuation, live rollout, PoseInsert handoff or RL training followed.** The numerical gate needs an explicit coverage condition before future promotion: every requested card/port must acquire a correct coarse anchor early enough to guide transport, and accepted wrong-port count must remain zero. A position p95 computed on only accepted easy cases cannot stand alone.

The next targeted repair must make middle-card identity observable, rather than guess a hidden card from average slot spacing. Audit actual identifiable visibility in all three original wide views over the failed episodes; then use explicit port-face/card-slot association on clear views or a safe active-view step when the requested target is hidden. The current projection label cannot supervise true visibility. Predeclare a coverage gate of **at least 9/10 complete development episodes acquiring a correct anchor within three seconds, including both requests for each middle card**, with zero accepted wrong identities and 3D p95 ≤20 mm; if the original start does not show the target, measure the acquisition time after a safe view-seeking step separately. Bound the next supervised repair to one architecture and one ≤2,000-update fit on one GPU, with a frozen no-training all-port control. The ten development resets selected this checkpoint, so a successful repair needs fresh non-final episode-grouped starts before live promotion. Only then consider a capped actor continuation. Complete live p95 must remain <300 ms; it was not measured for this unpromoted model.

## Reproduction and artifacts

```bash
.pixi/envs/default/bin/python scripts/build_sfp_balanced_anchor_labels.py \
  --transport-labels /var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/balanced_merged_sfp_v1/sfp_labels.jsonl \
  --output /var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/anchor_labels_v1

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=scripts:aic_utils/lerobot_robot_aic \
  .pixi/envs/default/bin/python scripts/train_sfp_balanced_anchor.py \
  --labels /var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/anchor_labels_v1 \
  --init outputs/experiments/2026-10-02_two_family_perception/conditional_locator_v3/best.pt \
  --output /home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/selected_anchor_native576_v1 \
  --updates 2000 --minutes 120
```

The training shell also set `LD_LIBRARY_PATH` to `.pixi/envs/default/lib` and preloaded its `libjpeg.so.8` and `libpng16.so.16` for host OpenCV compatibility. [Manifest summary](/var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/anchor_labels_v1/summary.json), [fit summary](/home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/selected_anchor_native576_v1/summary.json), [coverage and old-model comparison](/home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/selected_anchor_native576_v1/coverage_audit.json), [all-candidate frozen replay](/home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/selected_anchor_native576_v1/frozen_all_candidates.jsonl), [fixed-row-prior ceiling](/home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/selected_anchor_native576_v1/card_row_prior_ceiling.json), and the same artifact directory's `best.pt`, `history.json`, `development_predictions.jsonl`, `frozen_multiport_v5_development_predictions.jsonl`, and `frozen_multiport_v5_development_summary.json` preserve all results. [Tracked machine decision](2026-10-03-sfp-selected-anchor.summary.json).
