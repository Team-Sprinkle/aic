# SFP observed rail crop: bounded continuation

**October 4, 2026. Decision: keep the transport anchor gate open.** This is an offline continuation of the [middle-card identity repair](2026-10-04-sfp-middle-card-identity-repair.md). No actor, insertion policy, RL, or final scene was run.

## Question and observation-only method

The structured locator still confused neighboring cards in two requested middle-card episodes. I tested whether an approximate image crop around the **requested rail slot** would expose enough original-image detail to locate that card's port. The crop center is calculated from a frozen RGB prediction of both ports on another visible card, their estimated separation, the requested card and port code, measured TCP, fixed camera calibration, and the toolkit's 40 mm rail spacing. The true requested opening is read **after** crop placement only to score the proposal. A later crop refiner sees the original JPEG crop, camera identity, and requested code. It does not receive true geometry.

This depends on the source-card pair being correctly identified and the rail model holding across changed card positions. The 480×480 crop is only a coarse transport anchor; it does not establish insertion-grade 0.5 mm pose accuracy.

## Matched saved-episode results

The same 50 complete training resets and ten complete development resets from the balanced SFP set were used. Each saved time has three original 1152×1024 camera images. The strict source-pair rule required three views and ≤5 px reprojection error for **both** ports. A fixed, more permissive 12 px proposal rule was then checked without retraining the refiner or changing its final three-view ≤5 px acceptance.

| Crop proposal, label-only audit | Strict 5 px: train | Strict 5 px: development | 12 px: train | 12 px: development |
| --- | ---: | ---: | ---: | ---: |
| Proposed / sampled times | 267/692 | 59/136 | 435/692 | 102/136 |
| True target inside all three crops | 261/267 | 59/59 | 423/435 | 95/102 |
| Episodes with any suitable crop | 41/50 | 9/10 | 50/50 | 10/10 |
| Episodes with suitable crop by 3 s | 26/50 | 7/10 | 42/50 | 9/10 |

Under the 12 px rule, ten of the twelve training proposal misses had the true target outside at least one source image; two had an in-frame target outside the crop. On development, one of seven proposal misses was offscreen and six had an in-frame target outside the crop. A valid crop proposal is therefore useful but not a proof of correct target inclusion. These counts use simulator projection **only for audit**.

The first crop training attempt failed before training because a projected crop center could fall beyond an image boundary. Its `fit_v1` artifact is preserved. I fixed image clipping and black padding, then ran one 1,200-update GPU-2 fit on strict-rule training proposals: 267 times across 43 train scene groups. The selected update-800 checkpoint was chosen on 59 development proposals from nine groups. Its strict-rule result was **37/59 accepted**, zero accepted wrong requested ports, and **2.70 mm** accepted 3D p95. The all-view pixel p95 was **168.8 px**, showing unresolved bad predictions that final agreement rejects. The update-1,200 alternative accepted one wrong port and was rejected. This checkpoint was selected on the ten development starts and is not independent evidence.

Keeping that checkpoint frozen and changing only proposal tolerance to 12 px gave **59/102 accepted**, zero accepted wrong ports, and **3.98 mm** accepted 3D p95 on the same development set. The crop model alone acquired an anchor in nine episodes eventually. A read-only union of the structured, selected-only, old all-port, and crop predictions acquired one in **9/10 episodes by 3 s and 10/10 eventually**, with zero accepted wrong ports and zero >10 mm disagreements between coaccepted estimates on those saved times. The late episode is card 3/port 0: first acceptance **3.086 s** after its first sampled observation. A strict two-model consensus gave only 6/10 early episodes. The union is an exploratory diagnostic; accepting whichever model fires first has not been validated on fresh scenes and may create wrong-port stops.

## Why this is still blocked

The crop fit and model selection touched these development starts. Fresh, complete non-final Gazebo resets are needed to measure wrong-port acceptance, early coverage across all ten request codes, and complete live inference p50/p95/p99. The earlier operational gate remains zero wrong accepted ports, at least 9/10 early requests including middle cards, accepted coarse position p95 ≤20 mm, and full live p95 <300 ms. Even if the anchor passes, transport must reach the selected-port neighborhood safely before any perception-to-insertion switch. No autonomous full insertion is established.

An offline GPU-2 profile processed 20 saved three-camera observations through JPEG decoding, RGB resizing, crop extraction, and **all four frozen networks** sequentially. It measured **176.8 ms p50 / 201.8 ms p95 / 207.8 ms p99** on an RTX A6000. This leaves roughly 98 ms at p95 for camera transport, ROS, geometry, actor, controller, and scheduling, but those costs were not included or measured together. It therefore cannot pass the complete 300 ms live gate; it does show that the four-network option is relatively costly.

Rootless Docker initially failed to create a new container at **200/200 keys**. Later, kernel accounting fell to **199/200** before the bulk stopped-container cleanup. I removed 25 audited stopped September AIC containers; the count stayed 199/200, but a minimal one-at-a-time `docker run --rm` now succeeds. The old GPU-0 Isaac process remains running. This is enough to retry sequential fresh Gazebo starts, yet the quota remains fragile and the complete live timing check has not been run. The next bounded experiment is **ten new non-final reset groups**, one for each SFP request code, with the four predictors frozen. Run the prior structured locator and crop pipeline as a shadow proposal only; compare acquisition, wrong identity, >10 mm disagreement, and latency. Stop immediately if any accepted wrong port appears, if early coverage remains <9/10, or if p95 ≥300 ms. If the frozen union fails, test one observation-only source-pair/slot selection repair based on the recorded first-failure view; do not simply widen the threshold again on the same ten scenes. Keep final scenes sealed.

## Commands and artifacts

```bash
LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}" \
LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16" \
.pixi/envs/default/bin/python scripts/audit_sfp_observed_rail_crop.py \
  --candidates /home/nas/chmin/aic_archives/2026-10-04_sfp_observed_rail_crop_v1/frozen_train_candidates.jsonl \
  --labels /var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/anchor_labels_v1 \
  --output /home/nas/chmin/aic_archives/2026-10-04_sfp_observed_rail_crop_train_source12_v1 \
  --split train --source-reprojection-px 12

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=scripts:aic_utils/lerobot_robot_aic \
LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}" \
LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16" \
.pixi/envs/default/bin/python scripts/train_sfp_observed_rail_crop.py \
  --labels /var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/anchor_labels_v1 \
  --train-proposals /home/nas/chmin/aic_archives/2026-10-04_sfp_observed_rail_crop_train_v1 \
  --development-proposals /home/nas/chmin/aic_archives/2026-10-04_sfp_observed_rail_crop_v1 \
  --init outputs/experiments/2026-10-02_two_family_perception/conditional_locator_v3/best.pt \
  --output /home/nas/chmin/aic_archives/2026-10-04_sfp_observed_rail_crop_fit_v2 \
  --updates 1200 --minutes 90

.pixi/envs/default/bin/python scripts/summarize_sfp_anchor_union.py \
  --model structured /home/nas/chmin/aic_archives/2026-10-04_sfp_structured_slot_v1/predictions_step1400.jsonl \
  --model selected /home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/selected_anchor_native576_v1/development_predictions.jsonl \
  --model allport /home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/selected_anchor_native576_v1/frozen_multiport_v5_development_predictions.jsonl \
  --model crop /home/nas/chmin/aic_archives/2026-10-04_sfp_observed_rail_crop_source12_v1/frozen_crop_model_predictions.jsonl \
  --output /home/nas/chmin/aic_archives/2026-10-04_sfp_observed_rail_crop_source12_v1/frozen_four_model_union.json

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=scripts:aic_utils/lerobot_robot_aic \
LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}" \
LD_PRELOAD="$PWD/.pixi/envs/default/lib/libjpeg.so.8:$PWD/.pixi/envs/default/lib/libpng16.so.16" \
.pixi/envs/default/bin/python scripts/profile_sfp_four_anchor_offline.py \
  --labels /var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/anchor_labels_v1 \
  --proposals /home/nas/chmin/aic_archives/2026-10-04_sfp_observed_rail_crop_source12_v1 \
  --allport outputs/experiments/2026-10-02_two_family_perception/joint_locator_native_sfp_v5_heatmap/best.pt \
  --structured /home/nas/chmin/aic_archives/2026-10-04_sfp_structured_slot_v1/best.pt \
  --selected /home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/selected_anchor_native576_v1/best.pt \
  --crop /home/nas/chmin/aic_archives/2026-10-04_sfp_observed_rail_crop_fit_v2/best.pt \
  --output /home/nas/chmin/aic_archives/2026-10-04_sfp_observed_rail_crop_source12_v1/frozen_four_model_offline_latency.json \
  --samples 20
```

The strict train/development proposal audits are in `2026-10-04_sfp_observed_rail_crop_train_v1` and `2026-10-04_sfp_observed_rail_crop_v1`; the 12 px audits are in the corresponding `_train_source12_v1` and `_source12_v1` directories under `/home/nas/chmin/aic_archives/`. The latter contains `frozen_crop_model_predictions.jsonl`, `frozen_crop_model_summary.json`, `frozen_four_model_union.json`, and `frozen_four_model_offline_latency.json`. `2026-10-04_sfp_observed_rail_crop_fit_v2` holds `best.pt`, `history.json`, `summary.json`, and all six development prediction files. [Machine summary](2026-10-04-sfp-observed-rail-crop.summary.json).
