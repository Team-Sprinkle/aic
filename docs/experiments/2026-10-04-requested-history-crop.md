# Requested-port coarse perception with causal camera history

**Status:** bounded SFP diagnostic complete; no handoff, refiner connection, or actor promotion. Final scenes remain sealed.

## First two completed variants

The first model pooled each earlier camera image to one vector, then predicted a requested-port 2D crop center independently in each current view. It trained for the full bounded **1,000 updates** on 50 episodes, selected update 700 on ten development episodes, and replayed ten previously inspected stress episodes. With only its in-frame scores, it proposed 125/136 development crops and made **21 wrong-port centers**; 82/125 placed the target within 60 px in all views. On stress scenes it proposed 566/581 and made **282 wrong-port centers**. A strict, observation-only ≤5 px three-view reprojection check removed accepted wrong centers on this set, but reduced early acquisition to **1/10** on both development and stress scenes. The old frozen selected-only model reached **3/10** early on the same stress scenes. Replacing historical images with the current image left early acquisition at 1/10: history added later accepted episodes (6/10 versus 4/10 on stress) but did not repair initial identity. [Exact matched replay](</home/nas/chmin/aic_archives/2026-10-04_sfp_requested_history_crop_v1/matched_geometry_audit.json>).

The targeted shared-3D variant predicts one requested-port point in the TCP frame and projects it into all cameras, so its crops agree geometrically by construction. Its first attempt failed at update 100 due to a projection API shape error; that failed output directory is preserved as `2026-10-04_sfp_requested_history_3d_v1/`. I fixed the point-transform call and reran from scratch in `2026-10-04_sfp_requested_history_3d_v2/`. The 1,000-update selected checkpoint still had **15 wrong-port centers** in 134/136 accepted development rows and only **86/134** three-view crops with the 20 px margin. On the previously inspected stress scenes it had **313 wrong-port centers** in 573 accepted rows, 219/573 good crops, and 37.21/158.88 mm accepted 3D p50/p95. Its in-frame score accepted nearly everything, so geometric consistency alone is not an identity check. [Exact summary](</home/nas/chmin/aic_archives/2026-10-04_sfp_requested_history_3d_v2/summary.json>).

Both variants fail the zero-wrong and crop-coverage gates. The next bounded, distinct test retains a **4×5 spatial grid** from each historical camera image rather than pooling away card-row layout. It still predicts only the requested port. If it fails, stop this representation branch and inspect visual slot labels/data coverage before another fit.

## Spatial-history and targeted five-card continuation

The 4×5 spatial-history model completed 1,000 updates and selected update 700 on the same ten development episodes. It proposed 120/136 crops with **22 wrong-port centers** and 82/120 good 160 px crops. On the ten previously inspected stress scenes it made **370 wrong-port centers** in 565 proposals and only 170/565 good crops. A fixed ≤5 px three-view reprojection check reduced this to 26 accepted stress observations with zero wrong centers, but early acquisition was still just **3/10**, equal to the old selected-only head. Replacing history with the current image reduced its accepted stress episodes from 5/10 to 4/10 and early acquisitions from 3/10 to 2/10. Thus history helped some later continuity but did not clear requested-port identity. [Spatial fit summary](</home/nas/chmin/aic_archives/2026-10-04_sfp_requested_spatial_history_crop_v1/summary.json>) · [matched ablation](</home/nas/chmin/aic_archives/2026-10-04_sfp_requested_spatial_history_crop_v1/matched_geometry_audit.json>).

A data audit found a specific fit/stress mismatch. The 50-episode initial training set contains only **13 five-card episodes**; for card 1/port 1 and card 4/port 1 it contains **zero** five-card examples, while the stress check uses five cards in five of ten scenes. The earlier native-resolution perception archive has ten additional five-card *training* episodes, one for each requested card/port code, with **720 image triplets**. Their reset hashes do not overlap the original 50/10 split or the ten stress scenes. I added precisely these existing rows under `/home/nas/chmin/aic_archives/2026-10-04_sfp_requested_history_extra5_labels_v1/`, preserving the original split and their source manifest. One matched spatial-history retrain with this targeted addition completed at the 1,000-update cap on one GPU. No new simulator data were collected. The inspected stress set is not an independent promotion set.

That retrain completed at update 1,000 on **60 training episodes**. It improved the raw stress replay from 370 wrong centers in 565 proposals to **227 in 577**, and all-view crop coverage from 170/565 to **305/577**. This supports the specific five-card coverage hypothesis, but remains unsafe. With the same fixed ≤5 px camera-geometry check, it accepted 37/581 stress observations, zero wrong on this set, and acquired **4/10** episodes early versus the old selected-only model's **3/10**. The histories added one early episode over a current-frame ablation (4/10 versus 3/10), far below the 9/10 gate. [Fit summary](</home/nas/chmin/aic_archives/2026-10-04_sfp_requested_spatial_history_extra5_v1/summary.json>) · [matched geometry/history audit](</home/nas/chmin/aic_archives/2026-10-04_sfp_requested_spatial_history_extra5_v1/matched_geometry_audit.json>).

One final decoder-specific ablation added explicit $x,y$ coordinates to each output pixel, so the task-conditioned context could distinguish identical-looking card faces by image location. It used **the same 60 episodes, architecture depth, 1,000-update budget, development/stress groups and geometry check**. The selected update 700 reduced raw wrong stress centers to **173/572**, but only 303/572 crops had the target within a 60 px radius in all three views. After the fixed geometry check, just **23/581** stress observations passed with zero wrong; early acquisition fell to **3/10**, versus 3/10 for the old selected-only head. Replacing historical frames with current frames also gave 3/10 early. Explicit coordinates therefore did not make history useful enough for handoff. [Fit summary](</home/nas/chmin/aic_archives/2026-10-04_sfp_requested_spatial_history_xy_extra5_v1/summary.json>) · [matched ablation](</home/nas/chmin/aic_archives/2026-10-04_sfp_requested_spatial_history_xy_extra5_v1/matched_geometry_audit.json>).

## Interpretation and next upstream gate

The image history used here is causal, simultaneous across three cameras, and aligned by measured TCP motion. It begins at the **first recorded frame of each teacher observation episode**. Initial TCP-to-target distance is about **0.18 m median** in both fit/development and stress sets. One inspected five-card stress first center image visibly shows only part of the row; later approach images cannot recover an earlier, wider view that was never recorded. Other hard scenes do show the full row, so missing visual context is only one cause. The fit also has only 60 reset groups after targeted addition, and the label used for the score says *projected inside the frame*, not *visually identified*. The model can therefore assign a high in-frame score to a wrong face; its score cannot be a handoff confidence.

The candidate interface is still correct for future work: one requested-port crop center per current camera plus an abstain decision, with original-resolution 160 px crops passed to a pose-only refiner. **No candidate here clears the coarse identity gate**, so no refiner, PoseInsert actor, autonomous handoff, or RL was connected. A next experiment requires evidence that the requested numbered slot is actually identifiable in a wider/safer observation view, and true visual-presence or identity labels for ambiguous/occluded frames. A bounded observation-pose comparison should collect or reuse such views for each card-count/request combination, annotate whether the target face is identifiable independently of model predictions, and stop if correct identity is not visible across cameras. Only then retrain a requested-port model and test on new episode-grouped nonfinal starts. Do not use the inspected ten-scene stress replay as a promotion split.

## Hypothesis

The earlier coarse model emitted ten possible SFP port locations per view. Card slots look alike, so accepting a plausible but wrong channel is unsafe. A requested-port model can use public card/port code, all three cameras, and earlier views from the same episode to locate only the requested aperture. Earlier, wider views may identify a numbered slot after the current close view loses the row.

The current test takes current images plus the latest available causal frames at about 1.5, 3 and 6 seconds earlier. A shared frozen ResNet18 image encoder produces features; a small temporal/camera attention block combines them with measured relative TCP motion and the requested code. It predicts one current-image 2D crop center and one in-frame score for each camera. No all-port output head runs in this candidate. A 160 px native-resolution crop would be centered at each accepted prediction. The fine refiner is **not** in this initial gate: first test whether crops contain the requested opening and reject wrong ports. The crop center comes from images, task code and measured robot motion only; simulator port coordinates are training labels and posthoc scoring data.

## Data, gates and caveats

The fixed first fit uses 50 complete training episodes from `anchor_labels_v1`; the 10 complete validation episodes select the checkpoint. Ten already inspected fresh episodes are a separate stress replay, **not an independent promotion set**. Evaluation is grouped by episode and reset; no history crosses a reset. Same-scene rows are used only within the same split. The fit is capped at 1,200 updates and 60 minutes on one GPU. Its running command uses 1,000 updates.

Report proposal coverage, accepted wrong-port centers, early episode acquisition within 3 s, and whether the true projected opening lies within 60 px of each proposed 160 px crop center, leaving 20 px nominal margin. A useful result requires zero wrong-port crop centers, at least 9/10 early development episodes, and target coverage in at least 90% of accepted three-view crops. Only after that should a frozen candidate be tested on genuinely new non-final resets and then connected to a refiner, with complete p95 inference below 300 ms. A failure means inspect concrete wrong/missed scenes and test a distinct repair rather than relaxing thresholds on the same episodes.

The existing projection flag states that the simulator opening falls inside image bounds. It is **not a true visibility annotation** under cable/tool/card occlusion. The model's flag is therefore called an in-frame or crop-eligibility score, not calibrated visual confidence. Wrong-port identity and crop coverage are scored with labels only after predictions. The image encoder starts from the existing SFP selected-port checkpoint; this is a bounded feature test, not a fair architecture comparison against every historical model.

## Artifacts

Script: [train_requested_history_crop.py](../../scripts/train_requested_history_crop.py). Output directory: `/home/nas/chmin/aic_archives/2026-10-04_sfp_requested_history_crop_v1/`. It stores the feature cache and exact row/image manifest, best checkpoint, development and stress predictions, training history and summary. Preserve the original all-port comparisons for provenance: [corrected ten-scene replay](2026-10-04-sfp-fresh-anchor-shadow.md#corrected-all-channel-crop-replay-october-4).

The [machine-readable comparison](2026-10-04-requested-history-crop.summary.json) records every variant's metrics and archive path. The final targeted run is reproducible with:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=scripts:aic_utils/lerobot_robot_aic \
LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}" \
.pixi/envs/default/bin/python scripts/train_requested_history_crop.py \
  --labels /home/nas/chmin/aic_archives/2026-10-04_sfp_requested_history_extra5_labels_v1 \
  --stress-labels /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/labels_all_ten \
  --init /home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/selected_anchor_native576_v1/best.pt \
  --feature-cache /home/nas/chmin/aic_archives/2026-10-04_sfp_requested_spatial_history_extra5_v1 \
  --output /home/nas/chmin/aic_archives/2026-10-04_sfp_requested_spatial_history_xy_extra5_v1 \
  --spatial-history-xy --updates 1000 --minutes 60 --batch-size 8

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=scripts:aic_utils/lerobot_robot_aic \
LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}" \
.pixi/envs/default/bin/python scripts/audit_requested_history_crop.py \
  --labels /home/nas/chmin/aic_archives/2026-10-04_sfp_requested_history_extra5_labels_v1 \
  --stress-labels /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/labels_all_ten \
  --run /home/nas/chmin/aic_archives/2026-10-04_sfp_requested_spatial_history_xy_extra5_v1 \
  --selected-stress /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/all_ten_model_scores_all_channels_v2/selected.jsonl
```
