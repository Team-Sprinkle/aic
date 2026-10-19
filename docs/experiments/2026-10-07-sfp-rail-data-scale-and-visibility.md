# SFP rail endpoint: data scale versus label observability

This is a bounded follow-up to [the failed temporal board-reference gate](2026-10-07-sfp-temporal-board-reference.md). The question is whether 70 complete reset groups were too few, or whether projected CAD endpoints are sometimes not visually identifiable. It does not authorize automatic handoff, actor training or opening final scenes.

## Frozen protocol and decisions

1. Construct nested **20/40/70 complete-group** training subsets, balanced over the ten SFP requested card/port codes as far as the existing 70 groups allow. Do not select individual transitions. Keep the same ten untouched grouped development scenes, original 160×160 observation-centered crops, network, random seed, initialization, 2,000-update budget and checkpoint rule. The 70-group run already exists; fit only 20 and 40. This is a learning-curve diagnostic, not an independent test of a new architecture.
2. Compare the best development pixel median/p95 and unchanged causal temporal-board gate. Treat more of the *same projected labels* as promising only if the 20→40→70 p95 curve decreases consistently and by at least **20%** overall. The absolute 3/8 px and 9/10 board gates still apply. If the curve is flat or erratic, do not collect or train more of the same label by default.
3. Make a deterministic original-RGB development review packet with up to five distinct groups per camera/endpoint, showing the CAD projected point and broad image context but hiding model predictions. For each panel, mark whether a unique visual feature identifies that exact projected point, whether it is repeated/featureless/occluded, or whether the single reviewer is unsure. This is a **single-reviewer diagnostic**, not certified visibility ground truth and not the pending independent complete-public-state identity review. A mark on a feature does not alone establish absolute numbered slot identity.
4. If the learning curve helps and labels are visibly identifiable, the next experiment can add independent complete groups and test once on new nonfinal scenes. If labels are ambiguous, change the landmark or obtain an approved safe view exposing a fixed numbered origin before collecting more images. In neither case promote board anchoring from this replay alone.

The output and exact commands will be recorded below after completion.

## Result

The nested sets contained **7,129 / 9,742 / 16,145** labeled original-image crops from **20 / 40 / 70** complete training groups. The ten development groups and 557 projected-in-frame scored endpoints were identical throughout. With the same initialization, architecture, seed, 2,000 updates and development-p95 checkpoint selection:

| Training groups | Best checkpoint | Development median | Development p95 | Early accurate temporal board refs |
| ---: | ---: | ---: | ---: | ---: |
| 20 | step 1 | 11.85 px | 34.58 px | 1/10 |
| 40 | step 1 | 11.87 px | 34.67 px | 0/10 |
| 70 | step 1,250 | 9.92 px | 32.17 px | 3/10 |

The best 70-group p95 is only **7.0%** lower than the best 20-group p95, and 20→40 is not an improvement. The predeclared consistent ≥20% improvement condition fails; the absolute 3/8 px and 9/10 temporal gates fail by large margins. The 20/40 fits lowered training loss but their best held-out p95 was at the first update. One fixed subset order and random seed cannot prove that *no* larger, more diverse dataset would help. It does show that simply scaling these projected-label episodes from 20 to 70 groups did not put the model on a clear path to the needed accuracy. [Learning-curve plot](../../outputs/experiments/2026-10-07_sfp_rail_data_scale_v1/learning_curve.png) · [machine summary](2026-10-07-sfp-rail-data-scale-and-visibility.summary.json).

The nested subsets balanced the ten requested identities but were not perfectly balanced by card count: groups with 1/2/3/4/5 installed cards numbered **1/1/6/3/9** at 20 groups, **1/1/12/9/17** at 40, and **2/2/21/16/29** at 70. Thus the curve is evidence about the existing data distribution, not a claim that underrepresented sparse-card settings need no additional examples if a better landmark is later defined.

The deterministic review packet contained **30** development panels: five distinct groups for each camera/endpoint, eight unique groups total, with original full-image context plus a native 160×160 crop. The green mark showed the CAD projected endpoint; model predictions were hidden. In my **single-reviewer** assessment, **0** points were confidently identifiable as a unique, absolutely numbered rail-end visual mark; **23** were ambiguous/repeated or projected onto a featureless/obscured surface, and **7** were uncertain. [Both contact sheets, rubric and per-panel labels](../../outputs/experiments/2026-10-07_sfp_rail_data_scale_v1/visibility_review/). This does **not** establish a population visibility rate, and another reviewer may interpret the panels differently. It is enough to prevent treating these CAD projections as certified visible landmarks for a new scale-up. The separate independent complete-public-observation review remains pending.

**Decision:** stop collecting or fitting more episodes with this same endpoint label for now. The immediate repair is to define an actually observable fixed board origin, with explicit visibility/identity labels, or find a safe earlier observation exposing one. The prior +10° wrist inspection already failed its own position/identity gate. If an independently reviewed origin becomes available, build a new episode-grouped dataset and learning curve; reserve fresh nonfinal groups for its test. No board target association, transport pause, insertion, RL or final-scene test was enabled. The separate SFP safe-arrival gap remains 0/3.

Reproduction used:

```bash
.pixi/envs/default/bin/python scripts/make_sfp_rail_learning_curve_manifests.py \
  --rows /home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_feasibility_v3/rows.jsonl \
  --output outputs/experiments/2026-10-07_sfp_rail_data_scale_v1

# Run for N=20,40, replacing N in both paths; GPU 1, sequentially.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 \
  .pixi/envs/default/bin/python scripts/train_sfp_rail_crop_refiner.py \
  --labels /home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_feasibility_v3 \
  --train-predictions /home/nas/chmin/aic_archives/2026-10-06_sfp_slot_local_train_predictions_v1/predictions.jsonl \
  --development-predictions /home/nas/chmin/aic_archives/2026-10-05_sfp_slot_local_probe_v1/development_predictions.jsonl \
  --init /home/nas/chmin/aic_archives/2026-10-04_sfp_structured_slot_v1/best.pt \
  --train-groups-manifest outputs/experiments/2026-10-07_sfp_rail_data_scale_v1/train_groups_N.json \
  --output outputs/experiments/2026-10-07_sfp_rail_data_scale_v1/fitN

.pixi/envs/default/bin/python scripts/make_sfp_rail_visibility_review.py \
  --rows /home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_feasibility_v3/rows.jsonl \
  --predictions /home/nas/chmin/aic_archives/2026-10-05_sfp_slot_local_probe_v1/development_predictions.jsonl \
  --output outputs/experiments/2026-10-07_sfp_rail_data_scale_v1/visibility_review
```

`scripts/audit_sfp_temporal_board_reference.py` replayed each `fitN/development_predictions.jsonl` with the unchanged rail manifest, source labels, calibration and rule from [the original experiment](2026-10-07-sfp-temporal-board-reference.md), writing `temporalN/`. The 70-group score reused the prior frozen output, with no retraining. Exact selected group IDs, checkpoints, predictions, histories, causal decisions and review labels remain in the linked output tree.

After verification, the six generated `train_crops.npy` and `development_crops.npy` arrays from the 20/40/70 fits were deleted because `/data1` had only 5.5 GB free. They are deterministic caches of archived source JPEGs and can be recreated with the recorded commands; checkpoints, metrics, exact manifests, predictions, figures and source images were preserved.
