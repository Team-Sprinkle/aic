# SFP slot-local occupancy ablation (October 5)

## Hypothesis and fixed comparison

The preceding fixed-rail model localized its rail endpoints to 12.0 px median on grouped development, yet classified exact five-bit occupancy on only 5/10 reset groups and made three high-confidence occupied predictions for empty rails. Its occupancy head pooled each entire camera into one vector. In the original RGB examples, the occupied/empty evidence is local to each groove and card front. This bounded ablation asks whether **reading RGB features near each RGB-predicted rail slot** improves occupancy without simulator geometry at inference.

[`scripts/train_sfp_board_reference.py`](../../scripts/train_sfp_board_reference.py) now offers `--head slot_local`. It shares the earlier pretrained ResNet18, the same three 1152×1024 source images resized to 576×512, the same RGB rail-endpoint and in-frame heads, train/validation split, scene-balanced sampler, labels, losses, optimizer, update count and development-loss checkpoint rule. A small shared slot head samples nine positions around each of five interpolated slots in each view from the ResNet layer-3 map. **The crop/sample positions come only from this model's predicted endpoints**, with occupancy gradients stopped at those endpoint coordinates. Per-slot local features, a compact global context vector, camera support and slot index predict the five occupied bits. Board pose, scene YAML, projected rails and occupancy truth remain training/scoring labels only.

One-GPU budget: at most **2,000 updates or 120 minutes** on GPU 1. GPU 0's unrelated long-running Isaac process is untouched. A one-update smoke test at `/home/nas/chmin/aic_archives/2026-10-05_sfp_slot_local_smoke_v1` passed forward/backward, checkpoint and grouped evaluation. Select only by grouped-development loss, then score the three sparse autonomous actor groups. The **predeclared gate** is at least **8/10** exact occupancy maps on the ten grouped-development resets, at least **2/3** exact sparse actor maps, **zero** predicted-occupied probabilities ≥0.8 on truly empty development/actor rails, and no worse development endpoint p90 than the prior 26.4 px. Report exact per-scene predictions, original-RGB overlays, endpoint errors and full three-camera inference latency. If this fails, do not extend routine training or promote to requested-port identity; investigate a distinct observation cue and reviewed visual labels. Final scenes stay sealed and handoff remains disabled regardless of the outcome of this auxiliary test.

## Command

```bash
CUDA_VISIBLE_DEVICES=1 .pixi/envs/default/bin/python scripts/train_sfp_board_reference.py \
  --labels /home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_feasibility_v3 \
  --init /home/nas/chmin/aic_archives/2026-10-04_sfp_structured_slot_v1/best.pt \
  --output /home/nas/chmin/aic_archives/2026-10-05_sfp_slot_local_probe_v1 \
  --updates 2000 --minutes 120 --batch-size 4 --workers 4 --head slot_local
```

Results and decision will be appended after the fixed run. The earlier pooled-head run is preserved at `/home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_probe_v1`.

## Independent nonfinal stress set, frozen before scoring

After the fit passed its original auxiliary gate, I located ten separate nonfinal SFP scene groups with 581 original-resolution triplets at `/home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/labels_all_ten`. Their reset hashes have **zero overlap** with the 70 training, ten checkpoint-development and three actor-diagnostic groups above. They were used in older perception shadow experiments but not to train or select either board-reference head. Score both frozen heads on exactly these ten groups without changing weights or thresholds. The follow-up generalization gate is at least **8/10** exact early occupancy maps, zero confident false occupied rails, and an improvement of at least two exact maps over the pooled head. This gate is diagnostic only; requested-port identity and automatic handoff remain separate.

## Results

The slot-local fit completed **2,000 updates** at the cap in 688 s on GPU 1. The predeclared lowest-development-loss rule selected **update 1,200**. A later update temporarily fell to 6/10 maps before recovering; [history](</home/nas/chmin/aic_archives/2026-10-05_sfp_slot_local_probe_v1/history.json>) and [checkpoint](</home/nas/chmin/aic_archives/2026-10-05_sfp_slot_local_probe_v1/best.pt>) preserve this instability. No actor-diagnostic reset was used for selection. The existing Isaac process on GPU 0 was not interrupted.

| Episode-grouped split | Pooled whole-image occupancy | Slot-local occupancy | Local gate |
| --- | ---: | ---: | --- |
| Ten checkpoint-development resets, 3–5 cards | 5/10 exact maps; 3 confident false occupied rails | **10/10 exact; 0 false** | Pass: ≥8/10 and 0 false |
| Three saved sparse autonomous actor routes, 1–2 cards | 3/3 exact; 0 false | **3/3 exact; 0 false** | Pass: ≥2/3 and 0 false |
| Ten additional nonfinal teacher-route resets, 581 frames | 5/10 exact; 2 false | **10/10 exact; 0 false** | Pass: ≥8/10, 0 false, ≥2-map gain |

The extra ten reset hashes are disjoint from the checkpoint manifest and were not used to tune this head, although older perception work had already inspected these scenes. The [independent label manifest](</home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_independent10_labels_v1/summary.json>) retained 581/581 camera triplets with the ≤10 ms synchronization check. The [pooled](</home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_independent10_pooled_v1/summary.json>) and [local](</home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_independent10_slot_local_v1/summary.json>) heads were frozen and evaluated on exactly those images. This is a strong auxiliary occupancy result, not ten autonomous safe arrivals.

Rail endpoint p90 error against **posthoc CAD projections** was 25.7 px for the local head on checkpoint development, slightly better than the pooled head's 26.4 px, so the localization part of the original gate passed. On the extra ten groups, the local head's median/p90 was 15.8/35.3 px versus 15.1/39.7 px pooled. On the three sparse actor routes, local median/p90 was 16.0/86.2 px; late clipped/occluded views account for a long tail. [Development overlays](</home/nas/chmin/aic_archives/2026-10-05_sfp_slot_local_probe_v1/development_score>), [actor-route overlays](</home/nas/chmin/aic_archives/2026-10-05_sfp_slot_local_probe_v1/actor_diagnostic_score>) and [extra-scene overlays](</home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_independent10_slot_local_v1/score>) were inspected in original-size RGB. Red model endpoints generally lie near green projected rail labels, but projection-in-frame is not visibility. The [three-scene occupied/empty label review](</home/nas/chmin/aic_archives/2026-10-05_sfp_slot_local_label_review>) marks every rail and supports the YAML occupancy ordering in those examples.

The matched offline [latency benchmark](../../scripts/benchmark_sfp_board_reference.py) includes reading three JPEGs, resize, transfer and synchronized GPU forward on GPU 1. Over 200 repetitions, the local head was **71.7 ms p50 / 83.8 ms p95 / 87.2 ms p99**, versus **71.6 / 90.9 / 94.9 ms** pooled. Measurement artifacts: [local](</home/nas/chmin/aic_archives/2026-10-05_sfp_slot_local_probe_v1/latency_three_camera.json>) and [pooled](</home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_probe_v1/latency_three_camera.json>). This is below 300 ms for the board-reference model alone. It excludes ROS transport, the transport actor, precise selected-port refiner and controller; complete deployment latency is not established.

```bash
.pixi/envs/default/bin/python scripts/audit_sfp_board_reference_projection.py \
  --source /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/labels_all_ten \
  --camera-calibration configs/hierarchical_recovery/sc_visual_standoff_20261001.json \
  --sfp-calibration configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json \
  --output /home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_independent10_labels_v1 \
  --max-state-image-age-ms 10

CUDA_VISIBLE_DEVICES=1 .pixi/envs/default/bin/python scripts/evaluate_sfp_board_reference_checkpoint.py \
  --checkpoint /home/nas/chmin/aic_archives/2026-10-05_sfp_slot_local_probe_v1/best.pt \
  --labels /home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_independent10_labels_v1 \
  --output /home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_independent10_slot_local_v1
```

## Requested-port correspondence limit

I tested whether the new rail predictions could reject wrong candidates from a **frozen existing all-port detector** on the same ten saved nonfinal teacher routes. The first read-only rule compared port mouths with rail anchors directly in each image. It was flawed: the different 3D locations of port mouths and rail anchors create perspective/parallax; even correct ports projected a median **2.87 rail slots away** along that image axis. Its 3/10 to 2/10 early-accept drop is archived at `/home/nas/chmin/aic_archives/2026-10-05_sfp_rail_candidate_readonly_v1`; it is not a valid port verifier.

A second read-only causal rule triangulated the model's two rail endpoints into the measured TCP base frame, checked the fixed 160 mm rail span and reprojection consistency, retained up to three **past** valid references, and compared the frozen port proposal with the resulting 3D rail index. No true board or port pose entered that rule. It rejected all **27 wrong accepted rows** from the old detector, but a valid board reference was available in only **2/10** reset groups; early correct acceptance fell from **3/10** old detector to **1/10**. [Per-frame results](</home/nas/chmin/aic_archives/2026-10-05_sfp_rail_candidate_3d_readonly_v1/summary.json>) and [`scripts/audit_sfp_rail_candidate_3d.py`](../../scripts/audit_sfp_rail_candidate_3d.py) preserve this failure. A [reason-coded rerun](</home/nas/chmin/aic_archives/2026-10-05_sfp_rail_candidate_3d_diagnosis_v2/rows.jsonl>) attributes 392/581 frames to rail reprojection error above the frozen 5 px cutoff, 185 to too few two-view endpoints, two to implausible rail span, and only two to accepted new references. Thus occupancy is solved on these scenes, while multi-view endpoint geometry remains too inconsistent for this strict 3D gate. It was a replay of teacher routes and did not test autonomous stops.

**Decision:** Retain the slot-local checkpoint as the better **occupied-slot representation** for future experiments. Do not connect either read-only rail candidate rule or this head to automatic handoff. The next bounded repair must preserve board reference across changing/occluded views or learn requested-slot/port correspondence directly from the local slot representation, then demonstrate zero accepted wrong ports with high early coverage on new nonfinal autonomous starts. True selected-opening visibility, safe-pause labels and complete live p95 remain open. No insertion actor or RL was enabled and final scenes remain sealed.
