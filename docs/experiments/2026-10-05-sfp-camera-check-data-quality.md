# SFP camera-check data quality pilot (October 5)

## Decision

The request-expanded **geometry manifest is usable as a candidate locator dataset**, after a strict camera/state timestamp filter. It is **not yet a camera-check or safe-handoff dataset**. No visibility, selected-slot identifiability, or safe-pause head was trained from it. Projection into a camera is not proof that the requested opening is visible. The existing zero-wrong-port / at-least-9-of-10 early-request gate remains open; final scenes stay sealed.

This is the first bounded data-quality stage of the [approved plan](../answers/2026-10-05-high-quality-coarse-camera-data-plan.md), not a new model result.

## What was generated

[`scripts/build_sfp_camera_check_dataset.py`](../../scripts/build_sfp_camera_check_dataset.py) pairs every original 1152×1024 three-camera SFP moment with all ten public card/port request codes. It records reset group, images, request, physical port presence, projected selected-port pixel/depth where present, and TCP-to-port distance. The last three fields are **training/scoring geometry only**. The deployable model would receive images, public request and measured robot state; it would not receive this geometry. The review queue strips geometry so an RGB reviewer can inspect unmarked images first. The 166-row pilot is stratified across near/far, absent, partly/outside projections and reset roles.

The first immutable output, [`v1`](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v1/summary.json>), revealed 92 old-source triplets with a camera more than 10 ms from the recorded state, including two with different camera simulation stamps. The builder now excludes them. The corrected [`v2 summary`](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v2/summary.json>) contains:

| Role | Original triplets retained | Request-image rows | Reset groups |
| --- | ---: | ---: | ---: |
| Train | 1,664 | 16,640 | 66 |
| Development | 119 | 1,190 | 10 |
| Harder diagnostic | 581 | 5,810 | 10 |
| Saved actor-route diagnostic | 110 | 1,100 | 3 |

The timestamp filter excluded 73 stale train, 17 stale development, and 2 mixed-camera train triplets. The training rows include 1,460 physically absent requests, 2,372 partially projected/outside requests, 7,019 projected-all-three requests farther than 100 mm, and 5,789 projected-all-three requests within 100 mm. These are **request-image pairs, not independent episodes**. The 100 mm split is a sampling/diagnostic bin, not a safe-stop threshold.

The [`structural audit`](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v2/structural_audit.json>) checked all request IDs, exactly ten requests per triplet, disjoint train/evaluation reset groups, absent-port consistency, pilot geometry sealing, and 120 randomly sampled triplets / 360 decoded RGB files. It passed: all sampled files were RGB 1152×1024, maximum state/image age 10 ms, zero camera simulation-time spread, no broken files or request expansion errors. This sampling does not certify every image or visual visibility.

## Original-image spot check

I first inspected 12 unmarked v1 pilot triplets across training, development, harder diagnostic and actor-route sources and wrote [blind notes](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v1/blind_visual_audit_notes.json>). Only then did I inspect [four projected-point crops](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v1/projected_port_crops_audit.jpg>). For example, the far `a9a7d1184c870279dd58` request has markers plausibly on the same opening in all three cameras. The near `c9833b3feea6823cfde1` request is mathematically in all three frames, yet the center and right marker falls on the black gripper housing, and the left point is at the gripper edge. The near `c2496dfd03a5442f483b` target is similarly obscured in the side views. Calling these “visible” would be wrong. The v1 blind sheets are [1](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v1/blind_visual_sheet_1.jpg>), [2](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v1/blind_visual_sheet_2.jpg>) and [3](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v1/blind_visual_sheet_3.jpg>).

After creating v2, I also inspected [eight fresh unmarked selected samples](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v2/visual_audit_selection.json>) in [sheet 1](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v2/blind_visual_sheet_1.jpg>) and [sheet 2](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v2/blind_visual_sheet_2.jpg>). The rows show plausible one-to-five-card diversity, absent requests with other ports plainly visible, similar neighboring openings, and near gripper occlusion. [Notes](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v2/blind_visual_audit_notes.json>) conservatively keep exact visual identifiability **unlabeled** when the unmarked still does not establish numbered slot identity. This is a spot check, not an inter-reviewer agreement study or a full 166-item review.

A follow-up selected 18 projected-all-three near-port triplets from **18 distinct episodes** (the pilot had only one eligible saved-actor episode, so the target of 20 could not be filled under the one-per-episode rule). I inspected their [unmarked three-view sheets](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v2/near_port_blind_sheet_1.jpg>) before opening the [projected-point crop sheets](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v2/near_port_projected_crops_1.jpg>). Conservative single-reviewer [spot-check notes](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v2/near_port_projected_point_spotcheck.json>) mark **10/18 triplets and 14/54 camera points** with an obvious foreground obstruction at the projected keypoint. This deliberately says *keypoint obscured*, not that the entire requested aperture is invisible; the overlay-assisted spot check is **not** a certified visual label set. It does show why three projected-in-frame flags cannot serve as positive camera-check labels.

Saved scored command streams with measured controller TCP and wrench samples exist for 56/66 training resets and all 10 development resets; the other 10 training source rows do not name such a stream. The present manifest has no causally joined force/orientation/route-state pause target. Geometric distance alone cannot supply it. A physical safe-pause label must be built separately from a frozen standoff/orientation/force rule and checked against both teacher and off-route actor traces before it is used.

## Initial v2 gate before sparse-card continuation

Do **not** fit a visibility or pause classifier to the projection flag or the 100 mm distance bin; doing so would reproduce the earlier false-accept problem with more data. The next bounded action is an original-resolution, blind review of the **166-item pilot**, with `clear / partial / not identifiable / uncertain` per camera and a separate requested-slot identity decision. Reveal projections only after each judgment, adjudicate unclear middle-slot examples, and report counts/agreement by category. This full review has **not** yet been completed. The independent [ten-scene board-slot audit](2026-10-04-sfp-board-slot-reference.md) already found only 4/10 exact occupancy maps, with all five sparse layouts uncertain; the new samples do not overturn that warning. Stop this data branch if requested numbered ports remain unidentifiable in a large share of full-row early views; then test a safe observation posture/route change rather than more labels of the same view. Only if reviewed labels are reliable should the 500–1,000 triplet review and one ≤2,000-update, one-GPU matched candidate proceed. The eventual new-start identity gate remains zero accepted wrong ports, at least 9/10 early correct acquisitions, complete p95 latency below 300 ms, and no final-scene access.

## Sparse-card coverage follow-up (October 5)

A source-count audit found that the 66 v2 **training** resets contain only three-, four- and five-card scenes (1,198 / 264 / 202 retained triplets). The one- and two-card samples in v2 came from saved actor-route diagnostics, not training. This is a concrete coverage gap for the sparse layouts that defeated the independent slot audit. The 166-item original queue was rendered into 28 unmarked contact sheets under the v2 archive; only its first six sheets were inspected during this continuation. The complete blind annotation is still open.

I ran a four-case, train-source-only ordinary CheatCode pilot in rootless Gazebo. One-card slot 0/port 0 and slot 4/port 1 each scored the official full-insertion tier 3 = 75. Changing two verified two-card scenes to request the **second** card yielded only tier 3 = 38.28 and 25, so the declared ≥3/4 expansion gate **failed at 2/4**. I did not run the planned 18-case expansion. The failed re-requests remain diagnostic only. Two matched controls using each scene's original card 0/port 1 task both scored tier 3 = 75. This suggests the redirected teacher route was the problem; the controls do not establish that every second-card target is feasible.

The [four successful source selection](</home/nas/chmin/aic_archives/2026-10-05_sfp_sparse_camera_success_selection_v1/collection_manifest.json>) links the two one-card pilots and two two-card controls. Its [causal native projection derivative](</home/nas/chmin/aic_archives/2026-10-05_sfp_sparse_camera_success_projection_v1/summary.json>) has 2,030 triplets; the strict 10 ms filter retained **2,026**. [`v3`](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v3/summary.json>) combines these with the existing train source: 3,690 train triplets / 70 complete reset groups / 36,900 request-image rows. Development (10 groups), harder diagnostic (10) and actor-route diagnostic (3) are unchanged. The new training rows provide 20,260 sparse-card request pairs: 2,022 present and 8,088 absent in one-card scenes; 4,060 present and 6,090 absent in two-card scenes. They are highly correlated within four episodes, not 20,260 independent resets.

The [v3 structural audit](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v3/structural_audit.json>) passed all 4,500 distinct triplets, all ten requests per triplet, complete reset separation and 120 decoded triplets / 360 original RGB files. A separate [scene-file audit](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v3/sparse_scene_independent_audit.json>) compared every new request to the `nic_rail_i.entity_present` flags and task code in the saved YAML: **0 mismatches across 20,260 rows**. I visually inspected selected raw 1152×1024 one- and two-card center views, including [a present one-card request](</home/nas/chmin/aic_archives/2026-10-05_sfp_sparse_camera_train_v1/rollouts/cards1_slot4_port1/frames/1791218741707084622_22750000000_center.jpg>) and [a two-card approach with the other card outside the current center view](</home/nas/chmin/aic_archives/2026-10-05_sfp_sparse_camera_controls_v1/rollouts/cards2_slot4_original_card0_port1/frames/1791219236364399831_17750000000_center.jpg>). The images have plausible board, plug, housing and orange cable, but a scene-file presence label is still **not** a camera visibility or numbered-slot identity label. The [v3 blind queue](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v3/blind_review_queue.jsonl>) includes 12 sparse-card training examples out of 120 train examples. No visibility or safe-pause head was fitted from v3.

The next gate was a bounded **blind RGB identifiability review**, explicitly including sparse slots, absent requested slots, near-port plug occlusion and off-route actor views. A four-case original-resolution continuation below did not establish requested numbered-slot identity. The previously completed [wider-observation experiment](2026-10-04-sfp-wide-observation-gate.md) had already tested a +80 mm row-direction / +40 mm rise view at a 180 mm standoff: it made more card faces countable, but the strict blind review identified the requested numbered face in **0/4** scenes and one route failed its measured-pose gate. Do **not** repeat that view or collect routine data of the same type. The next distinct bounded repair is to test whether an observation-only detector can locate the board's fixed rail origin and occupied slots from existing native RGB. If that reference cannot be recovered on the saved grouped scenes, design a different view or external visual cue. Only after identity/visibility labels pass review should pause labels be constructed from causal orientation, force and route state, followed by a ≤2,000-update grouped-development fit. The four reserved final IK scenes stay sealed.

### Four original-resolution blind cases after v3

I reviewed four previously unseen v2 pilot cases (`37`, `44`, `53`, `59`) in all three original 1152×1024 cameras through anonymous image aliases, and saved [judgments before scoring](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v2/native_blind_aliases_20261005/blind_annotations_before_scoring.json>). Visible SFP faces were present, but I could not confidently tie the requested task-code number to one physical rail slot in any of the four; all four identity/visibility judgments remain **uncertain**, not negative. [Posthoc scoring](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v2/native_blind_aliases_20261005/scored_annotations.json>) found two requested ports physically present and two absent; one present target projected into all three cameras and the other into two. Projection remains no proof of visibility. This is one conservative reviewer and four cases, **not** completion of the 166-case review or an inter-reviewer agreement result. The twelve new sparse-card review sheets were also inspected, but their episode identities had already been seen during data validation, so they are **not counted as blind evidence**. Combined with the earlier independent 4/10 exact board-slot audit and the completed 0/4 wider-view strict review, this is sufficient to keep the visibility/identity training gate closed and avoid a routine 166-case annotation pass until the slot-reference hypothesis changes the images or labels.

Sparse collection and rebuild commands:

```bash
.pixi/envs/default/bin/python scripts/collect_sfp_sparse_camera_teacher.py
.pixi/envs/default/bin/python scripts/collect_sfp_sparse_camera_teacher.py --controls \
  --output /home/nas/chmin/aic_archives/2026-10-05_sfp_sparse_camera_controls_v1
.pixi/envs/default/bin/python scripts/build_sfp_native_projection_labels.py \
  --selection /home/nas/chmin/aic_archives/2026-10-05_sfp_sparse_camera_success_selection_v1 \
  --camera-calibration configs/hierarchical_recovery/sc_visual_standoff_20261001.json \
  --sfp-calibration configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json \
  --output /home/nas/chmin/aic_archives/2026-10-05_sfp_sparse_camera_success_projection_v1
.pixi/envs/default/bin/python scripts/build_sfp_camera_check_dataset.py \
  --train /home/nas/chmin/aic_archives/2026-10-04_sfp_coarse_combined_66_labels_v1 \
  --extra-train /home/nas/chmin/aic_archives/2026-10-05_sfp_sparse_camera_success_projection_v1 \
  --development /home/nas/chmin/aic_archives/2026-10-04_sfp_coarse_combined_66_labels_v1 \
  --diagnostic /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/labels_all_ten \
  --actor outputs/experiments/2026-10-05_spatial_transport_prerequisite_v1/actor_labels \
  --output /home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v3 \
  --max-state-image-age-ms 10
.pixi/envs/default/bin/python scripts/audit_sfp_camera_check_dataset.py \
  /home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v3 \
  --image-audit-count 120 --max-age-ms 10 --max-spread-ms 10
```

## Commands

```bash
.pixi/envs/default/bin/python scripts/build_sfp_camera_check_dataset.py \
  --train /home/nas/chmin/aic_archives/2026-10-04_sfp_coarse_combined_66_labels_v1 \
  --development /home/nas/chmin/aic_archives/2026-10-04_sfp_coarse_combined_66_labels_v1 \
  --diagnostic /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/labels_all_ten \
  --actor outputs/experiments/2026-10-05_spatial_transport_prerequisite_v1/actor_labels \
  --output /home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v2 \
  --max-state-image-age-ms 10

.pixi/envs/default/bin/python scripts/audit_sfp_camera_check_dataset.py \
  /home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v2 \
  --image-audit-count 120 --max-age-ms 10 --max-spread-ms 10
```

The [`v2 blind queue`](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v2/blind_review_queue.jsonl>), [`sealed projected points`](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v2/pilot_sealed_geometry.jsonl>) and [`all-request manifest`](</home/nas/chmin/aic_archives/2026-10-05_sfp_camera_check_dataset_v2/all_requests.jsonl>) are preserved with source-file hashes. No Gazebo process, new rollout, GPU training or active job was interrupted for this stage.
