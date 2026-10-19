# SFP visible board-origin candidate audit

The prior endpoint labels were CAD rail positions, often on plain or repeated surfaces. This bounded audit asks whether either of the two lower *outer board-outline corners* offers a more visibly grounded fixed origin. It uses existing original-RGB nonfinal development images only and does not train a model or command a robot.

The earlier [outline feasibility probe](2026-10-05-sfp-board-rail-reference-probe.md) found that all four board corners never project into one wrist camera view. Its candidate outline points come from the **collision envelope**, not a verified visual-mesh corner; this is a serious possible failure. Therefore inspect only lower corners 0/1, plus full-image context, and do not assume a green CAD projection coincides with a visible physical corner.

Generate a deterministic packet of five distinct development groups per camera/corner, with original full image and 160×160 native crop around the *CAD projection for audit only*. A single reviewer grades each panel as `unique_visible`, `ambiguous_or_occluded`, or `uncertain`, asking whether a distinct physical board corner is actually present at that pixel and can anchor absolute board order. Geometry-chosen crops are **not** deployable and cannot enter autonomous crop selection. The provisional visual feasibility gate is at least **24/30 unique-visible** panels, including examples from all three cameras and both corners, with no systematic projected-point/physical-corner offset. If it fails, stop this collision-envelope-corner branch before any model fit. A pass would still require independent review, observation-only crop selection, grouped new-scene tests and full latency before handoff.

The complete-public-observation independent review remains pending. Final scenes remain sealed; transport safe arrival and handoff remain open.

If the collision-envelope projection is systematically displaced from a visible silhouette corner, run one **privileged-window diagnostic**: threshold dark board pixels in a 220×220 original-image window centered on the CAD projection, remove thin floor lines, extract polygonal contour vertices and select the nearest vertex. This is only a way to inspect whether a physical corner could supply a new label. Count it as a useful candidate only if the red selected vertex visibly lies on the same physical board corner in **at least 24/30** fixed panels, spanning all three cameras and both lower corners. A failed result stops this simple silhouette-vertex route. Even a pass would need an observation-only full-image locator and a new grouped test before any actor use.

## Frozen image audit result

The [30-panel packet](../../outputs/experiments/2026-10-07_sfp_outline_corner_review_v1/) sampled five distinct development groups per camera/corner, five unique groups overall; the two corners were visible by *projection* in every selected panel. Original full images and native crops revealed the collision-box label often landed on a connector, floor line or background while the board silhouette and nearby bolt were offset. In my single-reviewer grading, **0/30** projected points were confidently an exact, uniquely identifiable physical outline corner, **17/30** were visibly offset/occluded, and **13/30** were uncertain. The fixed **24/30** visual gate fails. This is not an independent annotation or a population visibility estimate. The previous statement that all four outline corners never fit a single wrist frame remains true; focusing on two lower corners did not solve exact label alignment. [Selected native-size inspection](../../outputs/experiments/2026-10-07_sfp_outline_corner_review_v1/selected_native_corners.jpg) shows the green CAD point relative to the real silhouette.

The one deterministic [contour-vertex probe](../../outputs/experiments/2026-10-07_sfp_outline_corner_review_v1/contour_probe/) proposed a pixel in all 30 privileged windows, but by the same single-reviewer check only **4/30** were clearly the intended physical board corner; seven were uncertain and 19 selected a connector, bolt, floor-line intersection or another edge. The 24/30 contour gate fails. A candidate can be close in pixels yet be the wrong object. These crops were centered with CAD geometry **only for diagnosis** and are not autonomous proposals.

**Decision:** stop the collision-envelope-corner and simple silhouette-vertex branches before any model training. The board's visible bolts/corners may still support a future method, but they need **visual-mesh-aligned or manually verified physical landmark labels**, plus an observation-only full-image locator and independent review. This audit provides no deployable board origin and does not resolve selected-port identity or transport's 0/3 safe-arrival gap. No new Gazebo episode, automatic handoff, insertion, RL or reserved final-scene test occurred.

## Distinct mesh-screw label check (predeclared)

The board's `base_visual.glb` contains an M3 screw mesh with lower-corner screw-head centers near **(-0.145, -0.2075, 0.010) m** and **(+0.145, -0.2075, 0.010) m** in board coordinates. These are distinct from the collision-envelope corners at x=±0.150, y=-0.214, z=0.012 m. Before any fit, project the two screw centers through the same saved board poses, measured TCP and calibrated cameras. Inspect the *same 30 frozen panels*, with mesh-screw projections marked separately from the old envelope corner. Count a panel as usable only if the projected point lands within approximately **5 native pixels** of the visible center of the intended screw head and its board-side identity is clear from the full image. Require **at least 24/30** usable panels, representation from all three cameras and both screws, and early visibility in at least **9/10** complete development groups. Failure stops this screw-label branch before model training. A pass would permit only a grouped train/development screw detector feasibility fit; it would not certify autonomous crop selection or handoff.

The same 30-panel projection comparison gives a **19.7 px median** displacement between old collision-envelope corners and actual mesh-screw centers. In my single-reviewer inspection, **21/30** mesh-screw labels appeared within about 5 px of a distinct visible screw with clear board-side context, **four** were uncertain, and **five** were hidden by cards in the right camera's screw-0 view. This fails the original all-channel 24/30 gate. It is still a distinct improvement in label meaning: center screw 1 and right screw 1 were clear in all five sampled groups, while center screw 0 was clear in four and left screw 0 in three. [Projected mesh-screw contact sheets](../../outputs/experiments/2026-10-07_sfp_visual_screw_projection_v1/). These counts are single-reviewer and not independent certification.

Because the failure is concentrated in one physically blocked channel, run **one new, separately identified channel-specific feasibility audit** before any model fit. Use the same ten development groups, choose the first projected-in-frame observation within three seconds for `center/screw0`, `left/screw0`, `center/screw1`, and `right/screw1`, and review the original full image plus native crop. Require at least **9/10 groups** with screw 0 visible in center *and* left and screw 1 visible in center *and* right. A projected-in-frame mark is not a positive visibility label. If this fails, stop the two-camera-per-screw representation; a single-camera temporal method would need a separate control and gate.

**Channel-specific image result:** the [40-panel early packet](../../outputs/experiments/2026-10-07_sfp_visual_screw_early_review_v1/) has 39 projected-in-frame panels. The far-side screw-1 mark is visually clear in both center/right views in most groups; right is out of frame in one. The screw-0 center view is frequently covered by a NIC card or connector, so the strict two-camera-per-screw gate is at most **1/10** groups under my conservative single-reviewer classification, far below 9/10. Stop that representation before fitting it. Projection alone would have falsely counted hidden screw-0 points as visible.

The saved conservative per-panel single-reviewer file records **19 confidently visible screw-1 panels**, five uncertain screw-0 panels, 15 occluded/indistinct screw-0 panels and one screw-1 out-of-frame panel. It yields **0/10** groups with both screws independently confirmed in the specified two cameras. The earlier “at most 1/10” statement allowed the uncertain panels; either treatment fails the gate. These labels are not independent ground truth.

**Separate one-screw upper-bound check, before any detector fit:** screw 1 may provide an origin even when screw 0 is hidden. Using one **true mesh-screw 3D position only as a privileged diagnostic**, estimate each requested port from a train-only fixed board-layout offset and the circular-mean board yaw over the 70 training groups. Compare on the ten grouped development scenes with (a) true scene yaw to isolate local card/port variation, and (b) a train-mean screw location to isolate the value of the screw observation. Require at least **9/10** requested openings within 10 mm and zero instances where the estimate is closer to a different installed opening. If that fails, do not train a one-screw locator merely because the screw is visually visible. The oracle-screw diagnostic is not deployable and cannot select crops or stop transport.

**One-screw upper bound failed:** on the ten complete development groups, *true screw position plus train-only mean yaw and per-request local offset* yielded **22.62 mm median / 39.59 mm p95** requested-opening error, **0/10** within 10 mm and **three** estimates closer to a different installed opening. Giving the diagnostic the **true scene yaw** still left **15.02/22.18 mm** median/p95, only **3/10** within 10 mm and **two** wrong-opening nearest neighbors. Replacing the screw observation with the train-mean screw position worsened to **37.53/63.83 mm** and five wrong-opening neighbors. Thus locating a single bolt would help board translation, but one bolt plus a fixed layout and even perfect yaw does **not** identify the target opening reliably in these randomized scenes. Train-local port offsets themselves vary across complete reset groups: per-request median distance to its train median ranges about **3–17 mm**, with p95 up to **24 mm**. This is direct evidence that per-card/port geometry cannot be inferred precisely from the board origin alone. [Exact variants and per-group errors](../../outputs/experiments/2026-10-07_sfp_one_screw_board_prior_v1/) · [machine summary](2026-10-07-sfp-visible-board-origin-audit.summary.json).

**Resulting architecture decision:** do not train the one-screw board detector as a stand-alone target-position shortcut. A visible board screw may still be useful for coarse card-order context, but local port perception remains required, and the reviewed public-state identity problem remains unresolved. The independently reviewed complete-public-observation packet is the next information gate for whether a board cue is worthwhile at all. No actor, handoff or final scene was evaluated here.

Mesh-screw and upper-bound commands:

```bash
.pixi/envs/default/bin/python scripts/audit_sfp_visual_screw_projection.py \
  --rail-rows /home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_feasibility_v3/rows.jsonl \
  --source-rows /home/nas/chmin/aic_archives/2026-10-06_sfp_slot_crop_training_materialized_v1/labels/rows.jsonl \
  --review-manifest outputs/experiments/2026-10-07_sfp_outline_corner_review_v1/review_manifest.json \
  --camera-calibration configs/hierarchical_recovery/sc_visual_standoff_20261001.json \
  --sfp-calibration configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json \
  --output outputs/experiments/2026-10-07_sfp_visual_screw_projection_v1

.pixi/envs/default/bin/python scripts/make_sfp_screw_early_review.py \
  --screw-projections outputs/experiments/2026-10-07_sfp_visual_screw_projection_v1/development_screw_projections.jsonl \
  --output outputs/experiments/2026-10-07_sfp_visual_screw_early_review_v1

OPENBLAS_NUM_THREADS=1 .pixi/envs/default/bin/python scripts/audit_sfp_one_screw_board_prior.py \
  --rail-rows /home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_feasibility_v3/rows.jsonl \
  --source-rows /home/nas/chmin/aic_archives/2026-10-06_sfp_slot_crop_training_materialized_v1/labels/rows.jsonl \
  --sfp-calibration configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json \
  --output outputs/experiments/2026-10-07_sfp_one_screw_board_prior_v1
```

Reproduction:

```bash
.pixi/envs/default/bin/python scripts/make_sfp_rail_visibility_review.py \
  --candidate-kind outline \
  --rows /home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_feasibility_v3/rows.jsonl \
  --predictions /home/nas/chmin/aic_archives/2026-10-05_sfp_slot_local_probe_v1/development_predictions.jsonl \
  --output outputs/experiments/2026-10-07_sfp_outline_corner_review_v1

.pixi/envs/default/bin/python scripts/probe_sfp_visual_outline_corner.py \
  --manifest outputs/experiments/2026-10-07_sfp_outline_corner_review_v1/review_manifest.json \
  --output outputs/experiments/2026-10-07_sfp_outline_corner_review_v1/contour_probe
```
