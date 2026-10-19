# Selected-port handoff execution

Date: 2026-10-02. Active plan: [selected-port handoff](../SELECTED_PORT_HANDOFF_PLAN.md). This record separates completed diagnostics from gates still open. Reserved final configurations remain sealed. No active job was interrupted. `/data1` had about 1.2 GB free at startup, so this continuation writes only small manifests/diagnostics there and will place any large new artifacts on NAS.

## Gate 1: frozen baselines — complete

[Frozen baseline manifest](../../outputs/experiments/2026-10-02_selected_port_handoff/frozen_baselines_v1.json) records byte sizes and SHA-256 hashes for the transport action/head checkpoints, all-candidate and earlier target-conditioned locators, SFP local actor, and the two known stop/identity failure records. This run did not change their weights or the saved scenes.

The earlier [task-conditioned single-keypoint locator](../../outputs/experiments/2026-10-02_two_family_perception/conditional_locator_v3/summary.json) matters: it already takes a selected identity and outputs one keypoint, but it was trained using **projected-in-frame**, not true visibility, and a low-resolution single-stage view. On five untouched native SFP test episodes its near-target pixel p95 was **118.44 px**; the residual-conditioned repeat was **189.11 px** on the same named test set. These are not evidence against the newly proposed native-crop, selected-port, true-visibility design, but they rule out simply rerunning that older architecture as if it were a new idea.

## Gate 2: frozen transport proposal diagnosis — threshold change fails

Command:

```bash
.pixi/envs/default/bin/python scripts/audit_frozen_transport_arrival_proposals.py \
  --comparison outputs/experiments/2026-10-02_two_family_perception/sc_stop_frozen_development_seven_v1.json \
  --output outputs/experiments/2026-10-02_two_family_perception/sc_arrival_head_coarse_diagnostic_v1.json
```

The [machine result](../../outputs/experiments/2026-10-02_two_family_perception/sc_arrival_head_coarse_diagnostic_v1.json) scores saved **causal** handoff probabilities against **posthoc** teacher waiting-pose and route labels. “Coarse safe to inspect” was predeclared as TCP within 80 mm of the intended waiting pose, orientation within 10°, and maximum route deviation at most 50 mm. Those teacher quantities were used only after the model proposed a stop; they were never model inputs. This is a seven-route development diagnostic, not a selected deployment threshold.

| Probability threshold, two consecutive decisions after 15 s | Safe proposals | False proposals | Missed safe routes |
| ---: | ---: | ---: | ---: |
| 0.80 | 5 | 2 | 0 |
| 0.90 | 4 | 2 | 1 |
| 0.95 (earlier comparison) | 2 | 2 | 3 |
| 0.99 | 0 | 2 | 5 |

The two wrong-route traces receive high probabilities even at 0.99. Lowering the threshold regains good arrivals but **never removes the two false proposals**. The head was originally trained to recognize a narrow **teacher local-start time window** on successful routes, rather than safe arrival on the actor's own routes. This is a concrete reason to relabel and test actor-generated routes; threshold tuning alone fails. The coarse labels and seven paths are too few to prove a new stop rule. The head repair is therefore a bounded diagnostic using separately grouped recorded on-policy paths, with the original five comparison episode groups held out from fitting.

## Gate 3: true visibility source audit — open

Existing [SC RGBD pilot](../../outputs/experiments/2026-10-02_two_family_perception/sc_rgbd_visibility_pilot_v1/summary.json) has four scene groups and 364 scored crops, with pooled false-visible rate 5.3%, but the five-card port-1 fold falsely accepts 4/9 hidden examples. Its depth label is **geometric exposed fraction**, not reviewed human-identifiable visibility. The corrected `llb_0001` [label source](../../outputs/experiments/2026-10-02_two_family_perception/visibility_labels/llb_0001_fixed/summary.json) has 116 accepted rows; invalid or RGB-mismatched auxiliary samples were rejected. Its review montage was inspected: it shows the target entering/leaving view, but one scene cannot validate a two-family visibility gate. SFP still lacks trustworthy occlusion labels. Existing projection flags must not be reused as visibility.

## Remaining gates

The on-policy arrival-head and SFP visibility-source diagnostics below have finished and failed their gates. Obtain trustworthy selected-target visibility supervision and a verifier that rejects the known wrong-route/wrong-port cases before training or promoting a new selected-port model. Then compare it against the frozen all-candidate baseline, test two-stage handoff without insertion, and only then repair/integrate the insertion actor. No new RL, final-scene evaluation, or full insertion claim is implied by these diagnostics.

## Gate 2 continuation: on-policy head repair failed on frozen paths

The frozen encoder was replayed over existing actor-generated SC routes, with teacher waiting pose and route used **only after inference** for offline labels. [The cache](../../outputs/experiments/2026-10-02_selected_port_handoff/onpolicy_arrival_features_v1/summary.json) has 9,960 causal decisions from 40 audits / 12 complete episode groups. One audit was rejected because saved RGB was stale. All five groups in the original seven-path frozen comparison were excluded from fitting. Safe-to-inspect labels required TCP within 80 mm, orientation within 10°, and maximum route deviation within 50 mm.

```bash
CUDA_VISIBLE_DEVICES=1 .pixi/envs/default/bin/python scripts/cache_onpolicy_transport_arrival_features.py \
  --root outputs/experiments/2026-10-01_transport_reliability \
  --handoff outputs/experiments/2026-10-01_transport_reliability/frozen_actor_handoff/best.pt \
  --actor outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt \
  --imagenet /var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth \
  --frozen-comparison outputs/experiments/2026-10-02_two_family_perception/sc_stop_frozen_development_seven_v1.json \
  --output outputs/experiments/2026-10-02_selected_port_handoff/onpolicy_arrival_features_v1 --device cuda:0
.pixi/envs/default/bin/python scripts/diagnose_onpolicy_arrival_head.py \
  --features outputs/experiments/2026-10-02_selected_port_handoff/onpolicy_arrival_features_v1 \
  --frozen-comparison outputs/experiments/2026-10-02_two_family_perception/sc_stop_frozen_development_seven_v1.json \
  --handoff outputs/experiments/2026-10-01_transport_reliability/frozen_actor_handoff/best.pt \
  --output outputs/experiments/2026-10-02_selected_port_handoff/onpolicy_arrival_head_v1
```

The replacement linear head trained on five episode groups and selected on two groups scored 3 safe / 0 false / 0 missed on three validation audits, but only **1 safe / 6 false / 0 missed** on the seven frozen routes. The original head at the same selected threshold scored 5 safe / 2 false / 0 missed. [Exact grouped split, checkpoint and events](../../outputs/experiments/2026-10-02_selected_port_handoff/onpolicy_arrival_head_v1/summary.json). The repair was **not deployed**. An original-head 0.8 threshold still had false proposals on the other groups and frozen routes ([reassessment](../../outputs/experiments/2026-10-02_selected_port_handoff/arrival_proposal_reassessment_v1.json)). A train-derived nominal wrist-orientation veto removed the two false frozen proposals at 5° but missed one valid route. These seven development paths are now diagnostic; they cannot select a deployable threshold.

The two-stage safety gate should count **false accepted handoffs to insertion**. A false pause-and-inspect proposal may be harmless if precise perception rejects it and transport resumes. Excessive proposals and missed valid arrivals still matter. This revises the wording of gate 2, but the current head and verifier are not reliable or integrated.

## Gate 3 continuation: SFP RGBD label-source repeat failed

The earlier SFP depth problem was reproduced on one non-final five-card/card-0/port-0 start. One rootless run used an attached label-only RGBD camera with own-camera exclusion; the second used an independent world-follower camera with the normal scene mask. Ordinary actor RGB remained separate. Large artifacts were written to NAS because `/data1` was nearly full.

```bash
AIC_AUDIT_GPU=2 AIC_AUDIT_VISIBILITY_LABELS=true AIC_VISIBILITY_EXCLUDE_OWN_CAMERA=true \
  AIC_AUDIT_DEADLINE_SEC=240 DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock \
  bash scripts/run_recovery_gazebo_diagnostic.sh \
  /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sfp_visibility_pilot_card0_port0_cards5 stock
AIC_AUDIT_GPU=2 AIC_AUDIT_VISIBILITY_LABELS=true AIC_VISIBILITY_WORLD_FOLLOWER=true \
  AIC_AUDIT_DEADLINE_SEC=240 DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock \
  bash scripts/run_recovery_gazebo_diagnostic.sh \
  /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sfp_visibility_world_follower_v1 stock
```

The attached capture had 292 RGB/depth pairs; center/left/right had only 18/15/15 frames with >95% positive finite depth. The world-follower repeat had 470 pairs and only 23/19/19. **Every pair from simulated 5–40 s** had depth entirely `-inf` and flat-gray auxiliary RGB, while the ordinary wrist RGB showed cards, cable and gripper. The follower image was visibly misregistered even at 2 s. [Attached sensor quality](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sfp_visibility_pilot_card0_port0_cards5_sensor_quality.json), [follower quality](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sfp_visibility_world_follower_v1_sensor_quality.json), [ordinary RGB](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sfp_visibility_world_follower_v1/orig_10.jpg), [invalid auxiliary RGB](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sfp_visibility_world_follower_v1/aux_10.jpg). Nearest-frame RGB MAE is only a screen: wall-clock gaps reach 1.4–2.4 s and do not establish synchronization. The zero depth and gray images are decisive.

These SFP captures **cannot** supervise visibility. They agree with prior failed primary-RGBD, attached-camera and follower pilots, so further routine camera repeats stop. The next label-source experiment must use original wrist RGB with audited human-identifiability examples or an offline ray/render test from synchronized scene poses checked against those original images. Simulator geometry may supervise labels, but never enter the actor or select an evaluation crop. The SC geometric-exposure pilot is a provisional proxy only; its [false-accept montage](../../outputs/experiments/2026-10-02_selected_port_handoff/sc_visibility_false_accepts_review_v1.jpg) includes three essentially fully hidden openings assigned high visible probability. The selected-port estimator and live handoff remain gated.

As an RGB-only source check, existing native SFP projection rows from 29 expert episodes were sampled without touching final scenes. The [early three-camera gallery](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sfp_projection_review_early_10x3.jpg) shows two different five-card selected ports at 3, 6, 9, 12 and 15 seconds. The red point is the **ground-truth projected opening for audit only**. Several later frames place it on a plug that occludes the opening. Thus a geometrically correct projected point is visibly different from an identifiable opening; these gallery frames are **not** automatically labeled visible. The [selection metadata](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sfp_projection_review_early_10x3.json) gives exact image paths and times. The 29-episode scored RGB source has synchronized robot poses and can support a small independently reviewed original-RGB label pilot, without another Gazebo capture.

[Machine gate summary](../../outputs/experiments/2026-10-02_selected_port_handoff/gate_update_v1.json) records both failures. **Next bounded experiment:** on existing scored original RGB, review about 100 deliberately balanced selected-target examples across SFP and SC (clear, partially occluded, fully occluded, neighboring target visible, and offscreen), grouped by episode. Pilot an RGB-only selected-target visibility classifier and compare against the frozen geometric in-frame baseline on untouched groups. Budget: at most one workday of labeling/tooling and one GPU-hour of model fitting. Continue only if review agrees with labels on at least 95/100 examples and false visible calls on known hidden/wrong-port cases fall to zero without eliminating most clearly visible cases; otherwise repair the label definition/source before model training. Do not use this tiny pilot as a deployment claim. Once a checked larger label set exists, repeat the grouped selected-pose and full handoff gates from the active plan. No routine additional expert collection, RL, or final-scene test is justified now.

The first review-tooling step is complete, without assigning any visibility labels:

```bash
.pixi/envs/default/bin/python scripts/prepare_selected_visibility_review.py \
  --sfp outputs/experiments/2026-10-02_two_family_perception/sfp_native_projection_29_hostpaths/rows.jsonl \
  --sc outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/sc_rgbd_occlusion_audit.json \
  --output /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/selected_visibility_review_v1
```

The [100-image manifest](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/selected_visibility_review_v1/review_manifest.jsonl) has 50 SFP and 50 SC original-RGB panels. Each panel shows a full image and a local crop without projected markers; [overview](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/selected_visibility_review_v1/review_overview.jpg). SFP review crop centers come from privileged projection **for audit tooling only**; SC centers come from a frozen RGB locator. The eventual deployable crop selector must use RGB alone. Every `review_visibility` is still `null`; this batch is not a certified training set. The overview confirms examples with requested ports offscreen, visible candidate neighbors and gripper occlusion, but detailed independent labels remain the next gate.

## October 2 continuation: review batch correction and a specific SC failure

The first 100-panel batch was inspected. Its SC crop was centered on the **model prediction**, so on a wrong prediction it showed the wrong object and could have biased human visibility labels. This is a review-tooling error; v1 remains archived and must **not** be used for labels. [Corrected v2 review batch](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/selected_visibility_review_v2/review_manifest.jsonl) joins the existing independent SC aperture projections to center the **audit-only** crop, while recording the RGB locator point separately. The SFP audit crop was already projected. No privileged crop is a deployable input. The corrected 100 panels still have `review_visibility: null`; several SFP openings are covered by the moving plug, and a definitive image-only label is ambiguous without a carefully stated aperture definition. We did not invent positive labels or train a visibility classifier.

```bash
.pixi/envs/default/bin/python scripts/prepare_selected_visibility_review.py \
  --sfp outputs/experiments/2026-10-02_two_family_perception/sfp_native_projection_29_hostpaths/rows.jsonl \
  --sc outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/sc_rgbd_occlusion_audit.json \
  --sc-geometry \
    outputs/experiments/2026-09-30_transport_continuation/visibility_labels/port0_four_shiftplus/labels.jsonl \
    outputs/experiments/2026-09-30_transport_continuation/visibility_labels/port0_two_mixed/labels.jsonl \
    outputs/experiments/2026-09-30_transport_continuation/visibility_labels/port0_world_follower_masked_region_checked/labels.jsonl \
    outputs/experiments/2026-09-30_transport_continuation/visibility_labels/port1_five_shiftminus/labels.jsonl \
  --output /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/selected_visibility_review_v2
```

The corrected join exposed a concrete **selected-port localization error**, not merely a visibility-label problem. On a [clear left-camera SC port-0 frame](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_left_projection_locator_disagreement.jpg), the independent aperture projection is on the blue port (red) while the frozen RGB locator points to the gripper (green), about **210 px** away. The frozen model was evaluated with the same 288×256 resize, RGB conversion, family/camera code and 1152×1024 coordinate scaling as its training path, so this is not an obvious audit preprocessing mismatch. The visible right/center errors are much smaller, and left-camera port-1 in the tested five-card scene is also small; the failure is concentrated in these SC port-0 left views.

```bash
.pixi/envs/default/bin/python scripts/audit_sc_locator_geometry_disagreement.py \
  --audit outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/sc_rgbd_occlusion_audit.json \
  --labels \
    outputs/experiments/2026-09-30_transport_continuation/visibility_labels/port0_four_shiftplus/labels.jsonl \
    outputs/experiments/2026-09-30_transport_continuation/visibility_labels/port0_two_mixed/labels.jsonl \
    outputs/experiments/2026-09-30_transport_continuation/visibility_labels/port0_world_follower_masked_region_checked/labels.jsonl \
    outputs/experiments/2026-09-30_transport_continuation/visibility_labels/port1_five_shiftminus/labels.jsonl \
  --output outputs/experiments/2026-10-02_selected_port_handoff/sc_locator_geometry_disagreement_v1.json
```

On 270 geometrically exposed frames across four SC development scenes, the frozen locator's projected-aperture distance was **8.16 px median / 213.55 px p95**. Center had 4.11 / 11.04 px; right had 8.18 / 11.29 px; left had **209.75 / 216.33 px**, with 61/96 left exposed frames over 100 px. All 61 large exposed errors were left-camera frames; they came from the three port-0 scenes, while the tested port-1 left scene had 1.38 / 3.72 px. [Per-scene and per-camera machine result](../../outputs/experiments/2026-10-02_selected_port_handoff/sc_locator_geometry_disagreement_v1.json). These are development scenes and geometric-exposure proxies, not final-set or certified visibility metrics. They show that using a single selected RGB crop after transport without a cross-view identity check could lock onto the gripper.

A diagnostic triangulation with fixed per-camera calibration and observed RGB keypoints gave center+right 3D translation error **4.12 mm median / 6.04 mm p95** on 41 triplets where both cameras had geometrically exposed openings. Including the erroneous left point sometimes made 3D error much worse; an exploratory 10 px third-view reprojection gate rejected 21/41 such triplets, including 17 where left was geometrically hidden. This is an observation-only rejection hypothesis, not an insertion-grade 0.5 mm pose result or a tuned deployment threshold. It needs grouped new development starts and native crop refinement. [Machine report with all 175 synchronized triplets](../../outputs/experiments/2026-10-02_selected_port_handoff/sc_multiview_rejection_v1.json) was produced with `scripts/audit_sc_multiview_rejection.py`, the same four `--labels` paths listed above, and `--calibration configs/hierarchical_recovery/sc_visual_standoff_20261001.json`.

**Decision:** Keep the visibility-label and selected-port model gates open. Next target the port-0/left confusion using existing RGB and independently projected openings, with complete episode-grouped splits and an explicit wrong-object rejection metric. The first comparison should test a pose proposal from center+right with a left-view consistency veto against the current three-view result, then a small native-crop selected-port refiner only if the proposal is stable. Do not train insertion or RL from this shortcut. Retain the original-RGB review batch as an audit artifact, not a positive/negative visibility dataset.

### Decoder-only repair and scene-identity correction

One diagnostic update trained only the final 12-channel heatmap convolution of the frozen locator, using 50 geometrically exposed **port-0 left-camera** frames from `port0_four_shiftplus` and `port0_two_mixed`. The encoder and every other decoder layer were frozen; no task or true geometry entered inference. The 300-update run saved [a decoder-only checkpoint and exact results](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_decoder_repair_v1/summary.json). It initially appeared to pass an episode-named holdout (`port0_world_follower_masked`: left p95 220.9 → 4.4 px), while the port-1 five-card left view stayed at 3.84 px p95. **Scene-identity audit then found that this port-0 holdout has the exact same task-board scene configuration as training `port0_four_shiftplus`. That result is leaked as evidence of generalization and is not a promotion gate.** The training script now checks complete scene hashes and rejects any train/test overlap before fitting.

```bash
CUDA_VISIBLE_DEVICES=1 .pixi/envs/default/bin/python scripts/diagnose_sc_port0_left_locator_repair.py \
  --checkpoint outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/best.pt \
  --labels \
    outputs/experiments/2026-09-30_transport_continuation/visibility_labels/port0_four_shiftplus/labels.jsonl \
    outputs/experiments/2026-09-30_transport_continuation/visibility_labels/port0_two_mixed/labels.jsonl \
  --output /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_decoder_repair_v1 \
  --device cuda:0
```

A **new** development reset then changed task-board x/y/yaw and selected SC rail position; its scene SHA-256 is `631e6a52b7189150c4ca725d6bfb69f729c68ba1cc969e58714d7f68c7cb9847`, distinct from both training scenes. Stock CheatCode partially inserted (tier 3 = 39.67), which is enough to obtain a clear port view for this perception diagnostic; it is not a new successful expert trajectory. The auxiliary RGBD sensor failed after motion here too: only 3 strict labels passed, all hidden/offscreen. The updated label builder now maps Docker `/audit` paths to the host without changing label semantics. Instead, [ordinary scored RGB plus saved TF](</home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_independent_shift_v1/original_rgb_projection_v1/summary.json>) provided 108 one-second projection rows, all with visibility left unset. I inspected the [left-camera montage](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_independent_shift_v1/original_rgb_projection_v1/left_projection_review.jpg) *before* comparing checkpoints and fixed the clear-opening review window at 8.7–23.7 s. The opening is blue and unobscured in those 15 sampled frames; later gripper-overlap frames were excluded.

On that independent window, the frozen locator scored **35.0 px median / 233.75 px p95**, with 6/15 misses over 50 px. The saved decoder repair scored **2.53 px median / 4.16 px p95**, with 0/15 over 50 px. [Per-frame predictions and exact review window](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_independent_shift_v1/frozen_vs_decoder_v1.json). This repairs the specific left-view wrong-object failure on **one genuinely new reset**, but 15 correlated frames are insufficient to promote a general selected-port model. It also does not solve true visibility, right/center fusion, submillimeter 3D pose, or insertion.

Reproduction commands for this new reset were the rootless `run_recovery_gazebo_diagnostic.sh` with `AIC_AUDIT_GPU=2 AIC_AUDIT_SMOOTH_CAPTURE=1 AIC_AUDIT_VISIBILITY_LABELS=true AIC_VISIBILITY_EXCLUDE_OWN_CAMERA=true AIC_AUDIT_DEADLINE_SEC=240`, then `extract_moveit_training_stream.py` inside the official rootless evaluation image, `build_sc_original_rgb_projection_audit.py` on the resulting scored stream, and `evaluate_sc_port0_left_independent.py` with the original and decoder-only checkpoints. All exact inputs/outputs are under the new reset directory linked above. A second independent five-card reset completed the bounded replication; no further routine collection follows it.

### Independent five-card replication and 3D consequence

The training command above now uses only the two fitting scenes. The earlier four-label command is invalid after the scene-hash leak guard was added. A rerun to [this separate directory](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_decoder_repair_repro_v2/summary.json) produced **bitwise identical** last-convolution weight and bias (`torch.equal=True`, maximum absolute difference 0) to the saved diagnostic checkpoint. The encoder and other decoder layers were frozen; the 11 unrelated candidate output channels in the final convolution did not change. Independent resets were evaluated separately, without test labels in training.

The second SC port-0 development reset changed board x/y/yaw, moved the selected rail, and enabled five NIC cards. Its complete scene hash is `7a2b5db02ba70998163d0bf12dada87f60353c1768199cdf82b07089d0d20d56`, distinct from both fitting scenes and the first new reset. Stock CheatCode scored tier 3 = **40.42**, partial insertion; these are perception test images, **not** successful expert demonstrations. Original scored RGB and saved TF produced [102 projection rows](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_independent_fivecard_v1/original_rgb_projection_v1/summary.json). I inspected the requested opening in [left-camera](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_independent_fivecard_v1/original_rgb_projection_v1/left_projection_review.jpg) and [center/right-camera](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_two_new_scenes_center_right_review.jpg) original RGB before comparing checkpoints. The clear-opening window was fixed at 6.7–22.7 s, 16 frames per camera. The first new reset used 8.7–23.7 s, 15 frames per camera. Later gripper occlusions were excluded.

| New scene / camera | Frozen p95 pixel error | Repaired p95 pixel error | Frozen / repaired >50 px misses |
| --- | ---: | ---: | ---: |
| shifted, center | 8.92 | 5.14 | 0 / 0 |
| shifted, left | **233.75** | **4.16** | 6 / 0 |
| shifted, right | 7.96 | 6.91 | 0 / 0 |
| five-card, center | **172.28** | **5.81** | 2 / 0 |
| five-card, left | **513.91** | **11.60** | 15 / 0 |
| five-card, right | 7.35 | 8.20 | 0 / 0 |

The [before/after gallery](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_independent_before_after_gallery.jpg) was visually checked: **red** is the scored projection for audit, **green** is the frozen prediction, and **cyan** is the repaired prediction. The green marker hits the gripper in several clear views; the cyan marker is on the selected blue SC port. Exact [shifted](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_independent_shift_v1/frozen_vs_decoder_allcameras_v1.json) and [five-card](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_independent_fivecard_v1/frozen_vs_decoder_allcameras_v1.json) per-frame metrics and [scene-hash summary](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_decoder_independent_gate_v1.json) are saved. The descriptive 20 px criterion was written after seeing both results, so it is **not** a predeclared promotion gate. These 31 frames per camera are correlated within only two resets.

Saved RGB keypoints were then triangulated with fixed TCP-to-camera calibration. The scored TCP-relative port position was used **only** to measure error offline:

| New scene | Frozen / repaired three-view translation p95 | Frozen / repaired lateral p95 | Repaired axial p95 |
| --- | ---: | ---: | ---: |
| shifted | **42.76 / 2.86 mm** | **16.64 / 1.43 mm** | 2.62 mm |
| five-card | **139.07 / 3.78 mm** | **29.74 / 1.26 mm** | 3.77 mm |

The [shifted](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_independent_shift_v1/frozen_vs_decoder_portframe_v2.json) and [five-card](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_independent_fivecard_v1/frozen_vs_decoder_portframe_v2.json) records include every matched observation. The opening-frame local z axis defines axial error; local x/y define lateral error, matching the earlier native triangulation audit. **The gross wrong-object failure is repaired in these reviewed clear-view windows, but lateral p95 of 1.43/1.26 mm remains outside the 0.5 mm insertion corridor.** Orientation and true visibility were not measured. Do not attach this coarse output directly to insertion.

```bash
.pixi/envs/default/bin/python scripts/evaluate_sc_independent_triangulation.py \
  --prediction /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_independent_fivecard_v1/frozen_vs_decoder_allcameras_v1.json \
  --projection /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_independent_fivecard_v1/original_rgb_projection_v2/rows.jsonl \
  --calibration configs/hierarchical_recovery/sc_visual_standoff_20261001.json \
  --output /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_independent_fivecard_v1/frozen_vs_decoder_portframe_v2.json
```

The corrected [100-image original-RGB review batch](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/selected_visibility_review_v2/review_manifest.jsonl) has 50 SFP and 50 SC examples. Its SC audit crop is centered on an **independent scored aperture projection**, correcting v1's mistake of centering on the model's potentially wrong prediction. [Marked SC sheets](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/selected_visibility_review_v2/marked_sheet_08.jpg) were visually inspected. Every main-manifest `review_visibility` remains `null`: projected-in-frame and geometric exposure are not certified human-identifiable labels, and SFP plug occlusion leaves some cases ambiguous. V1 is superseded as a label source.

A separate [single-reviewer provisional triage](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/selected_visibility_review_v2/manual_triage_provisional_v1.jsonl) identified 20 obviously unidentifiable SC examples and 19 identifiable SC examples, leaving 11 SC and **all 50 SFP** as uncertain. These 39 labeled SC frames represent only **three complete scene groups**. Against this provisional triage, the frozen projected-in-frame score at 0.5 had **16 true accepts, four false accepts, three misses, and 16 correct rejects**; [indices 55, 60, 64, 69](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/selected_visibility_review_v2/manual_triage_provisional_v1_summary.json) are high-score hidden examples. Precision 0.80 and recall 0.84 are **pilot-only** because the batch was balanced by an exposure proxy, correlated within scenes, and reviewed once. It is neither a representative operating rate nor enough scene-group diversity for model selection. No visibility classifier was trained from it.

**Decision:** keep the arrival/verifier, visibility, selected-port 3D/orientation, and integrated handoff gates open. The next bounded repair is a checked three-way `identifiable / not identifiable / uncertain` label protocol on original RGB, split by complete episode/reset and including wrong-port examples. If labels cannot agree with reviewed images, stop rather than equating in-frame geometry to visibility. A selected-port original-resolution crop refiner then must pass held-out near-port lateral p95 ≤0.5 mm, zero known wrong-port/hidden false accepts, and complete live inference p95 <300 ms before insertion is attached. No autonomous insertion or final-scene opening is claimed.

## Continuation: native SC crop, second target, and grouped visibility review

The next bounded test reused the **frozen** 160×160 native SC crop refiner (`sc_native_refiner_selection10_v2/best.pt`) on the two genuinely new SC port-0 resets above. Each crop center came from the RGB coarse prediction, with the original 1152×1024 image cropped before resizing; scored geometry was read only after prediction. The refiner checkpoint had trained on 83 SC episodes and selected on ten other episodes. Both new scene-group IDs were absent from its 104-episode SC source. On reviewed clear-opening windows, the repaired locator put all **45/45 and 48/48** per-camera aperture projections inside its observation-selected crop. The old locator put only 40/45 and 33/48 inside, so refinement cannot repair its wrong-object crop. [Shifted pixel results](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_independent_shift_v1/coarse_native_refiner_v1.json), [five-card pixel results](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_independent_fivecard_v1/coarse_native_refiner_v1.json), and a [visually inspected overlay](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_independent_native_refiner_gallery_v1.jpg) preserve the evidence. In that overlay, red is the audit projection and yellow is the RGB-only refined prediction.

| New SC reset | Requested port | Reviewed synchronized frames | Repaired coarse lateral p95 | Coarse + native crop lateral p95 | Refined axial p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| shifted port 0 | 0 | 15 | 1.43 mm | **0.230 mm** | 0.331 mm |
| five-card port 0 | 0 | 16 | 1.26 mm | **0.175 mm** | 0.304 mm |
| five-card port 1 | 1 | 18 | see below | **0.149 mm** | 0.296 mm |

A third, targeted **port-1 five-card** development reset checked the other request identity. It shifted the board x by +11 mm, y by −14 mm, yaw by +0.019 rad and SC rail 1 by +8 mm from the existing development template; complete scene hash `78f2fdc38051654b6049fd7bca28a9c960b45800682265581ca869766d966b8b` and scene-group ID `02acba03288de376bba72dcd59d09dabdf95083b1a56c2f566a998f2ea64f9bb` are absent from the refiner's 104 SC episodes. Stock CheatCode reached tier 3 = **22.16**, partial insertion, not an expert-success trajectory. The [original three-camera RGB review](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port1_independent_fivecard_v1/original_rgb_projection_v1/three_camera_projection_review.jpg) fixed the clear-opening window at 9.8–27.8 s before scoring. Across 18 synchronized frames per camera, the frozen coarse model found requested port 1 at 3.26 px overall p95 and **0/54** >50 px misses. The port-0-only decoder update produced identical port-1 predictions. Native refinement gave center/left/right p95 0.82/0.67/0.66 px, with all **54/54** requested openings inside the RGB-selected crop. Three-view port-frame lateral/axial p95 were **0.149/0.296 mm**. [Per-frame pixel predictions](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port1_independent_fivecard_v1/coarse_native_refiner_v1.json), [3D errors](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port1_independent_fivecard_v1/coarse_native_refiner_portframe_v1.json), and the [visually inspected overlay](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port1_independent_fivecard_v1/port1_native_refiner_gallery_v1.jpg) are saved. [One machine summary of all three resets](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_selected_port_clearview_diagnostic_v2.json) records the precise scope. These are **three reset groups and correlated clear-view frames**, mostly 48–176 mm TCP-to-opening distance. They do not establish reliability under occlusion or full insertion.

To check time cost, an in-memory three-camera benchmark ran the 288×256 coarse locator, RGB-selected original-resolution crop, 160×160 refiner, and triangulation sequentially on one GPU. Component p95 was [23.57 ms port 0](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_independent_fivecard_v1/selected_port_chain_latency_v1.json) and [31.55 ms port 1](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port1_independent_fivecard_v1/selected_port_chain_latency_v1.json), 100 warmed iterations each. It excludes camera transport, JPEG decode, ROS, and either actor. **This is not the complete live <300 ms gate.** No model was deployed.

The checked-visibility source was also extended **without new simulation**. Existing SC successful-expert RGB provided [60 original-camera review panels](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/grouped_sc_visibility_review_v2/review_manifest.jsonl) over **20 complete scene groups**, balanced across ports 0 and 1. A conservative [single-reviewer triage](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/grouped_sc_visibility_review_v2/manual_triage_provisional_v1_summary.json) marked 20 offscreen route views unidentifiable and 20 clear approach views identifiable; **20 almost-contact views remain uncertain** because the plug covers the aperture while the housing may still locate it. Against these easy labels, the frozen projected-inframe score had 20 true accepts, three false accepts, zero misses, and 17 correct rejects. The earlier three-group manual batch had four high-score **occluded** false accepts. This larger batch increases scene diversity but **does not add certified occlusion labels** or any SFP visibility labels. It does not justify visibility-head fitting or a safe handoff.

```bash
CUDA_VISIBLE_DEVICES=1 .pixi/envs/default/bin/python scripts/evaluate_sc_independent_native_refiner.py \
  --coarse /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port1_independent_fivecard_v1/frozen_vs_decoder_allcameras_v1.json \
  --checkpoint outputs/experiments/2026-10-02_two_family_perception/sc_native_refiner_selection10_v2/best.pt \
  --output /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port1_independent_fivecard_v1/coarse_native_refiner_v1.json
.pixi/envs/default/bin/python scripts/evaluate_sc_independent_triangulation.py \
  --prediction /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port1_independent_fivecard_v1/coarse_native_refiner_v1.json \
  --projection /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port1_independent_fivecard_v1/original_rgb_projection_v1/rows.jsonl \
  --calibration configs/hierarchical_recovery/sc_visual_standoff_20261001.json \
  --output /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port1_independent_fivecard_v1/coarse_native_refiner_portframe_v1.json
```

**Gate decision:** the SC **clear-view position subtest** now passes on three independent development resets for both requested ports. Stage 3 true visibility, selected-port orientation on those resets, false-acceptance under target occlusion/wrong-port requests, complete live latency, stage 2 transport/verifier behavior, and autonomous insertion remain open. Do not use a true TF to choose a runtime crop; do not interpret the offline clear-view result as an automatic switch or successful insertion. The next bounded experiment should use original RGB with independently checked hard occlusion and same-image/different-request cases across complete scene groups. Its stopping rule is zero false accepted handoffs on the frozen wrong-route/wrong-port examples and new grouped development starts while retaining clearly visible valid arrivals. If the label source cannot support that check, repair the label source before policy training. Final scenes stay sealed.

### Bounded three-camera disagreement diagnostic

Before collecting more data or fitting a visibility classifier, I replayed the existing RGB crop cascade on the four high-score hidden SC examples from the corrected 100-image review and then all 20 provisionally hidden SC examples. This calculation uses the requested port code, three RGB predictions, and fixed camera calibration; scored target geometry is **not** supplied to the model or the agreement calculation. A provisional rule requires all three cameras, each frozen in-frame score at least 0.5, and a maximum three-view reprojection residual at most 2 px. The 2 px value is exploratory, chosen after looking at development data, and must not be treated as a calibrated deployment threshold.

On the **49 reviewed clear-view triplets from the three new SC resets**, the refined keypoints' three-view residual was at most 0.777 px (per-scene p95 0.660/0.628/0.423 px), and a new frozen-locator replay recovered the missing in-frame scores (minimum over all three cameras was 0.755). Thus the **complete exploratory rule retained 49/49** of these manually reviewed clear frames. On the old provisional hidden-review batch, that full rule accepted **0/20**. Three of the four most concerning high-score false-visible cases had all cameras but disagreed by 18.12–18.79 px. The fourth had only two reviewed cameras and was rejected by the three-camera requirement, even though those two predictions agreed to 0.094 px. This demonstrates that a two-view agreement check can still agree on a wrong target. [Four hard examples](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_hard_occlusion_consistency_v1.json), [all 20 provisional negatives](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_hidden20_consistency_v1.json), and [combined machine summary](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_selected_port_clearview_diagnostic_v4.json) preserve every result.

```bash
CUDA_VISIBLE_DEVICES=1 .pixi/envs/default/bin/python scripts/audit_sc_hard_occlusion_consistency.py \
  --review /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/selected_visibility_review_v2/review_manifest.jsonl \
  --audit outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/sc_rgbd_occlusion_audit.json \
  --indices 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 \
  --coarse outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/best.pt \
  --adapted /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_decoder_repair_v1/decoder_only.pt \
  --refiner outputs/experiments/2026-10-02_two_family_perception/sc_native_refiner_selection10_v2/best.pt \
  --calibration configs/hierarchical_recovery/sc_visual_standoff_20261001.json \
  --output /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_hidden20_consistency_v1.json
```

This rejects known examples in a small, single-reviewer, three-scene pilot; many of the 20 negatives are easy offscreen views. It says little about difficult near-contact occlusion, same-image/different-request identity, SFP, or a case where all three cameras agree on the same wrong port. Requiring all three cameras may also reject valid arrivals with one obstructed camera. **No verifier or handoff is promoted.** The next bounded test must challenge target identity and occlusion on independently reviewed complete new scene groups, measure valid-arrival recall as well as false accepts, and stop if the labels remain ambiguous. No new policy training follows from this diagnostic alone.

The missing clear-frame scores were recovered by rerunning `scripts/evaluate_sc_port0_left_independent.py` with the **same** frozen checkpoints, source rows, cameras and reviewed windows listed above, saving `frozen_vs_decoder_with_scores_v2.json` in each of the three reset directories. The adapted selected-port scores had minimum three-camera value 0.817/0.755/0.999 by scene; combined with the saved residuals, the full exploratory rule accepted **15/15, 16/16 and 18/18** clear triplets. The exact score files and rule counts are indexed in the v4 machine summary. This replay did not train weights or inspect final scenes.

I also inspected [original RGB at the closest scored observations of the three new resets](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_new_scene_near_contact_audit_v1.jpg). In both port-0 runs the plug overlaps the requested opening near 18 mm TCP-to-opening distance; port 1 stops around 46 mm with a much clearer opening. The montage uses red **audit-only** aperture markers. The overlap makes the first two cases **uncertain for direct visual identifiability**, so I did not add them as negative labels or claim the three-view rule handles contact. This motivates defining the handoff at an observable standoff and checking what the model does once the plug partly obscures the port.

A targeted replay of [eight complete three-camera triplets at those closest saved observations](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_new_scene_near_contact_consistency_v2.json) applied the same frozen models and exploratory rule: it accepted **2/2 provisionally clear port-1 views** at about 46 mm TCP-to-opening distance and **1/6 uncertain plug-overlapped port-0 views** at about 18–19 mm, rejecting the other five. The one accepted uncertain view is **not scored as a false acceptance**; the physical opening is partly covered and the visual-identifiability rubric has not been certified. This does show that camera agreement and high in-frame scores alone cannot define when the handoff is safe. The exact RGB audit and review manifest are `sc_new_scene_near_contact_audit_v1.jpg`, `sc_new_scene_near_contact_audit_v1.json`, and `sc_new_scene_near_contact_review_v1.jsonl` in the same NAS directory. The diagnostic script now reports accepted count separately from false accepts so uncertain and positive rows cannot be misreported as failures.

**Subsequent identity gate failure:** A same-RGB/different-request test found that this provisional rule accepts 16/16 requests for a physically absent SC port by locking onto the real neighboring port. The [full request-swap comparison and failed supervised repairs](2026-10-02-sc-absent-port-identity.md) supersede any interpretation of the 49/49 clear and 0/20 small hidden pilot as a safe selected-port verifier. The clear-view position result remains a narrow measurement only.
