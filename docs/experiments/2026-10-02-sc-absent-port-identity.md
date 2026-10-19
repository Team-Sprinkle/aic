# SC selected-port identity: bounded absent-request diagnostic

**Date:** October 2, 2026. **Current decision:** the original three-camera verifier failed the absent-port test. A later spatial identity head passed a narrow, fixed all-three-camera test on four new development resets, but the complete handoff remains disabled. This is development evidence; the reserved final split remains sealed.

## Hypothesis and stopping rule

The prior [clear-view SC crop test](2026-10-02-selected-port-handoff-execution.md#bounded-three-camera-disagreement-diagnostic) found precise port pixels and encouraging three-camera agreement. Its next gate required zero accepted requests for a hidden or wrong target while retaining visibly valid arrivals. I tested whether the frozen RGB locator, original-resolution refiner, and exploratory 2 px three-camera check distinguish the **requested** SC port from another visible SC port. A complete scene/reset is the unit of comparison. No teacher, ground-truth geometry, or privileged crop enters model inference; scored TF and scene configuration provide post-run labels only.

The task configuration itself guarantees that the requested port exists. The absent-port request below is a **counterfactual stress test**, not an estimated real deployment failure rate. It directly tests the planned selected-port rejection behavior. The local eval-like scenario generator samples the non-target SC port with probability 0.25 (`aic_engine/scripts/generate_random_trials_config.py`), so one-port scenes are a relevant visual condition. This is a property of that generator; it does not establish the hidden final distribution.

## Matched request swap: the current rule fails

Two new SC port-0 development resets each contain **only port 0**, verified from the scored `/tf` scene tree. I selected eight synchronized original-RGB three-camera triplets from the previously reviewed clear approach window of each reset. Each image was replayed twice: request the real, visible port 0, or change only the requested identity to nonexistent port 1. Scene geometry was used to certify absence and score pixels, never to choose an inference crop. A third two-port reset contributed eight clear port-1 positives. The model checkpoints, RGB images, camera calibration, and exploratory threshold stayed frozen.

| Frozen model and verifier | Real clear request accepted | Absent request falsely accepted |
| --- | ---: | ---: |
| Current locator + port-0 decoder repair + native refiner | **24/24** | **16/16** |
| Earlier full locator trained with absent-identity loss + native refiner | **19/24** | **14/16** |

For the current locator, nonexistent-port-1 predictions land on the *real port 0*: native pixel error to the visible port-0 opening was about **0.34–0.35 px median** on the two resets. Every absent triplet had three-view reprojection residual ≤0.481 px and the minimum in-frame score across cameras was ≥0.958. Thus camera agreement and high scores can be mutually consistent **on the wrong identity**. The [visually inspected overlay](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_absent_request_false_accept_gallery_v1.jpg) marks scored real port 0 in red and model-requested nonexistent port 1 in yellow; they overlap. The [machine summary](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_absent_request_identity_gate_v1.json) links every per-triplet record.

This does **not** invalidate the earlier submillimeter position measurement when the correct port was requested and visually clear. It shows that the **identity/rejection** part of a deployable selected-port estimator is unsolved.

## Why this happened

The original SC projection training source had **83 train + 21 validation scene groups, all with both SC ports physically present**. It had zero one-port SC groups. The original locator also masked the in-frame loss by `present`, so an absent candidate output received no negative loss on that frame (`scripts/train_two_family_port_locator.py`). Even the later absent-loss checkpoint had trained from a source without absent SC examples; it reduced some absent-slot errors in other tasks but did not solve this SC request swap. The later independent one-port resets are genuinely different scene groups; they were not in those 104 source episodes.

## Targeted repair attempts and their limits

I collected **four** additional one-port **port-1** development resets with rootless Gazebo on GPU 2: two assigned to fitting, two to development checks. They vary board x/y/yaw, selected rail position, and one/two/four/five NIC-card presence. Their [exact configs and reset hashes](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_one_port_identity_v1/manifest.json), scored streams, original RGB, and projection-only labels are saved under the same directory. Two older one-port port-0 development recordings supplied fitting examples; the two independent shifted port-0 resets above stayed out of fitting. A [visually inspected three-camera review](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_one_port_development_rgb_review_v1.jpg) shows the single requested blue connector in each of the four development configurations. Stock CheatCode insertion scores are irrelevant to the presence label; these are **not expert-success trajectories**.

First, I fitted **only the two SC rows of the frozen locator's final in-frame layer** for 500 updates, balancing one-port and two-port samples. The encoder, keypoint decoder, and every SFP output were unchanged. On four one-port development groups, absent-slot false positives fell from **385/404 to 334/404** per-camera labels. Present-port recall rose from 336/364 to 342/364, but on 21 original two-port validation groups it **fell from 554/567 to 495/567**. This checkpoint fails the stopping rule and is **not promoted**. [Exact split, command inputs and weights](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_presence_head_ablation_v2/summary.json).

Second, I tested a more expressive but still frozen-encoder availability probe with the same 500-update budget. One variant used pooled full-image features and camera identity; the other additionally used the **measured TCP pose**, a deployable input. Both fit all four one-port training groups perfectly but generalized poorly:

| Frozen-image probe | Development absent false positives | Development real-port recall | Original two-port validation real-port recall |
| --- | ---: | ---: | ---: |
| Image only | 202/404 | 307/364 | 557/567 |
| Image + measured TCP | **149/404** | **322/364** | 556/567 |

The four development groups are two one-port port-0 and two one-port port-1 resets; all four contribute errors. The image+TCP variant helps, but its absent false acceptance remains far above zero and it misses 42 present labels. These are **projection-in-frame** labels, not certified visual identifiability. Neither model passed a live handoff or contact/occlusion test. [Per-group metrics and checkpoints](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_selected_availability_probe_v2/summary.json).

A simple duplicate-prediction veto would reject the nonexistent port because its points collapse onto the real port in all three cameras. It would also reject the **real** port on the same one-port images, losing valid arrivals. I did not deploy this shortcut.

## Reproduction

```bash
# Confirm both requested identities on the same saved SC RGB. Port 1 is absent.
.pixi/envs/default/bin/python scripts/prepare_sc_absent_request_replay.py \
  --scene /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_port0_left_independent_shift_v1 \
  --start-s 8.7 --end-s 23.7 --count 8 \
  --output /home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_absent_request_shift_repro_new
# See the original saved audit.json/review.jsonl and consistency_v1.json; the matched five-card
# reset and positive-request files are indexed in sc_absent_request_identity_gate_v1.json.
```

The exact one-port fitting and development label paths, grouped counts, and checkpoints are in the two training summaries above. New Gazebo runs used `scripts/run_recovery_gazebo_diagnostic.sh` with `AIC_AUDIT_GPU=2`, `AIC_AUDIT_SMOOTH_CAPTURE=1`, the rootless Docker socket, and the frozen stock CheatCode. The official scored-bag extraction used the existing `aic_eval` image. No job was interrupted; the only background GPU0 Isaac probe was left alone. All AIC runs for this diagnostic have finished. Script compilation and `git diff --check` passed.

## Gate and next targeted repair

**The identity gate fails.** Do not promote the exploratory camera-agreement rule, either availability probe, a transport-to-insertion switch, or an insertion policy. The already-recorded SFP wrong-port replay remains a separate mandatory negative; these SC-only probes did not change SFP. The earlier 20/20 hidden-SC rejection result remains a small pilot and cannot compensate for this request-swap failure. Complete live p95 latency and autonomous insertion were not tested because the earlier safety gate failed. Final scenes remain sealed.

The next repair should **train explicit target identity/presence jointly with original-RGB spatial evidence**, rather than only pooled image features or a last-layer presence adjustment. Use scene-grouped one-port SC examples for **both requested ports** and two-port distractor scenes, plus checked `identifiable / unidentifiable / uncertain` labels. Keep a new set of development reset groups sealed from fitting and from threshold selection. A bounded first comparison is the same full-image encoder with a spatial selected-port head versus a compact original-resolution target crop with an image-derived proposal; both consume the task code and measured TCP, and neither receives simulator pose at runtime. Budget at most one workday of targeted one-port/occlusion data preparation and two GPU-hours of fitting/evaluation. Stop if any absent/wrong-port request is accepted on the new grouped development scenes, clear valid-arrival recall falls below 95%, or live complete inference p95 exceeds 300 ms. If those conditions hold, then replay frozen SC wrong-route and SFP wrong-port handoffs before actor attachment. No routine collection or RL proceeds from this failed gate.

## Spatial selected-port continuation on new reset groups

The bounded repair above was run without opening the final scenes. The [fixed scene manifest](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_v1/manifest.json) has eight new one-port fitting groups and four new one-port development groups, two per requested port. Board pose, selected rail, and one to five card presence vary. The 12 complete reset hashes are unique and do not overlap the previous one-port manifest. Rootless Gazebo generated native 1152×1024 RGB from all three wrist cameras and scored geometry for labels. All 12 captures and extractions completed. These are **perception examples**, not expert insertion demonstrations. The older four one-port fitting groups and 83 original two-port fitting groups were also used; the four new development groups and 21 original two-port validation groups were not fitted.

The [matched probe](../../scripts/train_sc_spatial_identity_probe.py) freezes the same RGB ResNet backbone and fits two heads for 800 updates each on identical minibatches with a fixed probability threshold of 0.5. Both explicitly consume the **requested SC port code**, camera identity, and measured TCP pose. The pooled control averages the deep feature map; the spatial head keeps its 16×18 feature grid through two convolutions before pooling. Neither receives Gazebo target pose at inference. Its training label is **projected in frame**, not human-identifiable visibility. The first fit encountered an evaluation-only image/request indexing bug before any held-out metric was produced; the corrected rerun is preserved as v2 and the incomplete v1 directory remains for provenance.

| Frozen-feature head | Development absent per-camera false accepts | Development real projected-in-frame recall | Original two-port validation real recall |
| --- | ---: | ---: | ---: |
| Pooled control | 125/304 | 116/272 | 529/567 |
| Spatial selected-port head | **9/304** | **255/272** | **558/567** |

The individual camera scores are not yet safe: in a port-1-only development reset, the spatial model called the physically absent port 0 present in seven left-view frames with probability 0.87–0.995. I inspected one native RGB and confirmed it shows the real blue port 1. Most missed projected-in-frame calls in another reset came from a right view where the gripper covers much of the connector; this is precisely why projection is an imperfect visibility label. The [exact 26 sampled errors and image paths](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_probe_v2/development_predictions_v2.json) preserve that distinction.

The original verifier already required all three cameras. I therefore replayed **every saved development image** with frozen weights, paired camera frames within 150 ms, and applied a fixed rule: each requested-port score must be ≥0.5. This yielded 109 synchronized triplets for each real and counterfactual absent request across the four reset groups. No threshold was selected from these scenes. The [complete machine summary](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_probe_v2/three_camera_summary_v1.json) has each reset separately.

| Head, all-three rule | Absent request accepted | Real request accepted when opening projected into all views |
| --- | ---: | ---: |
| Pooled control | 32/109 | 34/100 |
| Spatial selected-port | **0/109** | **96/100** |

Thus the **SC one-port identity subtest passes provisionally** with all-three agreement on these development resets. A two-of-three rule is unsafe here: it accepted 2/109 absent requests. The all-three rule can miss valid views when a camera is occluded, and its 96/100 denominator is projected-in-frame, not certified visual clarity. Only 8 synchronized triplets were available in one development group because of camera timestamp gaps; do not turn this small grouped result into a production reliability estimate. The matched [pooled scores](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_probe_v2/pooled_full_development_predictions_v1.json) and [spatial scores](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_probe_v2/full_development_predictions_v1.json) allow reanalysis.

A [100-iteration, preloaded-image component benchmark](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_probe_v2/component_latency_v1.json) ran the three-camera coarse locator, spatial identity head, and native crop refiner on GPU 1. It measured p50/p95/p99 **10.87/11.19/11.34 ms**. The locator backbone was run twice, making this a conservative estimate for those in-memory components, but this **excludes ROS image delivery, actors, triangulation, and controller command**. It is not the required complete live p95 <300 ms test.

I then replayed the **20 previously reviewed provisionally hidden SC triplets** without changing either checkpoint. Measured TCP was causally recovered from their original scored bags, and every reviewed image was matched to its saved controller state; the four recovered streams and all-frame projection-only state rows are in [`hidden_state_replay/`](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_probe_v2/hidden_state_replay). The spatial identity head **alone accepted 4/20** hidden examples with all three scores ≥0.5. The older native keypoint/reprojection rule accepted **0/20**, so the conjunction also accepted **0/20**. These negatives are a small, single-reviewer provisional set with repeated scene configurations; they do not certify hard occlusion. They show concretely that selected-port *identity* is not the same as selected-port *visibility*. [Every score and image path](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_probe_v2/hidden20_spatial_replay_v1.json).

The exact fitting, full-frame audit, three-camera summary, checkpoint, group metrics, and latency files are under [sc_spatial_identity_probe_v2](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_probe_v2). Reproduce the fit with `scripts/train_sc_spatial_identity_probe.py` using the source paths in `summary.json`; audit with `scripts/audit_sc_spatial_identity_errors.py --full-development`, then `scripts/evaluate_sc_identity_three_camera.py`. Data capture used `scripts/prepare_sc_spatial_identity_scenes.py` and sequential rootless `scripts/run_recovery_gazebo_diagnostic.sh` on GPU 2. The fit used one GPU (GPU 1), all jobs finished, and no reserved final scene was opened.

The following is the fitting command with the fixed input groups (run from the repo root); the [summary JSON](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_probe_v2/summary.json) retains the expanded exact file list:

```bash
archive=/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff
.pixi/envs/default/bin/python scripts/train_sc_spatial_identity_probe.py \
  --two-port-labels outputs/experiments/2026-10-02_two_family_perception/projection_labels/rows.jsonl \
  --one-port-train \
    "$archive"/sc_one_port_identity_v1/base_port0_four_projection_v1_state_projection_v2/rows.jsonl \
    "$archive"/sc_one_port_identity_v1/base_port0_two_projection_v1_state_projection_v2/rows.jsonl \
    "$archive"/sc_one_port_identity_v1/train_a/projection_state_v2/rows.jsonl \
    "$archive"/sc_one_port_identity_v1/train_b/projection_state_v2/rows.jsonl \
    "$archive"/sc_spatial_identity_v1/train_*/projection_state_v2/rows.jsonl \
  --one-port-development "$archive"/sc_spatial_identity_v1/development_*/projection_state_v2/rows.jsonl \
  --checkpoint outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/best.pt \
  --output "$archive"/sc_spatial_identity_probe_v2 --updates 800 --device cuda:1
```

The scene manifest came from `scripts/prepare_sc_spatial_identity_scenes.py --source-port0 "$archive"/sc_port0_left_independent_shift_v1 --source-port1 "$archive"/sc_port1_independent_fivecard_v1 --output "$archive"/sc_spatial_identity_v1`. Each scene ran sequentially with `AIC_AUDIT_GPU=2 AIC_AUDIT_SMOOTH_CAPTURE=1 AIC_AUDIT_DEADLINE_SEC=180 DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock bash scripts/run_recovery_gazebo_diagnostic.sh <scene-dir> stock`; the scored-bag stream was extracted inside the rootless official `aic_eval` image and projected using `scripts/build_sc_original_rgb_projection_audit.py --stride-s 1`. The old hidden replay used `--stride-s 0` to match every saved reviewed image to its measured TCP. The manifest and per-scene `capture.log`, `extract.log`, `project.log`, stream and RGB files retain the full provenance.

**Decision and next bounded experiment:** keep the automatic transport-to-insertion switch disabled. The 20-example old hidden replay is complete, but the labels remain provisional. First label a small, reset-grouped set of truly visible, gripper-covered, cable-covered, offscreen, and wrong-request SC triplets from original RGB; count `uncertain` separately. Replay the frozen SC wrong-route case through the new identity head **plus** the existing native keypoint/reprojection check. In parallel, diagnose the already-saved SFP wrong-port case with a task-conditioned identity test; this SC checkpoint does not fix SFP. Budget: one day of label review and one GPU-hour of frozen replay, with no new training or routine collection. Continue to integration only if there are zero false accepted SC/SFP handoffs, ≥95% recall on reviewed clear arrivals, and then complete **live** p95 below 300 ms. If those fail, repair the specific label/identity failure before actor work. The local insertion actor and RL remain gated.
