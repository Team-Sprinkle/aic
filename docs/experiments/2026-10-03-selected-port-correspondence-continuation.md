# Selected-port handoff continuation: SC posture and SFP correspondence

**Subsequent fixed-gate result:** The frozen integrated review failed on the known SC wrong-posture stop. A bounded one-port SC orientation continuation improved independent new-reset accuracy but did not reach 3°. See [the complete frozen gate and repair record](2026-10-03-frozen-handoff-gate-and-orientation-repair.md). This page preserves the earlier SFP correspondence variants and their original decision.

**Date:** 2026-10-03. **Decision:** no automatic transport-to-insertion handoff or actor promotion. The final scenes remain sealed. All reported geometry is for training labels or post-run scoring; autonomous perception receives saved RGB, public task code, measured TCP, and camera calibration only.

## Frozen SC wrong-route replay

The spatial SC identity checkpoint from [the earlier grouped experiment](2026-10-02-sc-absent-port-identity.md) was replayed on saved three-camera transport-stop observations without changing the route, scene, or checkpoint. It accepted **10/10** nearby proposals around the known wrong-route stop and **9/9** around the valid-route stop. The wrong stop had a scored TCP orientation error of **6.71°** even though its live visual orientation estimate was about **0.10°**. The target port was visibly present in both scenes. Thus selected-port identity alone cannot reject a wrong approach posture; the runtime orientation/standoff estimate needs an independent check. This result does not show a new autonomous insertion.

Commands: `scripts/replay_sc_identity_on_transport_stops.py` with the frozen `sc_spatial_identity_probe_v2/spatial.pt`, saved traces `live_sc_visual_standoff_llb_0009_v7_fastresize` (68.15 s) and `live_sc_visual_standoff_llb_0009_v11_recent_stop` (43.1 s), a ±2 s window, and GPU 1. The complete [wrong-route replay](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_probe_v2/wrong_route_stop_replay_v1.json) and [valid-route replay](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_probe_v2/valid_route_stop_replay_v1.json) contain per-camera scores and RGB paths.

The SC stop code uses a learned relative-orientation probe. At the v7 false stop, its **predicted remaining rotation** was 0.10°, yet the measured TCP was **6.18° from the training-set mean orientation** and **6.71° from the scored goal**. The v7 run used yaw blend 1.0. A later v11 run of the same task used 0.5 blend; at its valid stop, measured TCP was 0.54° from the mean and 2.07° from the scored goal. These runs also differed in route, so this is evidence of a biased or poorly calibrated orientation probe, not proof that changing blend alone cures the stop.

I replayed a 3° measured-TCP-to-training-mean guard on the six frozen SC traces with visual stops. It would reject the known false stop, but also reject **two of five valid stops**. A fixed orientation prior therefore loses too many good arrivals and is **not a repair**. The [per-trace guard replay](../../outputs/experiments/2026-10-01_transport_reliability/sc_orientation_prior_stop_replay_20261003.json) is post-hoc development analysis; the threshold was considered after seeing the failure. The next orientation check must estimate the requested port frame from observation or use a separately validated target-dependent orientation cue, then test clear and wrong-posture stops on grouped resets.

## Saved SFP wrong-port diagnosis

The frozen SFP cascade proposed the **neighboring card's port** twice in a saved `card1_port1` development reset, at 3.15 and 5.05 s. The requested `sfp_card_1_port_1` opening projected inside all three views; the predictions were approximately **150–172 native pixels** from that opening and **0.6–1.4 pixels** from `sfp_card_2_port_1`. This is a correspondence error, not an offscreen-request error that a binary presence head alone could cure. A different saved `card4_port0` reset contained two false accepts when the requested port was offscreen and the proposal landed on card 2. [Exact four wrong proposals and image paths](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sfp_wrong_port_correspondence_v1.json) were scored after inference with `scripts/audit_sfp_wrong_port_correspondence.py`.

## Bounded candidate-to-request probe

`scripts/train_sfp_candidate_correspondence_probe.py` freezes the earlier `joint_locator_v1` RGB backbone and fits a small spatial verifier conditioned on requested SFP card/port, camera identity, measured TCP, and a **vision-proposed image point**. The scored projection supplies positive/negative training pairs only. `scripts/evaluate_sfp_candidate_correspondence.py` replays the *actual* frozen cascade's three-camera proposals. The all-three threshold was fixed at 0.5. Source data had **234 complete training groups** and **32 original validation groups** (2,807/382 sampled camera views); there was no transition-random split. The actual offscreen pilot added **four training groups and two untouched development groups**. It supplied 262 capped training views where the requested port is offscreen but a distractor is visible; one development group supplied 105 such views. The same six recordings also contain 480 capped training and 240 development *onscreen requested-port* views. They are perception examples, not expert insertion trajectories.

| Probe | Saved wrong proposals rejected | Saved correct proposals retained | New held-out offscreen distractor views rejected | New held-out true-point views retained |
| --- | ---: | ---: | ---: | ---: |
| Original point verifier v1 | 2/4 | 69/69 | not tested | not tested |
| Request-swap v2 | 2/4 | 69/69 | not tested | not tested |
| Actual-offscreen-only v3, weight 0.5 | 4/4 | **0/69** | 105/105 | not included |
| Offscreen-only v4, weight 0.1 | 4/4 | **11/69** | 105/105 | not included |
| Matched onscreen+offscreen v5, weight 0.5 | **4/4** | **69/69** | **105/105** | **240/240** |

The v3 collapse was not caused by one-pixel keypoint jitter: replacing each correct proposal with the **exact scored point**, for diagnosis only, still accepted 0/69; zeroing normalized measured TCP state accepted only 1/69. The actual proposals had 0.3–1.2 px mean error to their requested opening. The evidence supports a scene/source shortcut when new offscreen negatives lacked matched positives. Adding both labels from the *same reset groups* removed that observed shortcut; lowering the negative weight alone did not. This is an inference from the controlled ablations, not proof of the learned mechanism.

V5 did retain all 21 accepted, requested-port proposals from a fresh frozen cascade replay on the **two held-out offscreen-pilot resets**. That cascade produced **no accepted wrong-port proposal** there; wrong-port rejection on these two resets remains untested. On those 21 accepted frames the cascade scored lateral position p95 **0.219 mm**, axial p95 **0.207 mm**, and orientation p95 **5.78°**. Three of 24 sampled frames were not accepted by the cascade. A preloaded-image perception-only benchmark measured p95 **18.62 ms** over 24 frames, with a cold-start p99 outlier around 331 ms; it excludes ROS delivery, handoff policy, insertion actor, and controller. Complete live p95 <300 ms is unmeasured.

V5 still falsely accepted **3/382** nearest-neighbor *point* negatives and **1/382** counterfactual request swaps on the original 32-group validation split, although it retained all 382 positive points. Those point-pair errors and the small, concentrated actual-proposal replay prevent a safety claim. The old saved replay has 69 correct frames from only one reset and four wrong frames from two resets. The new replay has 21 correct accepted frames from two resets and no wrong accepted frames. Neither is a reliable estimate of full-episode false handoff rate.

### Commands and artifacts

The exact scene configs, reset hashes, scored bags, native RGB and projection labels are in [the fixed six-scene manifest](/home/nas/chmin/aic_archives/2026-10-03_sfp_offscreen_request_v1/manifest.json) and its `projection_all_v1/rows.jsonl`. `scripts/prepare_sfp_offscreen_request_scenes.py` generated the configs. Each ran sequentially with rootless Docker using `AIC_AUDIT_GPU=2 DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock bash scripts/run_recovery_gazebo_diagnostic.sh <scene-dir> stock`; `scripts/build_sfp_native_projection_labels.py` built the scored labels. No final scene was opened.

```bash
common=/home/nas/chmin/aic_archives/2026-10-03_sfp_offscreen_request_v1
.pixi/envs/default/bin/python scripts/train_sfp_candidate_correspondence_probe.py \
  --labels outputs/experiments/2026-10-02_two_family_perception/projection_labels/rows.jsonl \
  --sfp-cache /var/tmp/chmin_aic_20260918_act/cache_all_verified_sfp_sc_v3 \
  --locator outputs/experiments/2026-10-02_two_family_perception/joint_locator_v1/best.pt \
  --offscreen-labels "$common"/projection_all_v1/rows.jsonl \
  --balanced-native-onscreen --counterfactual-requests --offscreen-loss-weight 0.5 \
  --updates 800 --device cuda:1 --output "$common"/correspondence_v5_balanced_native
.pixi/envs/default/bin/python scripts/evaluate_sfp_candidate_correspondence.py \
  --labels outputs/experiments/2026-10-02_two_family_perception/sfp_perturbed_recovery_development_projection_2/rows.jsonl \
  --cascade outputs/experiments/2026-10-02_two_family_perception/sfp_recovery_development_cascade_baseline_100frames.json \
  --correspondence "$common"/correspondence_v5_balanced_native/correspondence.pt \
  --output "$common"/correspondence_v5_balanced_native/old_wrong_proposal_replay_v1.json \
  --device cuda:1
```

The [V1](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sfp_candidate_correspondence_v1/summary.json), [V2](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sfp_candidate_correspondence_v2_swap/summary.json), [V3](/home/nas/chmin/aic_archives/2026-10-03_sfp_offscreen_request_v1/correspondence_v3_actual_offscreen/summary.json), [V4](/home/nas/chmin/aic_archives/2026-10-03_sfp_offscreen_request_v1/correspondence_v4_weak_offscreen/summary.json), and [V5](/home/nas/chmin/aic_archives/2026-10-03_sfp_offscreen_request_v1/correspondence_v5_balanced_native/summary.json) summaries preserve checkpoints and original grouped metrics. V3's [true-point and state ablation](/home/nas/chmin/aic_archives/2026-10-03_sfp_offscreen_request_v1/correspondence_v3_actual_offscreen/ablation_replay_v1.json), V5's [old failure replay](/home/nas/chmin/aic_archives/2026-10-03_sfp_offscreen_request_v1/correspondence_v5_balanced_native/old_wrong_proposal_replay_v1.json), [new reset cascade](/home/nas/chmin/aic_archives/2026-10-03_sfp_offscreen_request_v1/new_development_cascade_12frames_v1.json), and [new reset point replay](/home/nas/chmin/aic_archives/2026-10-03_sfp_offscreen_request_v1/correspondence_v5_balanced_native/new_development_proposal_replay_v1.json) contain per-frame results.

## Gate and next bounded test

The SFP correspondence subtest improves on the frozen wrong-port failures, but **the overall handoff gate remains open**. SC still has a wrong-posture accepted stop; SFP has original grouped point-negative errors; visually hidden or occluded selected ports lack certified identifiability labels; complete live latency and autonomous insertion remain untested. V5 is a diagnostic checkpoint, not a deployed verifier. Do not train the local actor or RL from this result.

Next use **existing saved non-final RGB traces** to construct one episode-grouped, image-reviewed test containing: SC wrong-posture proposals, SFP different-card wrong proposals, clear requested-port proposals, and cable/gripper occlusion. Freeze all models and the 0.5 threshold first. Budget: at most one workday of review and one GPU-hour of replay, with no routine scene collection or retraining. Stop if any known wrong or unsafe proposal is accepted, clear requested-port recall is below 95%, or orientation/standoff cannot be checked independently. Only then attach a *shadow* pause/inspect/resume controller, benchmark complete live p50/p95/p99 (<300 ms p95), and measure full autonomous insertion on new development starts. Final scenes stay sealed.
