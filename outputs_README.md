> **October 18 score target achieved:** **290.19/300**, all three public-sample insertions pass independent depth/lateral/orientation checks; all used-stage p95 inference is below300ms. The analyze/repair/evaluate loop stops after18 completed full samples. V56 preserves three quiet SCcontext frames while sharing initial grasp observations, and seats1.5s above the public1s requirement. Rule transport and explicit RGB/public-state insertion use unchanged perception weights; no learned BCactor or RLactions. Failures and unqualified fits are preserved. This is one selected public-sample pass, with small SCdepth margin and unresolved unusual-SFP-grasp coverage; final scenes stay sealed. Earlier sections are historical. [Results, commands and videos](docs/experiments/2026-10-18-five-repair-cycles.md).

## October 18: gated close-up pose update and SC insertion result

V34 passes **4/4 matched SC development cases** (zero/five obstacle cards, both ports), followed by a passing SC round in the unchanged public sample: **15.510 mm depth, 0.141 mm lateral error, 0.469° orientation**, toolkit completion, no harm. Five fresh agreeing images support one bounded plug-pose update at 25 mm; the port frame stays fixed. Tighter lateral centering and stronger lateral tracking are also needed. All used-stage p95 timings pass; these are explicit RGB/public-state control results, not BC or RL.

The complete sample is **170.57/300, 2/3 strict** because SFP trial 2 has a 3.53° biased grasp estimate before contact. Its actual roll exceeds the original SFP training range; close-up predictions remain wrong. Repeats stop, and the next targeted repair is SFP orientation coverage rather than a stiffness-only change. No new training or collection is running. The SC model's original 1.026° qualification failure remains explicit; four successful development cases do not establish general reliability. Final scenes remain sealed.

[Detailed evidence, commands, images and next repair](docs/experiments/2026-10-18-sc-explicit-controller-continuation.md).

## October18 close-up SC insertion milestone

`/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1/` now includes the successful `V21_close_grasp_development/`, matched failed `V25_no_close_matched_ablation/`, and `sample_V21_experimental/` (**3/3strict,257.36/300**). Per-trial physical JSONs/videos are under `sample_V21_experimental/per_trial/`; repeated sample state is `V21_three_sample_state.json`. Models and original sources are preserved; no new training/collection is queued. [Record](docs/experiments/2026-10-18-sc-explicit-controller-continuation.md).

## October18 explicit SCcontinuation

`/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1/` preserves fullsampleV17/repeat2scores, four freshSCcases, V18/V19/V20/V22failures,12near25mmperception holds,36-case acquisition curation, paired re-estimation/codec/solver audits, and completed plateau fits. `spacefilled_near_pose_dataset/` contains3,446 perception triplets across294 episodes (234 train/60 validation), including80 near training grasps and four unchanged near validation episodes. `spacefilled_near_fused_orientation_warm/` reaches1.026° near orientationp95 and is not qualified. `fine_feature_near_orientation/` is the one active architecture ablation; `final80_five_fresh_consensus/` preserves the held-out averaging diagnosis. These are explicit image controllers, not BC/RL successes. [Plan](docs/experiments/2026-10-18-sc-explicit-controller-continuation.md).

> **October 18 insertion repair:** Commands, plateau-training histories, split manifests, controller-context/persistence audits and frozen control comparisons are under `/home/nas/chmin/aic_archives/2026-10-17_insertion_control_repair_v1/` and `2026-10-17_insertion_control_repair_v2/`. Bounded nominal-centred SC perception captures: `2026-10-17_sc_nominal_grasp_capture_v1/`. Original sample result stays **165.42/300, 0/3 strict insertion**. [Record](docs/experiments/2026-10-17-insertion-control-and-grasp-repair.md).

> **October17 complete public sample:**165.42/300toolkit points,3/3near handoffs,0/3strict physical insertions; latest full-connector BC actor, recovery disabled. Results, scores, raw bags, per-trial videos and review images: `/home/nas/chmin/aic_archives/2026-10-17_full_connector_sample_configuration_v2/`. [Analysis and next repair order](docs/experiments/2026-10-17-full-connector-sample-configuration.md). Earlier running/queued blocks are historical; no newRL or final-scene access.

> **Four-rollout insertion/jerk probe complete:** `/home/nas/chmin/aic_archives/2026-10-17_full_connector_live_probe_v1/` contains all4raw runs, independent scores, corrected pre-freeze motion analysis, grasp-angle audits and regular H.26420fps clips in `videos/`. Strict physical insertion1/4; no queued retrain. [Record](docs/experiments/2026-10-17-full-connector-insertion-jerk-probe.md).

> **Orientation results and new BC diagnostic:** SC/SFP orientation models, prediction/GT overlays, teacher-phase and oscillation audits, and the full-SE(3) command derivative are under `/home/nas/chmin/aic_archives/2026-10-17_connector_orientation_and_sfp_oscillation_v1/`. See `continuation_state.json` for the one queued/running actor fit. [Results](docs/experiments/2026-10-17-connector-orientation-and-sfp-oscillation.md).

> **October 17 orientation/oscillation artifacts:** `/home/nas/chmin/aic_archives/2026-10-17_connector_orientation_and_sfp_oscillation_v1/` holds full connector-pose training, immutable split manifests and SFP command/motion diagnosis. Prior 14-rollout comparison is complete with zero strict full insertions. [Record](docs/experiments/2026-10-17-connector-orientation-and-sfp-oscillation.md).

<!-- history-horizon-live:start -->
## Live history/chunk experiment

- Stage: `frozen_near_port_live_comparison`. Completed training variants: **4/4**.
- New trained-model live comparison: **14/14** complete; no insertion promotion inferred from training.
- Runtime preserves measured transport history; accepted port pose stays fixed during the actor attempt. New acquisition uses fresh pose-window consensus.
- Same60,671anchor samples for4/8/12step targets; recorded commands only. No newRL or reserved-final testing.
- Artifacts: `/home/nas/chmin/aic_archives/2026-10-17_history_chunk_continuation_v1`.
<!-- history-horizon-live:end -->

> **October17 revised priority:** Transport now targets a visually/proprioceptively controlled **25mm stand-off**, superseding the earlier learned150mm-approach responsibility. The first SC pilot reaches24.20mm/0.547mm lateral, but the unchanged actor still fails full physical insertion. Audit confirms all four50ms insertion targets execute with zero insertion-stage stale drops; large replanning jumps, phase imbalance and unapproved plug-grasp orientation remain gaps. A separate four-run fresh/warm SC/SFP comparison is active on GPU5, recovery off to isolate the approach/actor; no new training or official tuning. [Audit and next decisions](docs/experiments/2026-10-17-near-handoff-and-insertion-continuity-audit.md).

> **Full-rollout scope update:** One episode per scenario, **62 total**, supersedes the earlier 186-episode schedule. Separate complete sample/eval YAML tests remain three repetitions each. The current rollout finishes before the scheduler hands over; models, rules and scenes remain unchanged. See [evaluation record](docs/experiments/2026-10-16-full-pipeline-three-per-scenario.md).

<!-- full-insertion-evaluation-live:start -->
## Full insertion evaluation continuation

- Stage: `scenario`. Artifacts: `/home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1`.
- Frozen rule transport → requested-port coarse/native refinement → shared 150 mm BC insertion actor.
- Planned: 62 scenario rollouts (62 groups × one), plus three complete sample YAML runs and three complete released-evaluation YAML runs.
- Hierarchical insertion recovery is rule-based; no new RL or training. Released evaluation is scoring only; reserved final scenes stay sealed.
- Counts and scores: see `progress.json`, `results_by_scenario.json` and `configuration_scores.json` in the artifact folder.
<!-- full-insertion-evaluation-live:end -->

<!-- sample-transport-live:start -->
## Live sample-derived transport results — 2026-10-16T20:25:57 UTC

- Adaptive SC development: **7/7 qualified**, seven planned.
- Unchanged SFP development: **22/22 qualified**.
- Frozen transport/handoff evaluation: **121/130 qualified**,130planned. This is not insertion.
- Rules/weights frozen: **True**. Stage: `frozen_transport_evaluation`.
- Decision: `frozen_evaluation_complete_requires_review`. No new training or reserved-final testing.
- Artifacts: `/home/nas/chmin/aic_archives/2026-10-16_sample_adaptive_transport_evaluation_v3`. Sample-derived and exposed official results stay separate; no evaluation tuning.
<!-- sample-transport-live:end -->

## Sample-derived conservative transport continuation — October16

Artifacts: `/home/nas/chmin/aic_archives/2026-10-16_sample_transport_continuation_v1/` (coordinator and gated frozen evaluation), `/home/nas/chmin/aic_archives/2026-10-16_sample_conservative_development_v3/` (29-case sample-only development), and `/home/nas/chmin/aic_archives/2026-10-16_sample_conservative_transport_v1/default3_v1/` (3/3 initial sample handoffs). [Experiment record](docs/experiments/2026-10-16-sample-conservative-rule-transport.md). No insertion or model training. The initial SC video is `default3_v1/sample_trial_3/five_cameras_20fps_grid_active.mp4` under the pilot root.

## October 16 development/test separation

`2026-10-16_official_transport_recheck_v1` is retrospective evaluation-only; its exact and derived stress scenes are excluded from training, rule design and threshold selection. See its `development_use_restriction.json` and [evaluation protocol](docs/EVALUATION_PROTOCOL.md). The pre-deadline development reference is `configs/evaluation_protocol/qualification_public_predeadline/{sample_config.yaml,provenance.json}`. Existing checkpoints need lineage auditing before a clean-development claim.

# Outputs Directory Guide

<!-- automatic-live-progress:start -->
## Latest recorded job state — 2026-10-16 07:24 UTC

This block supersedes earlier running/queued wording. Acquisition counts are **not insertion counts**. Completed update budgets are not automatically convergence.

- **BC fresh**: stop_reason=update_cap, updates_completed=30000, best_step=4600, stale_evaluations_at_end=254, summary_exists=True.
- **BC visual warm**: stop_reason=update_cap, updates_completed=30000, best_step=23100, stale_evaluations_at_end=69, summary_exists=True.
- **SC frozen acquisition new refiner effective**: completed=60, planned=60, qualified=44, wrong=0, harm=0, passed=False, episodes=60, wrong_accepts=0, summary_exists=True.
- **corrected intended acquisition20**: completed=20, planned=20, passed=False, family_groups={'sc': {'episodes': 10, 'qualified': 9}, 'sfp': {'episodes': 10, 'qualified': 9}}, episodes=20, qualified=18, wrong_accepts=0, summary_exists=True.
- **SC candidate acquisition20**: completed=20, planned=20, passed=True, family_groups={'sc': {'episodes': 20, 'qualified': 18}}, episodes=20, qualified=18, wrong_accepts=0, summary_exists=True.
- **autonomous insertion comparison**: status=perception_gate_failed, summary_exists=True.
- **conditional BC continuation**: extended=False, summary_exists=True.

Full-system promotion, new RL and reserved-final testing remain pending. Exact commands, weights, metrics and preserved failures are under `/home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1`.
<!-- automatic-live-progress:end -->


## October 15 rule/perception/hierarchical continuation

Bulk data and checkpoints are on NAS: `/home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1/`. `state.json` maps active artifacts; the [execution record](docs/experiments/2026-10-15-rule-perception-hierarchical-execution.md) distinguishes component results, live acquisition, privileged teachers and pending actor/RL work. Preserve both accepted and failed comparisons. Native grasp labels, command derivatives and new crops are separate copies; original experts are unchanged.


## October 15 frozen coverage evaluation — complete

Bulk root: `/home/nas/chmin/aic_archives/2026-10-14_frozen_pipeline_coverage_v1/`. All 140 planned rollouts completed; 28 groups have exactly five episodes. See [the full record](docs/experiments/2026-10-14-frozen-pipeline-coverage.md) for commands, exact model roles, failures and representative video paths. `plan.json` freezes 120 scenes; `summary.json`, `coverage_cells.csv`, `decision.json`, `approach_failure_diagnosis.json` and `final_frozen_hash_verification.json` are the authoritative outputs. `runs/` contains images, raw bags, physical tip curves, official scoring, videos and preserved invalid attempts. No full SC insertions passed; SFP was transport only and never visited its broad diagnostic approach region. No promotion, training, RL or final-scene access.


## October 14 autonomous transport execution artifacts

Bulk artifacts live onNAS: `/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/`. Main index: [execution record](docs/experiments/2026-10-14-transport-handoff-execution.md) and its JSON ledger. Important entries:

- `rule_tracked_fresh_arrival_metrics_v11.json`: sixfreshcorrectSCacquisitions, scoring-onlygeometry and per-scene latency.
- `tracked_crop_fresh_scenes_v1/manifest.json`: exactdevelopment resetgroups, nofinalscenes.
- `rule_common_llb_1405_v11_fresh/five_cameras_20fps_grid_active.mp4`: five-cardrail-limittransportvideo.
- `local_integrated_warm_llb_1200_teacher_gains_v3/five_cameras_20fps_grid_active.mp4`: localBCpilot with recordedteachercontrollergains.
- `cross_component_summary_v1.json`: preservedpre-timestamp-discoverycomponentresults; earliergraspprobe interpretation superseded bycorrectedlabels below.
- `sc_grasp_observation_labels_v3/timestamp_audit.json`: verifiedcommand/observationclock repair; originaldata unchanged.
- `sc_native_grasp_pose_v3_correct_time/` and `sc_grasp_keypoint_probe_v3_correct_time/`: correctlytimedmodels/checkpoints/predictions/metrics.
- `physical_camera_tcp_provenance_v2.json`: diagnostic-onlyphysicalcamera/FK comparison, postterminalposes excluded.
- `grasp_corrected_prediction_only_proof/`: network-onlymarkers/axes onheld-outframes; no truthmarkers.

Officialinsertionevents andstrictphysicalinsertion areseparate. Do notlabel shallow75-scorepilots as physicallyfull insertion. Models/failedtrials/rawdata arearchived, not git assets.


**October 14 active artifacts:** `/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/` contains endpoint predictions and marked images, repaired timestamp targets/provenance, rigid corridor screens, oracle executor pairs/videos, fixed-update actor fits, and preserved failed rule pilots. Read `docs/experiments/2026-10-14-transport-handoff-execution.json` before resuming or duplicating jobs. No final scenes or RL promotion.

**October 13 independent transport review:** CPU-only path plots, all-episode label audit, endpoint rescoring, source scripts and hashes are in `/home/nas/chmin/aic_archives/2026-10-13_transport_independent_review/`. This corrects interpretation of the older approach-band failures; no model was promoted. [Report and next plan](docs/experiments/2026-10-13-transport-independent-root-cause-review.md).

**October 13 SC pause/context and intentional-fit comparison:**
`outputs/experiments/2026-10-13_sc_pause_models_v1/` contains the
[pause audit](outputs/experiments/2026-10-13_sc_pause_models_v1/pause_audit.json),
[six-scene comparison](outputs/experiments/2026-10-13_sc_pause_models_v1/live_comparison_v2.json),
and container-visible 100 ms, dense 70 s, and intentional-fit 50 ms checkpoints. Immutable
100 ms labels/cache, complete training histories, 18 scored development
rollouts, two extra training-scene rollouts, five-camera videos and trace
analyses are under `/home/nas/chmin/aic_archives/2026-10-13_sc_*` and
`/home/nas/chmin/aic_archives/2026-10-13_route_*100ms*`. The original scored
teachers and previously cleaned derivative remain intact. All three new actors
reached 0/6 approach bands and insertions. [Full record and exact paths](docs/experiments/2026-10-13-sc-pauses-context-and-overfit.md).

**October 13 SC 12/6 autonomous reevaluation:**
`outputs/experiments/2026-10-13_sc_horizon12_reeval_v1/` contains the
[nine-run machine summary](outputs/experiments/2026-10-13_sc_horizon12_reeval_v1/summary.json),
[axial route plot](outputs/experiments/2026-10-13_sc_horizon12_reeval_v1/axial_progress.png),
and two saved-input previous-command ablations. The six new rootless Gazebo
rollouts, scored bags, source scenes, controller traces and five-camera videos
are under `/home/nas/chmin/aic_archives/2026-10-13_route_horizon12_execute6_sc_{repeat,newdev}_v1/`.
They reached 0/6 approach bands and 0/6 insertions without scored contact.
[Experiment record](docs/experiments/2026-10-13-sc-horizon12-reevaluation.md).

**October 13 transport action-horizon comparison:**
`outputs/experiments/2026-10-13_route_horizon_comparison_v1/` contains the
container-visible selected 8/4 and 12/6 checkpoints, the [machine summary](outputs/experiments/2026-10-13_route_horizon_comparison_v1/summary.json),
and [training curves](outputs/experiments/2026-10-13_route_horizon_comparison_v1/training_curves.png).
The full 200/300 ms feature/label caches, five bounded training phases,
complete scored bags and images for each 4/4, 8/4, and 12/6 live arm, and
the two-scene faster-sampling diagnostic are under
`/home/nas/chmin/aic_archives/2026-10-13_route_*`.
All three matched autonomous arms reached 0/6 safe buffers and 0/6 insertions.
[Experiment record](docs/experiments/2026-10-13-route-action-horizon.md).

**October 12 teacher-generator and released-eval route diagnosis:**
`/home/nas/chmin/aic_archives/2026-10-12_teacher_distribution_route_live_v1/`
has six exact held-out SFP/SC teacher-scene autonomous rollouts, frozen
`plan.json`, scene configs, bags, traces, frames, scores, and representative
five-camera videos. The first JPEG-86-named SFP archive is an unchanged-policy
repeat because its launcher omitted the Docker environment variable. It is at
`/home/nas/chmin/aic_archives/2026-10-12_teacher_distribution_sfp_codec86_v1/`;
the separate three-scene released qualification-family check is at
`/home/nas/chmin/aic_archives/2026-10-12_official_sfp_codec86_check_v1/`.
The local `outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/`
contains `teacher_distribution_analysis.json`,
`teacher_distribution_codec86_analysis.json`, `official_codec86_analysis.json`
(both unchanged-policy repeats despite their historical names),
and `teacher_distribution_decision.json`. Teachers inserted 6/6; actor 0/6.
No checkpoint was promoted. [Experiment record](docs/experiments/2026-10-12-teacher-distribution-transport-diagnosis.md).

**October 12 transport feedback and inference-cadence follow-up:**
`outputs/experiments/2026-10-12_route_consistent_diffusion_transport_v1/`
also holds four `input_ablation_*.json` saved-input replays, `zero_prior_ft_v1/`
and `lagged_prior_ft_v1/` bounded fine-tunes with their matched offline and
frozen-frame reports, `sample4_teacher_pilot_analysis.json`, and
`feedback_cadence_decision.json`. Complete true-codec and four-step Gazebo
bags, frames, traces, and videos are under NAS roots
`/home/nas/chmin/aic_archives/2026-10-12_teacher_route_true_codec_prewarm_v1/`,
`.../2026-10-12_teacher_route_true_codec_no_prewarm_v1/`, and
`.../2026-10-12_teacher_route_sample4_v1/`. The four-step SFP pilot reached
the buffer but contacted a card; SC missed. Neither pilot inserted.
[Full interpretation](docs/experiments/2026-10-12-route-feedback-and-inference-cadence.md).

**Actual Gazebo MoveIt replay:**
`/home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/gazebo_case5_live_replan_v1/`
contains the exact nonfinal randomized scene, live-start MoveIt route,
official scoring bag, measured waypoint/force trace,
`gazebo_replay_summary.json`, and the [five-camera H.264 recording](</home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/gazebo_case5_live_replan_v1/video/all_views_20fps.mp4>).
The separate `gazebo_case5_retreat_v1/` attempt aborted before executing
the saved path because the start pose differed by 23.96 mm. The successful
replay reached the buffered goal, with no insertion attempt. [Full result](docs/experiments/2026-10-08-sfp-requested-port-corridor-audit.md#requested-gazebo-replay-actual-robot-and-cable-motion).

**October 8 requested-port SFP corridor audit:**
`/home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/`
contains six task-selected MoveIt corridor plans with full TCP/joint paths,
scored actor-state provenance, `cable_proxy.json`, teacher measured-cable
comparison, [`cable_proxy_comparison.png`](</home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/cable_proxy_comparison.png>),
and `decision.json`. All six rigid plans succeeded, but only three cleared
the conservative full-cable screen; the safe-route gate stayed open. No
Gazebo replay, new checkpoint or autonomous video resulted.
[Case 2 offline planned-path video, H.264](</home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/case2_retreat_offline_h264.mp4>)
shows a frozen takeover camera image beside the animated planned TCP path;
it does not show physical cable motion.
[Full audit](docs/experiments/2026-10-08-sfp-requested-port-corridor-audit.md).

**October 8 SFP actor-state MoveIt probe:**
`/home/nas/chmin/aic_archives/2026-10-08_sfp_actor_state_moveit_six_v5_paths/`
contains exact commands, six saved nonfinal actor-state rigid plans or
no-target result, measured joint/TCP contracts, `shortcut_audit.json`, and
the inspected [`measured_vs_moveit_paths.png`](</home/nas/chmin/aic_archives/2026-10-08_sfp_actor_state_moveit_six_v5_paths/measured_vs_moveit_paths.png>).
Two development paths shortcut the measured expert detour. Earlier v1–v4
archives preserve the 20 mm no-target, missing late raw bag, and partial
teacher-case diagnoses. No Gazebo replay or actor training followed.
[Full record](docs/experiments/2026-10-08-sfp-actor-state-moveit-feasibility.md).

**October 8 actor-state waypoint-label feasibility:**
`/home/nas/chmin/aic_archives/2026-10-08_sfp_actor_recovery_label_pilot_v1/`
preserves the three immutable train-scene requests, exact command
environments, scored teacher-takeover bags, streams, scores, labels,
`summary.json`, `path_diagnosis.json`, `first_net10_audit.json`, and the inspected
[`takeover_tcp_paths.png`](</home/nas/chmin/aic_archives/2026-10-08_sfp_actor_recovery_label_pilot_v1/takeover_tcp_paths.png>).
Two teachers fully inserted but only one yielded a valid 20 mm-path
waypoint, so the predeclared pilot gate failed. There is no new actor
checkpoint or autonomous video. [Interpretation and next gate](docs/experiments/2026-10-08-sfp-actor-recovery-label-feasibility.md).

**October 8 SFP short waypoint feasibility:**
`outputs/experiments/2026-10-08_sfp_short_waypoint_feasibility_v1/`
holds the ridge probe, frozen one-second labels, exact commands/logs, offline
servo audit and six actor-boundary direction records. `arc20_labels/`
preserves the separate 20 mm measured-path ablation and its episode manifest.
Two matched 2,000-update one-GPU neural comparisons (teacher-only and
15%-boundary sampling) are archived under
`/home/nas/chmin/aic_archives/2026-10-08_sfp_short_waypoint_neural_matched_v1/`
and `.../2026-10-08_sfp_arc20_neural_matched_v1/`, including commands,
checkpoints, histories and grouped metrics. The live servo gate failed
(0/6 actor-boundary directions), so there are **no new videos or autonomous
rollouts**. [Record and next gate](docs/experiments/2026-10-08-sfp-short-route-waypoint-feasibility.md) ·
[machine decision](docs/experiments/2026-10-08-sfp-short-route-waypoint-feasibility.decision.json).

**October 8 SFP actor initialization diagnostic:**
`outputs/experiments/2026-10-08_sfp_pose_actor_training_diagnostic_v1/`
contains all-nine-episode causal joins, two 500-step capacity checks,
the full-data matched score, exact run/score commands, logs and `decision.json`.
The two 2,000-update warm and reset-head checkpoints, manifests, histories
and commands are under NAS archive
`/home/nas/chmin/aic_archives/2026-10-08_sfp_pose_partial_restart_matched_v1/`.
No live run or full-scratch model followed the failed pose-use gate.
[Interpretation](docs/experiments/2026-10-08-sfp-pose-actor-training-method.md).

**October 7 selected-port route-feedback ablation:**
`outputs/experiments/2026-10-07_sfp_route_branch_support_v1/` now also has
the saved-actor selected-port shadow predictions, a
[`predicted-versus-true original-image sheet`](outputs/experiments/2026-10-07_sfp_route_branch_support_v1/offroute_port_shadow_v1/first_accepted_prediction_vs_truth.jpg),
the 100-prediction geometry audit, action-input-use checks and
`route_feedback_decision.json`. The frozen 19,955-row native-image route
offset cache is
`/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/selected_port_route_values_v1/`.
Matched zero-versus-predicted-pose actor checkpoints, command lines and
failed offline gates are in NAS archives
`2026-10-07_sfp_port_route_feedback_matched_v1/` and
`2026-10-07_sfp_port_route_recovery_matched_v1/`. Neither was run live or
promoted. [Experiment and decision](docs/experiments/2026-10-07-sfp-early-branch-support.md#route-feedback-ablation-result-the-pose-is-visible-but-this-actor-ignores-it).

**October 7 current SFP transport continuation:**
`outputs/experiments/2026-10-07_sfp_route_branch_support_v1/` contains
early-route probes, strict context audits and `late15_decision.json`.
`/home/nas/chmin/aic_archives/history_context_live_scenes_v1/` holds the
independent three-scene 0/3-versus-0/3 autonomous comparison, official scores,
videos, causal traces and port-approach plot. The six safe 15-second
train-only takeovers are in `2026-10-07_sfp_late15_train_port{0,1}_v1/`;
`2026-10-07_sfp_late15_matched_v1/` contains matched 2,000-update
checkpoints, command lines, logs and the failed 3.5%-gain offline gate.
The untouched next live plan is
`/home/nas/chmin/aic_archives/history_context_late15_remaining_live_v1/`;
its scenes were **not run**. The immutable merged labels and 55,206-row
feature cache are under
`/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/` with names
`sfp_late15_context_merged_v1` and `temporal_cache_late15_v1`.
[Full record](docs/experiments/2026-10-07-sfp-early-branch-support.md).

**October 7 later-state SFP actor repair:**
`/home/nas/chmin/aic_archives/2026-10-07_sfp_later_takeover_gate_v1/`
contains six scored privileged full-insertion continuations after actor
prefixes. `2026-10-07_sfp_later_correction_matched_v2/` contains matched
2,000-update checkpoints, logs and the corrected offline gate. The fresh
autonomous three-scene comparison is at
`2026-10-07_sfp_later_correction_live_v1/`: exact plan, official/posthoc
scores, traces, route plot `cards5_routes_xy.png`, four five-camera videos
under `rollouts/`, and `summary.json`. Offline action accuracy improved but
live transport missed its 2/3 gate (corrected 1/3, control 0/3); neither actor
was promoted. [Full record](docs/experiments/2026-10-07-sfp-later-state-causal-repair.md)
· [machine summary](docs/experiments/2026-10-07-sfp-later-state-causal-repair.summary.json).

**October 7 stall-label comparison:**
`/home/nas/chmin/aic_archives/2026-10-07_sfp_paired_actor_comparison_v1/`
contains `data_quality_audit_v1.json`, `stall_filter_v1.json` and
`stall_raw_examples.png`. The separate
`/home/nas/chmin/aic_archives/2026-10-07_sfp_stall_clean_matched_cpu_v1/`
has two CPU checkpoints, exact commands, train logs and grouped/pair/task
scores. The filtered actor was not promoted. [Experiment record](docs/experiments/2026-10-07-sfp-stall-data-audit-and-cleaned-fit.md).

**October 7 fresh actor gate:**
`/home/nas/chmin/aic_archives/2026-10-07_sfp_paired_actor_live_starts_mounted_v2/`
has the fixed three-scene matched plan, successful teacher references,
baseline and candidate policy traces, frames/videos, official scores,
`posthoc_score.json`, `route_motion_diagnosis.json`, and `summary.json`.
Both policies reached 0/3 safe buffered arrivals. The control/candidate
checkpoints, matched training commands and grouped offline scores are at
`/home/nas/chmin/aic_archives/2026-10-07_sfp_paired_actor_comparison_v1/`.
The initial invalid checkpoint-mount launch remains archived at
`2026-10-07_sfp_paired_actor_live_starts_v1/` and is excluded from live
claims. [Full record](docs/experiments/2026-10-07-actor-proposer-coarse-continuation.md).

**October 7 actor/proposer/coarse continuation:**
`/home/nas/chmin/aic_archives/2026-10-07_sfp_diverse_crop_fit_v1/` holds the
selected dense-view requested-port crop checkpoint, training history and
60-sample board-plus-crop latency. Its independent dense stress result is in
`2026-10-07_sfp_diverse_crop_stress18_v1/`, and its older sparse regression
score in `2026-10-07_sfp_diverse_crop_old_sparse10_v1/`. The card occupancy
head is `outputs/experiments/2026-10-07_sfp_diverse_occupancy_head_v1/`.
`outputs/experiments/2026-10-07_sfp_pause_mlp_hardneg_v1/` contains the
frozen-action pause-head checkpoint and decision-by-decision saved-actor
shadow replay. The first scored paired route archive is
`/home/nas/chmin/aic_archives/2026-10-07_sfp_divergent_pairs_v1/`; the second
bounded collection is `2026-10-07_sfp_divergent_pairs_round2_v1/`. Both are
complete; 34 training and nine held-out matched first-action pairs passed
admission. None is
an automatically deployed actor/handoff. [Experiment record](docs/experiments/2026-10-07-actor-proposer-coarse-continuation.md).

**October 7 SFP diverse transport and proposal head:**
`outputs/experiments/2026-10-06_sfp_diverse_transport/` contains the matched
old-data and diverse-data 3,000-update actor checkpoints, grouped split
scores, same-image requested-code swaps, old/new frozen-action pause-head
checkpoints, saved autonomous-route replay and `decision_summary.json`.
`/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/` contains the
two immutable causal admission stages, final merged labels, frozen 50,679-row
three-view feature cache and broad pause-label cache. Raw teacher videos,
scored bags, five partial runs, one genuine retry failure and the preserved
Gazebo crash/rerun are under NAS archives
`2026-10-06_sfp_diverse_teacher_v1/` and `2026-10-07_sfp_diverse_retry_v1/`.
Neither actor nor pause head passed its autonomous promotion gate. [Full
record](docs/experiments/2026-10-06-sfp-diverse-transport-and-proposal.md).

**October 7 SFP visible board-origin audit:** `outputs/experiments/2026-10-07_sfp_outline_corner_review_v1/` contains collision-envelope corner image panels, native zooms, single-reviewer labels and failed contour proposals; `2026-10-07_sfp_visual_screw_projection_v1/` contains projections of actual board-mesh screws and comparison sheets; `2026-10-07_sfp_visual_screw_early_review_v1/` has 40 early channel panels and visibility assessments; `2026-10-07_sfp_one_screw_board_prior_v1/` has exact per-group oracle-screw/true-yaw/train-mean controls. These are read-only diagnostics, not runtime geometry or a trained detector. [Decision and commands](docs/experiments/2026-10-07-sfp-visible-board-origin-audit.md).

**October 7 SFP rail data-size audit:** `outputs/experiments/2026-10-07_sfp_rail_data_scale_v1/` contains nested complete-group manifests (`train_groups_{20,40,70}.json`), matched `fit20/` and `fit40/` checkpoints/predictions/histories, unchanged `temporal20/` and `temporal40/` replays, a [learning-curve plot](outputs/experiments/2026-10-07_sfp_rail_data_scale_v1/learning_curve.png), and `visibility_review/` with 30 original-image panels, rubric and explicit single-reviewer labels. The 70-group checkpoint/replay remain in the adjacent earlier refiner folders. Large generated crop arrays were removed after training to recover space; the exact script/manifests regenerate them from archived original images. None is deployable. [Decision and commands](docs/experiments/2026-10-07-sfp-rail-data-scale-and-visibility.md).

**October 7 SFP board-reference diagnostic:** `outputs/experiments/2026-10-07_sfp_temporal_board_reference_v1/` has frozen-RGB causal board fits; adjacent `_oracle0_v1/` and `_oracle3_v1/` are privileged pixel-projection controls; `2026-10-07_sfp_rail_train_bias_v1/` and `_temporal_board_reference_bias_v1/` preserve failed train-only static-bias results. `2026-10-07_sfp_rail_original_crop_refiner_v1/` contains 16,145 train crops, 570 development proposal crops, model checkpoint/history, corrected predictions, and [diagnostic crop sheet](outputs/experiments/2026-10-07_sfp_rail_original_crop_refiner_v1/development_diagnostic_crops.jpg); `2026-10-07_sfp_temporal_board_reference_refined_v1/` has the failed 3/10 early accurate replay. These are diagnostic artifacts, not deployable target/stop estimators. [Full record](docs/experiments/2026-10-07-sfp-temporal-board-reference.md).

**October 7 structural SFP cue diagnostics:** `outputs/experiments/2026-10-07_sfp_candidate_set_ceiling_v1/` has the frozen all-port candidate-set upper bound; `2026-10-07_sfp_slot_feature_port_cue_v{1,2}/` preserve no-update loader failures and `v3/` the complete pixel-head checkpoint/history/predictions; `2026-10-07_sfp_slot_feature_port_3d_v1/` has the complete shared-3D head and calibration failures; `2026-10-07_sfp_oracle_transport_cue_v{1,2}/` preserve the unscaled and normalized matched privileged offline controls. None is a deployed model. [Experiment, commands and decision](docs/experiments/2026-10-07-sfp-slot-feature-port-cue.md).

**October 7 SFP review and crop audit:** `/home/nas/chmin/aic_archives/2026-10-07_sfp_blind_review_followup_v1/` contains 30 disjoint unmarked RGB/history panels, five contact sheets, task and public TCP state by alias, a review form, and a sealed mapping for later scoring. No independent annotation exists yet. `/home/nas/chmin/aic_archives/2026-10-07_sfp_crop_feasibility_v1/` contains matched 320/480 px grouped-development JSON scores; 320 px missed its predeclared clean-crop gate (6/10 versus 9/10). These are diagnostic artifacts, not a trained model. [Protocol, commands and decision](docs/experiments/2026-10-07-sfp-observation-complete-review-and-tight-crop.md).

**October 6 SFP tilt-view diagnostic:** `/home/nas/chmin/aic_archives/2026-10-06_sfp_board_tilt_view_v1/` preserves offline projection feasibility, the two aborted command-bug runs (`card1_port0/`, `_v2/`), and the corrected +10° pilot (`_v3/`) with original-RGB comparison, blind view annotation, oracle/force events, scored bag and machine summary. The corrected view failed measured-position and identity gates; none of these files is a deployable policy or expert training episode. The separate 40-panel review packet is at `/home/nas/chmin/aic_archives/2026-10-06_sfp_requested_card_blind_review_v1/independent_review_packet.md`. [Record](docs/experiments/2026-10-06-sfp-board-tilt-view.md).

**October 6 visual identity gate:** `/home/nas/chmin/aic_archives/2026-10-06_sfp_requested_card_blind_review_v1/` has 40 unmarked original-RGB panels, manifests and 11 saved partial blind annotations; `sealed_mapping.json` is for post-review audit, not an inference input. `/home/nas/chmin/aic_archives/2026-10-06_sfp_opposite_wide_view_v1/card1_port0/` has the guarded opposite-side pilot, synchronized original camera images, unmarked baseline/view panel, blind annotation, contact/force events, score and machine summary. The view failed its identity gate and is diagnostic only. [Record](docs/experiments/2026-10-06-sfp-requested-card-identifiability.md).

**October 6 requested-card repair artifacts:** `/home/nas/chmin/aic_archives/2026-10-06_sfp_local_card_pair_fit_v1/` preserves the stopped JSON-serialization smoke; `fit_v2/` holds the complete 2,000-update three-point checkpoint, development predictions and summary. `/home/nas/chmin/aic_archives/2026-10-06_sfp_rail_to_port_prior_v1/` holds the 2,000-update rail/TCP offset prior and held-out view predictions. Both failed their gates; neither is deployed. [Record](docs/experiments/2026-10-06-sfp-local-card-pair-repair.md).

**October 6 joint-camera SFP follow-up:** `/home/nas/chmin/aic_archives/2026-10-06_sfp_joint_candidate_fresh10_v1/` holds ten frozen nonfinal reset configs and complete teacher camera/scoring captures. Companion `2026-10-06_sfp_joint_candidate_fresh10_*` directories hold 528 image/state triplets, 527 retained board rows, frozen RGB rail predictions, native crop proposals, joint and persistent scores, and a matched single-point score. `2026-10-06_sfp_joint_candidate_fresh10_failure_overlays_v1/` contains original-RGB wrong-card panels. Joint fusion failed its zero-wrong gate and is not deployed. [Record](docs/experiments/2026-10-06-sfp-joint-camera-candidates.md).

**October 6 SFP requested-card crop experiment:** Frozen RGB rail predictions and exact train/development proposal manifests are under `/home/nas/chmin/aic_archives/2026-10-06_sfp_slot_local_train_predictions_v1/` and `2026-10-06_sfp_slot_crop_training_materialized_v1/`. The selected 2,000-update crop checkpoint/history is `2026-10-06_sfp_slot_crop_fit_v1/`; additional ten-scene matched old/new scores are `2026-10-06_sfp_slot_crop_independent_old_checkpoint_v1/` and `2026-10-06_sfp_slot_crop_independent_new_checkpoint_v1/`. Original-image failure overlays are `2026-10-06_sfp_slot_crop_failure_views_v1/`. The model did not clear early-coverage handoff; these are diagnostics, not a deployed policy. [Record](docs/experiments/2026-10-06-sfp-slot-conditioned-crop-feasibility.md).

**October 5 SFP board-reference models:** `/home/nas/chmin/aic_archives/2026-10-05_sfp_board_reference_feasibility_v3/` holds the 70-train/10-development/3-actor native three-camera label manifest. `_sfp_board_reference_probe_v1/` is the failed pooled occupancy head (5/10 exact development maps); `_sfp_slot_local_probe_v1/` contains the improved checkpoint, predictions, annotated overlays and matched latency (10/10 development, 3/3 sparse actor maps). `_sfp_board_reference_independent10_labels_v1/` and the adjacent `_pooled_v1/` and `_slot_local_v1/` archives preserve the disjoint ten-scene comparison (5/10 versus 10/10). `_sfp_rail_candidate_3d_readonly_v1/` and `_diagnosis_v2/` preserve the failed requested-port consistency check. These are **perception diagnostics**, not a deployable handoff or expert action dataset. [Experiment and commands](docs/experiments/2026-10-05-sfp-slot-local-occupancy-ablation.md).

**October 5 spatial transport prerequisite:** `outputs/experiments/2026-10-05_spatial_transport_prerequisite_v1/actor_labels/` contains 110 original-RGB triplet labels from three saved sparse-card autonomous actor routes, with simulator geometry reserved for scoring. `frozen_rgb_identity_v2/` contains frozen selected-port and structured-slot predictions plus the machine summary. Both identified the requested port early on 3/3 routes, but the harder ten-request identity gate remains failed and no actor training followed. See [experiment record](docs/experiments/2026-10-05-spatial-transport-prerequisite.md).

**October 5 transport route/handoff continuation:** `outputs/experiments/2026-10-05_transport_identity_audit/` holds the matched action-only and route-conditioned actors (`route_handoff_*_v1/`), corrected expanded-data actor (`paired_coverage_actor_v2/`), two frozen stop heads (`sfp_frozen_handoff_v1/`, `_v2/`), and bounded failed task-selected head fits (`task_gated_full_coverage_v1/`, `task_gated_paired_early_v1/`). The canonical 326-train/10-validation SFP labels, unchanged 41,782-row image feature cache, and fitted region veto are under `/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/{paired_coverage_merged_v2,temporal_cache_v4,route_handoff_cache_v3}/`. NAS archives `/home/nas/chmin/aic_archives/2026-10-05_sfp_paired_coverage_v2/` and `_sfp_sparse_alternates_v1/` contain the new teacher plans/bags; `_sfp_sparse_paired_development_v1/` has four independent successful teachers; `_sfp_paired_development_actor_v1/` has six matched autonomous traces, videos and posthoc scores. The actor missed all three safe buffered arrival gates; no model here is deployed. [Experiment and machine summary](docs/experiments/2026-10-05-transport-route-handoff-continuation.md).

**October 5 transport and identity audit:** `outputs/experiments/2026-10-05_transport_identity_audit/` holds three saved SFP actor/teacher RGB comparison sheets, a 111-row causal autonomous camera/state audit, frozen native crop results, corrected same-observation task-code swaps (`ordinary_task_sensitivity_v3.json`), a failed bounded task-gated head (`task_gated_transport_v1/`), and the verified six-scene train-only alternate-request plan (`paired_request_plan_v2/`). After correcting a quota-field parser bug, `paired_request_pilot_v1/` holds six scored alternate-request Gazebo runs, host-extracted streams, progress and measured buffered-arrival metrics: five fully inserted. `paired_success_train_v2/`, `paired_original_train_v1/` and `paired_merged_train_v3/` hold the paired full-action transport labels; `/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/temporal_cache_v2/` has their corrected feature cache. `task_gated_paired_v1/` holds the matched fit that failed its offline gate. The four final IK configurations were not opened. [Experiment record](docs/experiments/2026-10-05-transport-identity-task-code-audit.md) · [machine summary](docs/experiments/2026-10-05-transport-identity-task-code-audit.summary.json).


**October 4 SFP side-view diagnostic:** `/home/nas/chmin/aic_archives/2026-10-04_sfp_side_view_v1/` preserves four final safety-guarded Gazebo teacher sweeps plus aborted safety pilots, original camera frames, scored streams, oracle contacts, 12 event-joined triplets, raw nine-panel grids, separate posthoc target-marker grids, frozen-model predictions and matched stage summary. Side views did not rescue the remaining hard requested-port case. [Methods, image names and decision](docs/experiments/2026-10-04-sfp-side-view-diagnostic.md), [machine summary](docs/experiments/2026-10-04-sfp-side-view-diagnostic.summary.json).

**October 4 SFP slot/pair repair:** `/home/nas/chmin/aic_archives/2026-10-04_sfp_slot_identity_v1/` holds the frozen fresh-scene wrong-port list, grouped original-RGB label manifest, two preserved stopped starts, bounded `fit_v3` checkpoint and predictions, detailed pixel/flag diagnosis, and the read-only sparse-history decisions. The fit failed before independent new-scene evaluation. [Experiment, commands and decisions](docs/experiments/2026-10-04-sfp-slot-pair-identity.md), [machine summary](docs/experiments/2026-10-04-sfp-slot-pair-identity.summary.json).

**October 4 fresh SFP shadow check:** `/home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/` holds the fixed ten-reset plan, ten scored Gazebo teacher bags and original camera images, host-extracted streams, 581-sample labels, all four frozen model predictions, candidate and crop proposals, failed OR-union audit, bounded structural diagnostic, offline latency and launch logs. The quota rose to 1024 and collection finished; requested-port identity accuracy now blocks handoff. [Experiment and exact commands](docs/experiments/2026-10-04-sfp-fresh-anchor-shadow.md), [machine summary](docs/experiments/2026-10-04-sfp-fresh-anchor-shadow.summary.json).

**October 4 rootless container cleanup:** `outputs/experiments/2026-10-04_rootless_key_quota_cleanup/` holds the exact IDs, writable-layer sizes and mounts for 25 stopped September AIC containers removed after audit. [Tracked summary](docs/experiments/2026-10-04-rootless-container-cleanup.summary.json). The active Isaac container and older uncertain containers were preserved. Key usage stayed 199/200 immediately after removal; the limit was later raised to 1024 and the ten-scene SFP check completed sequentially.


**October 4 SFP rail crop:** `/home/nas/chmin/aic_archives/2026-10-04_sfp_observed_rail_crop_v1/` and `_train_v1/` hold strict 5 px observation-selected proposal audits; `_source12_v1/` and `_train_source12_v1/` hold the fixed 12 px ablation and target-containment audits. `/home/nas/chmin/aic_archives/2026-10-04_sfp_observed_rail_crop_fit_v2/` holds the 1,200-update GPU-2 crop fit, selected update-800 checkpoint and prediction history; `fit_v1` preserves the failed pretraining image-boundary attempt. `_source12_v1/frozen_four_model_union.json` is a read-only matched ten-episode summary (9/10 early, 10/10 any, zero accepted wrong), not independent or live validation. [Record](docs/experiments/2026-10-04-sfp-observed-rail-crop.md).

**October 4 SFP middle-card identity repair:** `/home/nas/chmin/aic_archives/2026-10-04_sfp_middle_card_visibility_v2/` has six three-camera first/middle/last original-RGB review sheets and a provisional review JSON; the incomplete first render remains in `_v1/`. `/home/nas/chmin/aic_archives/2026-10-04_sfp_structured_slot_v1/` has the 2,000-update all-slot checkpoint, every 200-update development prediction, selected `summary.json`, and exact wrong-neighbor audit. The selected model improved accepted observations but reached only 7/10 early episode anchors; no actor continuation exists. [Experiment](docs/experiments/2026-10-04-sfp-middle-card-identity-repair.md).

**October 3 SFP selected-port anchor:** `/var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/anchor_labels_v1/` holds the 50/10 reset-grouped original-wide-RGB projection manifest. `/home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/selected_anchor_native576_v1/` holds the 2,000-update task-conditioned checkpoint, history, every development prediction, frozen multiport comparison, and `coverage_audit.json`. Its accepted 3D position p95 was 2.65 mm, but it acquired an anchor in only 4/10 episodes, missing all middle-card requests; no actor continuation exists. [Experiment record](docs/experiments/2026-10-03-sfp-selected-anchor.md).

**October 3 SFP route-stage/waypoint probe:** `/var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/route_stage_probe_cache_v2/` has the frozen 50-train/10-development stage and measured next-waypoint labels on original cached RGB features. `/home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/route_stage_waypoint_probe_v1/` has the 2,000-update checkpoint, constant/state-only baselines, 638 held-out predictions, metric history and [error plot](/home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/route_stage_waypoint_probe_v1/error_by_time_and_target_v2.png). The waypoint gate failed at 48.2 mm p95; no actor continuation exists for this probe. [Full record](docs/experiments/2026-10-03-sfp-route-stage-waypoint-probe.md).

**October 3 bounded corrective SFP continuation:** `/home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/balanced_bc/` contains `corrective_teacher_gate_3dev_v1/` (privileged feasibility), `corrective_teacher_train20_v1/` (frozen cap, first ten audited successes, one unadmitted second-tranche success, nine Docker quota no-launches), `corrective_10ep_fullaction_v1/` (single supervised continuation), and `corrective_10ep_autonomous_3dev_v1/` (three unchanged development rollouts, scores and bounded route/direction diagnosis). `/var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/` contains `corrective_train10_materialized_v2/`, `corrective_train10_merged_v2/`, and `corrective_train10_cache_host_v2/`. The actor reached **0/3** buffered arrivals; no model was promoted. [Full experiment](docs/experiments/2026-10-03-sfp-balanced-arrival-onpolicy-check.md#october-3-bounded-corrective-full-action-continuation-failed-live-gate).

**October 3 SFP cross-target pilot, balanced BC and failed autonomous gate:** `/home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1/` holds ten matched CheatCode feasibility controls, three earlier frozen cross-target routes and videos, **50** successful balanced training bags, **ten** successful disjoint development bags, the two matched shared-BC checkpoints, an extended-training control, all nine new development rollout audits under `balanced_bc/development_3x3_v1/`, and two explicitly privileged five-second teacher-prefix diagnostics. `/var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/` holds verified merged TCP-body labels, the frozen 39,161-sample visual cache, nine-run posthoc scores, prefix scores and a measured-TCP route-divergence plot. The three tested autonomous actor variants each reached **0/3** safe requested-port buffered visits; no weights were promoted. [Detailed record](docs/experiments/2026-10-03-sfp-balanced-arrival-onpolicy-check.md). The previous agent replay action arrays were not used for BC.

**October 3 SFP balanced-arrival diagnostics:** `/var/tmp/chmin_aic_20260918_act/sfp_balanced_arrival_20261003/` holds the 180-episode frozen RGB feature cache, episode-grouped arrival-head checkpoints and metrics, and exact autonomous trace replay. Small corrected provenance JSON is at `outputs/experiments/2026-10-03_sparse_history/sfp_arrival_probe_v1/older_agent_provenance_audit_v2.json`. [Experiment record](docs/experiments/2026-10-03-sfp-balanced-arrival-onpolicy-check.md).

## October 3 sparse-history and SFP arrival diagnostics

`outputs/experiments/2026-10-03_sparse_history/` contains frozen SC sparse-history replays, posthoc pose scores, and `sfp_arrival_probe_v1/` (three checkpoint ablations, episode-level event metrics, source hashes, and task-coverage audit). The SFP probe is diagnostic only; see [experiment record](docs/experiments/2026-10-03-sfp-arrival-probe-and-coverage.md).

**October 3 sparse SC history diagnostic:** [Tracked summary](docs/experiments/2026-10-03-sc-sparse-history-handoff.summary.json) and [experiment record](docs/experiments/2026-10-03-sc-sparse-history-handoff.md) map the frozen-image identity audits, 10/20/30 mm keyframe comparisons, 3D pose scores, two-port correspondence checks, and an extra one-port p1 challenge in `outputs/experiments/2026-10-03_sparse_history/`. [NAS live archive](/home/nas/chmin/aic_archives/2026-10-03_sparse_history_v1/) holds the matched baseline, an uninspectable route, the successful shadow-history inspection repeat, ROS bags, RGB, exact launch snapshots and logs. The history check remains opt-in and never commanded insertion.

**October 3 SC orientation/identity audit:** [NAS archive](/home/nas/chmin/aic_archives/2026-10-03_sc_orientation_repair_v1/) contains a newly frozen five-card reset, four live shadow runs, scored streams, command/measured/board orientation traces, RGB inspection panels, same-image quaternion/position swaps, an orientation-masked identity checkpoint and its failed absent-port comparison. All changes remain opt-in; no insertion actor ran. [Experiment record](docs/experiments/2026-10-03-sc-orientation-identity-repair.md), [machine summary](docs/experiments/2026-10-03-sc-orientation-identity-repair.summary.json).

**October 3 SC crop-identity diagnostic:** [NAS archive](/home/nas/chmin/aic_archives/2026-10-03_sc_crop_identity_v1/) has the four newly frozen reset configs, scored Gazebo RGB/stream captures, projection-only labels, 160-pixel crop classifier checkpoint, 144-triplet independent comparison, saved control/hidden replays, and sampled review panels. The classifier failed selected-port recall and absent-port rejection and is **not deployed**. [Experiment record](docs/experiments/2026-10-03-sc-original-rgb-crop-identity.md), [machine summary](docs/experiments/2026-10-03-sc-original-rgb-crop-identity.summary.json).

**October 3 live SC shadow handoff:** `/home/nas/chmin/aic_archives/2026-10-03_sc_shadow_handoff_v1/` holds the two same-scene integration pilots, four new scene-grouped development trials, matched direct-RGB latency replays, two bounded SFP arrival diagnostics, exact reset/launch manifests, camera frames, runtime traces, scored bags, predicted-opening overlays, 20 fps SC videos, and separate post-run TF scores. The [machine summary](docs/experiments/2026-10-03-live-sc-shadow-handoff.summary.json) indexes both four-run SC comparisons and the SFP false-stop test; the [experiment record](docs/experiments/2026-10-03-live-sc-shadow-handoff.md) gives commands and limits. All four SC starts reached a shadow **transport** handoff in both versions; no insertion occurred. Original JPEG first checks exceeded 300 ms; the matched faster verifier measured 176 ms p95 across six inspections. The SC-trained handoff head falsely paused the SFP diagnostic 162 mm from the target and remains disabled for SFP.

**October 3 geometric handoff repair:** The [tracked machine summary](docs/experiments/2026-10-03-geometric-handoff-repair.summary.json) maps hashes and results. `/home/nas/chmin/aic_archives/2026-10-03_geometric_handoff_repair_v1/` holds per-frame SC board-rail/new-reset metrics, six-stop replay, offline pipeline timing, and every SFP constructed candidate decision with cross-view geometry. No checkpoint was trained; source checkpoint paths and exact reproduction commands are in the [experiment record](docs/experiments/2026-10-03-geometric-handoff-repair.md). Automatic handoff remains disabled.

**October 3 native-crop handoff gate:** The [machine summary](outputs/experiments/2026-10-03_native_crop_handoff_gate/summary.json) records the failed Step 1–2 gate and hashes of selected checkpoints. The NAS archive `/home/nas/chmin/aic_archives/2026-10-03_native_crop_handoff_gate_v1/` holds the SC 160-pixel pooled and spatial crop fits, oracle-centering diagnostic, SFP 320-pixel pair-crop fit, matched frozen V5 scores, per-example predictions, reviewed montages, and `commands.sh`. Failed implementation attempts are retained. These checkpoints are **diagnostic only**; no new policy or full-rollout artifacts were produced. [Experiment record](docs/experiments/2026-10-03-native-crop-handoff-gate.md).

**October 3 frozen handoff gate:** Local machine decisions are `outputs/experiments/2026-10-02_two_family_perception/selected_port_frozen_gate_20261003.json` and `selected_port_orientation_repair_decision_20261003.json`; the privileged pair-axis ceiling is `sc_oracle_pair_orientation_ceiling_20261003.json`. Frozen SC stop identity and prior orientation replays are under `/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_probe_v2/`. SFP point-error images, corrected SC label streams, both one-port orientation checkpoints, synchronized new-reset evaluations and six-stop replays are under `/home/nas/chmin/aic_archives/2026-10-03_sfp_offscreen_request_v1/`. [Detailed artifact map and decision](docs/experiments/2026-10-03-frozen-handoff-gate-and-orientation-repair.md).

**October 3 selected-port continuation:** Fixed SFP scene configs, six rootless Gazebo captures, native projection labels, five candidate-correspondence checkpoints, replay metrics, and new development cascade results are under `/home/nas/chmin/aic_archives/2026-10-03_sfp_offscreen_request_v1/`. The small local [machine decision](outputs/experiments/2026-10-02_two_family_perception/selected_port_correspondence_decision_20261003.json) indexes results; [experiment record](docs/experiments/2026-10-03-selected-port-correspondence-continuation.md) explains what passed and what remains gated. Frozen SC wrong-route identity replays are under `/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_probe_v2/`.

**October 3 spatial SC identity continuation:** NAS root
`/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/` now also has
`sc_spatial_identity_v1/` (12 distinct one-port reset configs, scored Gazebo
streams, three-camera native RGB and projection-only labels) and
`sc_spatial_identity_probe_v2/` (matched pooled/spatial checkpoints,
scene-grouped metrics, all-frame predictions, exact error images,
`three_camera_summary_v1.json`, and component latency). The earlier v1 probe
directory preserves an evaluation-indexing failure before metrics were made.
The spatial head's fixed all-three-camera rule rejected 109/109 absent-port
requests and retained 96/100 real requests projected into every view on four
new development resets. This is not a deployable visibility/handoff result.
[Details and commands](docs/experiments/2026-10-02-sc-absent-port-identity.md#spatial-selected-port-continuation-on-new-reset-groups).

**October 2 selected-port identity gate:** The NAS root
`/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/` contains
`sc_absent_request_identity_gate_v1.json` (machine comparison),
`sc_absent_request_false_accept_gallery_v1.jpg` (visually checked wrong-port
overlay), `sc_one_port_identity_v1/` (four targeted one-port Gazebo resets,
exact scene configs, scored streams and RGB), `sc_presence_head_ablation_v2/`
(failed last-layer repair), and `sc_selected_availability_probe_v2/` (failed
image-only and image-plus-measured-TCP probes). These are diagnostic artifacts,
not deployable checkpoints. The verifier accepted all 16 counterfactual
absent-port requests; final scenes remain sealed. [Full artifact map and
commands](docs/experiments/2026-10-02-sc-absent-port-identity.md).

**October 2 transport-stop comparison:**
`outputs/experiments/2026-10-02_two_family_perception/` contains
`sc_stop_frozen_development_seven_v1.json` (seven complete same-path SC
stop traces), its PNG plot, `sfp_stop_identity_feasibility_100frames_v1.json`
(two SFP selected-port lock replays), and `sc_sfp_stop_decision_v1.json`
(compact result and hashes). The serial same-controller SC live pair, scene
snapshots, hashes, scores, traces, frames and five-view videos are on NAS at
`/home/nas/chmin/aic_archives/2026-10-02_transport_stop_comparison/`.
The visual stop remains development-only. [Full record](docs/experiments/2026-10-02-transport-stop-comparison.md).

**October 2 stage-preserving SFP correction comparison:** NAS root
`/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/` contains
`sfp_onpolicy_takeover20_fivecard_v1/` (12 complete scored attempts, ten
teacher-assisted full insertions, five-camera videos),
`sfp_dagger_teacher_chunks_v1.jsonl` (1,409 recorded post-takeover chunks),
`sfp_dagger_pair_bc_v1/` (1,392 admitted RGB-frame BC rows), and matched
`sfp_actor_dagger_control_v1/` / `sfp_actor_dagger_corrected_v1/` checkpoints
and logs. The local experiment root contains `sfp_dagger_selection_v1/`,
`sfp_v8_dagger_matched_first80_replay.json`, and
`sfp_dagger_matched_decision.json`. The corrected actor failed the declared
offline promotion gate; no autonomous result came from these checkpoints.
[Details](docs/experiments/2026-10-02-two-family-perception-local-bc.md#bounded-on-policy-correction-comparison-declared-before-training).

**October 2 actor-frame diagnostic:** The paired RGB frame relabel and two
matched native-SFP BC runs are on NAS under
`/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/` as
`sfp_native_pair_relabel_matched_v2/`, `sfp_actor_oldframe_matched_v1/`, and
`sfp_actor_pair_matched_v1/`. The earlier `sfp_native_pair_relabel_matched_v1/`
is invalid and was never used for training. The frozen pre-opening live
audits are `sfp_actor39_preloaded_axialhold_pair85_lat2_v7/` and
`sfp_actor39_pair85_lat2_lock1_v8/`, each with a five-camera 10 fps video.
Local machine summaries and RGB-only action replay are in
`outputs/experiments/2026-10-02_two_family_perception/`. See the
[experiment record](docs/experiments/2026-10-02-two-family-perception-local-bc.md#october-2-immediate-lock-failure-and-matched-actor-frame-repair).

**October 2 SFP selected-target and insertion continuation:** Under
`outputs/experiments/2026-10-02_two_family_perception/`, the
`sfp_native_confusion_scenes_v2/` manifest records ten collected training
starts and unused planned development starts. New raw expert RGB/scored runs
live on `/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/sfp_native_confusion_v2/`.
`sfp_native_confusion_labels_first10/` has 5,045 RGB projection rows;
`joint_locator_confusion_first10_v1/` and
`joint_locator_confusion_first10_low_lr_v2/` are rejected held-out target
identification continuations. `sfp_native_teacher_local_first10_bodyconverted.jsonl`
contains 1,914 causal recorded target chunks, and
`sfp_native_cascade_bc_first10_bodyconverted/` contains 1,478 accepted actor
rows. `sfp_native_cascade_bc_39_bodyconverted/` joins old and new grouped
actor data. `live_sfp_v4_oracle_contact_diagnostic/` preserves the scored
failure, five-camera video, and contact-pair trace. See the
[experiment record](docs/experiments/2026-10-02-two-family-perception-local-bc.md#october-2-follow-up-new-sfp-starts-body-frame-targets-and-live-contacts).

**October 2 native SFP and SC insertion artifacts:** `outputs/experiments/2026-10-02_two_family_perception/sfp_native_scene_selection_v1/` holds 29 scored full-success SFP expert replays and five-camera videos, including alternate targets. `sfp_native_projection_29_hostpaths/` holds grouped selected/alternate target labels; `sfp_native_cascade_bc_29_v4_ensemble/` holds 1,047 causal native insertion rows with recorded teacher targets and label-only pose-outlier filtering. `sfp_native_orientation_v2_highnative/` and `sfp_native_orientation_v3_runtime_matched/` preserve the orientation comparison; `sfp_native_cascade_benchmark_v3_allframes.json` preserves 722-frame RGB-only position, orientation, and lock metrics. `local_live_sc_llb0001_dagger_v2_guard2_v1/` contains the scored SC 0-insertion failure, posthoc plug-tip metrics, and five-camera video. Earlier `sfp_native_rgbd/` is a symlink to a checksum-matched NAS backup under `/home/nas/chmin/aic_archives/2026-10-02_two_family_perception/`, following the unusable depth-label pilot. [Interpretation and commands](docs/experiments/2026-10-02-two-family-perception-local-bc.md#october-2-continuation-complete-sfp-cascade-and-insertion-diagnosis).

**October 2 live local actor diagnostics:** `outputs/experiments/2026-10-02_two_family_perception/local_live_sc_llb0001_warm_v6_threads/` holds the scored failure, actual plug-tip stream, per-command/perception trace, measured latency, and [five-camera video](outputs/experiments/2026-10-02_two_family_perception/local_live_sc_llb0001_warm_v6_threads/five_cameras_10fps.mp4). The matched `local_live_sc_llb0001_warm_v7_static_lock/` retains the RGB-only static-pose ablation. `local_actor_warm_continue_v2/` is the bounded supervised continuation. `sfp_native_rgbd/scene_23_serial_world_follower/` preserves a successful SFP CheatCode capture with unusable blank auxiliary depth/RGB supervision; `scene_23_mask1/` repeats that label-only capture with corrected scene visibility mask. [Interpretation](docs/experiments/2026-10-02-two-family-perception-local-bc.md#live-sc-near-port-diagnosis-and-targeted-continuation).

**October 2 perception and local BC:** `outputs/experiments/2026-10-02_two_family_perception/` contains grouped three-camera projection labels; joint selected-port locator checkpoint and cached predictions; native SC crop refiner checkpoints and pixel/3D metrics; visual orientation checkpoint; SC RGBD visibility audit and four-scene pilot; selected-port pose cache; predicted-frame local BC rows and scratch/warm-start actor checkpoints. `local_live_sc_*` records preserve near-port integration diagnostics, including the failed concurrent-Gazebo attempt. The adjacent [experiment record](docs/experiments/2026-10-02-two-family-perception-local-bc.md) links three transport-only review videos and explains which metrics are offline, which labels are only projected in-frame, and why insertion is still open.

**October 2 route/stop artifacts:** In `outputs/experiments/2026-10-01_transport_reliability/`, `live_sc_visual_standoff_llb_0009_v11_recent_stop/` contains the repaired-stop five-card rollout; `visual_transport_cap_comparison.json` and `visual_transport_cap_fourcard_comparison.json` contain the rejected 30 mm cap comparisons with full scored runs alongside them. `sc_base_waypoint_probe{,_w512}/` holds the failed offline waypoint checkpoints, commands, histories and grouped metrics. [Interpretation](docs/experiments/2026-10-01-transport-reliability-continuation.md).

**October 1 transport reliability continuation:** `outputs/experiments/2026-10-01_transport_reliability/` contains grouped history/route-correction checkpoints and manifests, actor-view camera/pose audit labels, the explicit dual-port locator, training-only visual-standoff calibration, and all frozen autonomous development failures and passes. `visual_transport_development_comparison.json` maps every live run to terminal position/orientation, route p95, complete latency, dropped commands, raw force, and scored outcome. `live_sc_visual_standoff_llb_0122_v3_trigger80/five_cameras_10fps.mp4` shows the three-card handoff; `live_sc_visual_standoff_llb_0009_v7_fastresize/five_cameras_10fps.mp4` and `llb_0009_route_variability.png` show the wrong low five-card route; `live_sc_visual_standoff_llb_0009_v9_zfloor_repeat2/five_cameras_10fps.mp4` shows a guard-exercised transport pass. The runs are transport-only, not insertion successes; no reserved final scene was opened. [Experiment record](docs/experiments/2026-10-01-transport-reliability-continuation.md).

**October 1 five-way transport follow-up:** `outputs/experiments/2026-10-01_transport_five_way/imagenet_features/` contains 55,142 frozen three-view ImageNet feature rows, causal timestamps, route labels, split metadata, and camera-age metadata. `no_history/`, `long_history/`, `route/`, `diffusion/`, `sfp_only*/`, and `sc_only*/` preserve trained checkpoints, manifests, update histories, and validation summaries; `diffusion_failed_shape/` preserves a repaired first validation attempt. `matched_heldout.json`, `diffusion_sampling.json`, and `latency_local.json` contain controlled comparison metrics. `live_sfp_*` and `live_sc_*` contain same-scene development score bags, terminal-before-reset traces, posthoc standoff metrics or plots, and five-camera 5 fps MP4s. `outputs/experiments/2026-10-01_sfp_transport_decisive/` preserves the older frozen checkpoint's SFP scene, score, distance metrics, contact sheet, and five-camera video. These are development tests only; geometry is posthoc, not actor input. [Experiment record](docs/experiments/2026-10-01-transport-five-way-comparison.md).

**October 1 temporal transport:** `outputs/experiments/2026-10-01_temporal_transport/label_audit.json` contains the 104-SC/268-SFP stage, split, task, command-frame, and camera-timing audit; `transition_montage.jpg` is the inspected two-case transition montage. `frozen_features/` is the observation-only cached feature derivative for the bounded temporal/no-history comparison. Three completed supervised head checkpoints, per-run manifests/logs, `full_heldout_comparison.json` and `.png`, `tilt_return_boundary.json`, `full_heldout_handoff_check.json`, `commands.json`, and `decision.json` are under the same experiment root. No temporal Gazebo rollout passed the offline gate. [Experiment record](docs/experiments/2026-10-01-temporal-three-model-continuation.md). Plans and true geometry are offline labels, never deployable inputs.

**October 1 cleaned SC BC artifacts:** `outputs/trajectory_datasets/sc_llb_cleaned_v3/manifest.json` is the immutable 104-success selection and grouped 83/21 split. `materialized/streams/` contains scored-bag command/camera extractions; `raw_chunks/` contains full command-derived TCP-body transport and selected-port-frame local labels. `bc_chunks_motion_safe_v2/` is the preferred audited derivative, with `audit_motion_safe_v2.json`; `bc_chunks/` and `audit.json` preserve the earlier less conservative hold-cleanup ablation. The original source is `outputs/trajectory_datasets/sc_llb_score_verified/`. `bc_chunks_unfiltered_wrist_gravity_diagnostic/` preserves a rejected no-op filter run that mistook the tool's roughly 20 N gravity load for contact. Four supervised transport checkpoints, logs, and manifests are under `outputs/experiments/2026-10-01_sc_llb_cleaned_bc/`. The same root holds `offline_comparison.json`, `autonomous_comparison_cap20.json`, `autonomous_comparison_cap60.json`, `cap_comparison.json`, `teacher_target_cap_v3.json`, and `autonomous_transport/` for 16 scored bags, exact scene snapshots, action traces, images, measured-path plots, and selected 20 fps five-view videos. [Results and next gate](docs/experiments/2026-10-01-sc-llb-cleaned-bc.md#final-bounded-supervised-result-and-step-5-decision).

**September 30 latest:** teacher transport gate accepted. Two same-world Cartesian SC teachers fully inserted, providing v9 with 1,751 transport and 503 local chunks, plus the verified SFP source. The balanced-port BC retry finished 1,500 updates and failed both autonomous development starts: final planned-standoff errors 103 and 603 mm versus 76 and 74 mm for the prior masked actor; neither reached handoff or inserted. A bounded trace/video audit found early wrong-route commands without sustained cable snag. Stop routine feed-forward BC. A checkpoint-provenance audit found that this run reused the full action head from an older mixed-SC-route model and had only two new SC episodes; it was not a clean test of the new Low Lateral Bypass route. Audit clean route data and initialization before deciding on a route-subgoal model. No new training was started for this audit. True visibility, autonomous two-actor insertion and RL remain gated; reserved final scenes stay sealed. [Evidence and exact artifacts](docs/experiments/2026-09-30-transport-to-two-actor-continuation.md).

## September 30: accepted transport and two-actor continuation

**Latest v9 paired-port controlled comparison:** `outputs/experiments/2026-09-30_transport_continuation/transport_bc_paired_ports/` contains the 1,500-update checkpoint, history, training source snapshots and hashes. `autonomous_transport/paired_ports_port0_two/` and `paired_ports_port1_three/` contain each frozen full-start development test: official score, raw bag, exact launch snapshot, observation/proposal/executed-action/terminal trace, posthoc standoff analysis, first-divergence and task-code audits, and 20 fps `video/all_views_20fps.mp4` plus separate side/overhead videos. Both failed autonomous handoff and insertion. `matched_comparison.json` holds all ten candidate rollouts on the same two starts; `paired_port_bounded_diagnosis.json` records the single trace/video diagnosis. The v9 dataset is `outputs/trajectory_datasets/transport_local_verified/2026-09-30_v9_paired_ports/`. Final scenes remain sealed. [Result and next gate](docs/experiments/2026-09-30-transport-to-two-actor-continuation.md#paired-port-bc-result-and-bounded-failure-diagnosis).

`outputs/experiments/2026-09-30_transport_continuation/`: exact four-layout scene manifest, MoveIt plans, scored bags, original three-camera RGB plus overhead/side diagnostic views, measured route/handoff audits, and 20 fps five-view videos under `runs/CASE/video/`. Successful `port0_four_shiftplus` is full insertion; `port1_five_shiftminus` is a preserved local insertion failure. The other two layouts are route-only validation.

`outputs/trajectory_datasets/transport_local_verified/2026-09-30_v3/`: command-derived full-success SC training and explicitly separate held-out route-only transport rows. Four train episodes, 3,001 transport and 1,000 local chunks; two held-out episodes, 1,751 transport chunks. All files remain outside Git. **Do not use the interrupted `2026-09-30` output or the superseded v2.**

`visibility_runs/`, `visibility_labels/`, and `visibility_mask_validation/` under the experiment root preserve the label-only RGBD recorder, rejected blank/self-view frames, original-versus-auxiliary image alignment checks, and label overlays. These are not learned-model predictions or certified identifiable-visibility labels. [Decisions and commands](docs/experiments/2026-09-30-transport-to-two-actor-continuation.md).

## September 30: named contact and close-view force audit

`outputs/experiments/2026-09-30_force_oracle/` preserves frozen old-route success/failure controls and new tilt-before-return repeats. The [report](docs/experiments/2026-09-30-force-oracle.md) and [machine manifest](docs/experiments/2026-09-30-force-oracle-results.json) distinguish raw wrist delta from corrected Gazebo point forces. See `old_oracle_r2/oracle_video/port_contacts_20fps.mp4` and `old_oracle_r2/force_and_motion.png` for contact markers and the failed measured withdrawal. Raw contact JSONL, MCAP, trace, score, 20 Hz frames, camera calibration, reporting-fix source metadata and summaries are retained. These are privileged audit artifacts, excluded from Git. Compact raw evidence and videos are backed up under `/home/nas/chmin/aic_archive_20260930/force_oracle/`; source JPEG frames remain local.

The [September 30 tilt-before-return pilot](docs/experiments/2026-09-30-sc-tilt-before-return.md)
has an explicit beyond-row tilt, measured TCP/plug-tip motion, and a matched
older-route control. The [five-view 20 fps full-insertion video](outputs/experiments/2026-09-30_moveit_tilt_before_return/port1_tilt_hold_success_video/all_views_20fps.mp4)
shows one privileged teacher success. The [machine summary](docs/experiments/2026-09-30-sc-tilt-before-return-results.json)
maps all successes and failures to exact raw runs. The matched
[older-route near-opening failure video](outputs/experiments/2026-09-30_moveit_tilt_before_return/port1_old_tilt_matched_failure_video/all_views_20fps.mp4)
is saved in the same five-view format. Compact traces and scored
MCAPs are under `/home/nas/chmin/aic_archive_20260930/moveit_tilt_before_return_raw/`;
the [archive manifest](docs/experiments/2026-09-30-sc-tilt-before-return-archive.json)
records hash verification. Separate [five-card port-0](outputs/experiments/2026-09-30_moveit_tilt_before_return/port0_route_repeat_video/all_views_20fps.mp4)
and [shifted three-card](outputs/experiments/2026-09-30_moveit_tilt_before_return/threecard_route_repeat_video/all_views_20fps.mp4)
route-only repeat videos and their [machine results](docs/experiments/2026-09-30-sc-route-generality-repeats.json)
are also saved. These are not autonomous policy rollouts.

The [September 29 lowered MoveIt route continuation](docs/experiments/2026-09-29-moveit-low-cross-continuation.md)
maps route-only five-card ports 0/1 and shifted three-card videos, measured
card-top/plug-tip heights, and privileged insertion handoff failures. The
[port-0 full insertion video](outputs/experiments/2026-09-29_moveit_gazebo_pilot/low18_diagonal_insert_success_video/all_views_20fps.mp4)
has all five synchronized views at 20 fps. The
[machine results](docs/experiments/2026-09-29-moveit-low-cross-results.json)
map exact raw runs under `/var/tmp/chmin_aic_20260918_act/`. These are teacher
diagnostics, not autonomous policy rollouts. The original three-route record
below used a stricter clearance criterion and is preserved as a baseline.
The [verified NAS archive](docs/experiments/2026-09-29-moveit-low-cross-archive.json)
preserves scored MCAPs and exact inputs for 13 completed low-cross runs.

The September 29 MoveIt/Gazebo fixed-scene pilot saved three
[five-view 20 fps review videos](outputs/experiments/2026-09-29_moveit_gazebo_pilot/)
under `baseline_video/`, `lane20_video/`, and `strict_return_video/`.
The [experiment report](docs/experiments/2026-09-29-moveit-sc-route-pilot.md)
and [machine summary](docs/experiments/2026-09-29-moveit-sc-pilot-results.json)
map plans, executed joint traces, cable-clearance analyses, score files,
videos, code hashes, and the interrupted strict-return attempt. Raw scored
bags and source plans remain in
`/var/tmp/chmin_aic_20260918_act/moveit_sc_corridor_*_20260929/`; the
[verified archive](docs/experiments/2026-09-29-moveit-sc-pilot-archive.json)
is under `/home/nas/chmin/aic_archive_20260929/moveit_sc_pilot_raw/`.
These are route-only privileged teachers that reached standoff but failed
the cable-route-quality gate; none is a full insertion or BC-admitted episode.

The September 29 fixed-scene SC ±20 mm route diagnostic has three
[20 fps five-view review videos](outputs/experiments/2026-09-29_sc_low_cross_2cm_rescue/):
an unchanged-route partial, an outside-lane +20 mm partial, and an
outside-lane +20 mm full insertion. The adjacent `review_manifest.json`
maps each score, route plan, cable plot, port trace, contact sheets, and
video hash. The [experiment report](docs/experiments/2026-09-29-sc-low-cross-2cm-rescue.md)
and [all-run JSON](docs/experiments/2026-09-29-sc-low-cross-2cm-rescue.json)
cover 17 new privileged teacher rollouts plus the archived failure.
Raw source runs are under
`/var/tmp/chmin_aic_20260918_act/sc_low_cross_2cm_rescue_20260929/`;
the NAS copy is
`/home/nas/chmin/aic_archive_20260929/sc_low_cross_2cm_rescue_raw/`.
The [archive manifest](docs/experiments/2026-09-29-sc-low-cross-2cm-archive.json)
records verified hashes for all 17 scored MCAPs and compact result files.
These are not autonomous policy videos or BC-admitted episodes.

The [September 26 storage cleanup](docs/experiments/2026-09-26-output-space-cleanup.md)
records which old smoke/sweep artifacts were removed, which May policy-run
trees were backed up to NAS, and which expert/source data are protected.

The September 26 SC far-return teacher diagnostics are under
[`outputs/experiments/2026-09-26_sc_far_return/`](outputs/experiments/2026-09-26_sc_far_return/).
Start with the [low-cross five-view video](outputs/experiments/2026-09-26_sc_far_return/low_cross_candidate/all_views_20fps.mp4),
the [six one-axis videos](outputs/experiments/2026-09-26_sc_far_return/low_cross_variations/),
and the [route-sensitivity record](docs/experiments/2026-09-26-sc-far-return-route-sensitivity.md).
That output path is a symlink to NAS to preserve space on `/data1`.
The videos are privileged Gazebo teacher tests; they are not autonomous
policy rollouts. The shorter-forward and later margin variants did not all
fully insert.
The [machine summary](docs/experiments/2026-09-26-sc-far-return-summary.json)
maps score, cable proximity, force, and video for each run. The
[verified raw archive](docs/experiments/2026-09-26-sc-low-cross-raw-archive.json)
maps the nominal low-cross and three corrective-teacher runs to NAS copies;
source/archive sizes, MCAP hashes, and frame indexes were checked.

The September 25 SC lateral bypass videos and plots are under
`outputs/experiments/2026-09-25_sc_lateral_bypass/`. Start with the
[five-view 20 Hz video](outputs/experiments/2026-09-25_sc_lateral_bypass/wide_51500/all_views_20fps.mp4)
and the [experiment index](docs/experiments/2026-09-25-sc-lateral-bypass-and-policy-handoff.md).
The fixed wide run and port-0 pilot are scored teacher successes; the
port-1 wider-lane failure is preserved. Raw MCAPs are hash-verified on NAS
through archive manifests linked in the experiment index.

Reviewed: 2026-09-30. Start with the [documentation index](docs/README.md) and
[experiment ledger](docs/EXPERIMENTS.md) for the meaning and status of saved runs.

## Current storage map

The September 25 direct body-target shared BC ablation is under local
`/var/tmp/chmin_aic_20260918_act/shared_body_target_bc_batch3_20260925/`
and its lower-rate continuation under `shared_body_target_bc_batch3_cont_20260925/`.
Each has `run.sh`, `best.pt`, `summary.json`, `history.json`, and a scored
two-scene development evaluation. Their raw bags are hash-verified on NAS:
[first fit](docs/experiments/2026-09-25-shared-body-bc-batch3-dev2-mcap-archive.json)
and [continuation](docs/experiments/2026-09-25-shared-body-bc-batch3-cont-dev2-mcap-archive.json).
The teacher-rescue pilot and expanded training-only groups are under
`sc_body_early_rescue_pilot_20260925/` and
`sc_body_early_rescue_batch20_20260925/`; scored failures stay outside BC.
The [pilot archive](docs/experiments/2026-09-25-sc-body-early-rescue-pilot-mcap-archive.json)
and per-group archive manifests preserve the raw bags.
The matched body-target BC control and correction treatment are under
`shared_body_target_bc_rescue18_{control,treatment}_20260925/`; each contains
`run.sh`, `best.pt`, `summary.json`, `history.json`, `eval.log`, and the
scored two-scene development evaluation. The [machine gate](docs/experiments/2026-09-25-shared-body-bc-rescue18-matched-gate.json)
links checkpoint hashes and scores. Four evaluation MCAPs are losslessly
stored on NAS, indexed by the separate control and treatment archive
manifests linked in that gate. Neither actor inserted autonomously.

The later interleaved-rescue teacher episodes, narrow 0.5 s correction
selection, matched control/treatment checkpoints, and their archived Gazebo
bags are indexed by the [selection report](docs/experiments/2026-09-25-sc-body-interleaved-rescue-window0p5s-selection.json)
and [live gate](docs/experiments/2026-09-25-shared-body-bc-window0p5s-matched-live-gate.json).
The frozen parent four-target cadence run is under
`/var/tmp/chmin_aic_20260918_act/shared_body_target_bc_batch3_20260925/eval_new_dev2_chunk4/`;
its raw bags are indexed by the [NAS archive](docs/experiments/2026-09-25-shared-body-parent-chunk4-dev2-mcap-archive.json).
The [scored command/motion diagnosis](docs/experiments/2026-09-25-step5-body-cadence-scored-diagnosis.json),
[1 Hz replan trace](docs/experiments/2026-09-25-body-parent-replan1-trial999001-scored-trace.json),
[chunk trace](docs/experiments/2026-09-25-body-parent-chunk4-trial999001-scored-trace.json),
and [plot](docs/experiments/figures/2026-09-25-step5-body-cadence-first-divergence.png)
are durable in the repository. The extraction script is
`scripts/diagnose_sc_body_actor_trace.py`. These are posthoc scored
measurements; no geometry reached the autonomous actor.
Four H.264 three-camera [review clips](docs/experiments/2026-09-25-step5-review-videos.md)
under `outputs/experiments/2026-09-25_step5_review_videos/` show two
training-only teacher-rescued SC insertions and two failed autonomous
focused-BC treatment rollouts. The teacher and autonomous scenes differ;
the video index records timing and source-resolution limits.

To relieve the full `/data1` filesystem without losing an old checkpoint,
the May `outputs/agentic_reward_curriculum_20260529/...v704.../checkpoint_latest.pt`
file was copied to
`/var/tmp/chmin_aic_20260918_act/relocated_20260530_v704_checkpoint_latest.pt`,
verified by SHA-256
`fd7f4f012024ebe9fd4faf9a7baec7a8ce204e30fb48b6919077842a52eaac57`,
and replaced at its old path by a symlink. The old path still resolves to
the same bytes.

The latest shared-actor supervised gate is tracked in the
[execution record](docs/experiments/2026-09-24-shared-actor-sc-serl-execution.md).
The fresh full-start diagnostic is
`shared_corrected_fullstart_teacher_trace_20260925/` (`run.sh`, retained
frames, joined scored edges); its raw MCAP is SHA-verified in
[this archive manifest](docs/experiments/2026-09-25-shared-corrected-fullstart-teacher-trace-mcap-archive.json).
The phase-binned frozen locator audit is
[here](docs/experiments/2026-09-25-sc-global-opening-locator-phase-diagnostic.json),
and the bounded far-view locator ablation writes
`sc_global_far_locator_20260925/` and its adjacent `.log`. Neither is
an admitted BC source or a deployable policy.
The completed GPU run is `sc_global_far_locator_gpu_20260925/` (`run.sh`,
`locator.pt`, `locator_metrics.json`); its frozen predictions are in
`sc_global_far_locator_fresh_20260925/` and
`sc_global_far_locator_eight_20260925/`. The failed offscreen projection
ablation is in `sc_global_offscreen_locator_gpu_20260925/` and
`sc_global_offscreen_locator_eight_20260925/`. The separate, nondeployable
from-scratch true-pose actor upper bound writes
`shared_port_tcp_bc_scratch_allpose_oracle_20260925/`. The eight-scene
negative result is in
[this report](docs/experiments/2026-09-25-sc-global-far-eight-causal40.json).
The 2 mm capped full-start diagnostic is
`shared_corrected_fullstart_cap2mm_trace_20260925/` with its scored bag in
[the archive manifest](docs/experiments/2026-09-25-shared-corrected-fullstart-cap2mm-mcap-archive.json).
New training-only SC teacher collection and its exact generation/evaluation
commands are in `sc_diversity_batch2_20260925/` (`source_manifest.json`,
`run.sh`, and `postprocess.sh`). No row enters BC before admission.
The completed batch 2 labels are at
`sc_diversity_batch2_candidates_20260925/physical_waypoint_chunks.jsonl`
and `sc_diversity_batch2_lift_candidates_20260925/physical_waypoint_chunks.jsonl`;
their [original](docs/experiments/2026-09-25-sc-diversity-batch2-admission.json)
and [lift](docs/experiments/2026-09-25-sc-diversity-batch2-lift-admission.json)
admission reports record all checks. Matched BC control/treatment roots are
`shared_port_tcp_bc_early2mm_control_20260925/` and
`shared_port_tcp_bc_diversity_batch2_treatment_20260925/` with `best.pt`,
`summary.json`, `train.log`, and same-scene `eval_new_dev2_replan1/` score
files. The scored bags are in the corresponding
[control](docs/experiments/2026-09-25-shared-bc-diversity-batch2-dev2-control-mcap-archive.json)
and [treatment](docs/experiments/2026-09-25-shared-bc-diversity-batch2-dev2-treatment-mcap-archive.json)
archive manifests.
The SC 99%-sampling diagnostic is
`shared_port_tcp_bc_diversity_batch2_sc_only_probe_20260925/` with its
`run.sh`, checkpoint, offline `summary.json`, and complete same-scene
`eval_new_dev2_replan1/` engine score. The two raw bags are in the
[verified archive](docs/experiments/2026-09-25-shared-bc-sc99-dev2-mcap-archive.json),
and the [machine summary](docs/experiments/2026-09-25-shared-bc-sc99-gate.json)
records hashes. The 60-scene training-only collection source is
`sc_diversity_batch3_20260925/`; its `chunk_results.json` and adjacent
`_partNNN_NNN/` run roots hold exact scored group commands and bags.
Its admitted labels are in
`sc_diversity_batch3_candidates_20260925/physical_waypoint_chunks.jsonl`
with [scores](docs/experiments/2026-09-25-sc-diversity-batch3-scored.json)
and [admission](docs/experiments/2026-09-25-sc-diversity-batch3-admission.json).
The six-scene five-card source is `sc_fivecard_batch_20260925/`; its paired
80 mm teacher-lift replay is `sc_fivecard_lift_retry_20260925/`, with admitted
labels at `sc_fivecard_lift_candidates_20260925/physical_waypoint_chunks.jsonl`.
The [paired scores](docs/experiments/2026-09-25-sc-fivecard-lift-scored.json)
and [admission](docs/experiments/2026-09-25-sc-fivecard-lift-admission.json)
record the exact route and eligible frames. The matched continued shared-BC
roots are `shared_port_tcp_bc_batch3_{control,treatment}_20260925/`.
The [matched machine gate](docs/experiments/2026-09-25-shared-bc-batch3-matched-gate.json)
records both checkpoints, commands, offline metrics, and same-scene autonomous
scores; the corresponding [control](docs/experiments/2026-09-25-shared-bc-batch3-control-dev2-mcap-archive.json)
and [treatment](docs/experiments/2026-09-25-shared-bc-batch3-treatment-dev2-mcap-archive.json)
bags are hash-archived. The posthoc development teacher trace is
`sc_shared_bc_dev2_teacher_diag_20260925/`; it is excluded from BC.
All run roots below are under `/var/tmp/chmin_aic_20260918_act/`:
`shared_port_tcp_bc_nominal_clock_20260925/` (SC/SFP nominal decision-clock
correction), `sc_shared_early_dagger_newbatch{1,2,3}_20260925/` (nine admitted
scored teacher-bounded SC successes across three batches),
`shared_bc_takeover_after{15,20}nominal_20260925/` (teacher-prefix handoff
diagnostics), `shared_bc_takeover_after15nominal_native_20260925/` (original
resolution three-camera images and frozen-estimator audit),
`sc_unbounded_takeover_training_20260925/` (four failed training-side actor
takeovers and a **separate** 25-row safe local correction source),
`shared_port_tcp_bc_local_correction_{control,local,boost15}_20260925/`
(matched supervised ablations), and
`shared_port_tcp_bc_nominal_sc_tip_oracle_20260925/` (nondeployable true-tip
upper bound), `sc_bounded_takeover_rescue2s_20260925/` (four scored
teacher-rescue SC runs; three full successes) and its
`sc_bounded_takeover_rescue2s_candidates_20260925/` (2,046 causal chunks),
`shared_port_tcp_bc_rescue2s_{control,rescue,activeboost15}_20260925/`
(matched and active-row-sampling BC arms), and
`shared_port_tcp_bc_nominal_replan1_eval_20260925/` (same-checkpoint
first-target-only autonomous ablation),
`sc_early_takeover_rescue{2s,0p5s}_20260925/` and matching `_candidates_`
roots (one admitted scored success per four-scene batch),
`shared_port_tcp_bc_early_rescue_activeboost20_20260925/` (offline-rejected
BC), and `shared_port_tcp_bc_sc_allpose_oracle_20260925/` (privileged
all-phase pose upper bound). Each BC root has exact `run.sh`, selected `best.pt`, log and
summary. The raw scored MCAPs are SHA-verified on
`/home/nas/chmin/aic_archive_20260925/`; individual restore paths are in
the `docs/experiments/2026-09-25-*-mcap-archive.json` manifests. No result
here promotes an autonomous shared SC policy or opens RL/final scenes.

The genuinely new SC diversity source is
`sc_diversity_batch1_20260925/` (eight scored qualification-like starts,
five full successes) with admitted labels at
`sc_diversity_batch1_candidates_20260925/physical_waypoint_chunks.jsonl`.
The source and label checks are in
[admission](docs/experiments/2026-09-25-sc-diversity-batch1-admission.json).
`sc_diversity_batch1_lift_retry_20260925/` retries only the three failed
training starts; two full successes and 1,222 admitted chunks are under
`sc_diversity_batch1_lift_candidates_20260925/` with
[admission](docs/experiments/2026-09-25-sc-diversity-batch1-lift-admission.json).
`shared_port_tcp_bc_freshdiversity_{control,new5,new7}_20260925/`
holds the matched BC continuation and scored failed autonomous starts.
`shared_port_tcp_bc_scratch_new7_20260925/` is the longer offline-rejected
ImageNet-initialized fit. `sc_diversity_actor_early2mm_base_20260925/`
contains five teacher-rescued early-control trials; two successes and their
1,387 chunks are under `sc_diversity_actor_early2mm_candidates_20260925/`.
`shared_port_tcp_bc_early2mm_{control,corrected}_20260925/` holds the
matched continuation and failed fresh autonomous results.
Re-audited matrix/lift labels under
`sc_bc_matrix_pilot_candidates_20260925/` and
`sc_lift_fivecard_candidates_20260925/` are **duplicates** of the older
`sc_supplemental_candidates_v2_20260924/` source; they are provenance
audits, not additional training scenes.

The Gazebo SC native-image development ablation is documented in
[the September 24 record](docs/experiments/2026-09-24-sc-native-pose-ablation.md).
The current shared SFP/SC supervised target labels are
`/var/tmp/chmin_aic_20260918_act/sfp_verified_268_port_tcp_waypoints_20260924.jsonl`,
`sc_canonical_21_port_tcp_waypoint_chunks.jsonl`, and
`sc_supplemental_candidates_v2_20260924/port_tcp_waypoint_chunks.jsonl`
under the same `/var/tmp/chmin_aic_20260918_act/` root. Their tracked
[SFP](docs/experiments/2026-09-24-sfp-port-tcp-waypoint-labels.json),
[canonical SC](docs/experiments/2026-09-24-sc-canonical-port-tcp-waypoint-labels.json),
and [supplemental SC](docs/experiments/2026-09-24-sc-supplemental-port-tcp-waypoint-labels.json)
summaries include source hashes and frame contracts. The active one-GPU
offline BC smoke/full runs are under `shared_port_tcp_bc_smoke2_20260924/`
and `shared_port_tcp_bc_full_20260924/`; `best.pt`, `history.json`,
`summary.json`, and `train.log` identify checkpoint selection and progress.
These are supervised diagnostics until autonomous full-start Gazebo scoring.

The later shared BC arms are `shared_port_tcp_bc_command_full_20260924/`
(command consistency and initial-row balancing),
`shared_port_tcp_bc_time_full_20260924/` (causal elapsed time),
`shared_port_tcp_bc_sc_oracle_full_20260924/` (nondeployable SC near-port
ground-truth TCP-pose upper bound),
`shared_port_tcp_bc_sc_tip_oracle_full_20260924/` (nondeployable physical
plug-tip input), `shared_port_tcp_bc_joint_oracle_full_20260924/` (failed
joint TCP-plus-tip input), and
`shared_port_tcp_bc_lateral_full_20260924/` (near-port lateral objective).
Each has its own selected `best.pt`, `history.json`, and `summary.json`;
`grouped_offline_metrics_axis.json` separates lateral and axial command
errors for the comparable time/oracle arms. The runtime refuses the oracle
checkpoint. No checkpoint has passed the autonomous shared-policy gate.

`sc_yaw_boundary_refreeze_eval_20260924/` contains the **fourth independent
SC image suite**, frozen hash bundle, original images, predicted crops,
physical-tip labels, opening/tip/orientation predictions, tracked geometry,
and JSON scores. Its eight original MCAPs were compressed and verified on NAS;
[the archive manifest](docs/experiments/2026-09-24-sc-yaw-boundary-refreeze-mcap-archive.json)
contains SHA256 values and restore commands.

The SC near-port recorded-target spread is visualized at
[teacher_near_target_scatter.png](outputs/experiments/2026-09-24_shared_sc_bc_diagnostics/teacher_near_target_scatter.png),
with per-episode coordinates in
[2026-09-24-sc-teacher-target-diversity.json](docs/experiments/2026-09-24-sc-teacher-target-diversity.json).
`sc_bc_expansion_batch1_20260924/` holds eight fresh qualification-like
SC teacher starts, source scene hashes, exact `run.sh`, scored endpoints,
causal RGB records and physical-grasp calibration. Two full successes and
their 1,579 admitted chunks are under
`sc_bc_expansion_batch1_candidates_20260924/`; six failures remain outside
BC. The [bag archive](docs/experiments/2026-09-24-sc-bc-expansion-batch1-mcap-archive.json)
has verified restore commands. `sc_bc_expansion_lift_retry1_20260924/`
has three matched 8 cm teacher-route failures and its own verified bag
archive. `sc_bc_expansion_nearcap_retry1_20260924/` is the narrower 5 mm
teacher-bias retry. The supervised comparison with the two added successes
is `shared_port_tcp_bc_expanded2_full_20260924/`.

`sc_shared_dagger_smoke_20260924/` and
`sc_shared_dagger_smallclip_20260924/` contain failed isolated-scene
student correction probes. Their matched zero-student isolated controls also
ended at partial insertion. The repeatable full-success baseline needed
the original `trial_991001` then `trial_991002` order and is in
`sc_shared_dagger_sequence_control_20260924/`. The first ordered-pair
student run, `sc_shared_dagger_sequence_student_20260924/`, randomly
selected no student windows in its successful second scene and supplies no
corrective labels. `sc_shared_dagger_secondonly_student_20260924/` keeps
the first scene teacher-only and produced a scored full insertion with
400 student-active frames in the second scene. Its admitted 910 causal
chunks, with 317 student-active anchors, are in
`sc_shared_dagger_secondonly_candidates_20260924/corrective_waypoint_chunks.jsonl`.
These run roots contain exact `run.sh`, source scene config, images, frames,
and scored summaries; [the execution record](docs/experiments/2026-09-24-shared-actor-sc-serl-execution.md)
links their verified NAS MCAP archives. The one-GPU matched BC continuations
are `shared_port_tcp_bc_corrective_control_20260924/` and
`shared_port_tcp_bc_corrective_treatment_20260924/`. Neither is promoted
without guarded autonomous development scoring.

`sc_shared_dagger_multiscene_20260924/` is the later eight-scene SC
correction collection. Three scored full insertions were admitted, and
five failures were excluded. The three accepted scenes have 1,138 causal
chunks in `sc_shared_dagger_multiscene_candidates_20260924/`, with 467
student-active anchors. All eight raw MCAPs are in the
[verified NAS archive](docs/experiments/2026-09-24-sc-shared-dagger-multiscene-mcap-archive.json).
`shared_port_tcp_bc_corrective_lowweight_20260924/` is the one-episode
unit-weight ablation; `shared_port_tcp_bc_startaware_20260924/` adds
early-command loss and checkpoint selection while keeping the same actor.
`shared_port_tcp_bc_multiscene_startaware_20260924/` adds the three new
corrective scenes. The fresh actor development config and four-scene subset
are under `shared_bc_corrective_fresh_dev_20260924/`; scenes in the earlier
`sc_port_tracker_refreeze_eval_20260924/` suite are retired from actor
evaluation because they may now supply training labels.
The same start-aware checkpoint's first fresh four-scene SC rollout used
direct native-to-224 resizing and scored zero insertions with four scored
wrist/enclosure contacts. The
[camera parity audit](docs/experiments/2026-09-24-shared-bc-camera-preprocessing-parity.json)
shows millimeter-scale target changes from this preprocessing mismatch.
`shared_port_tcp_bc_startaware_20260924/eval_fresh_sc_four_preprocess_matched/`
is the paired rerun with training-matched 288×256 JPEG preprocessing and
live inference latency logging; its outcome is recorded in the
[execution record](docs/experiments/2026-09-24-shared-actor-sc-serl-execution.md).
The multiscene checkpoint's matched four-scene run and the start-aware
checkpoint's 20 mm target-cap ablation both scored 0/4; their raw MCAPs
are in the [multiscene](docs/experiments/2026-09-25-shared-bc-multiscene-fresh-sc-four-mcap-archive.json)
and [20 mm](docs/experiments/2026-09-25-shared-bc-startaware-fresh-sc-four-cap20mm-mcap-archive.json)
verified NAS archives. `shared_bc_fresh_teacher_reference_20260925/`
contains scored teacher routes and commands on two matched development
scenes; the first fully inserted. `shared_bc_early_student_probe_20260925/`
and `shared_bc_early_student_3mm_20260925/` contain clearly privileged
early-student diagnostics at 10 and 3 mm teacher-disagreement bounds. The
3 mm diagnostic inserted fully but is excluded from BC because it uses an
actor development scene. `shared_port_tcp_bc_initial_gate_20260925/`
contains the same-architecture first-command supervised continuation,
its replay command, selected checkpoint, grouped metrics, and a failed
autonomous scored rollout. `sc_shared_early_dagger_training_20260925/`
is an incomplete first collection that ran out of scratch before final
scoring; its bags are archived and no episode was admitted from it.
`shared_port_tcp_bc_initial_low_lr_20260925/` is a matched lower-learning-rate
startup-loss ablation on the same labels; its grouped offline scores improved,
but its fresh autonomous SC rollout still scored zero. The frozen-pose-head
ablation is `shared_port_tcp_bc_frozen_pose_start_20260925/` and also failed
that development start. `sc_shared_early_dagger_retry_20260925/` contains the
complete scored seven-scene replay; its three admitted successes and exact
causal labels are in `sc_shared_early_dagger_retry_candidates_20260925/`.
`shared_port_tcp_bc_early_dagger_{control,treatment}_20260925/` contain the
matched 1,500-update BC continuation, selected checkpoints, run scripts,
grouped metrics, and failed autonomous same-start evaluations. Both scored
Tier 3 zero. `sc_early_pulse_{8mm_training_probe,6mm_halfsec_training_probe}_20260925/`
contain the larger teacher-bounded startup probes, exact commands, images,
scores, and run scripts. Neither inserted fully; their raw bags have verified
NAS archives linked from the [execution record](docs/experiments/2026-09-24-shared-actor-sc-serl-execution.md).
`shared_port_tcp_bc_sc80_diagnostic_20260925/` is the 80% SC sampling
diagnostic, with checkpoint hash and exact command in `training_override.json`;
its fresh autonomous SC evaluation also failed. Its 3-second observation-only
startup hold is under `shared_port_tcp_bc_early_dagger_treatment_20260925/eval_fresh_sc_trial995001_hold3/`.
`sc_shared_early_dagger_newbatch1_20260925/` is a fresh four-scene
training-only bounded student collection; admission depends on final score
and causal label checks.

Raw 1152×1024 camera triplets, scored MCAP bags, run scripts, training-only
projected labels, learned and oracle crops, locator and pose checkpoints,
histories, and phase metrics are under
`/var/tmp/chmin_aic_20260918_act/sc_native_ablation_20260924/`,
`sc_native_ablation_extra_20260924/`, and
`sc_native_ablation_matched_20260924/` under that same root. The first
nine-scene pilot used a split incompatible with the prior checkpoint and is
retained as exploratory only. `sc_native_recollect_verified_20260924/`
contains the completed second scene-diversity pass. The expanded and
offset-augmented labels and checkpoints are under
`sc_native_expanded_20260924/`, `sc_native_offset_aug_20260924/`, and
`sc_native_multioffset_aug_20260924/`. New validation captures and frozen
spatial-locator results are under `sc_fresh_dev_20260924/`; new pose-only
training captures are in `sc_fresh_train_20260924/`. Tracked [capture](docs/experiments/2026-09-24-sc-native-ablation-capture.json)
and [matched label](docs/experiments/2026-09-24-sc-native-ablation-labels-matched.json)
manifests preserve scores and scene grouping. The oracle crops are privileged
diagnostics and cannot be used for autonomous crop selection.

The later SC pose work is under `sc_native_broadened_20260924/` (training
and 12 development scenes), `sc_independent_dev_20260924/` (first independent
scene check), `sc_yaw_extremes_20260924/` (12 yaw-edge training scenes),
`sc_native_yaw_aug_20260924/` (retrained position/orientation checkpoints),
and `sc_yaw_independent_20260924/` (six yaw-edge development scenes).
`sc_pose_frozen_eval_20260924/` contains the hashed perception bundle,
exact generator seeds, commands, and eight new development captures for a
one-shot check of the position-anchored orientation cold start. That check
found 1.608° orientation p95; the later board-rail yaw estimator and its
offline timing are saved in the same directory. Its separately frozen
eight-scene confirmation bundle is in
`sc_board_lines_frozen_eval_20260924/`. The
[native-image record](docs/experiments/2026-09-24-sc-native-pose-ablation.md#yaw-edge-continuation-and-frozen-confirmation)
maps the raw and filtered reports; these runs are pose diagnostics, not
shared-actor insertion results. Large JPEGs and bags stay under `/var/tmp/`.

`sc_bc_matrix_pilot_20260924/` is the scored teacher-data pilot for missing
SC card-count/port cells; its `source_manifest.json` identifies all 16 scenes,
including zero-card scenes explicitly derived from generated one-card starts.
Treat its captures as candidates only until score, image, and action audits
mark individual trajectories eligible for BC.
Its `dynamic_grasp_calibration.json` and `dynamic_tip_subset.json` compare
the **physical** scored plug tip with the old fixed TCP→tip proxy. The first
eight-scene board-rail confirmation and physical-tip failure are under
`sc_board_lines_frozen_eval_20260924/` (`frozen_bundle.json`,
`position_metrics.json`, `board_lines_yaw.json`,
`frozen_physical_tip_metrics.json`). Corrected 54-scene physical-tip labels,
the first retrained refiner, and its train/development metrics are in
`sc_native_yaw_aug_20260924/`. The expanded tip training set, including all
16 pilot scenes for **pose supervision only**, is in
`sc_physical_tip_augmented_20260924/`. The matched teacher lift probes are
under `sc_lift_route_probe_20260924/` and
`sc_lift12_route_probe_20260924/`; the latter remained unsuccessful on its
five-card scene. `sc_port_swap_lift_probe_20260924/` is an invalid diagnostic:
its target port was absent from the scene and it is excluded from route and
training conclusions.
Three bulky pilot MCAPs were compressed to
`/home/nas/chmin/aic_archive_20260924/sc_bc_matrix_pilot_20260924/`
after each decompressed stream matched its original SHA-256. The
[archive manifest](docs/experiments/2026-09-24-sc-pilot-mcap-archive.json)
gives exact original paths and restore commands. The current rootless ROS
container does not mount `/home/nas`, so restore a bag into its original
`/var/tmp` directory before using ROS bag tools inside that container.

The second frozen physical-tip suite is under
`/var/tmp/chmin_aic_20260918_act/sc_physical_pose_frozen_eval_20260924/`:
`frozen_bundle.json`, scored scene joins, physical tip labels, frozen RGB
predictions, original failed three-view metrics, corrected yaw metrics, and
the inspected hybrid-port development reports. Its eight raw MCAP streams
were SHA-256 verified after compression to
`/home/nas/chmin/aic_archive_20260924/sc_physical_pose_frozen_eval_20260924/`;
the [manifest](docs/experiments/2026-09-24-sc-physical-frozen-mcap-archive.json)
contains restore commands. The third separately frozen eight-scene
confirmation is under `sc_port_tracker_refreeze_eval_20260924/` with its own
bundle, seeds, RGB crops/predictions, frame-level port/tip/yaw/relative
metrics, and full offline latency. Its eight scored MCAP streams were
SHA256-verified after compression to
`/home/nas/chmin/aic_archive_20260924/sc_port_tracker_refreeze_eval_20260924/`;
[restore paths](docs/experiments/2026-09-24-sc-port-tracker-refreeze-mcap-archive.json)
are tracked. The 11 supplemental scored SC successes, including the
two-card/port-1 anti-windup recovery, are indexed under
`sc_supplemental_candidates_v2_20260924/`. Their exact BC admission audit
is [tracked](docs/experiments/2026-09-24-sc-bc-admission.json); failed
matching teacher runs remain in their source directories and are excluded.

The corrected Isaac SC card-route probes are under
`outputs/experiments/2026-09-24_isaac_sc_contact_routes/`: seven raw 20 Hz
physics traces, exact scripts/logs, and a named-force comparison plot.
The [tracked route analysis](docs/experiments/2026-09-24-isaac-sc-contact-routes.md)
and [machine summary](docs/experiments/2026-09-24-isaac-sc-contact-routes.json)
distinguish gripper/card, plug/card, and secondary cable/card contact. All
routes are privileged scripted diagnostics with unsafe peak force; none is
visual RL replay or an expert demonstration.

The repaired Isaac SC development probes are in
`outputs/experiments/2026-09-24_isaac_sc_fidelity/`: centered near-port and
full-start five-card traces, two deliberate local plug/port blockages, exact
run scripts, and logs. `camera_smoke_repaired/` now contains three decoded
wrist-camera frames, three short wrist videos, and a diagnostic overhead view
showing the orange cable and card row. The [camera follow-up](docs/experiments/2026-09-24-isaac-sc-fidelity-fix.md#camera-follow-up-september-24)
records the container-local driver-check override and its limits.
`camera_motion_clear_lane/` contains a 61-frame, 20 Hz moving probe and four
0/30/60-frame sheets; the distal cable shape is not yet Gazebo-validated. Their
[tracked mechanics manifest and SHA-256 values](docs/experiments/2026-09-24-isaac-sc-fidelity-summary.json)
and [repair record](docs/experiments/2026-09-24-isaac-sc-fidelity-fix.md)
distinguish these physics-only diagnostics from visual RL replay. The exact
development scene is tracked under
`configs/hierarchical_recovery/isaac_sc_fidelity/`; the generated robot and
port USDs are ignored but can be rebuilt with
`scripts/build_sc_isaac_fidelity_assets.sh` inside rootless IsaacLab. Replay
the four scenarios with `scripts/run_sc_isaac_fidelity_probe.sh`.

The [hierarchical recovery plan](docs/experiments/2026-09-24-hierarchical-recovery-continuation.md)
indexes the 20 Hz five-card stop, its scene YAML/route, synchronized videos,
and bulk MCAP root. The compact indexed set is
`outputs/experiments/2026-09-24_hierarchical_recovery/incident_index.json`:
24 retained Gazebo episodes from fixed-route and targeted suites (8 full,
11 partial, 5 none), plus a 21-episode canonical SC expert audit. Its five
no-insertion episodes are generic failure candidates, not proven cable snags.
The same ignored run root holds `wide_repeat_02_trace.jsonl` through
`wide_repeat_08_trace.jsonl`, `targeted_01_trace.jsonl` through
`targeted_10_trace.jsonl`, `trace_diagnostic.json`, and `trace_cohorts.json`;
regenerate them with
[the tracked extractor](scripts/extract_recovery_trace.py) and
[read-only analyzer](scripts/analyze_recovery_traces.py) from the retained
`/var/tmp/chmin_aic_cable_route_smooth_20260923/` and
`/var/tmp/chmin_aic_targeted_failure_20260923/` MCAP bags. These are
diagnostics, not training labels or validated live thresholds.
The three new normal-collision five-card Gazebo scene variations are indexed by
[their scored manifest](docs/experiments/2026-09-24-hierarchical-recovery-fresh-scenes.json):
tracked YAMLs under `configs/hierarchical_recovery/`, compact combined 20 fps
videos and overhead/side sheets under `artifacts/hierarchical_recovery/`, and
raw bags/frames under `/var/tmp/chmin_aic_hierarchical_recovery_20260924/`.
The SC target-frame audit is tracked as
[port TF edges](docs/experiments/2026-09-24-sc-expert-port-edges.json) and
[per-episode timing/label metrics](docs/experiments/2026-09-24-sc-port-target-audit.json).
Its 13,349-row derived target JSONL remains at
`outputs/experiments/2026-09-24_hierarchical_recovery/sc_port_tcp_teacher_targets.jsonl`
and is regenerated by `scripts/audit_sc_port_targets.py` from the retained
canonical episodes and scored TF bags.

| Location | What belongs here / what was found |
| --- | --- |
| `configs/` | Tracked recipe inputs. A config can still depend on ignored generated episodes, datasets, or checkpoints; inspect its paths before use. |
| `outputs/hf_combined/`, `outputs/trajectory_datasets/` | Derived/recorded LeRobot data. `expert_verified/` is the canonical BC collection; `clean_including_no_insert_trajs/` preserves the original 668 episodes across 23 collections. Successful recordings with unreliable action labels are indexed separately in `successful_pending_label_repair/`. See [dataset membership and S3 paths](docs/DATASETS.md). |
| `outputs/train/` | ACT, offline/online SERL, and legacy PPO training artifacts. It is not exclusively a PPO directory. |
| `outputs/experiments/2026-09-17_live_validation/` | Live control/reset checks, dataset audit, runtime regressions, configs, and `review/index.html` with MP4s and one-second snapshots. ACT/direct raw policy trials stay under their respective model run roots; see the [run report](docs/experiments/2026-09-17-live-validation.md). |
| `outputs/experiments/2026-09-19_full_world_training/` | Completed fresh tokenizer, strict 148-episode dynamics, supervised policy, and frozen 20-scene evaluation. `artifacts/` contains selected tokenizer/dynamics/control checkpoints, reconstruction sheets, final videos and compact reports; `artifact_sha256.txt` pins them. Bulk mutable intermediates remain under `/var/tmp/chmin_aic_20260919_full_world/`. |
| `outputs/experiments/2026-09-20_isaac_world_rl/` | Isaac transfer contract summary, exact run commands, selected non-promoted DAgger checkpoints, guide/autonomous videos, and per-episode varied-scene geometry metrics. `pose_probe/` contains the causal replay audit, episode-grouped manifests, frozen-feature and native-crop checkpoints/metrics/plots, complete latency results, corrected projected-label audit, opening-landmark continuation, and prediction-only visual review. Exact-keypoint triangulation passed its ceiling; the latest compact pretrained locator reached 0.278/0.670 mm median/p95 lateral error and failed the unchanged gate, so no conditioned policy or RL was started. `selected_varied_only/` is a failed supervised diagnostic, not a promoted controller. Bulk native RGB and replay remain under `/var/tmp/chmin_aic_20260920_isaac_world_rl/`. |
| `outputs/experiments/2026-09-21_temporal_multiview_perception/` | Frozen-spatial learned multiview and six-step causal temporal calibration, checkpoints, six new development reset configs, one-shot fresh metrics, latency, commands, summary, and hashes. Current-frame fusion improved the baseline to 0.307/0.573 mm near-port lateral median/p95 but failed the 0.25/0.5 mm gate; temporal p95 was worse at 0.667 mm. Bulk RGB/replay remains under `/var/tmp/chmin_aic_20260920_isaac_world_rl/pose_probe_temporal_multiview_development_collect1200/`. |
| `outputs/experiments/2026-09-22_cable_visibility_perception/` | Cable reset and force audits, closed-loop plug restoration, staged restore-to-CheatCode handoff, mask-derived offline visibility labels, static/natural/combined comparisons, selected checkpoint, prediction montage, latency, frozen error-tail audit, summary, artifact map, and hashes. The natural run retained 460/1,500 observations from all five included templates; the selected unseen-shape result was 0.322/0.626 mm lateral median/p95 with 7.18 ms p95 latency. Camera removal and fixed pixel-bias correction did not improve the tail. Further pose fitting is parked; the frozen estimate may support a supervised controller diagnostic, while RL remains gated on autonomous insertion. Bulk native RGB, masks, replays, and derived datasets remain under `/var/tmp/chmin_aic_20260920_isaac_world_rl/cable_visibility_20260922/`. |
| `outputs/experiments/2026-09-22_pose_conditioned_gru_policy/` | Matched 546,686-parameter action-only and frozen-pose-conditioned GRU checkpoint, episode-grouped metrics, eight-start development manifest, autonomous episode metrics, true/zero/shuffled pose-reliance audit, complete latency benchmark, three-camera endpoint sheets, MP4s, summary, artifact map, and hashes. Both arms achieved 0/8 insertion; generic BC largely ignored pose and the branch stopped before RL. Bulk live metrics and replay remain under `/var/tmp/chmin_aic_20260920_isaac_world_rl/`. |
| `outputs/experiments/2026-09-22_explicit_pose_correction/` | Balanced corrective YAMLs, dataset coverage and pair-selection audits, a frozen nominal GRU plus 66,596-parameter mandatory odd-symmetric pose correction, two failed offline selections, phase/motion-size slices, machine summary, artifact map, and hashes. Correct pose improved held-out translation command MAE by only 3.10% and missed the fixed dependence gate; all retained rows were approach-phase, live starts stayed unopened, and no RL ran. Bulk replays remain under `/var/tmp/chmin_aic_20260920_isaac_world_rl/cable_visibility_20260922/`. |
| `outputs/experiments/2026-09-22_rpdp_dppo/` | PoseDP/RPDP datasets and checkpoints, recorded-target 50 ms port-frame waypoints, verified connector-to-TCP adapter, low-force DAgger data, deterministic and diffusion BC comparisons, causal context, DPPO diagnostics, latency reports, commands, hashes, and machine summaries. The selected deterministic BC arm reached 6/6, 8/8, then 4/5 on near-port tests at 20.12 ms latest p95. These are local 8 mm surrogate seating results: the configs override the physical approximately 45.8--48.72 mm SFP depth and use privileged near-opening resets. They are not normal full-episode or official Gazebo insertions. Per-episode videos are in `bc_repair_direct/fresh_test5_videos/`. The later probabilistic RL continuation is in the next row; the reserved final split remains sealed. Bulk replay and native images remain under `/var/tmp/chmin_aic_20260920_isaac_world_rl/`. |
| `outputs/experiments/2026-09-22_serl_recovery/` | Four-component full-trajectory actor, measured-path recovery, outcome-relabeled prior, critic warm-up, online SAC checkpoints, policy-collapse audits, five matched eight-episode summaries, commands, artifact map, and machine summary. Frozen BC scored 3/8; two unregularized online actors and the first KL repair scored 0/8. Fixing full-mixture anchoring and restored-optimizer learning-rate override recovered 3/8 on the same episodes, with no improvement. A separate three-camera recording rerun is under `videos/`: BC 1/8, selected RL 1/8, and selected RL plus backtracking 3/8; its review page preserves all 72 per-episode clips and is not treated as a promotion. The durable interpretation is in [the video failure analysis](docs/experiments/2026-09-23-serl-video-failure-analysis.md). Complete live p95 inference was 22.67 ms. `online_selected/` is a safe-continuation diagnostic, not a promoted policy. Reserved final scenes remain sealed. Bulk replay and failed checkpoints remain in container `/var/tmp`. |
| `outputs/experiments/2026-09-23_sc_cable_bringup/` | Machine summary, artifact map, and two left-camera videos for the SC mechanics gate. The original-grasp five-card failure is retained as invalid cable-snag evidence because contact ablation traced it to gripper/card collision. The second video is a privileged gripper-clearance routing proxy, not an autonomous policy rollout. Bulk per-step JSON diagnostics remain under `/var/tmp/chmin_aic_20260920_isaac_world_rl/sc_snag_20260923/`. |
| `artifacts/prod_cheatcode_audit/targeted_failure_reproduction/` | Ten-trial post-fix Gazebo SC reproduction: exact YAML, manifest, official scores, compact plug/port analyses, summary, contact sheets, and three-camera videos for one axial partial and one large tracking failure. Bulk bags and one-Hz frames remain under `/var/tmp/chmin_aic_targeted_failure_20260923/`. The cable stayed visibly clear of the card field; these artifacts are not labeled cable snags. |
| `artifacts/prod_cheatcode_audit/ordinary_broad_followup/` | Exact 19-scene ordinary-development manifest/YAML, selected valid and invalid-attempt score summary, historical archive force review, measured plug/port analyses, and compact videos. `vlm_review/` contains the 190-attempt agent replay scan, exact three-card low-score route replay, regenerated five-card seed control, compressed route, scores, force/path plot, and machine summary. Scripts in the parent directory reproduce the audits. Bulk bags, engine logs, and camera frames remain under the documented `/var/tmp/chmin_aic_ordinary_*`, `/var/tmp/chmin_aic_vlm_route_replay_20260923/`, and `/var/tmp/chmin_aic_vlm_seed51500_20260923/` roots. These results do not establish a current cable snag. |
| `artifacts/prod_cheatcode_audit/ordinary_broad_followup/cable_route_probe/` | Fixed five-card Gazebo across-card and outside-left route comparison: seven diagnostic scores and exact scenes/plans, measured plug/TCP analyses, 20-link cable geometry samples, overhead/side plots, three-camera videos, and machine summary. One across-card no-insertion has a cable-trap pattern, but named cable/card contact and causal ablation remain missing. The old failed VLM recording was not found in checked S3 prefixes; an accepted S3 video was downloaded and hash-matched. Bulk bags and frames remain under `/var/tmp/chmin_aic_cable_route_probe_20260923/`; see the [experiment record](docs/experiments/2026-09-23-fixed-five-card-route-probe.md). |
| `outputs/experiments/<run-id>/` | Default for the repaired stateful/axial wrappers: high-level config, common flags, generated per-level episodes, retained training/evaluation cycles, summaries, events, and latest checkpoint. |
| `outputs/agentic_reward_curriculum_*/`, `outputs/one_day_insertion_pipeline/` | Historical generated episodes, commands, reset/controller diagnostics, rewards, checkpoints, metrics, and videos. Names/dates alone do not identify code or success criteria. |
| `outputs/gazebo_rl/` and per-model `runtime_eval*` directories | Gazebo bridge and saved-policy evaluations. The ACT runtime evaluator saves numbered attempts; read `scoring_yaml` from the checkpoint's `eval_summary.json`. A summary can describe a failed run. |
| `artifacts/` | Local diagnostic exports. June reset probes and video folders are present. This directory is also ignored. |
| `outputs/reentry_audit_20260917/june_run/` | Recovered June `events.jsonl` and final eval summary/config; see [audit hashes](docs/experiments/2026-09-17-reentry-audit.md). Contains metadata, not a copied model bundle. |
| `submission_handoff/` | Historical scripts. Handoff text is archived under `obsolete/submission_handoff/`. Referenced model/evaluation directories and `docker/aic_submission/Dockerfile` are missing locally; do not assume this is a complete submission bundle. |
| `isaac-lab-base:/tmp/...` | Some June training outputs reside only in the stopped container's filesystem. A host `/tmp` path with the same name is a different location. Recover before container removal/recreation. |

Exact baseline/run paths and checked availability are in the
[ledger artifact table](docs/EXPERIMENTS.md#artifact-locators). The old
`checkpoints/last` symlink in the ACT run currently resolves to step 375000;
specify step 175000 explicitly when reproducing that baseline.

## Storage convention for new experiments

Use `outputs/experiments/<UTC-date>_<short-name>/` as a new run root, with a
matching small report under `docs/experiments/`. Stateful/axial launchers now use
this default. Other scripts may still need an explicit output directory.

`run_one_day_insertion_pipeline.py` writes its generated `summary.md` under
`--output-root`; its dated May reports are preserved in `obsolete/docs/`.

Save the exact command, environment overrides, resolved configs, code commit
and dirty diff, dependency/image identity, dataset/checkpoint lineage, metrics,
scores, logs, and selected videos there. A portable policy bundle includes its
checkpoint plus the matching ACT export if required, JSON metadata, observation
normalizer/preprocessor, action schema, and evaluation config. Record a durable
backup location and checksums for selected models; an ignored directory alone
is not backed up by Git.

New `direct_visual` checkpoints embed their backbone, action head, normalization,
limits, and architecture metadata. They need no ACT export/normalizer at inference.
Keep the implementation/environment with them; DINOv2 additionally needs its
pinned model code/cache. See [direct visual policies](docs/DIRECT_VISUAL_POLICY.md).

Inside Isaac, set `AIC_STATEFUL_RUN_ROOT` to
`/workspace/isaaclab/aic/outputs/experiments/<run-id>` to write through the bind
mount. The repaired wrappers retain all cycle directories; plan disk space and
archive selected bundles deliberately. Old June artifacts still reside in
container `/tmp` and need recovery before container removal. Keep a fresh run ID for
reruns and configuration changes. Never use `latest`/`last` alone as model
provenance.

`.gitignore` excludes `outputs/`, `artifacts/`, `build/`, `install/`, `log/`,
the Pixi environment/lock, Isaac's downloaded `Intrinsic_assets`, and large
dataset/video/bag formats. A fresh clone needs these dependencies/artifacts
restored separately. Record availability as present, missing, or not checked;
retain missing-path references when they explain an old result.

## September verified ACT artifacts

`outputs/experiments/2026-09-17_act_verified_8h/` contains the score/image audit,
causal image caches, ACT runs, official evaluations and videos. Start with its
`results_latest.md`, `review/index.html`, and the
[experiment report](docs/experiments/2026-09-17-act-verified-8h.md). Each run has
training/normalization metadata, standard LeRobot checkpoints and source copies.
The 87-recording cache references earlier image shards; preserve those parents.

`selected_act_final/` contains the selected model, normalizer, export, frozen
selection/scene hashes, training lineage, final official results and terminal
review sheets. Read its `final_results.md` for the failed reliability target.
`environment/` preserves the resolved package list and Pixi manifests;
`source_snapshot/` records the dirty source/docs state and base commit.

Forty-two completed simulation bags were losslessly archived to adjacent
`.tar.zst` files after member-by-member decompression/hash verification, saving
54.08 GB. `verification/archived_diagnostic_bags.json`,
`verification/archived_completed_bags.json` and
`verification/archived_pre_final_bags.json` record each original location,
archive/member checksums and exact restore arguments. Frames, scores, caches
and models remain expanded. No S3 upload or external backup was performed.

## Historical detailed inventory

The original inventory below is retained for locating earlier expert/data
experiments. Its folder counts and words such as “current” describe the old
inventory, not a refreshed September listing. Only the selected artifacts
listed above and in the audit were checked in this review.

## Top-level folders

| Folder | Meaning |
| --- | --- |
| `configs/` | Ad hoc generated run configs. The current file, `vlm_backoff_real_run_1.yaml`, is a saved expert-teacher/VLM recovery run configuration. |
| `expert_datasets/` | Full live expert-generation attempts. These run the two-stage official teacher path: planner attempt, postprocessed trajectory, replay attempt, scoring, validation, and accepted metadata. Timestamped folders are individual runs. |
| `expert_matrix_configs/` | First full fixed-setting matrix of generated `request.yaml` and `engine_config.yaml` files. It contains 92 settings: 80 `sfp_to_nic` and 12 `sc_to_sc`. These are configs, not recorded datasets. |
| `expert_matrix_configs_smoke/` | Small smoke subset of the fixed-setting matrix. It contains 4 `sfp_to_nic` settings for quick validation of matrix generation/running. |
| `expert_matrix_configs_v2/` | Current fixed-setting matrix configs. Same 92 setting coverage as `expert_matrix_configs/`, with `trials_per_config: 1` recorded in the manifest so each matrix setting maps to exactly one engine trial. |
| `expert_matrix_fast_all_modes_*` | Real matrix sweep outputs from `scripts/run_expert_setting_matrix.py`. These run each fixed setting in one or more expert modes and save per-setting replay artifacts, summaries, scores, and GPT/debug output. |
| `expert_matrix_repair_*` | Targeted repair experiments for failed matrix settings. These are smaller reruns with modified candidate count or SC alignment/servo settings. |
| `expert_matrix_runs_dry/` | Dry/early matrix runner test output. The recorded row failed before simulation because dependencies were missing under plain `python3`; keep it only as runner-debug history. |
| `hf_datasets/` | Placeholder/export location for Hugging Face/LeRobot dataset publishing. It is currently empty. |
| `train/` | Isaac/low-dimensional PPO training outputs. Subfolders are separate training or smoke runs. |
| `trajectory_datasets/` | User-facing LeRobot trajectory dataset roots plus their generation requests, generated engine configs, raw/accepted datasets, score summaries, and evaluation notes. |

There is also a top-level file, `expert_matrix_reconstructed_run_configs.jsonl`,
which is a reconstructed audit log of matrix run configurations.

## Expert matrix config folders

`expert_matrix_configs/`, `expert_matrix_configs_smoke/`, and
`expert_matrix_configs_v2/` have this structure:

```text
matrix_manifest.yaml
sfp_to_nic/<setting_name>/request.yaml
sfp_to_nic/<setting_name>/engine_config.yaml
sc_to_sc/<setting_name>/request.yaml
sc_to_sc/<setting_name>/engine_config.yaml
```

The manifest is the index. Each setting points to the request, generated engine
config, task family, and intended derived dataset directory.

Setting names encode the scene:

| Pattern | Meaning |
| --- | --- |
| `matrix_sfp2nic_cards3_present124_target2_port1` | SFP module into NIC: 3 NIC cards are present, on rails/cards `1`, `2`, and `4`; target card is `2`; target SFP port is `1`. |
| `matrix_sc2sc_sc2_present01_target1_nic2` | SC plug into SC port: 2 SC ports are present, ports `0` and `1`; target SC port is `1`; there are 2 NIC distractor cards. |

## Expert matrix run folders

The `expert_matrix_fast_all_modes_*` folders are sweep results over the fixed
settings. They all use the same internal shape:

```text
matrix_results.jsonl
matrix_run_config.json        # present for newer runs
nominal/setting_<index>_<setting_name>/repeat_<NN>/
nominalrecovery/setting_<index>_<setting_name>/repeat_<NN>/
recovery/setting_<index>_<setting_name>/repeat_<NN>/
```

Mode meanings:

| Mode | Meaning |
| --- | --- |
| `nominal` | Clean insertion demonstration. Recovery/backoff is disabled; contact or force problems reject the attempt. |
| `nominalrecovery` | Starts with the nominal path but allows online recovery/backoff if contact makes it necessary. |
| `recovery` | Recovery-capable run used to collect or validate recovery behavior. In the current broad sweep, it is configured as "recover only if needed" rather than forcing a failure. |

Individual sweep folders:

| Folder | Meaning |
| --- | --- |
| `expert_matrix_fast_all_modes_v8_from_setting4/` | Early all-mode sweep starting at manifest setting 4. Contains 21 result rows. |
| `expert_matrix_fast_all_modes_v10_from_setting8/` | Follow-up all-mode sweep starting at setting 8. Contains 9 result rows. |
| `expert_matrix_fast_all_modes_v11_from_setting10/` | Larger continuation from setting 10. Contains 125 result rows; early rows include failures before later debug logging was improved. |
| `expert_matrix_fast_all_modes_v15_from_setting47/` | Continuation from setting 47. Contains 25 result rows and a posthoc `matrix_run_config.posthoc.json` capturing the run settings. |
| `expert_matrix_fast_all_modes_v16_from_setting55_logged/` | Current/newer logged continuation from setting 55. Contains 285 result rows, full `matrix_run_config.json`, embedded run configs in result rows, and `mid_sweep_code_changes.md`. |

## Repair experiment folders

| Folder | Meaning |
| --- | --- |
| `expert_matrix_repair_candidates_v1/` | Targeted setting-82 SC-to-SC nominal rerun with `candidates_per_scene: 3` and `max_total_attempts_per_repeat: 3`, testing whether trying more planner candidates fixes the failure. |
| `expert_matrix_repair_sc_align_v1/` | Same targeted setting-82 rerun, adding SC-specific precontact alignment settings: larger SC alignment cap/gain and stricter SC tracking gate. |
| `expert_matrix_repair_sc_servo_v1/` | Same targeted setting-82 rerun, adding SC guarded-insert lateral servo settings on top of the SC alignment changes. |

## Expert dataset folders

`expert_datasets/nominal_live_full_<timestamp>Z/` folders are one-shot live
nominal expert-generation runs from April 30, 2026. Each contains:

| Subfolder/file | Meaning |
| --- | --- |
| `generation_config.json` | Arguments/settings used for the run. |
| `generation_summary.json` | Accepted count, scores, planner/replay commands, validation result, and stop reason. |
| `planner_attempts/` | Planner-side dataset, planner debug artifacts, and `piecewise_trajectory.json`. |
| `replay_attempts/` | Postprocessed `smooth_trajectory.json`, replay LeRobot dataset, scoring results, temporary files, and debug analysis. |
| `accepted_metadata/` | Metadata copied for accepted trajectories. Empty or incomplete when the run did not pass acceptance. |
| `*.run.log` | Console log for the timestamped run. |

Current timestamped run outcomes:

| Folder | Outcome |
| --- | --- |
| `nominal_live_full_20260430T220209Z/` | 1 attempt, 0 accepted, score about 68.98; insertion happened but off-limit contact/force rejected it. |
| `nominal_live_full_20260430T221800Z/` | 1 attempt, 0 accepted, score about 94.02; insertion happened but validation rejected it. |
| `nominal_live_full_20260430T231409Z/` | 1 attempt, 0 accepted, score about 94.30; insertion happened but validation rejected it. |
| `nominal_live_full_20260430T231933Z/` | 1 attempt, 1 accepted, score about 95.02. |

## Trajectory dataset folders

`trajectory_datasets/` is the LeRobot dataset-generation area produced by
`aic_utils/lerobot_robot_aic/scripts/generate_trajectory_dataset.py`.

| Folder/file | Meaning |
| --- | --- |
| `sfp_to_nic/cheatcode/` | Datasets collected with the `CheatCode` policy for SFP-to-NIC insertion. `nic_cards_1/n1__test_n3`, `n2__act_smoke`, and `n10__act_smoke` are specific request sizes/suffixes. |
| `sfp_to_nic/vlm_planner/` | Dataset attempts generated with the VLM planner path before postprocessing. |
| `sfp_to_nic/vlm_planner_postprocessed/` | Postprocessed VLM-planner dataset outputs. |
| `evaluation_summaries/` | Markdown summaries from full evaluation iterations. |
| `planner_optimizer_validations/` | JSON validation artifacts for planner/optimizer gates, especially port-frame checks. |
| `runtime_attempts.jsonl` | Runtime attempt log across dataset-generation experiments. |
| `runtime_settings_comparison_2026_0425.json` | Saved comparison of runtime settings from April 25, 2026 experiments. |

Within generated dataset roots, the usual files/folders are:

| Subfolder/file | Meaning |
| --- | --- |
| `request.yaml` | User-level generation request. |
| `engine_config.yaml` | Generated AIC engine config. |
| `trials/` | Per-trial engine config slices. |
| `raw_dataset/` | Native LeRobot dataset root before filtering. |
| `accepted_dataset/` | Filtered LeRobot dataset root after applying score/success acceptance criteria. |
| `scores/` | Scoring outputs and score summary CSV. |
| `logs/` | Recording and filtering logs. |
| `generation_summary.json` | Summary of generated/accepted trajectories and paths. |

## Training folders

| Folder | Meaning |
| --- | --- |
| `train/isaac_rsl_rl_helper_smoke/` | Isaac Isaac online RL helper smoke-training output. |
| `train/isaac_rsl_rl_smoke/` | Isaac Isaac online RL smoke-training output. |
| `train/isaac_aic_lowdim_ppo/` | Low-dimensional PPO training output for the Isaac online RL AIC task. |

These training folders currently contain run directories under `aic_task/`.


### September 30 continuation: deployment input audit

The transport gate is accepted; full insertion remains a separate gate. SC dataset v5 fixes canonical TCP quaternions and the raw ROS joint-prefix order: all 4,752 transport rows exactly match the live state assembler. Earlier v3 BC is retained as an invalid-input diagnostic. Corrected rootless GPU7 BC is bounded at 1,500 updates; auxiliary depth visibility remains rejected. See [the continuation record](docs/experiments/2026-09-30-transport-to-two-actor-continuation.md). Autonomous handoff and RL have not passed.


### September 30 completed transport-policy comparisons

Under `outputs/experiments/2026-09-30_transport_continuation/`:

- `matched_comparison.json`: four autonomous failures; teacher transport remains accepted.
- `transport_bc_cartesian_matched/`: control-interface-matched BC, 1,500 updates.
- `transport_bc_rotation_scaled/`: same data/architecture, explicit rotation units, best update 1,000; source snapshots and hashes included.
- `autonomous_transport/cartesian_best_port{0_two,1_three}/`: frozen first comparison, bags/RGB/terminal traces and metrics.
- `autonomous_transport/rotation_scaled_port{0_two,1_three}/video/all_views_20fps.mp4`: recorded revised-policy failures; Ubuntu-compatible H.264. Side and overhead videos/contact sheets accompany them.
- `cartesian_matched_teachers/port0_four/video/transport_only_20fps.mp4`: reviewed transport-only prefix, excluding failed insertion; full failure retained separately.
- `cartesian_matched_teachers/port1_five/video/all_views_20fps.mp4`: fully inserted privileged teacher.
- `teacher_target_magnitude_audit.json`, `state_normalization_diagnostic.json`, `initial_control_reuse.json`: posthoc diagnostics, not training-success evidence.

Dataset `outputs/trajectory_datasets/transport_local_verified/2026-09-30_v8_cartesian/` contains 1,678 transport training chunks and 1,738 held-out diagnostic chunks; only 268 successful local chunks from this matched pair. Original successful local data remains separate. Do not call the failed-insertion prefix a full-success expert episode.


### October 14 insertion continuation artifacts

Bulk root: `/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1`.

- `local_tip_actor_{constant_translation,translation}_v1`: completed matched 6,000-update feature-input fits, including best/last checkpoints. Treatment failed the offline gate; `tip_actor_pilot_not_started.json` records why no rollout was launched.
- `local_sc_tip_target_dataset_v5`: preserved derivative with complete plug-tip targets and original TCP command labels.
- `point_target_comparison_frozen_plan_v1.json`: matched action-reference hypothesis, hashes, budgets, primary decoded-command metric and conditional rollout gate.
- `local_point_target_{tcp,tip}_v1`: comparison fits in progress on GPU 5.
- `point_target_conversion_check.json`: numerical training/runtime conversion check, with SFP unchanged.

These comparisons have not passed physical autonomous insertion. Reserved final scenes remain sealed.


### Causal continuation repair (October 14)

Under the same NAS execution root:

- `local_causal_continuation_*`: immutable actor-prefix/teacher-continuation trials. Failed versions and shallow contact events are retained.
- `causal_local_continuation_dataset_v3`: selected 1,765-row derivative; v4 is a byte-identical copy.
- `causal_dataset_12_windows_review.{jpg,json}`: reviewed native-camera sample grid and exact row/provenance metadata.
- `causal_local_matched_fit_frozen_plan_v1.json`: dataset/checkpoint/source hashes, two matched fit commands and frozen validation gate.
- `local_causal_actor_{control,correction}_v1`: control completed, correction fit running; best/last snapshots are preserved.
- `run_causal_actor_paired_pilot_v1.py`: waits for both fit summaries and launches the bounded comparison only if the correction arm passes.
- `observation_to_command_age_audit_v1.json`: autonomous timing did not exceed the 200 ms action horizon. No QoS change was promoted.

Video: `rule_then_oracle_tip_llb_1400_integral_rate2_v3/five_cameras_20fps_grid_active.mp4` is a **privileged controller diagnostic**, not a trained-policy success. `local_point_tip_llb_1400_pilot_v2/five_cameras_20fps_grid_active.mp4` is the failed, observation-only trained-policy rollout.

### October14 decoder and causal-history diagnostics

NAS execution root `/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1`: `local_conditional_decoder_{transformer,mlp}_v1/` retain6000-update fits; `conditional_decoder_pilot_not_started_v1.json` records failed gate. `local_history_frozen_plan_v1.json` stores commands and gates for the current sequential matched history fits; `local_history_{masked,causal}_v1/` retain selected/last checkpoints and metrics when complete. `local_pause_phase_audit_v1.json` and `local_causal_history_parity_v1.json` document holds and shared sampling checks. No new videos exist for offline-ineligible decoder models.

### October14 observation-only approach and partial insertion

NAS execution root: `local_history_comparison_summary_v1.json` and `local_history_pilot_not_started_v1.json` record failed6000-update masked/history gates. `observation_approach_frozen_plan_v1.json` freezes the separate component-ownership diagnostic. Its rollout is `observation_approach_llb_1400_v1/`, with `runtime_trace.jsonl`, raw bags, scored stream, oracle contacts, exact input hashes and **`five_cameras_20fps_grid_active.mp4`** (H.264/yuv420p,20fps). `observation_approach_terminal_metrics_v1.json` reports11.073mm depth,0.413mm lateral and0.938° orientation: partial insertion, not physical-gate success. `observation_approach_critical_review_v2.jpg` contains inspected five-view intervals; matchingJSON retains exact framepaths. Contact/event and latency reports use `observation_approach_contact_diagnosis_v1.json` and `observation_approach_latency_v1.json`. No current job remains active for these artifacts.

### October 16 active full-data insertion continuation

Under `/home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1/`:

- `shared_full_causal_grasp_actor_rows_v1/`:67,039 verified causal actor rows,500source episodes/377physical reset groups; originals unchanged.
- `shared_full_actor_image_cache_v1/`: complete immutable native-preprocessing cache on NAS.
- `shared_full_insertion_envelope_audit_v1.json` and `_wide_geometry.json`: passed wider approach and independent frame-label gates; failed strict attempts retained separately.
- `matched_shared_insertion_BC_plan_v1.json`, `shared_insertion_fresh_full_v1/`, `shared_insertion_visual_warm_full_v1/`: matched full-target BC arms, fresh active/warm queued; summaries/checkpoints written as fitting completes.
- `launcher_forwarding_audit_v1.json`: correction to earlier host-intended versus actual Docker settings. SFP live10 used0.90, not0.95; active SC broad60 uses the newer repair refiner. Original results preserved.
- `corrected_intended_perception_live20_v2/`: controlled corrected-container revalidation; v1 startup failure retained.
- `shared150mm_fresh_development_v1/`: frozen120-case development plan, scenes and physical-reset overlap audit; no final scenes.
- `shared150mm_matched_autonomous_v1/`: automatic prerequisite wait, then matched full-depth BC/recovery evaluation with preserved metrics, commands, effective environments and selected20fpsvideos. Pending gates are not successful insertions.

All large artifacts remain outside Git. See the execution record for qualification thresholds and limitations.

- `SC_effective_refiner_failure_paired_replay_v1/`: bounded9-frame old/new native comparison on3failedSCsettings; both reject9/9. Recorded-image timing caveat retained.
- `native_SC_rail_extension_v1/`: non-destructive native bank extension from29existingSCepisodes;4,526newcrops, QA montage/label/split audit. No newcollection.
- `SC_native_rail_repair_plan_v1.json`, `native_pose_SC_rail_repair_v1*`: cooperative GPU4 targeted refiner fit and exact trainer snapshot; active evaluation checkpoints unchanged.
- `shared_runtime_integration_pilot_v1/`, `shared_runtime_integration_SC_repeat_v2/`: **intermediate-model interface checks**, not completed-fit performance. First SC list-serialization failure retained; repaired SC/SFP actor commands, causal terminals and20fpsvideo saved.
# October 16 official-setting transport recheck

Artifact root: `/home/nas/chmin/aic_archives/2026-10-16_official_transport_recheck_v1` (bulk data remain on NAS). `plan.json` freezes 120 launches, upstream YAML/checkpoint/source hashes and exact-versus-stress scope. `current_job.json`, `progress.json`, `progress.md`, `runner.log` track execution; `summary.json` is written only after the matrix completes. `runs/CASE/` retains raw MCAP, native camera frames, runtime trace, effective container settings, scores and `transport_metrics.json`. This is rule transport and perception only, no insertion. [Protocol](docs/experiments/2026-10-16-official-transport-recheck.md).


### October 18 queued recovery comparison

`/home/nas/chmin/aic_archives/2026-10-17_insertion_control_repair_v1/post_repair_backoff_comparison_v1/`: `scheduler_state.json` tracks waiting/running/completion; later `runs/` retains12matched no/local/hierarchical-backoff BC rollouts and20fpsvideos, exact models/manifests/commands and independent pre-reset scores. Results are pending; this directory is not a promoted checkpoint or a newRL run.

### October 18 complete-sample score → repair → rescore cycle

`/home/nas/chmin/aic_archives/2026-10-18_three_trial_repair_cycle_v1/`: running scheduler states, frozen original three-trial sample `baseline/`, matched five-scene `dwell_repair_development/`, and independent `repaired/` rescore. Original/updated fullsample scores will be in`score_comparison.json`; all scenes remain predeadline public/development. No learned actor inference in the explicit-controller candidates. Do not rerun schedulers into existing roots.

Bounded rejected SC perception alternatives under insertion-repairV1:`SC_edge_registration_diagnostic_v1/`, `SC_edge_registration_rotation_only_v2/`, `SC_frozen_landmark_ridge_v1/`. Each retains held-out metrics/predictions; none is an approved runtime estimator or an expert trajectory dataset.
### October 19 transport-local-recovery artifacts

`/home/nas/chmin/aic_archives/2026-10-19_transport_local_recovery_sc_settle_v4/` retains the selected five-card SC **transport-only** recovered-contact development trace, exact launch environment, original source snapshot, measured 8.4 mm retreat / 9.5 mm lateral event, post-hoc oracle contacts, inspected stills and five-camera 20 fps video. The prior `sc_contact_v1`, `sc_partial_lateral_v2`, and `sc_both_sides_v3` roots retain rejected/aborted variants. `/home/nas/chmin/aic_archives/2026-10-19_V56_transport_local_recovery_official_v1/` retains the exposed released three-round matched score (98.749703/300; 1/3 strict insertions), three trial videos, source delta, traces, commands, and manual still review. No recovery fired in that three-round run, and reserved final scenes remain sealed. [Experiment record](docs/experiments/2026-10-19-transport-local-recovery.md).
