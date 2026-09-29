# Outputs Directory Guide

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

Reviewed: 2026-09-25. Start with the [documentation index](docs/README.md) and
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
