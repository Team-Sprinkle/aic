# Outputs Directory Guide

Reviewed: 2026-09-23. Start with the [documentation index](docs/README.md) and
[experiment ledger](docs/EXPERIMENTS.md) for the meaning and status of saved runs.

## Current storage map

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
