# Outputs Directory Guide

`outputs/` contains generated configs, recorded datasets, training artifacts, and
debug/evaluation runs. Most folders are reproducible artifacts from scripts in
`scripts/` or `aic_utils/lerobot_robot_aic/scripts/`.

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
| `train/isaac_stage5_helper_smoke/` | Isaac Stage 5 helper smoke-training output. |
| `train/isaac_stage5_smoke/` | Isaac Stage 5 smoke-training output. |
| `train/stage5_aic_lowdim_ppo/` | Low-dimensional PPO training output for the Stage 5 AIC task. |

These training folders currently contain run directories under `aic_task/`.
