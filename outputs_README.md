# Final artifacts and bulk data

The curated project-completion media are versioned in [docs/media/final_evaluation](docs/media/final_evaluation/README.md): three original 20-fps MP4s, three motion-trimmed GIFs and a hash/provenance manifest. Other datasets, checkpoints, raw bags and arbitrary generated videos remain outside Git.

| Artifact | Location |
| --- | --- |
| Selected released-config recording, 276.3383/300 | `/home/nas/chmin/aic_archives/2026-10-19_sc_sfp_released_repair_twenty_frozen/repeat_10/` |
| Fixed repeat batch, 19 scored / 20 planned | Same parent, `summary_stopped.json`, `progress.json`, and `repeat_*/score_summary.json` |
| Frozen policy/model bundle and configuration | Same parent, `frozen/`, `plan.json`, `command_environment.json` |
| First repaired all-task replay | `/home/nas/chmin/aic_archives/2026-10-19_sc_sfp_released_repair_cycle8_sc_marker/` |
| Earlier three-repeat confirmation | `/home/nas/chmin/aic_archives/2026-10-19_sc_sfp_released_repair_cycle9_sc_marker_repeat3/` |
| Latest exact public sample, 287.8994/300 | `/home/nas/chmin/aic_archives/2026-10-19_repaired_public_sample_video/` |
| Verified demonstration sources | `outputs/trajectory_datasets/expert_verified/`; [dataset contracts](docs/DATASETS.md) |
| Successful SC bypass sources and cleaned derivative | `outputs/trajectory_datasets/sc_llb_score_verified/`, `sc_llb_cleaned_v3/` |

A rollout contains the source config, actual container environment, policy trace, synchronized frames, scored MCAP bags and scoring YAML. Keep pre-reset terminal records and failures. The repo media are selected successful demonstrations from the exposed published config; repeat-batch insertion was 53/57, not 100%.

[Reproduction and sudo-free operation](docs/LOCAL_WORKFLOW.md) · [Technical artifact map](docs/answers/2026-10-19-final-project-completion-notes.md) · [Previous full artifact history](docs/answers/archive/2026-10-19-documentation/outputs_artifact_map_full_history.md)
