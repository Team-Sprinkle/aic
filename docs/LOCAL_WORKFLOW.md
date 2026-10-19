# Local workflow: final controller and rootless Docker

The final selected run uses `RunVisualConnectorServoQuietContextV56`, rule transport, requested-port RGB perception and explicit insertion. The old learned insertion actor remains in the codebase for comparisons, but it does not supply actions in the final demonstrations.

## Host setup

```bash
cd /data1/chmin/yj/ws_aic/src/aic
export PATH="$HOME/bin:$HOME/.pixi/bin:$PATH"
export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
docker info --format '{{json .SecurityOptions}}'
docker ps --format '{{.Names}} {{.Status}}'
nvidia-smi
```

`SecurityOptions` must include rootless. Use the provisioned user Docker daemon; routine operation needs no sudo. The workstation guide is `~/code/ws_aic/ROOTLESS_DOCKER_GUIDE.MD` when present. A session that cannot access the socket or `/dev/nvidia*` needs host access; changing socket permissions is not a repair.

Inspect active jobs before allocating GPUs. Project limit: **four concurrent GPUs**. Rootless launchers acquire cooperative GPU/domain locks and use separate Docker networks and `GZ_PARTITION` values. Do not interrupt another job. The machine's primary disk is nearly full, so write new bulk data to `/home/nas/chmin/aic_archives/`, after checking free space.

## Reproduce a complete released-config diagnostic

Choose a free GPU, an unused ROS domain and a **new** output directory:

```bash
.pixi/envs/default/bin/python scripts/run_sc_sfp_repaired_released_eval.py   --output /home/nas/chmin/aic_archives/NEW_RELEASED_REPLAY   --repeats 1 --sc-route sc_marker_scout --sfp-lift-m 0.06   --gpu 4 --domain 182
```

This runner downloads the released `eval_config.yaml`, verifies the copied scene YAML and frozen sources, records the actual container settings, and runs all three trials. It depends on the existing archived V56 bundle and local checkpoints; the repo's demonstration videos alone do not make training or runtime artifacts portable. Configuration/hash details are in [completion notes](answers/2026-10-19-final-project-completion-notes.md).

## Record the exact public sample

```bash
.pixi/envs/default/bin/python scripts/run_repaired_sample_video.py   --output /home/nas/chmin/aic_archives/NEW_SAMPLE_RECORDING   --gpu 4 --domain 183
```

This reuses the passing repaired snapshot, verifies the pinned public sample YAML, scores the three trials and creates five-camera 20-fps H.264 videos. The actual scene config inside an archive may be named `eval_config.yaml` even when its bytes are the public sample; the **hash**, not the filename, establishes provenance.

## Artifacts and review

Each rollout retains `runtime_trace.jsonl`, `frame_manifest.csv`, five-camera frames, the effective container environment, scene YAML, scored MCAP bags and `results/scoring.yaml`. Scoring and oracle contact recording run externally for diagnosis; runtime control uses public observations. A scoring award and an independently audited insertion-depth gate are different measurements.

The best released recording is `2026-10-19_sc_sfp_released_repair_twenty_frozen/repeat_10/` under the archive root. The stopped batch's `summary_stopped.json` includes all 19 scored runs and excludes the interrupted twentieth. The latest public sample is `2026-10-19_repaired_public_sample_video/`.

Recreate the curated repo media from the original best recording:

```bash
.pixi/envs/default/bin/python scripts/export_final_project_media.py   --source /home/nas/chmin/aic_archives/2026-10-19_sc_sfp_released_repair_twenty_frozen/repeat_10/rollout
```

Only the six selected MP4/GIF files and their provenance belong in the versioned [final media folder](media/final_evaluation/README.md). Dataset arrays, checkpoints, raw bags, arbitrary rollout videos and caches stay outside Git. The [old detailed workflow](answers/archive/2026-10-19-documentation/local_workflow_full_history.md) retains historical ACT/Isaac instructions.
