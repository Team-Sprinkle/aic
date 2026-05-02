# PR Validation Commands

Use this checklist before opening or merging a PR. Commands assume the repo root
is the current working directory.

## Basic Sanity

```bash
pixi install
```

```bash
pixi run python -m compileall aic_teacher_official aic_example_policies aic_utils aic_model
```

## Unit And Integration Tests

```bash
pixi run python -m pytest aic_teacher_official/test/test_official_teacher_pipeline.py -q
```

```bash
pixi run python -m pytest aic_model/test/test_policy_delta_pose.py -q
```

```bash
pixi run python -m pytest aic_utils/lerobot_robot_aic/test/test_generate_trajectory_dataset.py -q
```

```bash
pixi run python -m pytest aic_utils/gazebo_rl/test -q
```

For Hydra/config changes, also run:

```bash
pixi run python -m pytest aic_utils/lerobot_robot_aic/test/test_hydra_configs.py -q
```

## ACT Training Smoke

This runs real LeRobot ACT training for 2 steps and should produce a checkpoint.

```bash
pixi run python aic_utils/lerobot_robot_aic/scripts/hydra_train.py \
  --config-name experiment/hf_sfp2nic_card0_port0_act_20hz \
  run.output_dir=/data1/chmin/yj/ws_aic/src/aic/outputs/train/premerge_smoke/act_2step_premerge \
  run.name=act_2step_premerge \
  train.steps=2 \
  train.batch_size=2 \
  train.save_freq=2 \
  train.log_freq=1 \
  hardware.cuda_devices=[0] \
  hardware.num_devices=1 \
  hardware.distributed.enabled=false
```

Expected checkpoint:

```text
/data1/chmin/yj/ws_aic/src/aic/outputs/train/premerge_smoke/act_2step_premerge/checkpoints/000002/pretrained_model
```

## ACT-Backed Offline SERL Smoke

This runs ACT-backed hybrid SERL for 2 steps. It uses ACT as an actor/action
prior and keeps the critic initialized from scratch.

```bash
pixi run python aic_utils/lerobot_robot_aic/scripts/hydra_train.py \
  --config-name experiment/hybrid_nominal_sfp2nic \
  data=hf_sfp2nic_card0_port0_randomized \
  run.output_dir=/data1/chmin/yj/ws_aic/src/aic/outputs/train/premerge_smoke/serl_act_adapter_2step_premerge \
  run.name=serl_act_adapter_2step_premerge \
  train.act_checkpoint=/data1/chmin/yj/ws_aic/src/aic/outputs/train/premerge_smoke/act_2step_premerge/checkpoints/000002/pretrained_model \
  train.steps=2 \
  train.batch_size=1 \
  train.save_every=2 \
  train.action_horizon=4 \
  hardware.cuda_devices=[0] \
  hardware.num_devices=1 \
  hardware.distributed.enabled=false
```

Expected checkpoints:

```text
/data1/chmin/yj/ws_aic/src/aic/outputs/train/premerge_smoke/serl_act_adapter_2step_premerge/checkpoint_latest.pt
/data1/chmin/yj/ws_aic/src/aic/outputs/train/premerge_smoke/serl_act_adapter_2step_premerge/checkpoint_000002.pt
```

Confirm the run summary records:

- `actor_mode: act_adapter`
- `freeze_act: true`
- `critic_init: scratch`

## Dataset Generation Tests

For PR validation, run the generator tests rather than starting a long real
generation job:

```bash
pixi run python -m pytest aic_utils/lerobot_robot_aic/test/test_generate_trajectory_dataset.py -q
```

Real dataset generation uses the runtime container and can block the shared
`aic_eval` environment for a long time. Only run real generation when the PR
specifically changes runtime data collection.

## Runtime / Gazebo Smoke

Do not run host-native ROS/Gazebo commands on machines where the runtime is
provided by the AIC container. Pick the form that matches the server.

### Standard Distrobox Setup

Use this on machines following `docs/getting_started.md`, where `aic_eval` is a
distrobox container:

```bash
pixi run python aic_utils/gazebo_rl/scripts/gazebo_rl_train_short.py \
  --workspace-dir . \
  --engine-config <path-to-engine_config.yaml> \
  --sim-distrobox aic_eval \
  --max-iterations 2 \
  --max-steps 3 \
  --max-minutes 3 \
  --per-trial-timeout-sec 120 \
  --output-dir outputs/gazebo_rl_smoke
```

### Rootless Docker Setup On Knuth

Use this on the current Knuth rootless-Docker server. Start from the initialized
zsh environment so `DOCKER_HOST` and the AIC helpers are set:

```bash
LC_USER_ID=yoonjung zsh -l
cd /data1/chmin/yj/ws_aic/src/aic
```

Make sure no dataset-generation or eval job is already using `aic_eval`:

```bash
docker ps --format '{{.Names}} {{.Status}}'
docker exec aic_eval bash -lc 'ps -eo pid,ppid,etime,cmd | egrep "aic_model|aic_gz_bringup|aic_engine|aic-policy-recorder|rmw_zenoh|generate_trajectory" | grep -v egrep || true'
```

If the container is idle, run:

```bash
docker restart aic_eval
```

Then run the rootless Docker smoke:

```bash
pixi run python aic_utils/gazebo_rl/scripts/gazebo_rl_train_short.py \
  --workspace-dir . \
  --engine-config <path-to-engine_config.yaml> \
  --sim-docker-container aic_eval \
  --docker-host "$DOCKER_HOST" \
  --workspace-container /home/chmin/yj/ws_aic/src/aic \
  --host <host-ip-reachable-from-container> \
  --max-iterations 2 \
  --max-steps 3 \
  --max-minutes 3 \
  --per-trial-timeout-sec 120 \
  --output-dir outputs/gazebo_rl_smoke_premerge_docker
```

On Knuth, `147.47.206.241` was reachable from the container during validation:

```bash
pixi run python aic_utils/gazebo_rl/scripts/gazebo_rl_train_short.py \
  --workspace-dir . \
  --engine-config outputs/trajectory_datasets/sc_to_sc/cheatcode/sc_ports_1/n200__targeted_counts_20260502_081224/engine_config.yaml \
  --sim-docker-container aic_eval \
  --docker-host "$DOCKER_HOST" \
  --workspace-container /home/chmin/yj/ws_aic/src/aic \
  --host 147.47.206.241 \
  --max-iterations 2 \
  --max-steps 3 \
  --max-minutes 3 \
  --per-trial-timeout-sec 120 \
  --output-dir outputs/gazebo_rl_smoke_premerge_docker
```

If this fails with `Another world of the same name is running`, the shared
runtime container is already in use. Wait for the active generation/evaluation
job to finish or use a separate clean eval container.

## Before Opening The PR

Check the diff:

```bash
git status --short
git diff --stat
```

Review changed files:

```bash
git diff
```

Push the branch:

```bash
git push origin feat/hybrid-train
```

If GitHub CLI is available:

```bash
gh pr create \
  --base main \
  --head feat/hybrid-train \
  --title "<PR title>" \
  --body-file <pr_body.md>
```

If `gh` is not installed, open:

```text
https://github.com/Team-Sprinkle/aic/compare/main...feat/hybrid-train
```
