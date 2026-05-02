# Codex Notes for Knuth Rootless AIC Development

This checkout is used on a server where the host OS is not the official AIC
Ubuntu 24.04 environment and Docker runs rootless.

## Workspace Paths

- Host checkout: `/data1/chmin/yj/ws_aic/src/aic`
- Same checkout inside the AIC container: `/home/chmin/yj/ws_aic/src/aic`

Do not assume the upstream docs' default `~/ws_aic/src/aic` path.

## Shell Environment

Start interactive work from:

```bash
LC_USER_ID=yoonjung zsh -l
cd /data1/chmin/yj/ws_aic/src/aic
```

The yoonjung zsh config sets the rootless Docker socket and defines the AIC
helpers below.

## Rootless Docker and AIC Helpers

Use these helpers instead of running host-native ROS/Gazebo commands:

```bash
aic_shell      # enter the persistent aic_eval container
aic_eval       # run the evaluation stack in the aic_eval container
aic_eval_rviz  # run eval with RViz enabled
aic_policy     # run the default example policy in the aic_eval container
aic_status     # inspect ROS nodes, topics, lifecycle, recent logs/results
```

The default container name is `aic_eval`, created from
`ghcr.io/intrinsic-dev/aic/aic_eval:latest`.

Raw `docker` commands should use the rootless socket:

```bash
docker --host unix:///run/user/$(id -u)/docker.sock ps
```

In an initialized yoonjung zsh shell, `DOCKER_HOST` is already set and `docker`
is wrapped to use that socket.

## Testing Workflow

The AIC runtime normally needs two processes:

1. Start evaluation:
   ```bash
   aic_eval
   ```
2. In another shell or tmux pane, run a policy:
   ```bash
   aic_policy
   ```

Run these helpers as zsh functions, not through `timeout aic_eval` or a plain
bash shell. For example:

```bash
LC_USER_ID=yoonjung zsh -lc 'cd /data1/chmin/yj/ws_aic/src/aic && aic_status'
```

If a runtime smoke fails with `/aic_model` in `finalized` state or Gazebo says
another `aic_world` is running, restart the persistent container before retrying:

```bash
docker --host unix:///run/user/$(id -u)/docker.sock restart aic_eval
```

On this host `aic_policy` may spend several seconds updating/building the Pixi
environment the first time it starts. If the engine checks model readiness before
the policy reaches `unconfigured`, the score will be `0` with `Model validation
failed`; rerun after the Pixi build has completed, or prewarm `aic_policy` in a
clean container before starting the scoring run.

The most reliable runtime-eval smoke sequence on this host is to start the
simulator without the engine, start the policy under an isolated lifecycle node,
then start `aic_engine` directly and point it at that node:

```bash
docker --host unix:///run/user/$(id -u)/docker.sock restart aic_eval

docker --host unix:///run/user/$(id -u)/docker.sock exec -i aic_eval bash -lc '
  source /ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  cd /home/chmin/yj/ws_aic/src/aic
  /entrypoint.sh ground_truth:=false start_aic_engine:=false gazebo_gui:=false launch_rviz:=false
'

docker --host unix:///run/user/$(id -u)/docker.sock exec -i aic_eval bash -lc '
  source /ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  cd /home/chmin/yj/ws_aic/src/aic
  pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.WaveArm \
    -r __node:=aic_model_smoke
'

docker --host unix:///run/user/$(id -u)/docker.sock exec -i aic_eval bash -lc '
  source /ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  cd /home/chmin/yj/ws_aic/src/aic
  export AIC_RESULTS_DIR=/home/chmin/yj/ws_aic/src/aic/outputs/runtime_eval_smoke/manual_engine_<timestamp>
  mkdir -p "$AIC_RESULTS_DIR"
  ros2 run aic_engine aic_engine --ros-args \
    -p use_sim_time:=true \
    -p config_file_path:=/home/chmin/yj/ws_aic/src/aic/aic_engine/config/sample_config.yaml \
    -p model_node_name:=aic_model_smoke \
    -p model_discovery_timeout_seconds:=60 \
    -p model_configure_timeout_seconds:=90
'
```

This sequence was verified on 2026-05-01. It reached trial execution and wrote
`scoring.yaml`; `WaveArm` passed model validation and then failed task execution,
which is expected because it is not an insertion policy.

For `aic_example_policies.ros.CheatCode`, use `ground_truth:=true`; otherwise
the policy waits for task-board target transforms and cannot proceed:

```bash
/entrypoint.sh ground_truth:=true start_aic_engine:=false gazebo_gui:=false launch_rviz:=false
```

On this rootless/reused container, do not rely on the default `/insert_cable`
action endpoint for isolated smoke runs; another endpoint can remain visible in
the graph. Remap both the policy action server and the installed engine's
hard-coded action client with ROS remapping. Do not modify or rebuild
`aic_engine` just to change the action name:

```bash
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.CheatCode \
  -r __node:=aic_model_smoke \
  -r /insert_cable:=/insert_cable_smoke

ros2 run aic_engine aic_engine --ros-args \
  -r /insert_cable:=/insert_cable_smoke \
  -p use_sim_time:=true \
  -p config_file_path:=/home/chmin/yj/ws_aic/src/aic/aic_engine/config/sample_config.yaml \
  -p model_node_name:=aic_model_smoke \
  -p model_discovery_timeout_seconds:=60 \
  -p model_configure_timeout_seconds:=90
```

This was verified on 2026-05-02 with normal engine lifecycle management and all
three sample-config CheatCode trials passing in the rootless `aic_eval`
container. Result directory:

```text
outputs/runtime_eval_smoke/cheatcode_action_rosremap_lifecycle_20260502_003747
```

## ACT Checkpoint Runtime Eval During Training

For ACT checkpoint validation during a live multi-GPU training run, use the
checkpoint watcher instead of manually racing the engine against training. The
watcher starts the rootless runtime container, waits for a checkpoint directory,
loads the checkpoint with `aic_example_policies.ros.RunACT`, runs the official
`aic_engine`, and writes `runtime_eval/<step>/eval_summary.json`, logs, bag
files, and `scoring.yaml` under the training run directory.

Use a one-trial engine config for fast smoke evaluation:

```text
/home/chmin/yj/ws_aic/src/aic/outputs/runtime_eval_configs/sample_config_trial1.yaml
```

Start the watcher before launching training:

```bash
RUN_DIR=outputs/train/hf_smoke/act_4gpu_eval_during_training_$(date -u +%Y%m%d_%H%M%S)
printf '%s\n' "$RUN_DIR" > /tmp/aic_4gpu_eval_run_dir.txt

pixi run python scripts/evaluate_act_checkpoints_runtime.py \
  --run-dir "$RUN_DIR" \
  --workspace-host /data1/chmin/yj/ws_aic/src/aic \
  --workspace-container /home/chmin/yj/ws_aic/src/aic \
  --engine-config /home/chmin/yj/ws_aic/src/aic/outputs/runtime_eval_configs/sample_config_trial1.yaml \
  --poll-seconds 1 \
  --max-checkpoints 1 \
  --command-mode none \
  --start-delay-sec 1 \
  --max-runtime-sec 1 \
  --control-hz 4 \
  --sim-wait-sec 25 \
  --engine-timeout-sec 180
```

Then launch ACT training against the same run directory. Do not pre-create the
run directory; LeRobot refuses to train into an existing output directory.

```bash
RUN_DIR=$(cat /tmp/aic_4gpu_eval_run_dir.txt)
pixi run python aic_utils/lerobot_robot_aic/scripts/hydra_train.py \
  --config-name experiment/hf_sfp2nic_card0_port0_act_20hz \
  hardware.cuda_devices=[0,1,2,3] \
  hardware.num_devices=4 \
  hardware.distributed.nproc_per_node=4 \
  train.steps=120 \
  train.save_freq=10 \
  train.log_freq=10 \
  train.batch_size=4 \
  train.num_workers=4 \
  run.output_dir="$RUN_DIR"
```

Verified on 2026-05-02: the watcher picked up checkpoint `000010` while 4-GPU
training was still running, loaded the ACT checkpoint in `aic_eval`, ran one
engine trial, and wrote score `1` / engine return code `0` under:

```text
outputs/train/hf_smoke/act_4gpu_eval_during_training_20260502_051547/runtime_eval/000010
```

`--command-mode none` is a validation smoke: it verifies checkpoint loading,
task-vector inference path, ROS lifecycle/action wiring, runtime container
startup, bagging, engine execution, and scoring. It does not command robot
motion or measure insertion quality. Commanding motion with this early/untrained
ACT checkpoint has previously crashed `ros_gz_container`, so use `none` for
reliable checkpoint health checks during training and run motion eval only when
debugging the controller/runtime path.

For package/dependency setup, prefer reproducible Pixi changes from the host
checkout:

```bash
pixi install
pixi reinstall <ros-kilted-package-name>
```

Avoid assuming that plain host commands such as `ros2 ...`,
`source /ws_aic/install/setup.bash`, or `apt install ...` are valid outside the
container.

## Real Data Generation and Training Smoke Workflow

Use this when Codex needs a real run, not only unit tests or dry-run output.
Run host commands from:

```bash
LC_USER_ID=yoonjung zsh -l
cd /data1/chmin/yj/ws_aic/src/aic
```

Generate request/trial artifacts first from the host checkout. For quick smoke
runs, use a request with `target_accepted_trajectories: 1`, `max_attempts: 1`,
`acceptance.success_only: false`, and `acceptance.min_score: 0`, then run the
generator with `--dry-run`. This creates `request.yaml`, `engine_config.yaml`,
`trials/*.yaml`, `generation_summary.json`, and initial manifests without
starting Gazebo:

```bash
pixi run python aic_utils/lerobot_robot_aic/scripts/generate_trajectory_dataset.py \
  --request-yaml <request.yaml> \
  --dry-run
```

For the real run, restart the persistent container to clear stale ROS nodes from
earlier attempts:

```bash
docker restart aic_eval
```

Start the simulator/evaluation stack in one long-running shell. Keep this
process open until the run is complete:

```bash
docker exec -i aic_eval bash -lc '
  source /ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  cd /home/chmin/yj/ws_aic/src/aic
  /entrypoint.sh ground_truth:=true start_aic_engine:=false gazebo_gui:=false launch_rviz:=false
'
```

On this shared host there may already be another `/aic_model` or `/insert_cable`
action server running. For smoke runs, isolate both the lifecycle node and the
action endpoint. Start the policy with a remapped node and action name:

```bash
docker exec -d aic_eval bash -lc '
  source /ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  cd /home/chmin/yj/ws_aic/src/aic
  pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.CheatCode \
    -r __node:=aic_model_smoke \
    -r /insert_cable:=/insert_cable_smoke
'
```

Start the LeRobot recorder before the engine, and point it at the remapped
action status topic. Let it exit on its own after `--max_episodes`; do not
interrupt it while it is finalizing parquet/video files:

```bash
docker exec -i aic_eval bash -lc '
  source /ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  cd /home/chmin/yj/ws_aic/src/aic
  pixi run aic-policy-recorder \
    --dataset.repo_id=local/aic_real_smoke \
    --dataset.single_task="Insert cable into target port" \
    --dataset.root=/home/chmin/yj/ws_aic/src/aic/<output_dir>/raw_dataset \
    --dataset.fps=20 \
    --action_mode=cartesian \
    --status_topic=/insert_cable_smoke/_action/status \
    --max_episodes=1
'
```

Start the engine directly in another shell. Set `AIC_RESULTS_DIR` so scoring
lands under the generation output, and pass both the isolated lifecycle node and
the isolated action endpoint:

```bash
docker exec -i aic_eval bash -lc '
  source /ws_aic/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_zenoh_cpp
  cd /home/chmin/yj/ws_aic/src/aic
  export AIC_RESULTS_DIR=/home/chmin/yj/ws_aic/src/aic/<output_dir>/scores/manual_engine
  mkdir -p "$AIC_RESULTS_DIR"
  ros2 run aic_engine aic_engine --ros-args \
    -p use_sim_time:=true \
    -p config_file_path:=/home/chmin/yj/ws_aic/src/aic/<output_dir>/engine_config.yaml \
    -p model_node_name:=aic_model_smoke \
    -p insert_cable_action_name:=/insert_cable_smoke \
    -p model_discovery_timeout_seconds:=60 \
    -p model_configure_timeout_seconds:=90
'
```

The `insert_cable_action_name` parameter is a local source change. Before relying
on the isolated action endpoint in the container, rebuild/install `aic_engine`
inside the runtime image and confirm the parameter is accepted. If the build
fails on generated interface includes, capture that as an environment/build
blocker instead of assuming the installed container has the new parameter.

After the recorder exits cleanly, create or verify `<output_dir>/scores/score_summary.csv`,
run `filter_merge_lerobot_by_score.py`, and regenerate manifests so
`manifests/accepted.csv` reflects the real selection report. Then run a short
task-conditioned ACT training smoke:

```bash
pixi run python aic_utils/lerobot_robot_aic/scripts/train_act_policy.py \
  --dataset-root <output_dir>/accepted_dataset \
  --task-metadata <output_dir>/manifests/accepted.csv \
  --task-conditioning append-state \
  --output-dir <output_dir>/train_smoke \
  --job-name real_smoke_task_conditioned \
  --steps 1 \
  --batch-size 1 \
  --device cpu \
  --num-workers 0 \
  --chunk-size 16 \
  --n-action-steps 8 \
  --n-obs-steps 1
```

The task metadata lives in `<output_dir>/manifests/`. The native LeRobot
`raw_dataset/` and `accepted_dataset/` remain schema-compatible; task vectors
are joined during training by creating a derived task-conditioned dataset next
to the training output.
