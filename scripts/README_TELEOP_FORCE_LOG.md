# Teleop Force Logging (Isaac Lab + Gazebo)

This guide shows how to:
- sync local `teleop.py` into the running Isaac Lab container,
- install `aic_task` in the container Python environment,
- run teleop with `spacemouse` or `keyboard`,
- export force log CSV to host,
- log Gazebo force (`/observations`) to CSV for parity checks.

## 1) Start / Enter Docker

From host (`/home/brucekimrok/projects/IsaacLab`):

```bash
./docker/container.py start base
./docker/container.py enter base
```

Inside container you should see:

```bash
root@...:/workspace/isaaclab#
```

## 2) Install `aic_task` in Container

Inside container:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p -m pip install -e aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task
```

## 3) Sync Local `teleop.py` Into Container

From host:

```bash
bash aic/scripts/sync_python_to_container.sh
```

By default this copies:
- host: `aic/aic_utils/aic_isaac/aic_isaaclab/scripts/teleop.py`
- container: `/workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/teleop.py`

## 4) Run Teleop and Record Force CSV

Inside container:

### SpaceMouse

```bash
cd /workspace/isaaclab
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/teleop.py \
  --task AIC-Task-v0 \
  --num_envs 1 \
  --teleop_device spacemouse \
  --enable_cameras \
  --force_log_csv /workspace/isaaclab/aic/outputs/force_parity/teleop_force.csv \
  --save_on_reset_button
```

Notes:
- Right button reset (`R`) works in this path.
- With `--save_on_reset_button`, right button also saves CSV immediately.

### Keyboard

```bash
cd /workspace/isaaclab
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/teleop.py \
  --task AIC-Task-v0 \
  --num_envs 1 \
  --teleop_device keyboard \
  --enable_cameras \
  --force_log_csv /workspace/isaaclab/aic/outputs/force_parity/teleop_force.csv \
  --save_log_key K
```

Notes:
- Keyboard input is captured by the Isaac Sim window, not terminal.
- Click/focus the simulator window before pressing keys.

## 5) Stop and Save

When teleop exits (Ctrl+C or app close), CSV is written automatically.

You can also trigger save during runtime via key/button callback (depending on device mode).

## 6) Copy CSV to Host

From host:

```bash
docker cp isaac-lab-base:/workspace/isaaclab/aic/outputs/force_parity/teleop_force.csv \
  /home/brucekimrok/projects/IsaacLab/aic/outputs/force_parity/teleop_force.csv
```

If your container has a suffix, replace `isaac-lab-base` with your container name.

## 7) Log Gazebo Force CSV

### Option A: One-command parity runner (recommended)

From repo root (`/home/brucekimrok/projects/ws_aic/src/aic`):

```bash
bash scripts/run_gazebo_force_parity.sh
```

Default output:
- `outputs/force_parity/gazebo_force.csv`
- logs in `outputs/force_parity/logs/`

This script will:
- launch eval-style Gazebo bringup via distrobox (`aic_eval` by default),
- start `aic_model` policy node,
- run `gazebo_force_logger.py` to record `/observations`.

You can override config via env vars or `scripts/force_parity_config.env`, for example:

```bash
DURATION_S=30.0 GAZEBO_GUI=false LAUNCH_RVIZ=false bash scripts/run_gazebo_force_parity.sh
```

### Option B: Manual logger only

If Gazebo + policy are already running and publishing `/observations`, run:

```bash
pixi run python aic_utils/aic_isaac/aic_isaaclab/scripts/gazebo_force_logger.py \
  --out outputs/force_parity/gazebo_force.csv \
  --duration-s 20.0
```

CSV columns:
- `time_s`
- `force_z_n`
- `ee_x_m`
- `ee_y_m`
- `ee_z_m`
