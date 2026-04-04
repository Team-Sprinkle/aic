# lerobot_robot_aic

This package contains a [LeRobot](https://huggingface.co/lerobot) interface for the AIC robot.

## Usage

This describe some of the things you can do with LeRobot, for more information, see the official [LeRobot docs](https://huggingface.co/docs/lerobot/en/index).

The LeRobot driver is installed in a [pixi](https://prefix.dev/tools/pixi) workspace. In general, you can prefix a command with `pixi run` or enter the environment with `pixi shell`.

### Teleoperating with LeRobot

```bash
cd ~/ws_aic/src/aic
pixi run lerobot-teleoperate \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=<teleop-type> --teleop.id=aic \
  --robot.teleop_target_mode=<mode> --robot.teleop_frame_id=<frame_id> \
  --display_data=true
```

Options for `--teleop.type` (and setting `--robot.teleop_target_mode` accordingly):

- `aic_keyboard_ee` for cartesian-space keyboard control (and set `--robot.teleop_target_mode=cartesian`)
- `aic_spacemouse` for cartesian-space SpaceMouse control (and set `--robot.teleop_target_mode=cartesian`)
- `aic_keyboard_joint` for joint-space control (and set `--robot.teleop_target_mode=joint`)

Options for `--robot.teleop_frame_id` when `--robot.teleop_target_mode` is `cartesian`:
- `base_link` to send cartesian targets with respect to the robot's base link.
- `gripper/tcp` to send cartesian targets with respect to the `tcp` frame attached to the robot's gripper.

As an example,
```bash
cd ~/ws_aic/src/aic
pixi run lerobot-teleoperate \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_keyboard_ee --teleop.id=aic \
  --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=base_link \
  --display_data=true
```

For space mouse
```bash
pixi run lerobot-teleoperate \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_spacemouse --teleop.id=aic \
  --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=base_link \
  --display_data=true
```


:warning: Note: In addition to setting `--teleop.type` you must set `--robot.teleop_target_mode` because the `AICRobotAICController` class needs to know which type of actions to send to the controller and it doesn't have access to `--teleop.type`.

#### Cartesian space control

For cartesian control, in addition to setting `--teleop.type` and `--robot.teleop_target_mode`, you can also set `teleop_frame_id` (the reference frame used for cartesian control) which sets the reference frame. Set this to either the gripper TCP (`"gripper/tcp"`, the default) or the robot base link (`"base_link"`).

##### Keyboard

> Note on using the Shift+&lt;key&gt; commands: To stop, let go of &lt;key&gt; *before* letting go of Shift. Otherwise, the robot will continue rotating even after you let go of both Shift and &lt;key&gt;.

| Key     | Cartesian      |
| ------- | ---------- |
| w       | -linear y  |
| s       | +linear y  |
| a       | -linear x  |
| d       | +linear x  |
| r       | -linear z  |
| f       | +linear z  |
| q       | -angular z |
| e       | +angular z |
| shift+w | +angular x |
| shift+s | -angular x |
| shift+a | -angular y |
| shift+d | +angular y |
| n       | End current trial and advance to next board setup (`/cancel_task`) |

Press 't' to toggle between slow and fast mode.

View and edit key mappings and speed settings in `AICKeyboardJointTeleop` and `AICKeyboardJointTeleopConfig` in `aic_teleop.py`.

##### SpaceMouse

:warning: Note: In our experience, SpaceMouse teleoperation was laggier than keyboard teleoperation.

We used a 3Dconnexion SpaceMouse with the [pyspacemouse](https://github.com/JakubAndrysek/PySpaceMouse?tab=readme-ov-file#dependencies) library. To enable USB permissions, you may need to add the following to your `/etc/udev/rules.d/99-spacemouse.rules`:
``` bash
# Apply to all hidraw nodes for 3Dconnexion devices
KERNEL=="hidraw*", ATTRS{idVendor}=="046d", MODE="0666", GROUP="plugdev"
# Apply to the USB device itself
SUBSYSTEM=="usb", ATTRS{idVendor}=="046d", MODE="0666", GROUP="plugdev"
```
and then run
``` bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```
View and edit axis mappings and speed settings in `AICSpaceMouseTeleop` and `AICSpaceMouseTeleopConfig` in `aic_teleop.py`.

#### Joint space control

| Key | Joint          |
| --- | -------------- |
| q   | -shoulder_pan  |
| a   | +shoulder_pan  |
| w   | -shoulder_lift |
| s   | +shoulder_lift |
| e   | -elbow         |
| d   | +elbow         |
| r   | -wrist_1       |
| f   | +wrist_1       |
| t   | -wrist_2       |
| g   | +wrist_2       |
| y   | -wrist_3       |
| h   | +wrist_3       |
| n   | End current trial and advance to next board setup (`/cancel_task`) |

Press 'u' to toggle between slow and fast mode.

View and edit key mappings and speed settings in `AICKeyboardEETeleop` and `AICKeyboardEETeleopConfig` in `aic_teleop.py`.

### Recording Training Data

```bash
cd ~/ws_aic/src/aic
pixi run lerobot-record \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=<teleop-type> --teleop.id=aic \
  --robot.teleop_target_mode=<mode> --robot.teleop_frame_id=<frame_id> \
  --dataset.repo_id=<hf-repo> \
  --dataset.single_task=<task-prompt> \
  --dataset.push_to_hub=false \
  --dataset.private=true \
  --play_sounds=false \
  --display_data=true
```

:warning: Note (same as with `lerobot-teleoperate` above): In addition to setting `--teleop.type` you must set `--robot.teleop_target_mode` because the `AICRobotAICController` class needs to know which type of actions to send to the controller and it doesn't have access to `--teleop.type`.

Upon starting the command, you may see `WARN   Watchdog Validator ThreadId(13) zenoh_shm::watchdog::periodic_task: Some("Watchdog Validator")` which is safe to ignore; just look for `INFO ... ls/utils.py:227 Recording episode 0`.

LeRobot recording keys:

| Key         | Command          |
| ----------- | ---------------- |
| Right Arrow | Next episode     |
| Left Arrow  | Cancel current episode and re-record |
| ESC         | Stop recording   |

<!-- TODO: lerobot-record doesn't load the hil processor to handle teleop events (lerobot bug?) -->

### Recording Teleop Across Engine Trials (Multi-setup YAML)

For teleoperation collection across many board setups defined in an
`aic_engine` config (for example `outputs/configs/random_trials_eval_like.yaml`),
use `aic-teleop-record` together with a passive policy:

1. Launch simulation + engine with your config:
```bash
cd ~/ws_aic/src/aic
/entrypoint.sh ground_truth:=true start_aic_engine:=true \
  aic_engine_config_file:=/home/jk/ws_aic/src/aic/outputs/configs/random_trials_eval_like.yaml \
  shutdown_on_aic_engine_exit:=true
```

2. Run `aic_model` with the `NoOp` policy (keeps task active and lets you drive via teleop):
```bash
cd ~/ws_aic/src/aic
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.NoOp
```

3. Run teleop recorder (LeRobot-compatible dataset output):
```bash
cd ~/ws_aic/src/aic
pixi run aic-teleop-record \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_keyboard_ee --teleop.id=aic \
  --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=base_link \
  --dataset.repo_id=${HF_USER}/aic_teleop_eval_like \
  --dataset.single_task="Insert cable into target port" \
  --dataset.root=./outputs/lerobot_datasets \
  --dataset.push_to_hub=true \
  --dataset.fps=30 \
  --save_failed_episodes=true \
  --max_episodes=0
```

During recording:
- Press `n` in the teleop terminal to end the current board setup.
- `n` calls `/cancel_task`, which advances `aic_engine` to the next trial setup.
- With `aic-teleop-record`, episode boundaries follow `/insert_cable/_action/status`, so each trial can map to one episode automatically.

Alternatively, launch simulation + NoOp model + teleop recorder in one tmux
session:

```bash
cd ~/ws_aic/src/aic
bash ./aic_utils/lerobot_robot_aic/scripts/launch_teleop_recording_tmux.sh
```

Common overrides:

```bash
cd ~/ws_aic/src/aic
bash ./aic_utils/lerobot_robot_aic/scripts/launch_teleop_recording_tmux.sh \
  --session-name aic_teleop \
  --engine-config ./outputs/configs/random_trials_eval_like.yaml \
  --dataset-repo-id ${HF_USER}/aic_teleop_eval_like \
  --push-to-hub true \
  --teleop-type aic_keyboard_ee \
  --teleop-target-mode cartesian \
  --teleop-frame-id base_link
```

### Recording Autonomous Policy Rollouts

You can record trajectories generated by `aic_model` policies such as
`CheatCode` or `DiversifiedCheatCode` with `aic-policy-recorder`.

`aic-policy-recorder` writes a **native LeRobot dataset** (same schema family as
`lerobot-record`) so that:
- it is directly usable with `lerobot-train`
- it can be appended/merged seamlessly with teleoperation datasets via `--dataset.resume`

Recorder behavior:
- subscribes to `/observations` for observation frames
- subscribes to `/aic_controller/pose_commands` and `/aic_controller/joint_commands` for actions
- auto-segments episodes from `/insert_cable/_action/status`
- saves succeeded episodes by default (failed episodes can be kept via flag)
- exposes `/aic_policy_recorder/get_episode_save_status` so `aic_engine` can wait
  for per-goal episode save finalization before advancing

For diversified autonomous data collection, first generate a randomized
`aic_engine` trial config and launch bringup with it:

Inside the container, randomly generate the config file first.
```bash
python ~/ws_aic/src/aic/aic_engine/scripts/generate_random_trials_config.py \
  --output ./outputs/configs/random_trials_eval_like.yaml \
  --num_trials 10 \
  --episodes_per_setup 1 \
  --profile qualification_eval_like \
  --sfp_to_nic_weight 2 \
  --sc_to_sc_weight 1 \
  --seed 42
```

Use `--episodes_per_setup` to collect multiple episodes for each randomized
board setup. For example, `--num_trials 10 --episodes_per_setup 3` generates
30 total trials (3 episodes per setup).

Inside the container, spin up the simulation using the engine config file generated in the previous step.
```bash
cd ~/ws_aic/src/aic
/entrypoint.sh ground_truth:=true start_aic_engine:=true aic_engine_config_file:=/home/jk/ws_aic/src/aic/outputs/configs/random_trials_10.yaml shutdown_on_aic_engine_exit:=true
```

Run recorder in one terminal:
```bash
cd ~/ws_aic/src/aic
pixi run aic-policy-recorder \
  --dataset.repo_id=${HF_USER}/aic_mixed_dataset \
  --dataset.single_task="Insert cable into target port" \
  --dataset.root=./outputs/lerobot_datasets \
  --dataset.fps=30 \
  --action_mode=cartesian \
  --max_episodes=10
```

Run your policy in another terminal:
```bash
cd ~/ws_aic/src/aic
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.CheatCode
```

Alternatively, launch all three processes (simulation, policy, recorder) in a single tmux session:

```bash
cd ~/ws_aic/src/aic
bash ./aic_utils/lerobot_robot_aic/scripts/launch_policy_recording_tmux.sh
```

Optional (run directly without `bash`):
```bash
cd ~/ws_aic/src/aic
chmod +x ./aic_utils/lerobot_robot_aic/scripts/launch_policy_recording_tmux.sh
./aic_utils/lerobot_robot_aic/scripts/launch_policy_recording_tmux.sh
```

Common overrides:
```bash
cd ~/ws_aic/src/aic
bash ./aic_utils/lerobot_robot_aic/scripts/launch_policy_recording_tmux.sh \
  --session-name aic_run \
  --engine-config ./outputs/configs/random_trials_10.yaml \
  --policy-class aic_example_policies.ros.CheatCode \
  --dataset-repo-id ${HF_USER}/aic_mixed_dataset \
  --max-episodes 10
```

The script opens tmux windows named `simulation`, `policy`, and `recorder`.
Use `--no-attach` if you want to start the session in the background.
By default, recorder `--max_episodes` is auto-set to the number of trials in
`--engine-config`; use `--max-episodes` only if you want to override that.

The generated total trial count (`num_trials * episodes_per_setup`) acts as the
episode budget for that run. The recorder will stop early if `--max_episodes`
is reached first.

**NOTE: We have noticed that recording multiple trials in seqeunce in a single eval container spawned sometimes results in unstable robot pose initialization at the start fo the trial. More debugging to be done here; as a bypass, follow the below instruction for spinning up eval container per trial.**

If you need each episode/trial to start from a fresh bringup (new `/entrypoint.sh`
process every time), use the per-trial launcher:

```bash
cd ~/ws_aic/src/aic
bash ./aic_utils/lerobot_robot_aic/scripts/launch_policy_recording_per_trial.sh \
  --engine-config ./outputs/configs/random_10_trials_sfp2nic.yaml \
  --policy-class aic_example_policies.ros.CheatCode \
  --dataset-repo-id ${HF_USER}/sfp2nic \
  --dataset-root ./outputs/lerobot_datasets \
  --gazebo-gui true \
  --launch-rviz true \
  --push-to-hub true \
  --require-recorder-save-log true \
  --sudo-keepalive true
```

What this script does:
- reads trial IDs from your multi-trial engine config
- generates a one-trial temporary config for each trial
- runs `/entrypoint.sh` + policy + recorder for exactly one episode (`--max_episodes=1`)
- waits for recorder exit (episode save/finalize), then tears down remaining processes and repeats for the next trial
- appends new episodes with `--dataset.resume` after the first trial, and also on the first trial if the target dataset root already exists

Useful flags:
- `--sudo-keepalive true` to prompt for sudo once and keep credentials warm during the run
- `--per-trial-timeout-sec` to kill stuck trials
- `--recorder-drain-sec` grace period for recorder finalization after simulation exits
- `--require-recorder-save-log true` strict mode to fail a trial unless recorder log contains `Episode saved`
- `--push-to-hub true` push to hub
- `--continue-on-failure false` to stop on first failed trial
- `--tmp-dir` to keep generated single-trial YAMLs and per-trial logs
- `--gazebo-gui` whether to launch gazebo gui 
- `--launch-rviz` whether to launch rviz in each trial. 
- `--results-root` where the script writes the score summary and bags (default `./outputs/aic_results_per_trial`)

### Post-process CheatCode phases (alignment vs descent)

If your rollout rows include either timestamps or per-episode step indices, you can
add a `phase` label after recording without changing controller/backend code:

```bash
cd ~/ws_aic/src/aic
python3 aic_utils/lerobot_robot_aic/scripts/label_cheatcode_phases.py \
  --input /path/to/episodes.csv
```

You can also pass a LeRobot dataset root (for example `sample_data`), and the
script will process all `data/**/*.parquet` files:

```bash
python3 aic_utils/lerobot_robot_aic/scripts/label_cheatcode_phases.py \
  --input sample_data
```

Defaults match `CheatCode`:
- `alignment`: first `5.0` seconds
- `descent`: all later timesteps
- For `30 FPS` data without timestamps, this is a `150`-frame alignment window.

Useful overrides:
- `--timestamp-scale 1e-9` if timestamps are in nanoseconds
- `--fps 30` to force frame-rate-based labeling when timestamp is unavailable
- If `--fps` is omitted and `--input` points to a LeRobot dataset root, FPS is
  read from `meta/info.json` (for `sample_data`, this is `30`)
- `--episode-column <name>`, `--timestamp-column <name>`, `--step-column <name>`
  to force specific column names
- `--alignment-duration-sec` or `--sample-period-sec` if you intentionally changed
  policy timing

### Split phased dataset into separate LeRobot datasets

After adding the `phase` column, you can split one dataset into one output
dataset per phase (for example `alignment` and `descent`):

```bash
cd ~/ws_aic/src/aic
pixi run python3 aic_utils/lerobot_robot_aic/scripts/split_lerobot_by_phase.py \
  --input outputs/sample_datasets
```

This creates:
- `outputs/sample_datasets_alignment`
- `outputs/sample_datasets_descent`

Include camera videos in the split outputs:

```bash
cd ~/ws_aic/src/aic
pixi run python3 aic_utils/lerobot_robot_aic/scripts/split_lerobot_by_phase.py \
  --input outputs/sample_datasets \
  --include-videos
```

Useful flags:
- `--overwrite` to replace existing output directories
- `--phases alignment` (or `descent`) to export only selected phases
- `--suffix-template split_{phase}` to customize output dataset names
- `--phase-column <name>` if your phase label column is not `phase`

### Validating Dataset Compatibility

Before merging datasets, validate that teleop and policy datasets are schema-compatible:

```bash
cd ~/ws_aic/src/aic
pixi run aic-validate-dataset-compat \
  --base.repo_id=${HF_USER}/your_teleop_dataset \
  --candidate.repo_id=${HF_USER}/your_policy_dataset \
  --base.root=./outputs/lerobot_datasets \
  --candidate.root=./outputs/lerobot_datasets
```

The command checks:
- `robot_type`
- `fps`
- full feature-key set equality
- per-feature schema equality (`dtype`, `shape`, `names`) with dynamic video `info` ignored
- action feature key compatibility (for `action` / `action.*`)

Exit codes:
- `0`: compatible
- `1`: incompatible

Optional flags:
- `--allow_reordered_names`: ignore ordering differences in vector feature names
- `--ignore_robot_type`: skip robot type check
- `--json`: print machine-readable JSON summary


### Push local dataset to Hugging Face Hub

```bash
cd ~/ws_aic/src/aic
pixi run python aic_utils/lerobot_robot_aic/scripts/push_local_lerobot_ds_to_hub.py --local-dataset-dir <local-directory> --repo-id <huggingface-repo-id>
```

### Training

Once you have your LeRobot dataset, you can follow the [LeRobot tutorials](https://huggingface.co/docs/lerobot/en/index) for training.

```bash
cd ~/ws_aic/src/aic
pixi run lerobot-train \
  --dataset.repo_id=${HF_USER}/your_dataset \
  --policy.type=your_policy_type \
  --output_dir=outputs/train/act_your_dataset \
  --job_name=act_your_dataset \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/act_policy
```
