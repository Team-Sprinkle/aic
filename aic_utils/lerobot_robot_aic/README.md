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

For diversified autonomous data collection, first generate a randomized
`aic_engine` trial config and launch bringup with it:

Inside the container, randomly generate the config file first.
```bash
cd ~/ws_aic/src/aic
python generate_random_trials_config.py \
  --output ./outputs/configs/random_trials_10.yaml \
  --num_trials 10 \
  --seed 2026
```

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

The generated `N` trials act as the episode budget for that run. The recorder
will stop early if `--max_episodes` is reached first.

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
