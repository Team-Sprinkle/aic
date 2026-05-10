# Gazebo Trial to Isaac Config

This note documents how AIC Gazebo trial YAML files are mapped into the Isaac Lab
`AIC-Task-v0` scene.

## Gazebo to Isaac Conversion

The conversion entry point is:

```bash
python aic/aic_utils/aic_isaac/aic_isaaclab/scripts/apply_gazebo_trial.py \
  --trials_yaml aic/outputs/configs/fixed_1_trials_sfp2nic.yaml \
  --trial_id trial_1 \
  --sim
```

`apply_gazebo_trial.py` reads the Gazebo/aic_engine YAML config and extracts:

- `trials.<trial_id>.scene.task_board.pose`
- `trials.<trial_id>.scene.task_board.sc_rail_*`
- `trials.<trial_id>.scene.task_board.nic_rail_*`
- `robot.home_joint_positions`
- `task_board_limits.sc_rail`

The task board pose is converted from Gazebo world coordinates into the current
Isaac AIC world frame with:

- a 90 degree Gazebo-to-Isaac XY frame yaw offset
- a Z offset of `-1.14`
- an additional board yaw offset

After the board pose is converted, movable board parts are computed as
task-board-local offsets and composed into world poses:

```text
T_world_part = T_world_task_board * T_task_board_part
```

Current Isaac-local anchors in `apply_gazebo_trial.py`:

```text
sc_port:
  local anchor = (0.0067, -0.0362, 0.005)

sc_port_2:
  local anchor = (0.0067, -0.083, 0.005)
  local rotation = sc_port rotation plus 90 deg about local/world Z

nic_card:
  local x = -0.03235 - nic_translation
  local y = ISAAC_NIC_RAIL_Y_BY_INDEX[nic_index]
  local z = 0.0743
```

SC port X is clamped to `task_board_limits.sc_rail.{min_translation,max_translation}`
from the YAML config. This prevents the anchor plus rail translation from pushing
the SC asset off the board.

Robot joint initialization comes from `robot.home_joint_positions`, matching the
values used by `aic_engine` to build its home joint motion and reset request.

## Runtime Isaac Config Overrides

`apply_gazebo_trial.py` builds a runtime `AICTaskEnvCfg` through
`build_gazebo_aligned_env_cfg(...)`.

That function starts from:

```text
aic_task.tasks.manager_based.aic_task.aic_task_env_cfg:AICTaskEnvCfg
```

Then it overwrites these runtime config values:

- `env_cfg.scene.robot.init_state.pos`
- `env_cfg.scene.robot.init_state.rot`
- `env_cfg.scene.robot.init_state.joint_pos`
- `env_cfg.scene.task_board.init_state.pos`
- `env_cfg.scene.task_board.init_state.rot`
- `env_cfg.scene.sc_port.init_state.pos`
- `env_cfg.scene.sc_port.init_state.rot`
- `env_cfg.scene.sc_port_2.init_state.pos`
- `env_cfg.scene.sc_port_2.init_state.rot`
- `env_cfg.scene.nic_card.init_state.pos`
- `env_cfg.scene.nic_card.init_state.rot`

In `--sim` mode the script also disables reset-time scene randomization:

```python
env_cfg.events.reset_robot_joints = None
env_cfg.events.randomize_board_and_parts = None
```

After `env.reset()`, it calls `apply_robot_home_joints(...)` to write the home
joint state directly into the robot articulation and actuator targets. This is
needed because reset events or default articulation state can otherwise leave
the robot at zero joints.

## Static Defaults in aic_task_env_cfg.py

The base Isaac task config lives at:

```text
aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py
```

That file contains static defaults for:

- the task board asset
- `sc_port`
- `sc_port_2`
- `nic_card`
- reset-time randomization offsets in `randomize_board_and_parts`
- light/heavy/no-randomization profile overrides
- default robot joint positions

These defaults are used whenever the task is launched normally, without
`apply_gazebo_trial.py` overriding the runtime config. They also matter during
reset randomization unless `randomize_board_and_parts` is disabled.

Current custom SC port values that must remain consistent between
`apply_gazebo_trial.py` and `aic_task_env_cfg.py`:

```text
sc_port offset:
  (0.0067, -0.0362, 0.005)

sc_port_2 offset:
  (0.0067, -0.083, 0.005)

sc_port_2 rotation:
  (0.999391, 0.0, 0.0, 0.034903)
```

If a custom trial looks correct when printed by `apply_gazebo_trial.py` but not
inside the Isaac scene, check both places:

1. `build_gazebo_aligned_env_cfg(...)` in `apply_gazebo_trial.py`
2. static `sc_port_2` init state and randomization offsets in `aic_task_env_cfg.py`

## TODOs

- SC ports should be arranged along the board-local Y axis. The current custom
  overwrite in `aic_task_env_cfg.py` still needs validation against the actual
  SC port rail geometry and USD asset frame. In particular, confirm that
  `sc_port_2` uses the intended Y-only separation and the required 90 degree
  Z-axis rotation in the rendered Isaac scene.
- Replace hand-maintained duplicate SC/NIC constants with one shared source of
  truth for `apply_gazebo_trial.py` and `aic_task_env_cfg.py`.
- Use IK and camera observations to detect or estimate the SFP port pose instead
  of relying only on the fixed `ISAAC_SFP_PORT_LOCAL_BY_NAME` offsets.
- Add an automated check that expands a Gazebo trial, applies the Isaac config,
  and verifies board-local relative poses for `sc_port`, `sc_port_2`, `nic_card`,
  and SFP ports.
