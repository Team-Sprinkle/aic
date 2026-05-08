# Gazebo -> Isaac Parity Notes

This document explains how parity is handled in:

- `aic/aic_utils/aic_isaac/aic_isaaclab/scripts/cheatcode_modified_eval.py`

## Docker workflow requirement

Isaac Lab runs inside Docker in this setup. If you edit Python files on the host,
you must sync changes into the running container before rerunning parity scripts.

- Sync host Python changes into container:
  - `aic/scripts/sync_python_to_container.sh`
- Install/update task package dependencies in the Isaac container environment:
  - `./isaaclab.sh -p -m pip install -e aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task`

## Why a single frame transform is not enough

A single world transform (`Rz + T`) is necessary, but not sufficient, to match Gazebo and Isaac scenes.

The two setups are built differently:

- Gazebo scene is composed through launch + xacro + SDF, with rail-relative part placement.
- Isaac scene is composed from pre-authored USD assets and IsaacLab `init_state` roots.
- Some parts are defined relative to board-local anchors in one stack, but as independent rigid objects in the other.
- Different import pipelines (URDF/SDF vs USD) can shift local pivots, authored root frames, and orientation conventions.

Result: even after correct global frame conversion, individual objects can still appear shifted or rotated.

## What the script does today

When `--enable_gazebo_parity` is enabled, the script:

1. Loads a Gazebo trial YAML (`--gazebo_trial_config`, `--gazebo_trial_id`).
2. Applies a global Gazebo->Isaac transform for board/robot bootstrap.
3. Reconstructs SC/NIC placement from rail-relative trial parameters.
4. Hides absent trial entities by moving them out of scene.
5. Applies robot home joints from YAML when provided.

## Why per-object offsets are required

Because object local frames are not guaranteed to match across the two simulators, each spawned object can require its own residual correction after global mapping.

`cheatcode_modified_eval.py` exposes per-object world offsets:

- `robot_offset_x`, `robot_offset_y`, `robot_offset_yaw`
- `task_board_offset_x`, `task_board_offset_y`, `task_board_offset_yaw`
- `sc_port_offset_x`, `sc_port_offset_y`, `sc_port_offset_yaw`
- `sc_port_2_offset_x`, `sc_port_2_offset_y`, `sc_port_2_offset_yaw`
- `nic_card_offset_x`, `nic_card_offset_y`, `nic_card_offset_yaw`

These are not a replacement for the global transform. They are residual calibration terms used after global mapping to account for simulator/asset-local differences.

## Practical calibration workflow

1. Start with explicit global transform values and explicit per-object offsets.
2. Set all per-object offsets to `0` first to observe pure global-map error.
3. Tune board first, then robot, then rails/parts (SC ports, NIC card).
4. Keep corrections minimal and record them as CLI args for reproducibility.

## Physics note

This parity path also applies optional simulation-step alignment (`--no_match_gazebo_physics` to disable), but physics-step matching does not fix frame/pivot mismatches by itself.

## Example command

```bash
isaaclab -p /workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/cheatcode_modified_eval.py \
  --enable_gazebo_parity \
  --gazebo_trial_config /workspace/isaaclab/aic/aic_engine/config/sample_config.yaml \
  --gazebo_trial_id trial_1 \
  --robot_offset_x 0 --robot_offset_y 0 --robot_offset_yaw 0 \
  --task_board_offset_x 0 --task_board_offset_y 0 --task_board_offset_yaw 0 \
  --sc_port_offset_x 0 --sc_port_offset_y 0 --sc_port_offset_yaw 0 \
  --sc_port_2_offset_x 0 --sc_port_2_offset_y 0 --sc_port_2_offset_yaw 0 \
  --nic_card_offset_x 0 --nic_card_offset_y 0 --nic_card_offset_yaw 0
```
