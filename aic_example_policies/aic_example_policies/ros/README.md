# HybridTeleop (ROS)

This guide shows how to run the HybridTeleop policy and switch control between:

- Wrapped policy (`WaveArm` in this example)
- Manual teleoperation (`keyboard_ee`)

HybridTeleop starts in `policy` mode by default.

## 1) Start simulation and engine

In terminal 1:

```bash
cd ~/ws_aic/src/aic
/entrypoint.sh ground_truth:=true start_aic_engine:=true
```

## 2) Start `aic_model` with HybridTeleop

In terminal 2:

```bash
cd ~/ws_aic/src/aic
pixi run ros2 run aic_model aic_model --ros-args -p policy:="aic_example_policies.ros.HybridTeleop" -p hybrid.wrapped_policy:="aic_example_policies.ros.WaveArm" -p hybrid.teleop_type:="keyboard_ee"
```

## Runtime controls

While HybridTeleop is running:

- `space`: toggle active source (`policy` <-> `teleop`)
- `enter`: exit HybridTeleop loop

When in `teleop` mode with `keyboard_ee`, use the keyboard teleop bindings to command the robot.

## Notes

- TODO: On source toggle, policy and teleop execution contexts continue running in parallel and only command output is gated. This can create internal state discontinuity when handing control back and forth, which may be problematic for policies that assume continuous closed-loop state progression.
