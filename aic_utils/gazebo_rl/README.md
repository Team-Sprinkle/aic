# Gazebo RL

`aic_utils/gazebo_rl` is the low-throughput, high-fidelity reinforcement learning path for AIC. It wraps the existing Gazebo, ROS, `aic_engine`, `aic_model.Policy`, `aic_controller`, and scoring stack instead of trying to turn Gazebo into a pure synchronous step simulator.

Isaac Lab remains the high-throughput training environment under `aic_utils/aic_isaac`. This package is intended for short real rollouts, sim-to-sim adaptation checks, and validation against the same stack used by challenge evaluation.

## Architecture

```text
trainer / GazeboRLEnv
      |
      | newline-delimited JSON over localhost TCP
      v
gazebo_rl.bridge_policy.GazeboRLBridgePolicy
      |
      | get_observation(), move_robot(), send_feedback()
      v
existing aic_model.Policy API
      |
      v
existing Gazebo + ROS + aic_engine + scoring stack
```

The policy loaded by `aic_model` is the boundary. `GazeboRLBridgePolicy` converts observations to plain dictionaries, receives clipped relative TCP delta actions, and sends real `move_robot()` commands in frame `gripper/tcp`.

## Smoke

Run one short random-action rollout using the local pixi launch path:

```bash
pixi run python aic_utils/gazebo_rl/scripts/gazebo_rl_smoke.py \
  --max-steps 5 \
  --ground-truth true \
  --gazebo-gui false \
  --launch-rviz false
```

Or run through a user-created distrobox evaluation container:

```bash
pixi run python aic_utils/gazebo_rl/scripts/gazebo_rl_smoke.py \
  --sim-distrobox <your_eval_container> \
  --max-steps 5 \
  --ground-truth true \
  --gazebo-gui false \
  --launch-rviz false
```

## Short Training

Run at most five iterations or five minutes:

```bash
pixi run python aic_utils/gazebo_rl/scripts/gazebo_rl_train_short.py \
  --sim-distrobox <your_eval_container> \
  --max-iterations 5 \
  --ax-minutes 5 \
  --max-steps 25 \
  --ground-truth true \
  --gazebo-gui false \
  --launch-rviz false
```

Outputs are written under `outputs/gazebo_rl/`, including `checkpoints/` and `run_summary.json`.

## Policy Loading

The bridge policy can be loaded by `aic_model` with:

```bash
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=gazebo_rl.bridge_policy.GazeboRLBridgePolicy
```

`--sim-distrobox` is optional. If omitted, the runner uses the local source launch path:
`pixi run ros2 launch aic_bringup aic_gz_bringup.launch.py ...`.

If provided, `--sim-distrobox` must be the name of a distrobox container that the user already created locally. `aic_eval` is not a toolkit resource; it may be the name a user chose while following setup instructions, but users should pass whatever container name they created.

With `--sim-distrobox <your_eval_container>`, the runner starts the evaluation container using:

```bash
distrobox enter -r --no-tty <your_eval_container> -- /entrypoint.sh ground_truth:=true start_aic_engine:=true
```

Use `--ground-truth false` for non-oracle state observations. The trainer environment starts the TCP server and passes connection settings through:

- `AIC_GAZEBO_RL_HOST`
- `AIC_GAZEBO_RL_PORT`
- `AIC_GAZEBO_RL_COMMAND_DT_SEC`
- `AIC_GAZEBO_RL_MAX_STEPS`
- `AIC_RESULTS_DIR`

## Limitations

- Low throughput: every step is a real ROS/Gazebo/controller tick.
- Sparse reward: v1 uses a small per-step penalty and terminal score parsed from `scoring.yaml`.
- State-only observations: images are intentionally not required for the initial proof.
- Intended for sim-to-sim adaptation and validation, not broad RL exploration.
- The runner defaults to the existing launch flow and supports a configurable distrobox wrapper, but local environments may need workspace-specific distrobox naming or `/entrypoint.sh` wiring.
