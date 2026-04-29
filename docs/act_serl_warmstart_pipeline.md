# ACT Warm-Start Smoke Pipeline

This is the first minimal end-to-end imitation-learning path for AIC:

```text
Gazebo CheatCode expert rollouts -> native LeRobot dataset -> ACT smoke training
```

It deliberately uses CheatCode-based trajectories only. It does not use VLM
planner trajectories, SERL offline RL, Isaac Lab RL, or Gazebo recovery rollouts.
The repo has a `Team-Sprinkle/mip` dependency and `RunMIP` runtime policy
integration, but no direct in-repo MIP ACT training wrapper for this flow; this
smoke path uses LeRobot training directly.

The runtime policy/control interfaces are left untouched. In particular,
`aic_model/aic_model/policy.py` continues to support both Cartesian
`MotionUpdate` and joint-space `JointMotionUpdate` commands through
`MoveRobotCallback`. Action mode is a recorder/dataset/training concern.

## Generate Dataset Artifacts Without Gazebo

```bash
cd ~/ws_aic/src/aic
python aic_utils/lerobot_robot_aic/scripts/generate_trajectory_dataset.py \
  --request-yaml aic_utils/lerobot_robot_aic/config/data_generation_templates/sfp_to_nic_minimal_10_cheatcode.yaml \
  --target-accepted-override 2 \
  --max-attempts-override 3 \
  --dry-run \
  --skip-recording
```

This writes request, engine config, per-trial YAMLs, and a generation summary.
It does not launch Gazebo or record data.

## Generate 10 CheatCode Trajectories

Run this only in an environment where the AIC simulation/runtime is available:

```bash
cd ~/ws_aic/src/aic
python aic_utils/lerobot_robot_aic/scripts/generate_trajectory_dataset.py \
  --request-yaml aic_utils/lerobot_robot_aic/config/data_generation_templates/sfp_to_nic_minimal_10_cheatcode.yaml \
  --target-accepted-override 10 \
  --max-attempts-override 15
```

Expected accepted dataset root:

```text
outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n10__act_smoke/accepted_dataset
```

## Inspect Dataset Schema

```bash
cd ~/ws_aic/src/aic
python aic_utils/lerobot_robot_aic/scripts/inspect_dataset_schema.py \
  outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n10__act_smoke/accepted_dataset
```

The inspector reads `meta/info.json`, prints feature keys, FPS, robot type, and
infers whether the action schema is Cartesian or joint-like from feature names.

## Train ACT Smoke Policy

```bash
cd ~/ws_aic/src/aic
pixi run python aic_utils/lerobot_robot_aic/scripts/train_act_policy.py \
  --dataset-root outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n10__act_smoke/accepted_dataset \
  --output-dir outputs/train/act_smoke \
  --job-name act_smoke \
  --steps 200 \
  --batch-size 4 \
  --device cuda
```

The wrapper builds a `lerobot-train` command with `--policy.type=act`,
`--dataset.video_backend=pyav`, local `--dataset.root`, and wandb disabled by
default. Use `--dry-run` to print the exact command first.

## Future Task Compatibility

- Add future SFP-to-NIC and SC-to-SC request YAMLs under the same request format.
- Set `generation.action_mode` to `cartesian` or `joint` according to the recorder
  action schema being collected.
- Keep task-family details in request YAMLs and dataset metadata rather than
  narrowing `policy.py`.
- Reuse the same schema inspector and training wrapper before training on larger
  mixed datasets.
