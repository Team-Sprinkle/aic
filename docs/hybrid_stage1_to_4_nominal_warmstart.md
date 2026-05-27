# Hybrid Stage 1-4 Nominal Warm Start

This is the current bounded cleanup path:

```text
Gazebo CheatCode/no-contact nominal data -> canonical schema inspection -> ACT smoke -> offline SERL smoke
```

It does not use VLM planner trajectories, Gazebo recovery, long Isaac PPO
training, or true online SERL/SAC.

## Canonical Schema Inspection

Inspect an accepted LeRobot dataset with the legacy schema view:

```bash
pixi run python aic_utils/lerobot_robot_aic/scripts/inspect_dataset_schema.py \
  outputs/trajectory_datasets/hybrid_nominal_sfp2nic_cheatcode_n10/accepted_dataset \
  --json
```

Inspect the canonical Gazebo/ACT/SERL/Isaac metadata view:

```bash
pixi run python aic_utils/lerobot_robot_aic/scripts/inspect_hybrid_schema.py \
  --dataset-root outputs/trajectory_datasets/hybrid_nominal_sfp2nic_cheatcode_n10/accepted_dataset \
  --action-horizon 8 \
  --json
```

The canonical summary reports task family, simulator source, action mode,
single-step action dimension, action horizon, observation mode, low-dimensional
observation dimension, camera keys, and low-dimensional keys. It does not
assume Cartesian-only data or SFP-to-NIC-only tasks.

## Generate Nominal Gazebo Dataset

Dry-run request/config generation without recording:

```bash
pixi run python aic_utils/lerobot_robot_aic/scripts/generate_trajectory_dataset.py \
  --request-yaml aic_utils/lerobot_robot_aic/config/data_generation_templates/sfp_to_nic_hybrid_nominal_10.yaml \
  --dry-run \
  --skip-recording
```

Real 10-accepted-trajectory CheatCode/no-contact generation:

```bash
pixi run python aic_utils/lerobot_robot_aic/scripts/generate_trajectory_dataset.py \
  --request-yaml aic_utils/lerobot_robot_aic/config/data_generation_templates/sfp_to_nic_hybrid_nominal_10.yaml \
  --target-accepted-override 10 \
  --max-attempts-override 15
```

Expected accepted dataset root:

```text
outputs/trajectory_datasets/hybrid_nominal_sfp2nic_cheatcode_n10/accepted_dataset
```

2026-04-30 result: complete. The generator ran 15 attempts with per-trial eval
container restarts enabled. It produced 14 raw saved episodes, 13 score-passing
episodes, and a capped accepted dataset with exactly 10 episodes and 5399
frames. One attempt failed and one saved episode scored below `90.0`; both were
excluded from `accepted_dataset`.

Relevant generation fixes:

- The YAML sets `restart_sim_container: true` and `per_trial_timeout_sec: 900`.
- The recorder now records the actual per-camera image shape instead of
  assuming all cameras match the left camera.
- The filter supports `--max-selected-episodes`; the generator passes the
  target accepted count so the accepted dataset is exactly `n10`.
- The generator allows a nonzero recording return when enough raw data exists,
  so filtering still runs after isolated failed attempts.

Final canonical schema summary:

```json
{
  "task_family": "sfp_to_nic",
  "simulator_source": "gazebo",
  "obs_mode": "image_lowdim",
  "obs_dim": 32,
  "action_mode": "cartesian",
  "action_dim": 6,
  "action_horizon": 8,
  "camera_keys": [
    "observation.images.center_camera",
    "observation.images.left_camera",
    "observation.images.right_camera"
  ],
  "lowdim_keys": ["observation.state"]
}
```

## Train ACT Smoke Policy

Run only after the accepted dataset exists:

```bash
pixi run python aic_utils/lerobot_robot_aic/scripts/train_act_policy.py \
  --dataset-root outputs/trajectory_datasets/hybrid_nominal_sfp2nic_cheatcode_n10/accepted_dataset \
  --output-dir outputs/train/hybrid_act_nominal_n10 \
  --job-name hybrid_act_nominal_n10 \
  --steps 200 \
  --batch-size 4 \
  --chunk-size 16 \
  --n-action-steps 8 \
  --n-obs-steps 1 \
  --device cuda
```

2026-04-30 result: complete, 200 steps on CUDA.

Final checkpoint:

```text
outputs/train/hybrid_act_nominal_n10/checkpoints/000200/pretrained_model/model.safetensors
```

Config confirmed: `steps=200`, `batch_size=4`, `chunk_size=16`,
`n_action_steps=8`, `n_obs_steps=1`.

## Run Offline SERL Smoke

Run only after the same accepted dataset exists:

```bash
pixi run python aic_utils/lerobot_robot_aic/scripts/train_offline_serl.py \
  --dataset-root outputs/trajectory_datasets/hybrid_nominal_sfp2nic_cheatcode_n10/accepted_dataset \
  --output-dir outputs/train/hybrid_offline_serl_nominal_n10 \
  --job-name hybrid_offline_serl_nominal_n10 \
  --steps 200 \
  --batch-size 32 \
  --action-horizon 8 \
  --hidden-dim 256 \
  --num-layers 3 \
  --device cuda \
  --save-every 200
```

Expected checkpoint:

```text
outputs/train/hybrid_offline_serl_nominal_n10/checkpoint_latest.pt
```

2026-04-30 result: complete, 200 steps on CUDA. Metadata:

```json
{
  "obs_dim": 32,
  "action_dim": 48,
  "action_horizon": 8,
  "action_mode": "cartesian",
  "normalization_stats": ["action_mean", "action_std", "obs_mean", "obs_std"]
}
```

Inspect compatibility metadata after a successful run:

```bash
pixi run python aic_utils/aic_isaac/scripts/check_policy_checkpoint_compatibility.py \
  --checkpoint outputs/train/hybrid_offline_serl_nominal_n10/checkpoint_latest.pt \
  --json
```

This does not mean the checkpoint loads into Isaac PPO. Direct ACT/SERL to
RSL-RL hidden-layer checkpoint loading is still not implemented for the current
camera PPO architecture, but a conservative action-prior bridge now exists.

## ACT -> SERL Warm Start

Offline SERL can consume an ACT checkpoint:

```bash
pixi run python aic_utils/lerobot_robot_aic/scripts/train_offline_serl.py \
  --dataset-root outputs/trajectory_datasets/hybrid_nominal_sfp2nic_cheatcode_n10/accepted_dataset \
  --output-dir outputs/train/hybrid_offline_serl_nominal_n10_actwarm \
  --job-name hybrid_offline_serl_nominal_n10_actwarm \
  --steps 200 \
  --batch-size 32 \
  --action-horizon 8 \
  --hidden-dim 256 \
  --num-layers 3 \
  --device cuda \
  --act-checkpoint outputs/train/hybrid_act_nominal_n10/checkpoints/000200/pretrained_model
```

The implemented bridge validates the LeRobot ACT checkpoint and transfers
`model.action_head.bias` into `actor.mean_head.bias`, repeated over the SERL
action horizon. Full ACT transformer-to-MLP hidden-layer transfer is not
claimed.

## SERL -> Isaac PPO Warm Start

The Isaac online RL wrapper now accepts an offline SERL checkpoint:

```bash
pixi run python aic_utils/aic_isaac/scripts/train_isaac_rsl_rl.py \
  --task AIC-Task-v0 \
  --num-envs 4 \
  --max-iterations 1 \
  --seed 1 \
  --headless \
  --randomization-profile light \
  --output-dir outputs/train/isaac_online_serl_warmstart \
  --init-policy-checkpoint outputs/train/hybrid_offline_serl_nominal_n10_actwarm/checkpoint_latest.pt
```

For the current lowdim SERL -> camera PPO architecture mismatch, this applies a
partial action-prior initialization to the PPO actor output bias/std and writes
`params/offline_serl_warmstart.json` in the Isaac run directory.

## Gazebo SERL Transfer Validation

Validate an offline SERL checkpoint through the existing Gazebo RL bridge:

```bash
pixi run python aic_utils/gazebo_rl/scripts/serl_transfer_validate.py \
  --checkpoint outputs/train/hybrid_offline_serl_nominal_n10_actwarm/checkpoint_latest.pt \
  --workspace-dir . \
  --sim-distrobox aic_eval \
  --engine-config outputs/trajectory_datasets/hybrid_nominal_sfp2nic_cheatcode_n10/trials/trial_000001.yaml \
  --max-steps 600 \
  --per-trial-timeout-sec 900 \
  --output-dir outputs/gazebo_rl/serl_transfer_validation/hybrid_nominal_n10
```

This runs the SERL actor as the external policy behind
`gazebo_rl.bridge_policy.GazeboRLBridgePolicy`, parses Gazebo scoring output,
and writes `transfer_validation_summary.json`. Recovery rollout collection can
use the existing CheatCode trajectory generator on failed transfer trials, but
true same-state oracle takeover/recovery suffix recording remains future work.
