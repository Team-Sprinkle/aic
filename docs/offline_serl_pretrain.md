# Offline SERL Pretraining

This adds a minimal offline SERL-style smoke path:

```text
Gazebo expert LeRobot dataset -> lowdim replay transitions -> actor-critic + BC pretraining checkpoint
```

It operates only on stored LeRobot/Gazebo expert data. It does not implement
Isaac online RL, Gazebo online RL, recovery intervention data collection, VLM
planner trajectories, or ROS policy execution.

## Inputs

Use an `accepted_dataset` created by the CheatCode trajectory generator, for
example:

```text
outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n10__act_smoke/accepted_dataset
```

The smoke implementation uses `observation.state` and vector `action` columns.
Images are left in the dataset but skipped by `--obs-mode lowdim`.

Set `--action-horizon` to train the actor and critics on flattened future action
chunks. The default is `1`, which is a single-step action. For example, a
Cartesian action with six values and `--action-horizon 8` becomes a 48-value
actor output while the runtime action schema metadata remains Cartesian.

If the dataset has no `reward` column, `--reward-mode dataset` falls back to a
final-success reward of `1.0` on each episode's last frame and `0.0` elsewhere.
You can also set `--reward-mode final_success` or `--reward-mode zero`
explicitly.

## Dry Run

```bash
cd ~/ws_aic/src/aic
pixi run python aic_utils/lerobot_robot_aic/scripts/train_offline_serl.py \
  --dataset-root outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n10__act_smoke/accepted_dataset \
  --output-dir outputs/train/offline_serl_smoke \
  --job-name offline_serl_smoke \
  --steps 10 \
  --batch-size 4 \
  --device cpu \
  --dry-run
```

## Smoke Training

```bash
cd ~/ws_aic/src/aic
pixi run python aic_utils/lerobot_robot_aic/scripts/train_offline_serl.py \
  --dataset-root outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n10__act_smoke/accepted_dataset \
  --output-dir outputs/train/offline_serl_smoke \
  --job-name offline_serl_smoke \
  --steps 200 \
  --batch-size 4 \
  --action-horizon 8 \
  --hidden-dim 256 \
  --num-layers 3 \
  --device cuda
```

Outputs:

```text
outputs/train/offline_serl_smoke/checkpoint_latest.pt
outputs/train/offline_serl_smoke/train_config.json
outputs/train/offline_serl_smoke/metrics.jsonl
```

The checkpoint contains the actor, twin critics, target critics, optimizer
state, dataset schema summary, training config, and normalization statistics.
It is a metadata-compatible handoff artifact for future bridge work, but Stage 5
can now consume it as a conservative PPO action-prior initialization; see
`isaac_rl_stage5.md`.

## ACT Warm Start

`--act-checkpoint` accepts a LeRobot ACT `pretrained_model` directory or
`model.safetensors` file:

```bash
pixi run python aic_utils/lerobot_robot_aic/scripts/train_offline_serl.py \
  --dataset-root outputs/trajectory_datasets/hybrid_nominal_sfp2nic_cheatcode_n10/accepted_dataset \
  --output-dir outputs/train/hybrid_offline_serl_nominal_n10_actwarm \
  --steps 200 \
  --batch-size 32 \
  --action-horizon 8 \
  --hidden-dim 256 \
  --num-layers 3 \
  --device cuda \
  --act-checkpoint outputs/train/hybrid_act_nominal_n10/checkpoints/000200/pretrained_model
```

Because ACT is a vision transformer and offline SERL is a lowdim MLP, the
implemented transfer is deliberately conservative: it validates the ACT
checkpoint metadata and initializes the SERL actor output bias from ACT
`model.action_head.bias`, repeated over the configured action horizon. The
checkpoint records this in `warmstart_metadata`.

## Current Limitations

- The actor is a lowdim Gaussian MLP. Use
  `--hidden-dim`, `--num-layers`, and `--action-horizon` to make the smoke model
  wider, deeper, or chunked without changing runtime policy interfaces.
- ACT warm-start transfers an output action prior only; it does not map ACT
  transformer hidden layers into the MLP.
- Rewards are replay-data rewards when present, otherwise final-success or zero
  fallback modes.
- The checkpoint can initialize Isaac PPO's action prior and can run through the
  Gazebo SERL transfer validator, but true online SERL/SAC is still not present.
- Runtime `policy.py` and command abstractions remain untouched.
