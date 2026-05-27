# Offline SERL Pretraining

This documents two offline SERL-style smoke paths:

```text
Gazebo expert LeRobot dataset -> lowdim replay transitions -> actor-critic + BC pretraining checkpoint
Gazebo expert LeRobot dataset -> frozen ACT + trainable adapter + vision critics + BC/TD pretraining checkpoint
```

Both operate only on stored LeRobot/Gazebo expert data. They do not implement
Isaac online RL, Gazebo online RL, recovery intervention data collection, VLM
planner trajectories, or ROS policy execution.

## Inputs

Use an `accepted_dataset` created by the CheatCode trajectory generator, for
example:

```text
outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n10__act_smoke/accepted_dataset
```

The lowdim implementation uses `observation.state` and vector `action`
columns. Images are left in the dataset but skipped by `--obs-mode lowdim`.
The vision implementation uses `observation.state`, ACT camera keys, and vector
`action` chunks through the official LeRobot dataset API. Its default actor is
the ACT-adapter architecture: ACT produces a base action chunk and a small MLP
adapter predicts a regularized correction.

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
It is a metadata-compatible handoff artifact for future bridge work, but Isaac online RL
can now consume it as a conservative PPO action-prior initialization; see
`isaac_online_rl.md`.

## Lowdim ACT Warm Start

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

## Vision Offline SERL

`train_vision_offline_serl.py` keeps the lowdim path intact and adds a
vision-capable actor-critic path. By default it reconstructs the LeRobot
`ACTPolicy` from the ACT checkpoint, freezes ACT, and trains a small adapter:

```text
obs -> ACT -> a_ACT
observation.state + a_ACT -> adapter MLP -> delta
final action = a_ACT + adapter_scale * delta
```

The adapter final layer is zero-initialized, so the initial policy is exactly
ACT. The actor loss includes RL actor loss, BC loss on the final action, adapter
magnitude penalty, optional smoothness penalty, and ACT-preservation loss. Twin
vision critics encode the same camera observations plus low-dimensional state
and score flattened action chunks. `--actor-mode act_direct` remains available
for the older direct-ACT actor.

Default production settings for the current ACT -> offline SERL warm start are:

```text
actor_update_mode: q_bc
critic_arch: multiplicative
adapter_activation: gelu
critic_activation: gelu
state_encoding: fourier
state_encoding_indices: 0 1 2 13 14 15
state_encoding_num_bands: 4
state_encoding_max_freq: 8.0
state_encoding_scale: 10.0
adapter_lr: 1e-5
critic_lr: 1e-4
bc_weight: 5.0
reward_mode: dataset
```

The Fourier encoding is model-side. It appends sinusoidal bands for TCP xyz
and TCP-error xyz after the dataset/runtime feature vector has already been
materialized as 32D, 42D, 72D, or 82D. Do not add these Fourier features in the
dataset postprocessor. The 40D contact/recovery features and the 10D task
vector remain dataset/runtime features, so offline training and online Gazebo or
Isaac execution see the same physical state contract.

Dry run:

```bash
pixi run python aic_utils/lerobot_robot_aic/scripts/train_vision_offline_serl.py \
  --dataset-root outputs/trajectory_datasets/hybrid_nominal_sfp2nic_cheatcode_n10/accepted_dataset \
  --act-checkpoint outputs/train/hybrid_act_nominal_n10/checkpoints/000200/pretrained_model \
  --output-dir outputs/train/hybrid_vision_offline_serl_nominal_n10 \
  --job-name hybrid_vision_offline_serl_nominal_n10 \
  --steps 10 \
  --batch-size 2 \
  --device cuda \
  --action-horizon 8 \
  --actor-mode act_adapter \
  --freeze-act \
  --adapter-activation gelu \
  --critic-activation gelu \
  --state-encoding fourier \
  --state-encoding-indices 0 1 2 13 14 15 \
  --dry-run
```

Smoke training:

```bash
pixi run python aic_utils/lerobot_robot_aic/scripts/train_vision_offline_serl.py \
  --dataset-root outputs/trajectory_datasets/hybrid_nominal_sfp2nic_cheatcode_n10/accepted_dataset \
  --act-checkpoint outputs/train/hybrid_act_nominal_n10/checkpoints/000200/pretrained_model \
  --output-dir outputs/train/hybrid_vision_offline_serl_nominal_n10 \
  --job-name hybrid_vision_offline_serl_nominal_n10 \
  --steps 200 \
  --batch-size 2 \
  --device cuda \
  --action-horizon 8 \
  --actor-mode act_adapter \
  --freeze-act \
  --adapter-activation gelu \
  --critic-activation gelu \
  --state-encoding fourier \
  --state-encoding-indices 0 1 2 13 14 15 \
  --bc-weight 1.0 \
  --adapter-penalty-weight 1e-3 \
  --act-preservation-weight 1e-2 \
  --cql-weight 0.0
```

Guarded training experiments can additionally pass:

```bash
  --adapter-delta-clip 0.05 \
  --action-clip 0.05
```

`--adapter-delta-clip` clamps the learned correction before adding it to ACT;
`--action-clip` clamps the final action chunk. These are useful when testing
deployment stability, but they do not replace adapter regularization. The
metrics log both raw and clipped correction norms when clipping is active.
With `action_clip` enabled, the final action can differ from raw ACT even when
the adapter is zero-initialized because the final ACT-plus-adapter action is
clamped. Leave `--action-clip` unset when you specifically need to verify that
step-zero policy output is exactly ACT.

2026-04-30 result:

```text
dataset: outputs/trajectory_datasets/hybrid_nominal_sfp2nic_cheatcode_n10/accepted_dataset
ACT checkpoint: outputs/train/hybrid_act_nominal_n10/checkpoints/000200/pretrained_model
adapter vision SERL checkpoint: outputs/train/hybrid_vision_offline_serl_adapter_nominal_n10/checkpoint_latest.pt
state_dim: 32
single_action_dim: 6
action_horizon: 8
flattened_action_dim: 48
cameras: observation.images.center_camera, observation.images.left_camera, observation.images.right_camera
ACT trainable tensors loaded: 153
ACT state tensors loaded: 234
actor parameters initialized from ACT: 51,580,806 / 51,679,718
actor coverage: 99.80860576677296%
ACT trainable parameters: 0
adapter parameters: 98,864
adapter trainable parameters plus log_std: 98,912
initial delta norm: 0.0
final step delta norm: 1.6950193643569946
skipped tensors: none
```

The checkpoint contains the ACT-backed actor state, twin critics, target
critics, optimizers, train config, dataset summary, and warm-start report.

## Online SERL Rewards

Offline and online SERL should optimize the same task geometry, not Isaac's
random `ee_pose` command. The online defaults now use simulator-only ground
truth for reward calculation while keeping the deployable policy observation
unchanged.

Gazebo online SERL uses ground-truth TF from `observation.oracle`:

```text
progress_weight: 1.0
distance_weight: 0.5
close_weight: 0.3
orientation_weight: 0.05
terminal_score_weight: 1.0
```

The dense terms compare the plug frame to the target port frame. If ground-truth
poses are unavailable, Gazebo falls back to the legacy `-0.01` non-terminal /
`total_score / 100` terminal reward.

Isaac online SERL enables target-object rewards in
`aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`:

```text
target_reward_body: sfp_module_link
target_scene_name: sc_port or sc_port_2 from target_port_index
target_reward_distance_weight: 0.5
target_reward_close_weight: 0.3
target_reward_orientation_weight: 0.05
target_reward_reaching_weight: 1.0
target_reward_lateral_weight: 0.0
disable_command_pose_rewards: true
```

With exact target-body/target-port position and orientation alignment, the dense
distance/close/orientation part reaches `0.85` before reaching bonus,
smoothness, and safety penalties.

Guarded few-step run:

```text
outputs/train/hybrid_vision_offline_serl_adapter_clipped_fewstep/checkpoint_latest.pt
```

This used `steps=3`, `adapter_penalty_weight=0.1`,
`act_preservation_weight=1.0`, `adapter_delta_clip=0.05`, and
`action_clip=0.05`. Final step metrics included
`raw_adapter_delta_norm=0.03374718874692917`,
`adapter_delta_norm=0.03374718874692917`, and
`final_minus_act_norm=1.163395643234253`; the final-minus-ACT norm is dominated
by final action clipping, not adapter overwrite.

## Current Limitations

- The actor is a lowdim Gaussian MLP. Use
  `--hidden-dim`, `--num-layers`, and `--action-horizon` to make the smoke model
  wider, deeper, or chunked without changing runtime policy interfaces. This
  applies to `train_offline_serl.py`, not the vision path.
- Lowdim ACT warm-start transfers an output action prior only; it does not map
  ACT transformer hidden layers into the MLP.
- Vision ACT-adapter warm-start uses frozen ACT as the base actor and trains a
  small correction adapter by default. `--no-freeze-act` and `--act-lr` are
  available for partial ACT finetuning experiments.
- Rewards are replay-data rewards when present, otherwise final-success or zero
  fallback modes.
- The lowdim checkpoint can initialize Isaac PPO's action prior and can run
  through the Gazebo SERL transfer validator, but PPO is now a legacy
  smoke/baseline path. The primary online path is Isaac SERL/SAC with the same
  ACT-adapter actor.
- Direct vision SERL -> Isaac PPO/RSL-RL weight transfer is no longer the
  primary plan because the PPO actor architecture differs from the ACT-adapter
  SERL actor. The intended path is online Isaac SERL/SAC using the same
  ACT-adapter actor checkpoint.
- Runtime `policy.py` and command abstractions remain untouched.
