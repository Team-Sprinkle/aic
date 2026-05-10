# Gazebo Online SERL Status

Last updated: 2026-04-30 on branch `feat/hybrid-train`.

This note documents the current Gazebo RL policy loading and training path for
the hybrid ACT-adapter SERL actor. Human recovery data collection is intentionally
out of scope here.

## Current Status

The Gazebo RL environment can now load and train the same ACT-adapter SERL actor
used by offline vision SERL and Isaac online SERL.

Implemented:

- `GazeboRLEnv` runs the existing Gazebo, ROS, `aic_engine`, `aic_model.Policy`,
  controller, and scoring stack through localhost IPC.
- The bridge policy can send live RGB images from Gazebo observations through
  IPC when `include_images=true`.
- `ACTAdapterSERLGazeboPolicy` can load offline, Isaac-online, and Gazebo-online
  ACT-adapter SERL checkpoints.
- `gazebo_serl_train.py` can run short online SERL training directly in Gazebo,
  update the adapter actor and twin critics, and save a reloadable checkpoint.
- `serl_transfer_validate.py` can roll out lowdim SERL or ACT-adapter SERL
  checkpoints through the same Gazebo RL bridge.

Not implemented yet:

- Human recovery/demo intervention.
- Long scored Gazebo validation for a candidate checkpoint.
- Failure-specific buffer writing and offline refresh orchestration.
- A tuned Gazebo-online SERL policy. Current Gazebo run is a wiring/training
  proof, not a solved insertion policy.

## Actor Architecture

The primary hybrid actor is:

```text
obs -> ACT -> a_ACT
state + a_ACT -> adapter MLP -> delta
final action = a_ACT + adapter_scale * delta
```

Runtime behavior:

- ACT is loaded from a TorchScript export.
- The adapter is loaded from an ACT-adapter SERL checkpoint.
- The actor outputs a flattened action chunk with shape:
  `action_horizon * single_action_dim`.
- Gazebo executes only the first 6D Cartesian action from the chunk.
- `adapter_delta_clip` clamps the adapter correction before adding it to ACT.
- `action_clip` clamps the final ACT-plus-adapter action chunk.

Current canonical dimensions for the nominal SFP-to-NIC artifacts:

```text
state_dim: 32
single_action_dim: 6
action_horizon: 8
flattened_action_dim: 48
camera keys:
  observation.images.center_camera
  observation.images.left_camera
  observation.images.right_camera
```

## Gazebo RL Runtime Flow

```text
gazebo_serl_train.py / serl_transfer_validate.py
  -> GazeboRLEnv
  -> IPC server on localhost
  -> GazeboRLRunner starts evaluation stack
  -> aic_model loads gazebo_rl.bridge_policy.GazeboRLBridgePolicy
  -> bridge sends observations/images to trainer
  -> trainer returns 6D Cartesian delta action
  -> bridge calls existing move_robot() policy API
```

The bridge remains inside the official `aic_model.Policy` boundary. It does not
train. It only serializes observations, waits for actions, sends controller
commands, and reports done/error states.

Live images are serialized as resized JPEG payloads:

```text
source: aic_model_interfaces/Observation center_image/left_image/right_image
IPC encoding: jpeg_rgb8 base64
policy tensor shape: 1 x 3 x 256 x 288
```

`--allow-zero-images` exists only for interface tests. Real transfer/training
should use live images.

## Main Files

Gazebo RL:

- `aic_utils/gazebo_rl/gazebo_rl/gym_env.py`
- `aic_utils/gazebo_rl/gazebo_rl/runner.py`
- `aic_utils/gazebo_rl/gazebo_rl/observation.py`
- `aic_utils/gazebo_rl/gazebo_rl/serl_policy.py`
- `aic_utils/gazebo_rl/gazebo_rl/serl_train.py`
- `aic_utils/gazebo_rl/scripts/gazebo_serl_train.py`
- `aic_utils/gazebo_rl/scripts/serl_transfer_validate.py`

Supporting artifacts:

- ACT checkpoint:
  `outputs/train/hybrid_act_nominal_n10/checkpoints/000200/pretrained_model`
- ACT TorchScript:
  `outputs/train/hybrid_act_nominal_n10/act_policy_ts_cuda.pt`
- Offline ACT-adapter SERL checkpoint:
  `outputs/train/hybrid_vision_offline_serl_adapter_nominal_n10/checkpoint_latest.pt`
- Gazebo-online SERL checkpoint:
  `outputs/gazebo_rl/online_serl/adapter_2step_latest/checkpoint_latest.pt`

## Gazebo Online SERL Training

Dry-run command used to verify real checkpoint loading without starting Gazebo:

```bash
pixi run python aic_utils/gazebo_rl/scripts/gazebo_serl_train.py \
  --checkpoint outputs/train/hybrid_vision_offline_serl_adapter_nominal_n10/checkpoint_latest.pt \
  --act-torchscript outputs/train/hybrid_act_nominal_n10/act_policy_ts_cuda.pt \
  --output-dir outputs/gazebo_rl/online_serl/adapter_dry_run \
  --device cuda \
  --dry-run
```

Dry-run result:

```json
{
  "status": "dry_run",
  "state_dim": 32,
  "action_dim": 48,
  "single_action_dim": 6,
  "action_horizon": 8,
  "include_images": true,
  "allow_zero_images": false,
  "adapter_delta_clip": 0.05,
  "action_clip": 0.05
}
```

Actual short Gazebo online SERL command:

```bash
pixi run python aic_utils/gazebo_rl/scripts/gazebo_serl_train.py \
  --checkpoint outputs/train/hybrid_vision_offline_serl_adapter_nominal_n10/checkpoint_latest.pt \
  --act-torchscript outputs/train/hybrid_act_nominal_n10/act_policy_ts_cuda.pt \
  --output-dir outputs/gazebo_rl/online_serl/adapter_2step_latest \
  --workspace-dir . \
  --sim-distrobox aic_eval \
  --device cuda \
  --max-episodes 1 \
  --max-steps 2 \
  --updates 1 \
  --batch-size 1 \
  --max-minutes 10 \
  --per-trial-timeout-sec 300 \
  --ground-truth true \
  --gazebo-gui false \
  --launch-rviz false \
  --adapter-delta-clip 0.05 \
  --action-clip 0.05
```

Result:

```text
checkpoint: outputs/gazebo_rl/online_serl/adapter_2step_latest/checkpoint_latest.pt
metrics: outputs/gazebo_rl/online_serl/adapter_2step_latest/metrics.jsonl
train config: outputs/gazebo_rl/online_serl/adapter_2step_latest/train_config.json
run summary: outputs/gazebo_rl/online_serl/adapter_2step_latest/run_summary.json
```

Run summary:

```json
{
  "episodes_completed": 1,
  "steps_completed": 1,
  "updates_done": 1,
  "elapsed_sec": 44.723284710998996
}
```

Final metric row:

```json
{
  "reward": -0.01,
  "critic_loss": 0.0003967539523728192,
  "actor_loss": 0.06374824792146683,
  "raw_adapter_delta_norm": 1.6684852838516235,
  "adapter_delta_norm": 0.329450786113739,
  "final_action_norm": 0.3193478584289551,
  "final_minus_act_norm": 1.1606016159057617
}
```

The run requested `--max-steps 2`, but stopped after one real step because the
requested `--updates 1` target was reached.

Reload check for the Gazebo-trained checkpoint:

```bash
pixi run python aic_utils/gazebo_rl/scripts/gazebo_serl_train.py \
  --checkpoint outputs/gazebo_rl/online_serl/adapter_2step_latest/checkpoint_latest.pt \
  --act-torchscript outputs/train/hybrid_act_nominal_n10/act_policy_ts_cuda.pt \
  --output-dir outputs/gazebo_rl/online_serl/adapter_2step_reload_dry_run \
  --device cuda \
  --dry-run
```

Result: successful reload with the same dimensions and live-image requirement.

## Transfer Validation

Clipped ACT-adapter transfer validation command:

```bash
pixi run python aic_utils/gazebo_rl/scripts/serl_transfer_validate.py \
  --policy-kind act_adapter_serl \
  --checkpoint outputs/train/isaac_online_serl_adapter/2026-04-30_21-11-49_online_serl_adapter_1k_guarded/checkpoint_latest.pt \
  --act-torchscript outputs/train/hybrid_act_nominal_n10/act_policy_ts_cuda.pt \
  --workspace-dir . \
  --sim-distrobox aic_eval \
  --device cuda \
  --max-steps 3 \
  --per-trial-timeout-sec 300 \
  --ground-truth true \
  --gazebo-gui false \
  --launch-rviz false \
  --adapter-delta-clip 0.05 \
  --action-clip 0.05 \
  --output-dir outputs/gazebo_rl/serl_transfer_validation/act_adapter_clipped_3step_latest
```

Result:

```text
summary: outputs/gazebo_rl/serl_transfer_validation/act_adapter_clipped_3step_latest/transfer_validation_summary.json
real_steps: 3
include_images: true
allow_zero_images: false
total_reward: -0.02
action_norm_mean: 0.10949402778003287
action_norm_max: 0.10949986228346104
adapter_delta_norm_mean: 0.3464101552963257
raw_adapter_delta_norm_mean: 54.15252685546875
classification: no_score
```

The run intentionally stopped after 3 steps, so no terminal scoring file was
expected. The clamp bounded execution, but the raw adapter correction from that
Isaac checkpoint was still too large for a candidate policy.

## Tests Run

Focused Gazebo tests after adding online SERL training:

```bash
pixi run python -m pytest aic_utils/gazebo_rl/test/test_serl_train.py \
  aic_utils/gazebo_rl/test/test_serl_policy.py -q
```

Result:

```text
9 passed
```

Full relevant suites:

```bash
pixi run python -m pytest aic_utils/gazebo_rl/test -q
pixi run python -m pytest aic_utils/lerobot_robot_aic/test -q
pixi run python -m pytest aic_utils/aic_isaac/test -q
pixi run python -m pytest aic_model/test -q
git diff --check
```

Results:

```text
aic_utils/gazebo_rl/test: 31 passed
aic_utils/lerobot_robot_aic/test: 29 passed, 1 warning
aic_utils/aic_isaac/test: 4 passed
aic_model/test: 2 passed
git diff --check: passed
```

Note: running several pytest suites in parallel caused idle `pixi` processes in
this environment, so the suites were rerun serially.

## Implementation Details

`gazebo_rl.serl_train.ReplayBuffer`

- Stores real Gazebo transitions.
- Stores actor observations as:
  `{"state": tensor, "images": dict[str, tensor]}`.
- Samples batches for critic and adapter updates.

`GazeboOnlineSERLTrainer`

- Holds the loaded `ACTAdapterSERLGazeboPolicy`.
- Reuses its TorchScript ACT base and adapter actor.
- Loads twin vision critics from the source checkpoint.
- Maintains target critics.
- Updates critics with one-step TD targets.
- Updates actor with:
  `-Q(obs, actor(obs)) + adapter_penalty + ACT_preservation`.

Saved checkpoint keys:

```text
actor
critic1
critic2
target_critic1
target_critic2
actor_optimizer
critic_optimizer
online_gazebo_serl_config
step
```

`ACTAdapterSERLGazeboPolicy` now recognizes:

- direct offline vision SERL checkpoints with `vision_offline_serl_config`
- Isaac online SERL checkpoints with `online_serl_config`
- Gazebo online SERL checkpoints with `online_gazebo_serl_config`

## Practical Interpretation

This is now a real Gazebo online RL training path, but it is intentionally
low-throughput. It should be used for:

- short high-fidelity adaptation checks
- validating that the policy loads under the real Gazebo/ROS stack
- producing small Gazebo-online checkpoints
- transfer and scoring validation of candidate policies

It should not be used for broad exploration or long initial training. Isaac
remains the high-throughput online training environment; Gazebo should validate
and lightly adapt candidate policies against the evaluation stack.

## Next Recommended Work

Before involving human recovery, the next policy-quality step is to train a
candidate with intrinsically small raw adapter corrections, then run a longer
scored Gazebo validation. The current clamps prevent unsafe execution, but they
do not make an unstable raw adapter a good policy.

