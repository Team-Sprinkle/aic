# Working Tree Changes Since HEAD - 2026-05-13

This summarizes the local changes in the working tree relative to the latest
commit on `feat/hybrid-train`.

## Scope

Changed files:

- `aic_example_policies/aic_example_policies/ros/RunACT.py`
- `aic_example_policies/aic_example_policies/ros/RunACTAdapterSERL.py`
- `aic_example_policies/aic_example_policies/ros/RunACTTorchScript.py`
- `aic_utils/gazebo_rl/gazebo_rl/serl_policy.py`
- `scripts/evaluate_act_checkpoints_runtime.py`
- `aic_utils/aic_isaac/scripts/isaac_episode_configs.py`
- `aic_utils/aic_isaac/scripts/train_isaac_online_serl.py`
- `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
- `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/probe_target_reward.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/events.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py`

At the time this was written, `git diff --stat HEAD` showed:

```text
11 files changed, 2381 insertions(+), 137 deletions(-)
```

## ACT/SERL Chunk Execution

The runtime policy paths were updated so ACT and ACT-adapter SERL execute
predicted chunks consistently as `chunk_size=8`, `n_action_steps=4`.

### Gazebo ACT Runtime

`RunACT.py`

- Reads `AIC_ACT_N_ACTION_STEPS`, defaulting to `4`.
- Validates `1 <= n_action_steps <= chunk_size`.
- Overrides `ACTConfig.n_action_steps` before policy construction.

`RunACTTorchScript.py`

- Reads `AIC_ACT_N_ACTION_STEPS`, defaulting to `4`.
- Reads TorchScript metadata `chunk_size`.
- Predicts a full ACT chunk, queues the first `n_action_steps` actions, and pops
  one action per control tick.
- Clears the queued actions at the start of each `insert_cable` request.
- Validates TorchScript output shape before using it.

### Gazebo ACT-Adapter SERL Runtime

`RunACTAdapterSERL.py`

- Reads `AIC_SERL_N_ACTION_STEPS`, falling back to `AIC_ACT_N_ACTION_STEPS`,
  defaulting to `4`.
- Uses the new chunk API in the Gazebo SERL policy and queues the first
  `n_action_steps` actions.
- Clears the queue at each new task.

`aic_utils/gazebo_rl/gazebo_rl/serl_policy.py`

- Added `ACTAdapterSERLGazeboPolicy.act_chunk(obs, n_action_steps=...)`.
- `act()` remains available for compatibility and now returns the first action
  from `act_chunk(..., n_action_steps=1)`.

### Isaac Online SERL Runtime

`aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`

- `--act_only_action_horizon` default changed to `0`, meaning "read
  `chunk_size` from TorchScript metadata".
- Added `--n_action_steps`, default `4`.
- Validates `n_action_steps <= action_horizon`.
- Isaac rollout now predicts a full action chunk and queues only the first
  `n_action_steps` before recomputing.
- Replay stores the executed 6D TCP action, not the full chunk.
- Critic input is fixed to the executed 6D action representation
  (`first_executed_6d`).

## Isaac Policy Frequency

Isaac policy/control rate was aligned to the 20 Hz Gazebo expert dataset rate.

`aic_task_env_cfg.py`

- Default physics timestep remains `1/120`.
- Added `AIC_ISAAC_POLICY_HZ`, default `20.0`.
- Computes `decimation` from `sim.dt` and policy Hz.
- Sets render interval to match decimation.

`train.py` and `train_isaac_online_serl.py`

- Added `--policy_hz` / `--policy-hz`.
- Forward policy Hz through environment variables and training metadata.

Validated in Isaac audit:

```text
sim_dt = 0.008333333333333333
decimation = 6
policy_dt = 0.05
policy_hz = 20.0
physical_chunk_duration_isaac_s = 0.4
physical_chunk_duration_expert_s = 0.4
```

## Near-Gate Episode YAML and Reset Fixes

The near-gate curriculum previously behaved like it was moving the scene toward
the robot. It now keeps the scene fixed and moves the robot/cable reset body.

`isaac_episode_configs.py`

- Added quaternion helpers for wxyz orientation math.
- Added NIC card and SFP port orientation/entrance geometry.
- For `sfp_to_nic`, target pose now includes:
  - target seated pose
  - entrance pose
  - target orientation
  - insertion axis
  - body-start orientation for the plug tip
- Near-gate metadata now records:
  - `reset_body_name`, defaulting to `sfp_tip_link`
  - `body_start_position_world`
  - `body_start_orientation_wxyz`
  - entrance-relative axial/lateral distances
- Validates axial and lateral distances are non-negative.
- Axial distance now means the tip starts outside/above the port entrance plane,
  not already below/past it.

`events.py`

- `reset_robot_tcp_to_episode_start` now supports `body_start_position_world`
  and `reset_body_name`.
- Supports 6D damped IK reset when an orientation is provided.
- Preserves compatibility with older `tcp_start_position_world` YAMLs.
- Records reset reports per env, including:
  - reset body name
  - initial/final position error
  - initial/final orientation error
  - whether orientation was used
  - episode id
- Records reset event order so audits can verify
  `randomize_board_and_parts` runs before `reset_robot_tcp_to_episode_start`.

Validated in Isaac audit:

```text
reset body: sfp_tip_link
reset note: 6D damped IK reset
final position error: about 0.0003 m in env0, about 0.000055 m in env1
final orientation error: about 9.27e-05 rad in env0, about 2.52e-05 rad in env1
```

## Reward Target Orientation and Source Parity

`rewards.py`

- Added `_episode_target_orientation_w`.
- Orientation rewards now prefer episode YAML target orientation when present.
- If the episode YAML does not provide target orientation, the code falls back
  to target asset root orientation plus configured offset.

`train.py`

- Diagnostics print whether target position and target orientation come from the
  same source.
- Audits print target pose, target asset root pose, and distances/orientation
  errors from `wrist_3_link`, `gripper_tcp`, and `sfp_tip_link`.

Validated in Isaac audit:

```text
target position source: episode_yaml
target orientation source: episode_yaml
sfp_to_nic target scene: nic_card
target reward body: sfp_tip_link
```

## Force/Wrench Reward and Diagnostics

The Isaac force proxy was changed from semantic/fixed TCP frames to a physical
body that reports usable wrench data.

`aic_task_env_cfg.py`

- Added optional `ContactSensorCfg` on `wrist_3_link`.
- Contact sensor is disabled by default and enabled only if
  `AIC_ISAAC_ENABLE_CONTACT_SENSOR=1`.
- Changed `force_delta_penalty` default body from `gripper_tcp` to
  `wrist_3_link`.

`rewards.py`

- `force_delta_penalty` now checks:
  - `body_incoming_wrench_w`
  - `body_incoming_wrench_b`
  - `body_incoming_joint_wrench_b`
  - optional `contact_forces` sensor fallback
- Resets previous force on episode reset.

`probe_target_reward.py`

- Uses `wrist_3_link` for force probing.
- Reads `body_incoming_joint_wrench_b` if other wrench tensors are absent.

`train.py`

- Force diagnostics print wrench tensor source, selected body, body names, per
  body norms, contact sensor summaries, and warnings when force is zero.

Validated in Isaac audit:

```text
force source: body_incoming_joint_wrench_b
selected force body: wrist_3_link
force penalty became nonzero during the audit, so the signal is not completely dead
```

## Isaac Online SERL Diagnostics

`train.py` received extensive lightweight diagnostics that are enabled with
`--debug_diagnostics` or audit mode.

Added or expanded diagnostics for:

- ACT/SERL action scale and actual Isaac IK action scale.
- Realized TCP delta vs requested TCP delta.
- Rotation action realization.
- Reward body/target resolution per reward term.
- Reward weights before and after env creation.
- Target position/orientation source mismatch warnings.
- Distances from `wrist_3_link`, `gripper_tcp`, and `sfp_tip_link` to target.
- Reset/control/reward body mismatch.
- Near-gate reset orientation error.
- Quaternion convention warnings for wxyz vs xyzw identity mistakes.
- Force/wrench/contact sensor values by body.
- Lateral reward warning when world-axis lateral reward is enabled.
- Terminal/truncation handling and TD bootstrap `done`.
- Previous action features and action-manager action state.
- Frequency diagnostics: sim dt, decimation, policy dt, expert fps, chunk duration.
- ACT chunk inference behavior for Gazebo and Isaac runtime paths.
- Checkpoint compatibility:
  - state dim
  - action dim
  - action horizon
  - camera key order
  - normalizer dim
  - state encoding config
  - task vector layout
- ACT freeze correctness.
- Gripper/cable/tip poses.
- Randomization profile.
- Camera freshness and image value statistics.
- Stage C process isolation:
  - CUDA visible devices
  - torch device
  - seed
  - run name
  - output dir
  - checkpoint paths

## 82D State Schema Diagnostics

`train.py`

- Added exact state schema names and index ranges:
  - base 32D
  - contact/recovery 40D
  - task vector 10D
- Diagnostics print, for env0:
  - raw value
  - normalizer mean
  - normalizer std
  - normalized value
  - feature name
  - feature index
- Prints top absolute normalized dimensions.
- Warns when any normalized state dimension exceeds `abs(value) > 10`.
- Prints all contact/recovery features and task-vector values.
- Prints env-origin audit for `num_envs > 1` to check world-position leakage.

Validated in Isaac audit:

```text
No abs(normalized)>10 dimensions were observed.
Largest normalized value was wrist_force.z around -8.23.
num_envs=2 did not show env-origin leakage into ACT TCP position state.
```

Known remaining concern:

```text
Isaac tcp_error dims 13:19 are currently zero-filled.
Gazebo expert datasets contain controller tcp_error fields.
The audit warns about this explicitly.
```

## Replay, Critic, BC, and Loss Diagnostics

`train.py`

- Replay buffer diagnostics now print latest/middle transition action, reward,
  done, metadata.
- Added replay age distribution diagnostics with oldest/middle/latest sample,
  reward, age, done, episode id, and target distance.
- Added before/after episode metadata for each transition to catch auto-reset
  next_obs contamination.
- `done_for_bootstrap` uses `terminated` only by default, not time-limit
  truncation.
- Added loss-scale summary:
  - actor Q loss
  - adapter penalty contribution
  - ACT preservation contribution
  - expert BC contribution
  - critic loss
  - batch reward stats
- Expert BC nearest-neighbor diagnostics print:
  - selected query dimensions
  - nearest expert dimensions
  - normalized NN distance
  - nearest expert action
  - dataset schema
- Expert chunk loading now records terminal repeated action slot fraction.

Validated in Isaac audit:

```text
done_for_bootstrap_mean = 0.0
episode_changed_after_step = false for the short audit
replay stored executed 6D actions
critic action representation = first_executed_6d
```

## Camera/Image Diagnostics

`train.py`

- Added camera freshness checks comparing consecutive raw frames.
- Added image dtype/range/mean/std before and after ACT normalization.
- Added `--force_camera_render_before_read` for camera freshness debugging.
- Added `--disable_ppo_resnet_observation_terms`, default true, while still
  reading raw camera tensors for SERL.

Validated in Isaac audit:

```text
camera freshness likely_stale = false for center, left, and right cameras
raw image tensors were float32 in 0..about 0.94
normalized image tensors were finite
```

## Wrapper and CLI Updates

`train_isaac_online_serl.py`

- Forwards the new online SERL options:
  - `--n-action-steps`
  - `--policy-hz`
  - diagnostics/audit flags
  - action-frame flags
  - contact sensor flag
  - camera render/freshness flags
  - truncation handling flag
- Plan JSON now records the new settings.
- Validation allows audit runs with warmup greater than audit step count.

## Validation Performed

Static and unit checks:

```bash
python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py
.pixi/envs/default/bin/python -m pytest aic_utils/aic_isaac/test/test_isaac_online_serl.py -q
```

Result:

```text
5 passed
```

Real Isaac audit:

```text
outputs/debug/isaac_serl_audit/2026-05-13_21-56-43_issues36_56_schema_audit
```

Key audit results:

- Isaac action scale was overridden from env default `0.05` to `1.0`.
- Runtime action manager reported `DifferentialInverseKinematicsAction` scale `1.0`.
- `sfp_to_nic` reward target was `nic_card`, reward body was `sfp_tip_link`.
- Target position and orientation both came from episode YAML.
- Chunk execution was `8,4`:
  - step 1 recomputed a chunk and queued 3 remaining actions
  - steps 2-4 drained the queue
- Policy frequency matched expert data at 20 Hz.
- Cameras were not stale.
- Replay stored executed 6D actions for the critic.
- `num_envs=2` did not show env-origin leakage in ACT TCP state.
- No state normalization values exceeded `abs(value)>10`.
- Force signal used `body_incoming_joint_wrench_b` on `wrist_3_link`.

## Remaining Known Concern

The main remaining issue from this diagnostics pass is:

```text
Isaac 82D state currently zero-fills tcp_error dims 13:18.
Gazebo expert data records controller tcp_error in those dims.
```

This is now visible in diagnostics. It may or may not be a major issue,
depending on how informative the Gazebo controller tcp_error was during expert
training. It should be revisited before long online runs.

## Follow-Up Offline SERL Runtime Fix

After this summary was first written, another concrete offline SERL evaluation
issue was found:

- The official runtime evaluator could inherit stale `AIC_SERL_ADAPTER_DELTA_CLIP`
  and `AIC_SERL_ACTION_CLIP` environment variables.
- The bad offline SERL 1050 evaluation log showed:

```text
adapter_delta_clip=0.02
action_clip=0.05
```

but the checkpoint was trained with:

```text
adapter_delta_clip=1e-5
action_clip=None
```

That means evaluation could let a much larger raw adapter output through than
the checkpoint was trained/evaluated with. `scripts/evaluate_act_checkpoints_runtime.py`
now unsets these SERL runtime overrides by default and adds explicit optional
CLI flags if an override is actually desired.

`aic_utils/gazebo_rl/gazebo_rl/serl_policy.py` was also updated so the Gazebo
runtime loader reconstructs `adapter_arch` and `adapter_layer_norm` from the
checkpoint config. The 1050 checkpoint uses a plain MLP without layer norm, so
this was not the cause of that specific failure, but it prevents future gated or
layer-norm adapter checkpoints from being loaded with the wrong architecture.

## Offline SERL 8,4 Contract Check

The offline SERL trainer now validates and records the ACT/dataset/action
contract before training:

- ACT `chunk_size` must be at least offline SERL `action_horizon`.
- ACT `observation.state` dimension must match dataset state dimension.
- ACT per-step action dimension must match dataset action dimension.
- ACT camera keys must match the offline SERL camera keys.
- The saved run config records that Gazebo runtime should use
  `n_action_steps == action_horizon`.

Validation run:

```bash
.pixi/envs/default/bin/python aic_utils/lerobot_robot_aic/scripts/train_vision_offline_serl.py \
  --dataset-root outputs/hf_combined/clean_sfp_to_nic_sc_to_sc_task_conditioned_contact_features_s3latest_20260512_003120_h264_dense_rewards \
  --act-checkpoint outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k/checkpoints/175000/pretrained_model \
  --output-dir outputs/train/clean_sfp_sc/offline_serl/q_bc/smoke_8_4_act175k_20260513 \
  --steps 1 \
  --batch-size 2 \
  --device cpu \
  --num-workers 0 \
  --action-horizon 4 \
  --actor-update-mode q_bc \
  --freeze-act \
  --adapter-arch mlp \
  --adapter-delta-clip 1e-5 \
  --critic-arch multiplicative \
  --bc-weight 10 \
  --act-preservation-weight 1.0 \
  --reward-mode dataset \
  --val-fraction 0.05 \
  --val-every 1 \
  --val-max-batches 1 \
  --save-every 1
```

Result:

```text
ACT checkpoint: chunk_size=8, n_action_steps=8, state_dim=82, action_dim=6
offline SERL: action_horizon=4, action_dim=24
contract errors: []
expected warning: offline SERL uses the first 4 actions from the 8-action ACT chunk
checkpoint written: outputs/train/clean_sfp_sc/offline_serl/q_bc/smoke_8_4_act175k_20260513/checkpoint_latest.pt
```

The Gazebo runtime loader was then checked against the smoke checkpoint with
the ACT TorchScript base. It now loads:

```text
state_dim=82
action_dim=24
action_horizon=4
single_action_dim=6
adapter_delta_clip=1e-5
action_clip=None
```

`pixi run ...` is currently blocked by this checkout's `pixi.lock` missing the
newer lockfile `version` field, so validation used the existing Pixi Python
directly at `.pixi/envs/default/bin/python`.
