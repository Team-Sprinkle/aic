# AIC Training Debug Findings - 2026-05-13

This note captures the recent ACT, offline SERL, and online Isaac SERL findings so
we can resume debugging without relying on chat history. Values below are split
between measured artifacts and current hypotheses.

## Current Artifact Index

### ACT

- Main ACT run:
  `outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k`
- Best ACT checkpoint selected for downstream use:
  `outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k/checkpoints/175000/pretrained_model`
- TorchScript export used by Isaac/Gazebo runtime:
  `outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k/act_policy_ts_175000_cpu.pt`
- CUDA TorchScript used inside Isaac:
  `outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k/act_policy_ts_175000_cuda0.pt`

### Offline SERL

- Current/offline retrain run:
  `outputs/train/clean_sfp_sc/offline_serl/q_bc/20260513_offline_serl_v2rewards_clip1e-6_bc10_gpu1`
- Dataset used:
  `outputs/hf_combined/clean_sfp_to_nic_sc_to_sc_82d_rewardsonly_v2_rgbpatch_20260513_120214`
- ACT warm-start:
  `outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k/checkpoints/175000/pretrained_model`
- Best validation checkpoint currently written by the run:
  `outputs/train/clean_sfp_sc/offline_serl/q_bc/20260513_offline_serl_v2rewards_clip1e-6_bc10_gpu1/checkpoint_best_val.pt`

### Online Isaac SERL

- Previous broad-reward run, used as latest online checkpoint source:
  `outputs/train/online_serl_broad_rewards_from_latest_20260513_162059/gpu0_broad/2026-05-13_16-21-44_online_broad_from_latest/checkpoint_latest.pt`
- Hung near-gate restart, stopped:
  `outputs/train/online_serl_broad_near10_curriculum_from_latest_20260513_173446`
- Active debug restart from latest checkpoint:
  `outputs/train/online_serl_near10_curriculum_debug_clip005_from_latest_20260513_174319/gpu0_near_curr_clip005`
- Active metrics:
  `outputs/train/online_serl_near10_curriculum_debug_clip005_from_latest_20260513_174319/gpu0_near_curr_clip005/2026-05-13_17-44-02_online_near10_curr_clip005/metrics.jsonl`
- Active latest checkpoint:
  `outputs/train/online_serl_near10_curriculum_debug_clip005_from_latest_20260513_174319/gpu0_near_curr_clip005/2026-05-13_17-44-02_online_near10_curr_clip005/checkpoint_latest.pt`

## Data And Feature Logic

- Policy state is 82D:
  - 32D original observations.
  - 40D contact/recovery features.
  - 10D task vector.
- Offline dataset summary for the current offline SERL run:
  - `state_dim`: 82.
  - `action_horizon`: 4.
  - `single_action_dim`: 6.
  - `action_dim`: 24.
  - `num_episodes`: 414.
  - `num_frames`: 321704.
  - cameras: center, left, right.
- RGB issue:
  - Expert dataset videos/images appeared channel-flipped, e.g. orange cable looked blue.
  - Current offline dataset path includes `rgbpatch`, indicating the temporary dataset-side RGB patch was applied.
  - Current Isaac runs use `--no-swap-rgb-channels`; based on current configuration, Isaac is not applying a runtime channel swap.
- Rewards:
  - Offline SERL requires recomputed reward columns because we redesigned rewards to be denser.
  - Online Isaac SERL computes reward live, so no reward column recomputation is needed for online data.

## ACT Findings

### Settings And Selection

- ACT was trained with:
  - 82D state input.
  - ACT image inputs from the three cameras.
  - chunk size 8 in the saved ACT config.
  - ACT run checkpoints at many intervals up to at least 375k.
- The 175k checkpoint was selected as the best practical ACT checkpoint based on the earlier validation review.
- Exact structured ACT training/validation metric files were not found under the final ACT run directory during this documentation pass. The checkpoint/eval artifacts are present, but loss trend values should be recovered from the original console/log source if exact ACT validation-loss numbers are needed later.

### Runtime/Normalization Fixes

- Runtime policies must clamp zero or near-zero normalizer std entries to `1.0`.
- This matters because task-conditioned single-task dimensions can be constant, making saved normalizer std zero.
- Without this clamp, normalized state can contain inf/NaN, causing ACT to produce non-finite actions.
- `RunACT` and `RunACTTorchScript` include this clamp.
- Important interpretation: normalization should only standardize continuous dimensions. One-hot/task bits should not be corrupted by zero-std division. The runtime clamp prevents NaNs; the saved ACT/offline normalization logic should still be inspected before any new long run.

### Behavioral Status

- ACT can approximately navigate near the port/gate.
- ACT still fails reliable final insertion in official Gazebo evaluation.
- Therefore, online SERL should mainly improve the near-gate insertion behavior, not relearn the whole approach from scratch.

## Offline SERL Findings

### Current Run Configuration

Run:
`outputs/train/clean_sfp_sc/offline_serl/q_bc/20260513_offline_serl_v2rewards_clip1e-6_bc10_gpu1`

Key settings:

- actor update mode: `q_bc`.
- actor mode: ACT adapter.
- ACT frozen: true.
- ACT warm-start: 175k ACT checkpoint.
- `adapter_delta_clip`: `1e-6`.
- `bc_weight`: `10.0`.
- `act_preservation_weight`: `1.0`.
- `adapter_penalty_weight`: `0.001`.
- `adapter_lr`: `1e-5`.
- `critic_lr`: `3e-5`.
- `batch_size`: 32.
- `action_horizon`: 4.
- critic image encoder: `small_conv`.
- critic arch: `multiplicative`.
- critic hidden/feature dim: 256.

Warm-start report:

- ACT parameters: 34,341,126.
- Actor parameters: 34,452,790.
- Adapter parameters: 111,640.
- Parameters loaded from ACT: 34,341,126.
- Percent actor parameters loaded: 99.6759%.
- Initial adapter delta norm: 0.0.
- Skipped tensors: none.

### Validation Trend

Validation metrics from `validation_metrics.jsonl`:

| Step | Val TD/Critic Loss | Actor Loss | Q Mean | Reward Mean | Adapter Delta Norm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 700 | 0.0197249 | -2.40585 | 2.40587 | 0.779804 | 4.89898e-6 |
| 1400 | 0.0241768 | -4.71796 | 4.71804 | 0.779804 | 4.89898e-6 |
| 2100 | 0.0371868 | -7.04947 | 7.04963 | 0.779804 | 4.89898e-6 |
| 2800 | 0.0560618 | -9.34204 | 9.34229 | 0.779804 | 4.89898e-6 |
| 3500 | 0.0733196 | -11.5876 | 11.5879 | 0.779804 | 4.89898e-6 |
| 4200 | 0.118158 | -13.7210 | 13.7215 | 0.779804 | 4.89898e-6 |

Interpretation:

- Validation critic/TD loss monotonically worsened after step 700.
- Actor loss becoming more negative is not necessarily good. Here it mostly tracks critic Q increasing, while validation TD loss worsens.
- The best validation checkpoint is probably step 700, which matches the existence of `checkpoint_best_val.pt`.
- Because `adapter_delta_clip=1e-6`, this run barely changes the ACT action. This is useful as a conservative smoke test, but it is unlikely to fix insertion by itself.

### Training Tail

From current `metrics.jsonl`, 4602 rows were present during this check.

Last row at step 4602:

- `reward_mean`: 0.756334.
- `td_loss` / `critic_loss`: 0.0557120.
- `actor_loss`: -14.3237.
- `q_mean`: 14.3231.
- `bc_loss`: 4.7159e-9.
- `raw_adapter_delta_norm`: 0.0289615.
- clipped `adapter_delta_norm`: 4.89898e-6.

Recent training windows:

| Window | Avg TD/Critic Loss | Min | Max | Avg Actor Loss | Avg Reward |
| ---: | ---: | ---: | ---: | ---: | ---: |
| last 100 | 0.650191 | 0.0181315 | 9.99781 | -13.8465 | 0.735931 |
| last 500 | 0.471099 | 0.0109770 | 9.99781 | -13.2893 | 0.733332 |
| last 1000 | 0.443946 | 0.0109770 | 11.2342 | -12.5878 | 0.732975 |

Interpretation:

- Critic has large spikes, up to about 10-11 TD loss.
- Validation is the stronger signal here, and it is worsening.
- This supports treating prior poor offline SERL performance as likely from offline SERL training/critic behavior rather than ACT needing retrain.

### Prior Bad Offline SERL Hypotheses

Likely contributors:

- Earlier configs allowed overly aggressive adapter movement, e.g. `adapter_delta_clip=0.10`, `bc_weight=1.0`, `act_preservation_weight=0.01`.
- A `0.10` delta-pose adapter clip is huge for insertion-scale control and can easily move far away from ACT behavior.
- Poor critic optimization can push the adapter in the wrong direction even if ACT normalization is correct.
- Current conservative run avoids large deviation, but also barely adapts.

## Online Isaac SERL Findings

### Stage/Architecture

- Current training is Stage C: one independent learner per GPU.
- We are currently using GPU0 only for the active online Isaac run.
- Earlier Stage D centralized learner was explored, but not adopted. Throughput did not justify spending more time on it for the current deadline.

### Current Curriculum

Generated curriculum:
`outputs/train/isaac_curriculum_sfp_near10_70_full_30_20260513_1735`

Composition:

- 700 slots: `02_sfp_near_gate`, 10 mm axial/lateral near-gate starts.
- 300 slots: `01_sfp_full`.
- There were 320 unique near-gate and 320 unique full SFP child YAMLs available, so the 1000 slots cycle through unique YAMLs as needed.
- Order is near-gate first, then full SFP. This was chosen for debugging insertion behavior quickly.

Earlier intended long-run curriculum order remains:

1. SFP full episodes.
2. SFP near-gate episodes.
3. SC-to-SC full episodes.
4. SC-to-SC near-gate episodes.

For immediate debugging, the near-gate-heavy curriculum is more informative because it puts the policy directly into the insertion reward basin.

### Hung Near-Gate Run

Run:
`outputs/train/online_serl_broad_near10_curriculum_from_latest_20260513_173446`

Settings:

- Started from:
  `outputs/train/online_serl_broad_rewards_from_latest_20260513_162059/gpu0_broad/2026-05-13_16-21-44_online_broad_from_latest/checkpoint_latest.pt`
- `adapter_delta_clip`: 0.01.
- `num_envs`: 8.
- `batch_size`: 4.
- `save_latest_every_steps`: 1800.
- `save_every_steps`: 7200.
- Reward weights:
  - target distance tanh effective table weight: 12.0.
  - target distance exp effective table weight: 13.5.
  - target distance progress effective table weight: 3.0.
  - target orientation gated exp effective table weight: 3.0.
  - terminal success effective table weight: 30.0.
  - force delta penalty effective table weight: 6.0.

Observed:

- Run stalled at step 116.
- Metrics/log timestamps stopped updating at:
  - metrics: 2026-05-13 17:37:08 UTC.
  - train log: 2026-05-13 17:37:04 UTC.
- Process was still alive in Isaac but no longer writing new training rows.
- It was stopped manually.

Important values:

- Step 1:
  - `reward_mean`: 0.118525.
  - `target_distance_tanh`: 0.142354.
  - `target_distance_exp`: 0.002234.
  - `target_distance_progress`: 0.0.
  - `joint_acc`: -0.024948.
  - force delta penalty: 0.0.
  - `adapter_delta_abs_max`: 0.01.
  - `adapter_delta_norm`: 0.048990.
- Last 100 steps:
  - avg `reward_mean`: -0.098567.
  - avg `target_distance_tanh`: 0.002597.
  - avg `target_distance_exp`: approximately 0.
  - avg `target_distance_progress`: -0.077545.
  - avg force penalty: 0.0.
- Last 10 steps:
  - avg `reward_mean`: -0.123633.
  - avg `target_distance_tanh`: 1.38e-5.
  - avg `target_distance_progress`: -0.063933.

Interpretation:

- The reset/curriculum worked: all active envs were `02_sfp_near_gate`, and TCP reset final errors were under 2 mm. Mean TCP reset final error was about 0.00147 m; max was about 0.00189 m.
- The policy started in the reward basin, but immediately moved away.
- The adapter was saturated at the per-dimension clip from the first update, suggesting `0.01` was too permissive for this checkpoint/curriculum.
- The failure was not a force-contact problem; force penalties were zero.

### Active Debug Restart With Smaller Adapter Clip

Run:
`outputs/train/online_serl_near10_curriculum_debug_clip005_from_latest_20260513_174319/gpu0_near_curr_clip005`

Settings:

- Started from the same latest checkpoint:
  `outputs/train/online_serl_broad_rewards_from_latest_20260513_162059/gpu0_broad/2026-05-13_16-21-44_online_broad_from_latest/checkpoint_latest.pt`
- `adapter_delta_clip`: 0.005.
- `num_envs`: 8.
- `batch_size`: 4.
- `save_latest_every_steps`: 200.
- `save_every_steps`: 6000.
- Same near-gate-heavy curriculum.
- Same dense/broad reward weights as the hung run.

Early comparison to the `0.01` run:

- Step 1 `reward_mean`: 0.124301, slightly higher than 0.118525.
- Step 1 `adapter_delta_abs_max`: 0.005, half the previous run.
- Step 1 `adapter_delta_norm`: 0.024495, half the previous run.
- Step 1 `final_action_norm`: 0.02361, versus about 0.04615 in the `0.01` run.

Current measured point during this documentation pass:

- Rows: 562.
- Last step: 562.
- Last row:
  - `reward_mean`: -0.016994.
  - `target_distance_tanh`: 0.004499.
  - `target_distance_exp`: approximately 9.82e-22.
  - `target_distance_progress`: -0.015446.
  - force delta penalty: 0.0.
  - `joint_acc`: -0.003671.
  - `joint_pos_limits`: -0.001737.

Recent windows:

| Window | Avg Reward | Min Reward | Max Reward | Avg Actor Loss | Avg Critic Loss | Avg Env Steps/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| last 10 | -0.021747 | -0.032524 | -0.008551 | 2.40481 | 0.019058 | 13.3840 |
| last 50 | -0.018259 | -0.032654 | -0.006768 | 2.33447 | 0.015746 | 13.4399 |
| last 100 | -0.014597 | -0.046208 | 0.010024 | 2.32761 | 0.015889 | 13.3366 |
| last 500 | -0.022932 | -0.586490 | 0.048572 | 2.12291 | 0.016057 | 13.5900 |
| all 562 | -0.023294 | -0.586490 | 0.124301 | 2.11255 | 0.015896 | 13.7130 |

Reward-term windows:

| Window | Avg Target Tanh | Avg Target Exp | Avg Progress | Avg Orientation Gated | Avg Force Penalty |
| ---: | ---: | ---: | ---: | ---: | ---: |
| last 10 | 0.004663 | ~0 | -0.016792 | 0.0 | 0.0 |
| last 50 | 0.005484 | ~0 | -0.013357 | 0.0 | 0.0 |
| last 100 | 0.006377 | ~0 | -0.008941 | 0.0 | 0.0 |
| last 500 | 0.004640 | ~0 | -0.003800 | 0.0 | 0.0 |
| all 562 | 0.010278 | 0.0000208 | -0.010102 | 0.0 | 0.0 |

Interpretation:

- The `0.005` clamp avoids the early hard failure/stall observed at `0.01` and gives higher throughput.
- The policy still often leaves the insertion reward basin after starting near the gate.
- Progress is becoming less negative in later windows, but still not consistently positive.
- Orientation and success rewards remain zero, so the current policy is not reaching aligned/insertion states.
- Force penalty remains zero, so the current issue is not contact/collision.
- Adapter is still saturated at the smaller clip. Raw adapter delta abs max is about 0.01207, so both `0.01` and `0.005` clamp the adapter.

### Online Throughput

Observed from active `0.005` run:

- Typical step wall time: about 0.59-0.60 s for 8 envs.
- Effective throughput: about 13-14 env steps/s.
- First step is slow because of initialization, about 4.27 s.

## Current Debug Hypotheses

1. **Online adapter is still pushing away from the target.**
   - Evidence: starts with positive target-distance reward, then target-distance tanh collapses and progress turns negative.
   - Force penalty is zero, so this is not primarily collision.

2. **Adapter clip controls stability but does not fix directionality.**
   - `0.01` was too aggressive and saturated immediately.
   - `0.005` is more stable and faster, but still saturated and not solving insertion.

3. **Offline SERL critic quality is a risk.**
   - Validation TD loss worsens steadily from 0.0197 at step 700 to 0.118 at step 4200.
   - Actor loss gets more negative mainly because critic Q gets larger; that is not proof of policy improvement.

4. **Conservative offline adapter clip protects ACT but cannot improve much.**
   - `adapter_delta_clip=1e-6` effectively preserves ACT.
   - This is useful for validating code/data/reward plumbing, but not enough for insertion improvement.

5. **Reward is denser than before but still not guiding orientation/insertion yet.**
   - Target distance tanh activates near the gate.
   - Target exp is only nonzero very close to target.
   - Orientation gated and success terms stay zero in current online rollouts.
   - More insertion-specific curriculum or action regularization may be needed before increasing adapter freedom.

## Recommended Next Debug Steps

1. Keep the active `0.005` run running long enough to see whether progress turns positive over several thousand steps.
2. Evaluate latest/periodic checkpoints in official Gazebo as soon as a stable checkpoint exists.
3. For online SERL, compare:
   - `adapter_delta_clip=0.0025` for extra stability.
   - `adapter_delta_clip=0.005` current baseline.
   - Possibly `0.01` only after adding stronger preservation or action-direction debugging.
4. Add per-step target distance and signed axial/lateral error logging, not only reward terms. This will show whether the policy is moving laterally, axially backward, or rotating incorrectly.
5. Consider a near-gate-only insertion micro-curriculum before returning to full episodes.
6. For offline SERL, prefer checkpoint step 700 or `checkpoint_best_val.pt` over later checkpoints unless validation improves.
7. Avoid using offline SERL checkpoints with large adapter clips such as `0.10` for Gazebo submission unless a direct Gazebo eval proves improvement.

## Isaac Lab Online Training Implementation Stages

This section documents the implementation state and intended semantics of the
Stage A/B/C/D Isaac online SERL training paths discussed during debugging.

### Shared Building Blocks

Core online trainer:

- Wrapper script:
  `aic_utils/aic_isaac/scripts/train_isaac_online_serl.py`
- Isaac-side trainer:
  `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
- Design note:
  `aic_utils/aic_isaac/docs/isaac_online_serl_design.md`

Actor contract:

- Load ACT TorchScript for the frozen base policy.
- Load an ACT-adapter SERL checkpoint when starting from offline/online SERL.
- Actor computes:
  - ACT base action chunk.
  - adapter delta action.
  - final action = ACT base action + clipped adapter delta.
- ACT is frozen by default.
- Adapter, critics, and optimizer state are checkpointed in `checkpoint_latest.pt`
  and periodic `checkpoint_*.pt` files.

Online replay/training loop:

- Runs `AIC-Task-v0` in Isaac Lab.
- Reads raw camera tensors from Isaac cameras.
- Builds LeRobot-compatible ACT observations.
- Converts TCP delta action to Isaac IK action.
- Stores transition tuples in replay.
- Performs off-policy actor/critic updates.
- Logs `metrics.jsonl`, including reward terms, losses, adapter norms, action
  norms, force metrics, and episode metadata.

Episode YAML/config path:

- Minimal YAML materializer:
  `aic_utils/aic_isaac/scripts/isaac_episode_configs.py`
- Stage-C launcher:
  `aic_utils/aic_isaac/scripts/launch_isaac_serl_curriculum.py`
- Child YAMLs fully specify per-episode scene/task information:
  - task family.
  - target port/card.
  - start position.
  - `start_near_gate`, if enabled.
  - asset poses/randomization.
  - curriculum metadata.
- Isaac reset event consumes these via:
  `AIC_ISAAC_EPISODE_CONFIG_DIR`
- Reset/event implementation:
  `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/events.py`

Near-gate reset logic:

- For near-gate child YAMLs, the reset should move the robot TCP/cable near the
  gate while keeping board/assets at the child YAML scene pose.
- Implementation function:
  `reset_robot_tcp_to_episode_start`
- It solves a small damped positional IK problem for the arm joints and writes
  joint state before the episode begins.
- Current observed reset quality in the near-gate curriculum:
  - mean TCP final reset error: about 0.00147 m.
  - max TCP final reset error: about 0.00189 m.
  - tolerance: 0.002 m.

Reward implementation:

- Dense target reward terms are configured through command-line arguments and
  mapped into Isaac reward table weights.
- Current active reward table weights observed in Isaac:
  - `target_distance_tanh`: 12.0.
  - `target_distance_exp`: 13.5.
  - `target_distance_progress`: 3.0.
  - `target_orientation_gated_exp`: 3.0.
  - `target_success_once_bonus`: 30.0.
  - `force_delta_penalty`: 6.0.
  - small joint/action regularizers also active.
- Force penalty is based on force delta norm, not duration above threshold.

### Stage A - Single-GPU Single Trainer Smoke

Intent:

- Prove the ACT-adapter online loop can start Isaac, read cameras, build ACT
  observations, execute actions, update replay, train critics/adapter, and write
  checkpoints.
- Debug correctness before introducing multi-GPU or curriculum sharding.

Implementation:

- Use `train_isaac_online_serl.py` directly with one GPU and one trainer.
- Usually use small values like:
  - `num_envs=1-4`.
  - very small `steps` and `updates`.
  - short wall-time guard.
- No multi-GPU coordination.
- No cross-process communication.

Verified behavior from earlier design note:

- A tiny artifact-producing run completed 3 Isaac steps and 2 online updates.
- A 300-step/100-update guarded run stopped after 107 Isaac steps and 100 updates
  in about 4.09 minutes.
- A 1k-step guarded run completed 1000 Isaac steps and 993 updates in about
  5.90 minutes.

Use cases:

- Import and camera readiness smoke.
- Checkpoint format smoke.
- Reward-term logging smoke.
- Fast diagnosis of crashes or NaNs.

Limitations:

- Not intended for throughput.
- Does not test curriculum sharding.
- Does not use multiple GPUs.

### Stage B - Single-GPU Curriculum/Child-YAML Run

Intent:

- Test the user-facing curriculum YAML flow without multi-GPU complexity.
- Confirm minimal YAMLs are materialized into child YAMLs and Isaac consumes
  them in order.
- Confirm reset randomization and task vector metadata are consistent.

Implementation:

- Use the same single trainer as Stage A.
- Add `--episode-config-dir` pointing to a generated child-YAML directory.
- Episode configs are consumed sequentially by Isaac reset events.
- Per-step metrics include `episodes`, which records:
  - `episode_id`.
  - `source_request`.
  - `task_family`.
  - `target_port_index`.
  - `target_card_index`.
  - `start_near_gate`.
  - `tcp_reset` report when applicable.

Important behavior:

- This stage is where we validated the near-gate reset direction:
  - board/assets stay at the child YAML scene pose.
  - robot TCP/cable is moved near the gate.
- This fixed the earlier conceptual issue where randomization could move the
  board/assets while not moving the TCP as intended for near-gate curriculum.

Use cases:

- Validate SFP full vs SFP near-gate curriculum.
- Validate SC-to-SC child YAMLs.
- Check task/card/port distribution.
- Debug reward values at signature positions.

Limitations:

- Still single trainer.
- No multi-GPU throughput.

### Stage C - Multi-GPU Independent Trainers

Intent:

- Use multiple GPUs without needing distributed RL infrastructure.
- Each GPU receives a shard of the curriculum and trains independently.
- Pick the best resulting checkpoint by metrics/Gazebo eval.

Current implementation:

- Stage-C-only launcher:
  `aic_utils/aic_isaac/scripts/launch_isaac_serl_curriculum.py`
- It materializes minimal YAMLs into full child YAMLs through
  `materialize_many_episode_configs`.
- It shards child YAMLs by GPU in curriculum order.
- It launches one independent `train_isaac_online_serl.py` process per GPU shard.
- It writes a `launch_plan.json` with commands, shard paths, and the curriculum
  summary.

Launcher contract:

- `--minimal-yaml-dir`: folder containing minimal YAMLs.
- `--filenames`: concatenated/listed filenames to include, preserving curriculum
  order.
- `--output-dir`: where curriculum, shard dirs, plans, logs, and per-GPU outputs
  are written.
- `--max-gpus`: maximum GPU shards to materialize.
- `--checkpoint`: ACT/SERL checkpoint to start from.
- `--act-torchscript`: ACT TorchScript path.
- `--stage C`: only supported value.
- `--run`: actually launch processes; without it, render plan only.

Sharding behavior:

- Episodes are assigned round-robin by global curriculum order.
- Example with 3 minimal YAMLs `a,b,c`, each 100 trajectories, and 4 GPUs:
  - gpu0 gets `a_001`, gpu1 gets `a_002`, gpu2 gets `a_003`, gpu3 gets `a_004`,
    then round-robin continues.
  - After `a_100`, assignment continues with `b_001`, then `c_001`, etc.
- Each shard directory contains only that GPU's assigned episode YAMLs.
- Each child YAML has `curriculum` metadata:
  - `global_episode_index`.
  - `gpu_id`.
  - `shard_index`.
  - `num_shards`.
  - local shard episode index.

Training behavior:

- Each GPU owns:
  - Isaac simulator instance/process.
  - env batch.
  - replay buffer.
  - actor.
  - critics.
  - optimizers.
  - metrics.
  - checkpoints.
- No gradients are synchronized.
- No replay is shared.
- No policy parameters are synchronized.
- Therefore this is parallel experimentation/checkpoint search, not true
  multi-GPU learning.

Why Stage C became default:

- It was the most robust path under time pressure.
- It avoids low-latency IPC and synchronization bugs.
- It can run different curriculum variants or hyperparameters per GPU.

Limitations:

- GPUs do not learn from each other.
- Total wall-clock exploration increases, but one policy does not get aggregate
  data from all GPUs.
- Best checkpoint selection is external, typically based on reward trends and
  official Gazebo evaluation.

Current active use:

- We are effectively using a Stage-C-style single-GPU shard on GPU0 for the
  near-gate debug run:
  `outputs/train/online_serl_near10_curriculum_debug_clip005_from_latest_20260513_174319/gpu0_near_curr_clip005`

### Stage D - Centralized Learner With Rollout Workers

Intent:

- True high-throughput multi-GPU online RL.
- Multiple rollout workers simulate independently on different GPUs.
- A central learner consumes transitions from all workers, updates one policy,
  and periodically broadcasts updated policy weights to workers.

Desired lifecycle:

1. Learner starts with ACT/SERL checkpoint.
2. Worker processes start on separate GPUs with local Isaac envs.
3. Workers receive current policy weights.
4. Each worker rolls out its assigned episodes at its own pace.
5. Workers stream transitions or chunks of transitions to the learner.
6. Learner appends transitions to centralized replay.
7. Learner performs actor/critic updates.
8. Every N updates or seconds, learner publishes updated actor weights.
9. Workers swap to the newer policy without restarting Isaac.
10. Checkpoints come from the central learner.

What would be synchronized:

- Policy parameters from learner to workers.
- Transition batches from workers to learner.
- Optionally worker stats/reward summaries to learner.
- Gradients would not need to be synchronized if using an asynchronous replay
  learner architecture.

Why this is better in principle:

- A single policy learns from all GPUs' experience.
- Faster data collection can improve sample diversity.
- It avoids the inefficiency of picking the best among independent Stage-C runs.

Implementation/testing status:

- Stage D was explored conceptually and with low-latency profiling.
- Observed aggregate rollout throughput in one test was about 24.88 env steps/s.
- Learner train latency in that test:
  - avg: 0.368 s/update.
  - min: 0.166 s/update.
  - max: 2.36 s/update.
- This was not sufficiently better than the simpler Stage-C path to justify the
  engineering risk during the current deadline window.
- Stage D is not the current default and is not the current active training path.

Main unresolved Stage-D engineering risks:

- Robust IPC transport between Isaac workers and learner.
- Backpressure when learner is slower than rollout workers.
- Policy versioning and safe hot-swap on workers.
- Replay serialization cost for image observations.
- Failure recovery when one worker crashes or stalls.
- Ensuring reward/debug images/metrics remain traceable to episode YAML and
  policy version.

Decision:

- Do not spend more current deadline time on Stage D unless Stage C cannot make
  any progress and there is enough time to stabilize the IPC/worker lifecycle.

### Practical Recommendation

For the current submission window:

- Use Stage A for quick smoke tests after code changes.
- Use Stage B to validate new curriculum YAMLs or reset/reward logic.
- Use Stage C for actual Isaac training and parallel experiments.
- Do not use Stage D for the current main run unless we explicitly pause
  training for infrastructure work.

For future work:

- Stage D is the correct long-term high-throughput architecture.
- It should be implemented as a separate, testable worker/learner system with:
  - explicit transition schema.
  - small-image or feature-only transport option.
  - policy version numbers.
  - watchdogs for stalled workers.
  - deterministic replay logs for postmortem debugging.
