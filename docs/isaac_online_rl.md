# Isaac Online RL

Isaac online RL now treats online SERL/SAC with the ACT-adapter actor as the primary
future hybrid path. The existing Isaac Lab + RSL-RL PPO stack remains available
as a legacy smoke test, baseline, and backup trainer:

```text
Gazebo expert data -> ACT -> offline ACT-adapter SERL -> Isaac online SERL/SAC
Gazebo expert data -> ACT / offline SERL smoke checkpoints -> Isaac Lab PPO/RSL-RL legacy smoke
```

Online SERL/SAC now has an artifact-producing short-run implementation. The
current PPO path still starts PPO from scratch, resumes an RSL-RL-native
checkpoint, or applies a conservative offline SERL action-prior initialization
before PPO training.

## Primary Future Method: Online SERL/SAC

- Intended actor: the same ACT-adapter actor produced by
  `train_vision_offline_serl.py --actor-mode act_adapter`.
- Default training: ACT frozen, adapter and critics trainable.
- Host launcher: `aic_utils/aic_isaac/scripts/train_isaac_online_serl.py`.
- Isaac trainer: `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`.
- Design doc: `aic_utils/aic_isaac/docs/isaac_online_serl_design.md`.
- Current status: short-run capable. It loads an ACT TorchScript export, keeps
  Isaac camera sensors enabled, reads raw camera RGB tensors, collects replay,
  updates critics and adapter, and saves a real online checkpoint.

Dry-run checkpoint inspection:

```bash
pixi run python aic_utils/aic_isaac/scripts/train_isaac_online_serl.py \
  --checkpoint outputs/train/hybrid_vision_offline_serl_adapter_nominal_n10/checkpoint_latest.pt \
  --act-torchscript outputs/train/hybrid_act_nominal_n10/act_policy_ts_cuda.pt \
  --output-dir outputs/train/isaac_online_serl_adapter \
  --device cuda \
  --steps 10 \
  --batch-size 2 \
  --warmup-steps 0 \
  --dry-run
```

First artifact-producing command was run inside the Isaac Lab container:

```bash
./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py \
  --task AIC-Task-v0 \
  --num_envs 1 \
  --seed 1 \
  --checkpoint /workspace/isaaclab/aic/outputs/train/hybrid_vision_offline_serl_adapter_nominal_n10/checkpoint_latest.pt \
  --act_torchscript /workspace/isaaclab/aic/outputs/train/hybrid_act_nominal_n10/act_policy_ts_cuda.pt \
  --output_dir /workspace/isaaclab/aic/outputs/train/isaac_online_serl_adapter \
  --run_name online_serl_adapter_short \
  --steps 8 \
  --updates 2 \
  --batch_size 2 \
  --headless \
  --device cuda
```

Result:

```text
outputs/train/isaac_online_serl_adapter/2026-04-30_20-18-32_online_serl_adapter_short/checkpoint_latest.pt
```

The run completed 3 Isaac steps and 2 online updates before stopping after the
requested update count.

Sanity run:

```text
outputs/train/isaac_online_serl_adapter/2026-04-30_20-55-36_online_serl_adapter_sanity_300/checkpoint_latest.pt
```

Requested `steps=300`, `updates=100`, `batch_size=8`, and
`max_wall_time_minutes=30`. It stopped by `target_updates`, not by wall time,
after 107 Isaac steps, 100 online updates, and 4.087678201599981 elapsed
minutes. This indicates a 1k-step run is likely under 30 minutes on the current
L40S host if the same configuration remains stable, but the adapter norm should
be watched.

1k guarded run:

```text
outputs/train/isaac_online_serl_adapter/2026-04-30_21-11-49_online_serl_adapter_1k_guarded/checkpoint_latest.pt
```

Requested `steps=1000`, `updates=1000`, `batch_size=8`,
`max_wall_time_minutes=30`, `adapter_penalty_weight=0.01`, and
`act_preservation_weight=0.1`. It stopped by `max_steps`, not wall time, after
1000 Isaac steps, 993 updates, and 5.902379688616626 elapsed minutes. The
adapter correction grew to `adapter_delta_norm=24.33574676513672`, so this is a
throughput/stability artifact, not yet a good policy candidate.

Bounded execution guards are now part of the ACT-adapter path:

- `--adapter-delta-clip` clamps the adapter correction before it is added to
  ACT.
- `--action-clip` clamps the final action chunk after ACT plus adapter.
- Isaac online SERL and Gazebo transfer default both guards to `0.05`.
- Offline vision SERL exposes the same arguments for training-time guarded
  experiments, but leaves them disabled unless requested.

## Gazebo Transfer Boundary

The Gazebo transfer validator now has an ACT-adapter SERL policy mode:

```bash
pixi run python aic_utils/gazebo_rl/scripts/serl_transfer_validate.py \
  --policy-kind act_adapter_serl \
  --checkpoint outputs/train/isaac_online_serl_adapter/2026-04-30_21-11-49_online_serl_adapter_1k_guarded/checkpoint_latest.pt \
  --act-torchscript outputs/train/hybrid_act_nominal_n10/act_policy_ts_cuda.pt \
  --workspace-dir . \
  --device cuda \
  --output-dir outputs/gazebo_rl/serl_transfer_validation/act_adapter_latest
```

This path reconstructs the TorchScript ACT base plus the trained adapter and can
consume either offline or online ACT-adapter SERL checkpoints. For
`--policy-kind act_adapter_serl`, the validator defaults `--include-images` to
true, and the Gazebo bridge serializes live `center_image`, `left_image`, and
`right_image` fields from `aic_model_interfaces/Observation` into resized
`jpeg_rgb8` IPC payloads. `--allow-zero-images` is available only for explicit
interface validation and must not be used for scored transfer.

Actual 3-step live-image transfer wiring run without clamps:

```text
outputs/gazebo_rl/serl_transfer_validation/act_adapter_3step_latest/transfer_validation_summary.json
```

It used `--sim-distrobox aic_eval`, completed 3 real Gazebo steps with
`include_images=true` and `allow_zero_images=false`, and stopped by
`max_steps`. The short run produced no terminal score file, so classification is
`no_score`. The adapter correction was large
(`adapter_delta_norm_mean=54.55885442097982`), consistent with the 1k Isaac
throughput checkpoint being a stability artifact rather than a deployment
candidate.

Actual 3-step live-image transfer wiring run with clamps:

```text
outputs/gazebo_rl/serl_transfer_validation/act_adapter_clipped_3step_latest/transfer_validation_summary.json
```

It used `--sim-distrobox aic_eval`, `--adapter-delta-clip 0.05`, and
`--action-clip 0.05`. It completed 3 real Gazebo steps with
`include_images=true` and `allow_zero_images=false`. The short run produced no
terminal score file, so classification is still `no_score`, but execution was
bounded: `action_norm_max=0.10949986228346104`,
`adapter_delta_norm_mean=0.3464101552963257`, and
`raw_adapter_delta_norm_mean=54.15252685546875`. This confirms the clamp works;
the raw adapter is still too aggressive and should be retrained before a longer
scored transfer run.

## Legacy/Backup Method: PPO/RSL-RL

- Environment: `AIC-Task-v0`
- Isaac Lab env type: `ManagerBasedRLEnv`
- RL stack: RSL-RL PPO
- Config: `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_cfg.py`
- Base training entrypoint: `aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py`
- Isaac online RL wrapper: `aic_utils/aic_isaac/scripts/train_isaac_rsl_rl.py`

## Current Action Interface

The active action interface is differential IK relative pose control:

```text
DifferentialInverseKinematicsActionCfg
command_type="pose"
use_relative_mode=True
body_name="wrist_3_link"
scale=0.05 for none/light, 0.06 for heavy
```

A joint-position action config is present in comments but is not active.

## Current Reward Terms

- `end_effector_position_tracking`: reach/pose tracking, L2 distance penalty.
- `end_effector_position_tracking_fine_grained`: reach/alignment reward near target using tanh.
- `end_effector_position_tracking_exp`: reach/alignment reward near target using an exponential kernel.
- `end_effector_orientation_tracking`: orientation alignment penalty.
- `end_effector_orientation_tracking_fine_grained`: orientation alignment reward near target.
- `reaching_bonus`: sparse success-style bonus when the end effector reaches the commanded pose threshold.
- `action_rate`: smoothness regularization.
- `joint_vel`: smoothness regularization.
- `joint_acc`: smoothness regularization.
- `joint_torques`: smoothness/contact-safety regularization.
- `joint_pos_limits`: safety regularization.

Insertion-aware target-geometry terms are available for Isaac online SERL:

- `target_distance_tanh`: body-to-target-object distance reward.
- `target_distance_exp`: close-range body-to-target-object distance reward.
- `target_orientation_tanh`: body-to-target-object orientation reward.
- `target_reaching_bonus`: sparse bonus when the body target point reaches the target frame threshold.
- `target_lateral_error`: lateral error penalty to a target object.

For SFP-to-NIC, the default target frame is derived from the Gazebo SDF assets:
the reward target is the SFP port entrance/opening, not the deeper
`sfp_port_*_link` inserted frame. Port 0/1 link frames are defined in
`aic_assets/models/NIC Card/model.sdf`; the entrance offset is
`(0, 0, -0.0458)` in the Gazebo port frame. Isaac uses the SFP module tip
position `(0, -0.02365, 0)` as the body point and root/body quaternions for
orientation. The Gazebo SDF semantic orientation offsets were validated as a
poor Isaac controller target and are not the default.

## Randomization Profiles

Select a profile through the wrapper:

```bash
--randomization-profile none
--randomization-profile light
--randomization-profile heavy
```

Internally this sets `AIC_ISAAC_RANDOMIZATION_PROFILE`.
For the online SERL wrapper, Isaac IK action scaling is controlled separately
with `--isaac-action-scale` and defaults to `1.0`, because ACT/SERL actions are
already physical TCP deltas.

`none`:

- disables reset joint noise;
- fixes light intensity/color;
- disables board/part pose noise;
- disables lowdim observation noise;

`light`:

- preserves the previous default behavior;
- robot joint reset noise: `(-0.05, 0.05)`;
- light intensity: `(1500, 3500)`;
- light color: `(0.5, 0.5, 0.5)` to `(1.0, 1.0, 1.0)`;
- task board x/y noise: `(-0.005, 0.005)`;
- SC port x offsets and NIC card y offsets;
- lowdim observation noise already present in the policy observation group;

`heavy`:

- robot joint reset noise: `(-0.12, 0.12)`;
- light intensity: `(800, 5000)`;
- light color variation;
- task board x/y/z/yaw noise;
- SC port x/y/z/yaw offsets;
- NIC card x/y/z/yaw offsets with y snapping preserved;
- larger lowdim observation noise;

Physics/material randomization, cable stiffness randomization, controller
stiffness/damping randomization, and camera-pose jitter are documented future
work. The current assets/config do not expose stable semantic handles for those
without more invasive Isaac scene work.

## PPO Legacy Smoke Command

```bash
cd ~/ws_aic/src/aic
pixi run python aic_utils/aic_isaac/scripts/train_isaac_rsl_rl.py \
  --task AIC-Task-v0 \
  --num-envs 4 \
  --max-iterations 1 \
  --seed 1 \
  --headless \
  --randomization-profile heavy \
  --output-dir outputs/train/isaac_rsl_rl_smoke
```

Use `--dry-run` to print the underlying Isaac Lab command without launching
Isaac Sim.

Optional insertion reward weights:

```bash
  --insertion-distance-weight 0.5 \
  --insertion-close-weight 0.3 \
  --insertion-orientation-weight 0.0 \
  --insertion-reaching-weight 1.0 \
  --insertion-lateral-weight 0.0
```

## Checkpoint Bridge Status

Checkpoints are not fully interchangeable, but there is now a bounded bridge:

- ACT produces a LeRobot ACT checkpoint.
- Lowdim offline SERL can consume an ACT checkpoint through `--act-checkpoint`
  as a conservative action-prior bridge.
- Vision offline SERL can consume an ACT checkpoint through `--act-checkpoint`
  and, by default, trains an ACT-adapter actor with ACT frozen and a
  zero-initialized correction adapter.
- Offline SERL produces either a lowdim MLP actor-critic checkpoint or a vision
  ACT-adapter actor-critic checkpoint.
- Isaac PPO uses RSL-RL's PPO actor-critic architecture.

The Isaac online RL wrapper accepts `--init-policy-checkpoint` for an offline SERL
checkpoint and forwards it to the Isaac Lab RSL-RL train entrypoint. The train
entrypoint applies a conservative warm start:

- exact-shape tensors are copied when future SERL/RSL architectures match;
- current lowdim SERL -> camera PPO initializes the PPO actor output bias and
  `std` from the first single-step action prior in SERL normalization stats;
- hidden-layer transfer is not claimed when observation/action architectures
  differ.

Use RSL-RL `--resume`, `--load-run`, and `--checkpoint` for RSL-RL-native
checkpoint resume.

Example SERL-initialized PPO command:

```bash
pixi run python aic_utils/aic_isaac/scripts/train_isaac_rsl_rl.py \
  --task AIC-Task-v0 \
  --num-envs 4 \
  --max-iterations 1 \
  --seed 1 \
  --headless \
  --randomization-profile light \
  --output-dir outputs/train/isaac_online_serl_warmstart \
  --init-policy-checkpoint outputs/train/hybrid_offline_serl_nominal_n10/checkpoint_latest.pt
```

Use the compatibility checker for metadata inspection:

```bash
pixi run python aic_utils/aic_isaac/scripts/check_policy_checkpoint_compatibility.py \
  --checkpoint outputs/train/offline_serl_chunked_smoke/checkpoint_latest.pt \
  --expected-obs-dim 32 \
  --expected-action-dim 48
```

## Current Limitations

- Isaac online SERL/SAC is short-run capable, but not yet tuned or scaled.
- PPO/RSL-RL is implemented and useful for smoke/baseline runs, but it is no
  longer the primary hybrid-transfer architecture.
- Lowdim offline SERL still exists; its ACT warm-start transfers only an output
  action prior, not full transformer weights.
- Vision offline SERL now has an ACT-adapter actor with frozen ACT by default,
  and Isaac online rollout/update code can now run that actor through ACT
  TorchScript plus the trained adapter.
- Isaac PPO warm-start from offline SERL is a partial action-prior
  initialization, not full actor-critic hidden-layer transfer for the current
  camera-enabled PPO architecture.
- The optional insertion-aware rewards use approximate target object poses.
- Gazebo online RL, recovery intervention, and Isaac-to-Gazebo recovery loops
  are not implemented yet.
