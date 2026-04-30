# Isaac RL Stage 5

Stage 5 uses the existing Isaac Lab + RSL-RL PPO stack for online RL:

```text
Gazebo expert data -> ACT / offline SERL smoke checkpoints -> Isaac Lab PPO/RSL-RL training
```

This is not true online SERL/SAC yet. The current Isaac path starts PPO from
scratch or resumes an RSL-RL-native checkpoint.

## Current Training Method

- Environment: `AIC-Task-v0`
- Isaac Lab env type: `ManagerBasedRLEnv`
- RL stack: RSL-RL PPO
- Config: `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_cfg.py`
- Base training entrypoint: `aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py`
- Stage 5 wrapper: `aic_utils/aic_isaac/scripts/train_isaac_ppo_stage5.py`

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

Optional insertion-aware terms are added but disabled by default:

- `target_distance_tanh`: body-to-target-object distance reward.
- `target_lateral_error`: lateral error penalty to a target object.

These use available object root poses as approximate targets. They should stay
low-weight or disabled until the Isaac assets expose semantic cable-tip and
port-insertion frames.

## Randomization Profiles

Select a profile through the wrapper:

```bash
--randomization-profile none
--randomization-profile light
--randomization-profile heavy
```

Internally this sets `AIC_ISAAC_RANDOMIZATION_PROFILE`.

`none`:

- disables reset joint noise;
- fixes light intensity/color;
- disables board/part pose noise;
- disables lowdim observation noise;
- uses action scale `0.05`.

`light`:

- preserves the previous default behavior;
- robot joint reset noise: `(-0.05, 0.05)`;
- light intensity: `(1500, 3500)`;
- light color: `(0.5, 0.5, 0.5)` to `(1.0, 1.0, 1.0)`;
- task board x/y noise: `(-0.005, 0.005)`;
- SC port x offsets and NIC card y offsets;
- lowdim observation noise already present in the policy observation group;
- action scale `0.05`.

`heavy`:

- robot joint reset noise: `(-0.12, 0.12)`;
- light intensity: `(800, 5000)`;
- light color variation;
- task board x/y/z/yaw noise;
- SC port x/y/z/yaw offsets;
- NIC card x/y/z/yaw offsets with y snapping preserved;
- larger lowdim observation noise;
- action scale `0.06`.

Physics/material randomization, cable stiffness randomization, controller
stiffness/damping randomization, and camera-pose jitter are documented future
work. The current assets/config do not expose stable semantic handles for those
without more invasive Isaac scene work.

## PPO Smoke Command

```bash
cd ~/ws_aic/src/aic
pixi run python aic_utils/aic_isaac/scripts/train_isaac_ppo_stage5.py \
  --task AIC-Task-v0 \
  --num-envs 4 \
  --max-iterations 1 \
  --seed 1 \
  --headless \
  --randomization-profile heavy \
  --output-dir outputs/train/isaac_stage5_smoke
```

Use `--dry-run` to print the underlying Isaac Lab command without launching
Isaac Sim.

Optional insertion reward weights:

```bash
  --insertion-distance-weight 0.05 \
  --insertion-lateral-weight -0.01
```

## Checkpoint Bridge Status

Current checkpoints are not directly interchangeable:

- ACT produces a LeRobot ACT checkpoint.
- Offline SERL produces a minimal lowdim MLP actor-critic checkpoint.
- Isaac PPO uses RSL-RL's PPO actor-critic architecture.

The wrapper exposes `--init-policy-checkpoint`, but it fails clearly because
direct checkpoint loading is not implemented. Use RSL-RL `--resume`,
`--load-run`, and `--checkpoint` for RSL-RL-native checkpoint resume.

Use the compatibility checker for metadata inspection:

```bash
pixi run python aic_utils/aic_isaac/scripts/check_policy_checkpoint_compatibility.py \
  --checkpoint outputs/train/offline_serl_chunked_smoke/checkpoint_latest.pt \
  --expected-obs-dim 32 \
  --expected-action-dim 48
```

## Current Limitations

- Isaac online RL is PPO/RSL-RL, not off-policy SERL/SAC.
- Offline SERL is still a lowdim smoke path.
- Offline SERL does not consume ACT checkpoints yet.
- The optional insertion-aware rewards use approximate target object poses.
- Gazebo online RL, recovery intervention, and Isaac-to-Gazebo recovery loops
  are not implemented yet.
