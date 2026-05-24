# Agentic Reward/Curriculum Best Run 2026-05-24

## Strict Success

Strict final-window insertion was reproduced in Isaac with a guarded diagnostic wrist-IK servo.

Run folder:

`outputs/agentic_reward_curriculum_20260524_controller_contact/runs/2026-05-24_09-23-09_strict_candidate_x015_zneg024_ax60_images`

First strict post-step row:

| metric | value |
| --- | ---: |
| step | 23 |
| env | 0 |
| signed depth `s` | `0.008056168 m` |
| lateral error `r` | `0.000225569 m` |
| semantic tip orientation `theta` | `0.029892141 rad` |
| module consistency | `0.806509197` |
| strict_success | `true` |

## Evidence

- Metrics: `metrics.jsonl`, `strict_metrics.csv`, `strict_success.json`.
- Reproducibility: `run_config.json`, `command.txt`, `git_status.txt`, `git_diff.patch`.
- Visual sanity frames: `step_images/step_000023/env_0000_center_camera.png`, `env_0000_left_camera.png`, `env_0000_right_camera.png`.
- Videos: `center_camera_insertion.mp4`, `left_camera_insertion.mp4`, `right_camera_insertion.mp4`.

Command:

```bash
docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py --headless --enable_cameras --rendering_mode performance --task AIC-Task-v0 --num_envs 1 --seed 1 --device cuda:0 --output_dir aic/outputs/agentic_reward_curriculum_20260524_controller_contact/runs --run_name strict_candidate_x015_zneg024_ax60_images --episode_config_dir aic/outputs/agentic_reward_curriculum_20260524_final_window_probe/generated_episode_configs/20260524_finalwindow_env1_poseenvlog_084746_step155_ep2 --override_start_signed_depth_m 0.00652 --override_start_lateral_m 0.000137 --override_start_orientation_rotvec_world 0.015 0.0 -0.024 --derive_reset_position_from_orientation --episode_length_s 2.0 --near_gate_reset_max_iterations 120 --near_gate_reset_position_tolerance 0.0002 --near_gate_reset_orientation_tolerance 0.005 --approach_steps 0 --probe target_tip_stabilize --probe_steps 30 --probe_translation_step_m 0.00006 --target_tip_stabilize_goal_depth_m 0.008 --target_tip_stabilize_axial_step_m 0.00006 --target_tip_stabilize_lateral_step_m 0.00002 --target_tip_stabilize_orientation_gate_depth_m 1.0 --image_log_every 1 --max_logged_image_steps 40'
```

## Why This Is Not A False Positive

The success row is post-step geometry, not reward return. It satisfies seated depth, lateral error, semantic tip orientation, and module consistency simultaneously. Earlier false positives had positive tip depth with poor module consistency or large lateral error; this row has `r=0.226 mm` and module consistency `0.807`.

The center, left, and right images at step 23 were visually checked and are consistent with the SFP module entering the cage, not bypassing laterally.

## Limitation

This is a focused final-window diagnostic success, not a learned ACT/SERL policy success over randomized starts. The reset was generated from a near-success checkpoint trajectory and used a scripted target-tip depth hold with a partial reset-orientation trim. The next step is to port the successful recipe into the config-driven SERL eval guard and evaluate it from held-out near-window starts.

## Validation

```bash
.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py
.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py
```

Latest result: `30 passed in 12.95s`.
