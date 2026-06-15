# Agentic Reward/Curriculum Best Run 2026-05-24

## Corrected Full-Depth Near-Final Trajectory Success

After the full-depth correction, the strongest current run is now a guarded
target-tip servo trajectory from a near-final reset, not just a seated reset:

`outputs/agentic_reward_curriculum_20260524_depth_correction/runs/2026-05-25_12-40-17_full_depth_insertion_local43_goal0p0456_targettip_no_rot_final4_images_env40`

This is a privileged diagnostic/servo trajectory from a local final-window
start. It is not a learned ACT/SERL policy result and should not be cited as
policy training success. It does demonstrate corrected full-depth insertion
under the strict post-step checker with three-camera evidence.

Strict final post-step row:

| step | env | configured start depth mm | configured lateral mm | x-rot rad | s mm | r mm | theta rad | module consistency | strict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 4 | 40 | 43.5 | 0.25 | 0.000 | 45.990 | 0.454 | 0.02947 | 0.844 | true |

Evidence files:

- Metrics: `metrics.jsonl`, `wrist_contact_summary.json`,
  `strict_success_rows.json`, `strict_success_rows.csv`.
- Reproducibility: `run_config.json`, `command.txt`, `git_status.txt`,
  `git_diff.patch`, `agent_decision.json`.
- Run note: `full_insertion_success.md`.
- Final strict visual sanity frames:
  - `step_images/step_000004/env_0040_center_camera.png`
  - `step_images/step_000004/env_0040_left_camera.png`
  - `step_images/step_000004/env_0040_right_camera.png`
- Videos:
  - `env0040_center_insertion.mp4`
  - `env0040_left_insertion.mp4`
  - `env0040_right_insertion.mp4`
- Higher-quality full-episode videos from the same 60-env strict trajectory:
  - `outputs/agentic_reward_curriculum_20260524_depth_correction/runs/2026-05-25_12-51-10_full_depth_insertion_local43_goal0p0456_targettip_no_rot_quality448_fullvideo_env40/env0040_center_full_episode_20fps_quality448.mp4`
  - `outputs/agentic_reward_curriculum_20260524_depth_correction/runs/2026-05-25_12-51-10_full_depth_insertion_local43_goal0p0456_targettip_no_rot_quality448_fullvideo_env40/env0040_left_full_episode_20fps_quality448.mp4`
  - `outputs/agentic_reward_curriculum_20260524_depth_correction/runs/2026-05-25_12-51-10_full_depth_insertion_local43_goal0p0456_targettip_no_rot_quality448_fullvideo_env40/env0040_right_full_episode_20fps_quality448.mp4`
  - `outputs/agentic_reward_curriculum_20260524_depth_correction/runs/2026-05-25_12-51-10_full_depth_insertion_local43_goal0p0456_targettip_no_rot_quality448_fullvideo_env40/env0040_three_camera_full_episode_20fps_quality448.mp4`

Command:

```bash
docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py --headless --enable_cameras --rendering_mode performance --task AIC-Task-v0 --num_envs 60 --seed 15 --device cuda:0 --output_dir aic/outputs/agentic_reward_curriculum_20260524_depth_correction/runs --run_name full_depth_insertion_local43_goal0p0456_targettip_no_rot_final4_images_env40 --episode_config_dir aic/outputs/agentic_reward_curriculum_20260524_depth_correction/generated_episode_configs/full_depth_insertion_local43_sweep60 --episode_length_s 2.0 --near_gate_reset_max_iterations 200 --near_gate_reset_position_tolerance 0.0001 --near_gate_reset_orientation_tolerance 0.003 --approach_steps 0 --probe target_tip_stabilize --probe_steps 4 --probe_translation_step_m 0.00001 --probe_rotation_step_rad 0.0 --target_tip_stabilize_goal_depth_m 0.0456 --target_tip_stabilize_axial_step_m 0.00001 --target_tip_stabilize_lateral_step_m 0.000005 --target_tip_stabilize_orientation_gate_lateral_m 0.0005 --target_tip_stabilize_orientation_gate_depth_m 0.030 --target_tip_stabilize_orientation_error_threshold_rad 0.030 --target_tip_stabilize_rotation_compensation_clip_m 0.0 --image_log_every 1 --max_logged_image_steps 4 --save_image_env_indices 40'
```

Why this is not a false positive:

- The success row is post-step semantic geometry from `sfp_tip_link`, not reward
  return or a single pre-step row.
- Signed depth clears the corrected full-depth threshold derived from the
  Gazebo SFP port entrance (`~45.8 mm`) with the existing `0.5 mm` tolerance.
- Lateral error is below `0.5 mm`.
- Semantic tip orientation is below `0.030 rad`.
- Module/body consistency is above `0.80`.
- Center, left, and right final images are consistent with a seated SFP body,
  not a lateral bypass.

Remaining limitation:

The run starts in a local final window at configured `s=43.5 mm`. It is a
privileged final insertion servo, not a full policy rollout from approach and
not retrained online SERL. The next step is to port this final-window recipe
into the config-driven train/eval guard and then retrain/evaluate from the best
checkpoint under the corrected full-depth success criterion.

Video interpretation note:

The full-episode video starts with the SFP already visually inside the cage
because frame 0 is the local final-window reset, not a pre-contact approach.
The reset metric is `s=43.501 mm`, while the final strict frame is
`s=45.990 mm`. This video proves the last millimeters of seating under the
corrected strict checker; it does not show the entrance-crossing phase from a
non-contact start.

## Corrected Full-Depth Strict Seated Diagnostic

After correcting the Isaac SFP target depth from the old shallow `8 mm` value to
the Gazebo-derived semantic port depth of about `45.8 mm`, the best current
corrected-depth diagnostic run is:

`outputs/agentic_reward_curriculum_20260524_depth_correction/runs/2026-05-24_13-00-03_full_depth_reset_sweep_depth_lat_x64_posehold_images_env15_61`

This run is a guarded reset/pose-hold seated-state diagnostic, not a learned
ACT/SERL policy rollout and not yet a far-start insertion trajectory. It is
useful because it demonstrates that the corrected full-depth strict checker is
physically reachable in Isaac with semantic tip/body metrics and three-camera
evidence.

Strict post-step rows:

| step | env | configured depth mm | configured lateral mm | x-rot rad | s mm | r mm | theta rad | module consistency | strict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 15 | 45.8 | 0.2 | 0.010 | 45.636 | 0.335 | 0.02751 | 0.929 | true |
| 3 | 61 | 47.0 | 0.2 | 0.000 | 46.864 | 0.330 | 0.02842 | 0.812 | true |

Evidence files:

- Metrics: `metrics.jsonl`, `wrist_contact_summary.json`,
  `strict_success_rows.json`, `strict_success_rows.csv`.
- Reproducibility: `run_config.json`, `command.txt`, `git_status.txt`,
  `git_diff.patch`.
- Visual sanity frames:
  - `step_images/step_000001/env_0015_center_camera.png`
  - `step_images/step_000001/env_0015_left_camera.png`
  - `step_images/step_000001/env_0015_right_camera.png`
  - `step_images/step_000003/env_0061_center_camera.png`
  - `step_images/step_000003/env_0061_left_camera.png`
  - `step_images/step_000003/env_0061_right_camera.png`

Command:

```bash
docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py --headless --enable_cameras --rendering_mode performance --task AIC-Task-v0 --num_envs 64 --seed 15 --device cuda:0 --output_dir aic/outputs/agentic_reward_curriculum_20260524_depth_correction/runs --run_name full_depth_reset_sweep_depth_lat_x64_posehold_images_env15_61 --episode_config_dir aic/outputs/agentic_reward_curriculum_20260524_depth_correction/generated_episode_configs/full_depth_reset_sweep_depth_lat_x64 --episode_length_s 1.0 --near_gate_reset_max_iterations 200 --near_gate_reset_position_tolerance 0.0001 --near_gate_reset_orientation_tolerance 0.003 --approach_steps 0 --probe pose_hold --probe_steps 3 --probe_translation_step_m 0.000005 --probe_rotation_step_rad 0.0 --image_log_every 1 --max_logged_image_steps 3 --save_image_env_indices 15,61'
```

Why this is not a false positive:

- The success rows are post-step metrics from `sfp_tip_link`, not reward return.
- Signed depth clears the corrected full-depth threshold (`target_depth` is
  about `45.8 mm`, with `0.5 mm` depth tolerance).
- Lateral error is below `0.5 mm`.
- Semantic tip orientation is below `0.030 rad`.
- Module/body consistency is above `0.80`.
- The saved center, left, and right images are consistent with a seated SFP body
  rather than a lateral bypass.

Remaining limitation:

This is a seated reset/hold diagnostic. The next required step is to reproduce
the same strict state from a shallower near-gate start using a guarded insertion
servo, then transfer the successful final-window target/guard parameters into
the curriculum/reward loop.

## Superseded Shallow Success

This run was originally recorded as strict success under the old Isaac shallow
depth criterion. It is now superseded after comparing against the Gazebo SFP
port frames and cage geometry.

It should be treated as **shallow tip insertion**, not full SFP-to-NIC seating.
The official Gazebo SDF defines the SFP port entrance at `-0.0458 m` relative to
`sfp_port_*_link`, and the cage collision bodies are `0.04872 m` deep. Therefore
full Isaac semantic insertion should target about `45.8 mm` from the entrance
frame, plus any explicit outward entrance offset, not `8 mm`.

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
| old strict_success | `true` |
| corrected full-depth success | `false` |

## Evidence

- Metrics: `metrics.jsonl`, `strict_metrics.csv`, `strict_success.json`.
- Reproducibility: `run_config.json`, `command.txt`, `git_status.txt`, `git_diff.patch`.
- Visual sanity frames: `step_images/step_000023/env_0000_center_camera.png`, `env_0000_left_camera.png`, `env_0000_right_camera.png`.
- Videos: `center_camera_insertion.mp4`, `left_camera_insertion.mp4`, `right_camera_insertion.mp4`.

Command:

```bash
docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py --headless --enable_cameras --rendering_mode performance --task AIC-Task-v0 --num_envs 1 --seed 1 --device cuda:0 --output_dir aic/outputs/agentic_reward_curriculum_20260524_controller_contact/runs --run_name strict_candidate_x015_zneg024_ax60_images --episode_config_dir aic/outputs/agentic_reward_curriculum_20260524_final_window_probe/generated_episode_configs/20260524_finalwindow_env1_poseenvlog_084746_step155_ep2 --override_start_signed_depth_m 0.00652 --override_start_lateral_m 0.000137 --override_start_orientation_rotvec_world 0.015 0.0 -0.024 --derive_reset_position_from_orientation --episode_length_s 2.0 --near_gate_reset_max_iterations 120 --near_gate_reset_position_tolerance 0.0002 --near_gate_reset_orientation_tolerance 0.005 --approach_steps 0 --probe target_tip_stabilize --probe_steps 30 --probe_translation_step_m 0.00006 --target_tip_stabilize_goal_depth_m 0.008 --target_tip_stabilize_axial_step_m 0.00006 --target_tip_stabilize_lateral_step_m 0.00002 --target_tip_stabilize_orientation_gate_depth_m 1.0 --image_log_every 1 --max_logged_image_steps 40'
```

## Why This Is Not Full Insertion

The row is post-step geometry, not reward return, and it satisfies lateral,
orientation, and module-consistency gates at `s=8.056 mm`. However, the depth
target was wrong for full SFP seating. It only targets a shallow point near the
front of the cage.

The center, left, and right images at step 23 are consistent with the SFP module
entering the cage, not bypassing laterally, but they do not show the long
Gazebo-like insertion travel seen in the reference screenshots.

## Limitation

This is a focused final-window diagnostic shallow insertion, not a learned
ACT/SERL policy success over randomized starts and not full seating. The next
step is to rerun the guarded final-window search against the corrected full-depth
target.

## Validation

```bash
.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py
.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py
```

Latest result: `30 passed in 12.95s`.
