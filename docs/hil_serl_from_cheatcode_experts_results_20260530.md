# HIL-SERL From Cheatcode Experts Results 2026-05-30

This report records the current cheatcode-transfer / HIL-SERL state. It is not a success claim.

## Strict Success Status

No held-out run currently has `strict_success=true` under the existing strict checker. The closest preserved Isaac artifact remains the v865 near-success capture:

| run | strict_success | best s mm | best r mm | theta rad | module s mm | module r mm | consistency | label |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `v865_v860_seed_repro_near_success_capture_video` | false | 45.190 | 0.240 | 0.03521 | 21.555 | 0.604 | 0.860 | near_full_orientation_blocked |

Strict failure reason: semantic-tip orientation is still above the 0.030 rad threshold, so this is a near-success teacher artifact only.

## Preserved Diagnostic Artifact

Run folder:

`outputs/agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_21-34-44_v865_v860_seed_repro_near_success_capture_video/`

Preserved evidence:

- `command.txt`
- `run_config.json`
- `git_status.txt`
- `git_diff.patch`
- `metrics.jsonl`
- `wrist_contact_summary.json`
- `env0000_center_full_episode_20fps_quality.mp4`
- `env0000_left_full_episode_20fps_quality.mp4`
- `env0000_right_full_episode_20fps_quality.mp4`
- `step_images/`

The run stopped early because `--stop_on_near_success_capture` hit the configured near-success thresholds. That stop condition is only for preserving evidence and does not modify strict success.

## Residual-Target Extraction

Script:

`aic_utils/aic_isaac/scripts/extract_privileged_residual_targets.py`

Patch made today:

- `post_step_module_geometry` is now supported in addition to legacy `post_step_all_body_insertion_geometry.sfp_module_link`.
- This fixes module `s/r` extraction for contact-realization diagnostic logs.

Validation:

`python -m py_compile aic_utils/aic_isaac/scripts/extract_privileged_residual_targets.py`

`python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_randomized_near_gate_curriculum.py`

Result: `32 passed`.

### v865 Extraction

Output:

`outputs/agentic_reward_curriculum_20260529/expert_trajectories_20260530/near_success_v865/`

Files:

- `privileged_residual_targets.csv`
- `privileged_residual_targets_summary.json`

Summary:

| label | rows |
| --- | ---: |
| `tip_depth_false_positive` | 177 |
| `centered_high_theta_module_near` | 17 |
| `near_full_orientation_blocked` | 6 |

Best centered row:

- step 199
- `s=45.190 mm`
- `r=0.240 mm`
- `theta=0.03521 rad`
- `module_s=21.555 mm`
- `module_r=0.604 mm`
- gap to full target `1.674 mm`

Recommendation from the extractor: test bounded final-window orientation/contact micro-recovery before more reward-only training.

## Existing Teacher Replay Audit

Existing replay-oriented teacher runs were scanned before launching any new high-disk collection.

| run | strict rows | best s mm | r mm | theta rad | module s mm | module r mm | consistency | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `v645_teacher_v642_deep_module_probe_env4` | 0 | 45.81 | 0.202 | 0.04220 | 22.17 | 0.705 | 0.708 | reject for imitation; theta/consistency not strict |
| `v648_teacher_v647_recordedvel_final_orientation_probe` | 0 | 47.35 | 1.268 | 0.03302 | 23.71 | 1.882 | 0.002 | reject; tip-depth false positive / bypass |
| `v649_teacher_v647_microbackoff_orientation_probe` | 0 | 48.64 | 0.395 | 0.03740 | 25.00 | 0.749 | 0.000 | reject; tip-depth false positive / contact |

Combined residual extraction:

`outputs/agentic_reward_curriculum_20260529/expert_trajectories_20260530/teacher_replay_v645_v648_v649/`

Label counts:

| label | rows |
| --- | ---: |
| `tip_depth_false_positive` | 999 |
| `contact_spike` | 954 |
| `lateral_bypass` | 26 |
| `near_full_orientation_blocked` | 10 |
| `near_full_orientation_contact_blocked` | 6 |
| `prejump_realization_mismatch` | 5 |

Decision: do not use these replays as positive imitation data. They contain useful negative/failure evidence, but using them as expert data would teach tip-depth exploitation and contact-heavy artifacts.

## HIL-SERL Implication

The current online SERL expert hook expects a LeRobot-style dataset under `--expert_dataset_root` with parquet columns `observation.state`, `action`, `episode_index`, and `frame_index`. The residual CSVs are not a full imitation dataset because contact-realization diagnostics do not log policy observation vectors or images.

Safe next HIL-SERL path:

1. Keep v865 as a labeled near-success teacher artifact, not as a strict success.
2. Fix the Isaac cheatcode/teacher final-window orientation behavior first.
3. Collect a new teacher dataset only after the controller produces rows with low theta and nonzero module consistency.
4. Convert only successful or near-success rows into LeRobot/BC data; keep false-positive/contact rows as negative audit data.

## Next Controller Change

The bounded next code path should target final-window orientation without losing module following:

- activate only after `s > 45 mm`, `r < 0.5 mm`, and module consistency is high;
- predict/reject orientation commands that lower module consistency or increase module lateral error;
- keep axial motion clamped or zero during orientation trim;
- retain wrist IK plus explicit tip-motion compensation;
- stop capture when strict theta is reached or when module consistency degrades.

If this still cannot reduce theta below 0.030 rad while preserving module consistency, the blocker is controller/contact realization rather than reward-only policy learning.
