# Gazebo vs Isaac Cheatcode Comparison 2026-05-30

## Scope

This report starts the cheatcode-transfer branch of the insertion work. It compares the known successful Gazebo official-evaluation cheatcode rollout against the available Isaac cheatcode-style rollouts and current strict full-depth diagnostics. It does not claim Isaac success.

Strict success remains full-depth SFP insertion using post-step semantic geometry: `sfp_tip_link` depth near the episode target (`~45.8-46.9 mm`), low `r`, semantic-tip `theta < 0.030 rad`, `sfp_module_link` consistency, no lateral bypass, and center/left/right visual sanity.

## Artifacts

Comparison folder:

```text
outputs/agentic_reward_curriculum_20260529/cheatcode_compare_20260530/
```

Gazebo source:

```text
outputs/validation/reward_probe_20260510_031535/gazebo_actual_cheatcode_h264/
outputs/validation/reward_probe_20260510_031535/gazebo_actual_cheatcode_results/scoring.yaml
```

Isaac reused source pending a current rerun:

```text
outputs/analysis/isaac_cheatcode_policy_video_20260515_105551/videos/
outputs/analysis/isaac_cheatcode_policy_video_20260515_105551/summary.json
outputs/analysis/isaac_cheatcode_staged_slow_20260515_110326/summary.json
outputs/one_day_insertion_pipeline/cheatcode_policy_validation_20260518/runs/
```

The comparison folder contains copied videos, 1 FPS extracted frames, command/provenance notes, git status, and a relevant git diff patch.

## Video Comparison

| System | Video | Duration | Resolution | FPS | Frames | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Gazebo official cheatcode | `videos/gazebo_center_camera_h264.mp4` | `41.60 s` | `288x256` | `20` | `832` | scoring success |
| Isaac reused cheatcode | `videos/isaac_reuse_center_camera.mp4` | `6.05 s` | `224x224` | `20` | `121` | not strict success |

Gazebo scoring:

```text
total: 94.230113862879691
tier_3: Cable insertion successful.
duration: 25.32 seconds
contacts: No contact detected.
insertion force: No excessive force detected.
```

Selected visual checkpoints:

| State | Gazebo center frame | Isaac reused center frame | Observation |
| --- | --- | --- | --- |
| pre-contact / approach | `gazebo_frames/center/t_001.png` | `isaac_reuse_frames/center/t_001.png` | Gazebo begins as a long controlled approach; Isaac reused video is much shorter and not a full official-equivalent episode. |
| entrance alignment | `gazebo_frames/center/t_006.png` | `isaac_reuse_frames/center/t_002.png` | Gazebo maintains alignment before insertion. Isaac comparison is insufficient for full-depth behavior. |
| shallow insertion | `gazebo_frames/center/t_012.png` | `isaac_reuse_frames/center/t_003.png` | Gazebo visibly progresses inward. Isaac reused diagnostics remain old shallow/near-gate evidence. |
| mid insertion | `gazebo_frames/center/t_020.png` | `isaac_reuse_frames/center/t_004.png` | Gazebo continues axial seating. Isaac has no comparable deep insertion segment. |
| late insertion | `gazebo_frames/center/t_030.png` | `isaac_reuse_frames/center/t_005.png` | Gazebo is near seated. Isaac reused rollout is not evidence of full insertion. |
| full insertion | `gazebo_frames/center/t_042.png` | `isaac_reuse_frames/center/t_006.png` | Gazebo official eval is seated by scoring; Isaac reused visual is not strict success. |

## Isaac Metrics Snapshot

Old Isaac staged slow cheatcode-style run:

```text
outputs/analysis/isaac_cheatcode_staged_slow_20260515_110326/summary.json
```

Best old metric:

```text
step: 127
distance_m: 8.387 mm
orientation_error_rad: 0.0735
force_norm: 24.37
terminal_unit: 1.0
```

This was an old shallow target/reward condition, not corrected full-depth strict success.

Corrected full-depth audit from 2026-05-24:

```text
target depth: ~45.8 mm
full_depth_targettip_smoke_cameras best s: 21.36 mm, r: 0.240 mm, theta: 0.04646 rad, strict: false
full_depth_targettip_long_ax120 best s: 19.60 mm, r: 0.182 mm, theta: 0.04825 rad, strict: false
```

Recent no-guard v705/v706 policy training is also not success. v705 reached only shallow tip depth with module lag; v706 was stopped after the pivot to this cheatcode-transfer plan.

## Failure Classification

Current Isaac status is not a reward-only issue. The closest corrected-depth evidence shows:

```text
failure label: near_success_orientation_blocked + near_success_module_consistency_blocked
secondary label: controller_realization_mismatch
false-positive risk: tip-depth-only positive s without module/body consistency
```

Gazebo succeeds with an absolute pose cheatcode that continuously descends the gripper/TCP through a long insertion trajectory. Isaac has mostly been exercising clipped differential actions, target-tip guides, and retention/guard layers. Those layers can align `r/theta`, but they have not produced full module-following insertion.

## Immediate Gap

The reusable Gazebo visual evidence confirms that real full insertion is a long axial seating motion, not the old `8 mm` shallow condition. Isaac must therefore be compared against a current, full-length cheatcode-only rollout from a comparable near-gate start. The next run should be an Isaac teacher/cheatcode diagnostic with:

- strict full-depth checker unchanged,
- center/left/right high-quality videos saved separately,
- 1 FPS snapshots,
- post-step `s/r/theta`, module `s/r`, consistency gate/error, force/contact,
- command/config/git status/git diff in the run folder.

