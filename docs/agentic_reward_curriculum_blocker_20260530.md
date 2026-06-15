# Agentic Reward/Curriculum Blocker Report - 2026-05-30

Strict full SFP-to-NIC insertion has not been achieved under the existing strict checker. No run should be treated as a
success from reward, positive tip signed depth, or a visually ambiguous near-seat state.

## Strict Target

The active strict checker uses post-step semantic geometry:

- Full semantic tip depth near the episode target, currently about `45.8-46.9 mm`.
- Lateral error near `<= 0.5 mm`.
- Semantic tip orientation below `0.030 rad`.
- Offset-aware `sfp_module_link` consistency.
- No lateral bypass or impossible simulator artifact.
- Center/left/right visual sanity when claiming success.

## Closest Candidate

The closest preserved artifact is:

```text
outputs/agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_21-34-44_v865_v860_seed_repro_near_success_capture_video/
```

Best captured row:

| metric | value |
|---|---:|
| strict_success | false |
| tip s | `45.190 mm` |
| tip r | `0.240 mm` |
| tip theta | `0.03521 rad` |
| module s | `21.555 mm` |
| module r | `0.604 mm` |
| module consistency gate | `0.860` |
| remaining depth gap to `46.864 mm` | `1.674 mm` |
| remaining theta gap to `0.030 rad` | `0.00521 rad` |

Video evidence exists for this run:

- `env0000_center_full_episode_20fps_quality.mp4`
- `env0000_left_full_episode_20fps_quality.mp4`
- `env0000_right_full_episode_20fps_quality.mp4`

This is a near-success teacher/contact artifact, not strict insertion. It fails the orientation threshold and does not
have a reproduced held-out strict-success row.

## Experiments Attempted

| family | best evidence | outcome |
|---|---|---|
| Full-depth target correction | `docs/isaac_sfp_depth_audit_20260524.md` and later strict runs | old `8 mm` target was corrected; full target is now around `45.8-46.9 mm` |
| Guarded reward/curriculum policy training | v529/v530 and earlier guided runs | overfit to shallow/single reset or learned entrance hover; no strict success |
| Randomized reset curriculum | v850/v852 accepted reset validation | reset distribution fixed, but no no-guard/randomized policy achieved deep module-following insertion |
| Architecture/history ablations | v814 ResNet/history, v826 ConvNeXt/history, v855/v856 no-guard ConvNeXt/history | no strict success; improved representation did not solve contact/controller realization |
| Isaac teacher/cheatcode full-depth probes | v707-v865 family | best run v865 reached near-full tip depth with tight r, but theta stayed above threshold |
| Orientation axis/pose-hold sweeps | v872-v874 | pure/forced rotations could lower theta only slightly or collapsed r/module consistency |
| Two-body constrained rotation compensation | v875-v876 | preserved module consistency on the good corridor, but did not reduce theta |
| Collider transfer audit | v877-v879 | naive Gazebo-active SFP collider replacement regressed badly to `34.294 mm` tip depth and zero consistency |
| Expert extraction for HIL-SERL | `expert_trajectories_20260530/near_success_v865/` and `teacher_replay_v645_v648_v649/` | extracted labels are mostly false positives/near-success failures; no positive strict demonstrations to imitate |

## Latest Collider/Consistency Diagnosis

Offline SDF audit:

```text
outputs/agentic_reward_curriculum_20260529/collision_audits/sfp_nic_collision_geometry_20260530_v877/
```

Isaac-stage prim audit:

```text
outputs/agentic_reward_curriculum_20260529/collision_prim_audits/2026-05-30_22-21-59_v878_v870_shrunk_collision_prim_audit/
```

Key findings:

- Gazebo-active SFP body AABB is `13.750 x 47.300 x 8.452 mm`.
- NIC opening is `14.000 x 8.949 mm`, with `48.720 mm` cage depth.
- The closest Isaac teacher uses four shrunk body-shell colliders, not the full Gazebo-active SFP collider set.
- The strict consistency checker is offset-aware. It does not simply require `sfp_module_link` to reach the same absolute
  `45.8 mm` target as `sfp_tip_link`.
- Replacing the SFP collision with Gazebo-active boxes in v879 regressed to `s=34.294 mm`, `r=0.697 mm`,
  `theta=0.04005`, and consistency near zero under high contact.

## Blocker Classification

Primary blocker:

```text
controller/contact realization mismatch
```

Supporting evidence:

- Reward/curriculum changes can select near-gate and near-full-depth states, but do not create the final seated,
  module-following contact trajectory.
- The best guide can achieve lateral alignment and nearly full semantic tip depth only with a permissive shrunk
  body-shell collider setup.
- Final orientation correction of roughly `0.005 rad` cannot be realized without either over-inserting the tip,
  increasing lateral error, or collapsing module consistency.
- Restoring more Gazebo-like active SFP detail collisions blocks axial progress earlier rather than improving transfer.

Secondary blockers:

- Semantic-tip orientation residual near the final millimeter.
- Module/body consistency lags behind tip-depth progress.
- Contact force proxy remains large and insensitive in many deep-contact probes.

## Why HIL-SERL Is Blocked

HIL-SERL / imitation-heavy SERL needs positive or at least reliably near-success expert trajectories. The extracted
teacher data currently contains near-success failures and many tip-depth false positives. Training on those as positive
demonstrations would teach reward exploitation or entrance hover rather than strict full insertion.

## Single Next Recommended Code Change

Implement a diagnostic-only compliant seated-contact teacher mode that separates the final millimeter into two phases:

1. Use the current shrunk body-shell collider and target-module guide to reach the v865 corridor.
2. In the final window, switch from pure wrist pose increments to a bounded compliant module-following command:
   keep tip lateral fixed, keep module lateral within threshold, limit positive tip depth to the strict target, and
   solve only for the smallest wrist orientation/translation update that reduces semantic-tip theta.

The diagnostic should reject any command whose predicted or realized next-step state increases tip depth beyond the
target while module consistency drops. If that compliant final-window command cannot reduce theta below `0.030 rad`
from the v865 state, the remaining issue is likely the Isaac cable/contact asset or articulation model, not reward,
curriculum, HIL-SERL, or visual architecture.

## Stop Decision

Strict insertion was not achieved. The bounded reward/curriculum, no-guard training, randomized reset, architecture,
cheatcode transfer, collision-transfer, orientation compensation, and expert-extraction paths have been attempted or
blocked by the lack of a strict successful Isaac teacher. The current evidence supports stopping reward-only and
architecture-heavy tuning until the final-window controller/contact realization blocker is resolved.
