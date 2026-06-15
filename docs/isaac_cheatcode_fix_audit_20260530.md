# Isaac Cheatcode Fix Audit 2026-05-30

## Findings

The working Gazebo official-evaluation cheatcode and the current Isaac insertion path are not equivalent controllers.

Gazebo official baseline from upstream `main`:

```text
aic_example_policies/aic_example_policies/ros/CheatCode.py
```

It computes a gripper pose from the port and plug TFs, then uses absolute pose targets:

```text
target_z = port_z + z_offset - plug_tip_gripper_offset_z
z_offset starts at 0.2 m
approach: 100 steps at 0.05 s
descent: z_offset -= 0.0005 m every 0.05 s until -0.015 m
```

The local Gazebo cheatcode file has been modified relative to upstream:

```text
target_z = port_z + z_offset + plug_tip_gripper_offset_z
uses set_delta_pose_target instead of set_pose_target
adds scoring insertion-event early exit
adds handoff, settle, and minimum-jerk descent
```

The local official-teacher replay path is a separate, richer teacher:

```text
aic_teacher_official/aic_teacher_official/OfficialTeacherReplay.py
aic_teacher_official/aic_teacher_official/expert_generator/nominal_expert.py
```

Important teacher properties:

- Uses ground-truth TFs and port/plug geometry.
- Supports `AIC_OFFICIAL_TEACHER_ONLINE_CHEATCODE_INSERTION=true`.
- Computes guarded insertion depth from TFs when `cheatcode_z_mode=tf_depth`.
- Streams absolute `base_link` TCP exact-position targets for final insertion.
- Default expert generator appends a `70 mm` guarded insert segment at `0.0013 m/s`.
- Online teacher default insertion speed is millimeter-scale (`AIC_OFFICIAL_TEACHER_CHEATCODE_INSERTION_SPEED_MPS`, default `0.012` in the current local file).

Isaac guide/guard path:

```text
aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py
```

Relevant functions and flags:

- `_cheatcode_transform_guided_policy_action(...)`
- `_target_guided_policy_action(...)`
- `_apply_insertion_action_guard(...)`
- `--target_action_guide_mode`
- `--target_action_guide_step_size`
- `--target_action_guide_target_tip_axial_step_m`
- `--tcp_translation_action_clip`
- `--tcp_rotation_action_clip`
- `--insertion_action_guard*`
- `--target_success_*`

Recent Isaac policy/guide runs used differential action clips such as `30 um` translation per step, and many guard/recovery modes were layered on top. At `20 Hz`, `30 um/step` is only `0.6 mm/s`; crossing a corrected `~46 mm` insertion target from a 40 mm outside start needs a long, stable controller. The Gazebo teacher instead commands absolute target positions on a smooth long descent and lets the controller realize them.

## Depth Convention

The `8 mm` strict target was already corrected. The current full-depth target is the episode target, usually:

```text
sfp_port_*_link_entrance to target: ~45.8 mm
with Isaac entrance axis offset: ~46.7 mm
```

The `48.72 mm` cage depth is the collision cage length. The semantic port-frame insertion target is slightly shorter, and the strict checker now uses the episode target depth instead of the old shallow override.

## Why Gazebo Progresses Axially While Isaac Stalls

The evidence points to a controller/transfer mismatch before a learning architecture issue:

1. Gazebo official eval succeeds by issuing a long absolute pose descent through the port.
2. Isaac attempts have mostly used small differential action increments, wrist IK, and guide/guard overrides.
3. The Isaac guard can improve lateral/orientation alignment, but it often clamps or backs out axial motion when near the entrance.
4. Corrected full-depth Isaac diagnostics reached shallow to mid depth but failed orientation/module consistency before seating.
5. The local Gazebo cheatcode itself has diverged from upstream, including a z-offset sign change and delta-pose control path, so it should not be assumed to represent the official upstream convention without explicit validation.

Most likely blocker:

```text
controller_realization_mismatch + action scaling/clipping + frame/sign convention risk
```

Secondary blockers:

```text
orientation residual near final seating
module/body consistency lag
possible contact/collision sensitivity at deep insertion
```

## Smallest Testable Change

Do not start with more reward-only training. The first bounded test should be a config-driven Isaac teacher/cheatcode diagnostic that mimics Gazebo teacher semantics as closely as possible:

1. Use semantic `sfp_tip_link` and `sfp_module_link` diagnostics.
2. Keep wrist IK, but command a Gazebo-style long insertion target through gripper/TCP control.
3. Use millimeter-scale absolute/servo target progression rather than only `5-30 um` actor deltas.
4. Keep strict checker unchanged.
5. Save center/left/right videos, snapshots, metrics, command/config, git status/diff.
6. Compare against Gazebo frames and classify: axial progress, lateral bypass, orientation residual, module lag, force/contact, or realization mismatch.

Suggested first Isaac probe:

```text
train_v707_isaac_teacher_cheatcode_compare_full_depth
```

Use the existing teacher-collection command builder as the starting point:

```bash
.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/build_isaac_teacher_collection_command.py \
  --run-name train_v707_isaac_teacher_cheatcode_compare_full_depth \
  --episode-config-dir <validated 40/10 or near-gate config> \
  --steps 1400 \
  --num-envs 1 \
  --target-depth-m 0.046864 \
  --translation-clip-m 0.0012 \
  --axial-step-m 0.0012 \
  --video-crf 16 \
  --video-fps 20
```

If this probe makes deep visual progress but fails `theta/module`, tune teacher lateral/orientation compensation. If it cannot move axially despite millimeter-scale commands, stop reward tuning and debug IK/contact/action realization.

## HIL-SERL Implication

HIL-SERL/imitation-heavy SERL should wait until Isaac can generate useful teacher trajectories. The teacher trajectories must include executed actions, semantic post-step metrics, and failure labels. Failed teacher rollouts remain valuable, but they should not be used as positive demonstrations unless they satisfy strict full insertion or are explicitly labeled as near-success/failure.

## Addendum - Collider and Module-Consistency Audit v877-v879

Outputs:

- Offline SDF/report audit:
  `outputs/agentic_reward_curriculum_20260529/collision_audits/sfp_nic_collision_geometry_20260530_v877/`
- Isaac-stage prim audit:
  `outputs/agentic_reward_curriculum_20260529/collision_prim_audits/2026-05-30_22-21-59_v878_v870_shrunk_collision_prim_audit/`
- Gazebo-active SFP collider probe:
  `outputs/agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_22-23-04_v879_v865_gazebo_active_sfp_colliders_metrics/`

The offline SDF audit confirms:

- Gazebo-active SFP body AABB: `13.750 x 47.300 x 8.452 mm`.
- NIC port opening: `14.000 mm` wide, `8.949 mm` high.
- NIC cage depth: `48.720 mm`.
- The v865/v876 near-success teacher used runtime `shrunk_body_boxes`: four long body-shell cubes, not the full Gazebo-active collider set.

The Isaac-stage prim audit of the near-success reset confirms the best preserved state is already near the semantic tip target:

| body | signed depth | lateral | target depth | axial error to target |
|---|---:|---:|---:|---:|
| `sfp_tip_link` | `45.464 mm` | `0.414 mm` | `45.800 mm` | `0.335 mm` |
| `sfp_module_link` | `21.829 mm` | `0.796 mm` | `45.800 mm` | `23.971 mm` |

The `sfp_module_link` row should not be interpreted as requiring the module frame to reach the same absolute
`45.8 mm` target as the tip. The strict success / consistency logic is offset-aware: it stores the reset-time axial
gap between the rewarded tip and the consistency body, then checks the consistency body against
`target_depth - reference_gap`. The module-geometry diagnostic's `target_depth` field is useful for raw body geometry,
but not by itself a proof that the strict checker is wrong.

v879 tested the most direct collider-transfer hypothesis by replacing the Isaac SFP converted mesh with Gazebo-active
SFP module boxes instead of the four shrunk body boxes. It regressed badly:

| run | collider mode | best s | best r | theta | consistency | decision |
|---|---|---:|---:|---:|---:|---|
| v865 | shrunk body boxes | `45.190 mm` | `0.240 mm` | `0.03521` | `0.860` | closest near-success, strict false |
| v879 | Gazebo-active SFP boxes | `34.294 mm` | `0.697 mm` | `0.04005` | `~0.000` | reject |

v879 also showed high contact proxy force and no module-following progress, so naively adding all Gazebo-active SFP
detail boxes is not the smallest viable fix. The remaining blocker is still best classified as
`controller/contact realization mismatch`: the teacher can put the semantic tip close to the final target only with a
permissive body-shell collision model, but it cannot reduce the final `~0.005 rad` semantic-tip theta or maintain a
deep module-consistent seated state under the stricter active-collider contact.
