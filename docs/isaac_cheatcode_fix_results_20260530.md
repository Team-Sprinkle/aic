# Isaac Cheatcode Fix Results - 2026-05-30

This report tracks the Isaac cheatcode transfer experiments against the strict full-depth SFP-to-NIC insertion checker. Success is not claimed unless `strict_success=true` under the existing post-step checker and center/left/right visual sanity passes.

## Code changes tested

- Full-rate video logging is now the default for Isaac teacher collection command generation (`--image-log-every=1`), with center/left/right videos saved as separate high-quality files.
- Reward preset flag parsing now respects explicit negated flags such as `--no-terminate_on_target_success`.
- `cheatcode_transform` guide translation now routes through `_root_translation_delta_to_policy_frame(...)`, fixing an Isaac IK XY sign double-inversion that caused catastrophic lateral divergence in v710.
- `cheatcode_transform` can now use separate semantic bodies for translation and orientation. Translation can optionally switch from the semantic tip to a deep/module body after a configured tip-depth activation.
- The deep/module translation switch is latched for the rest of the episode after activation, so transient tip-depth retreat does not disable the module-following phase.

Validation after the latest code change:

```text
.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py
.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_randomized_near_gate_curriculum.py
32 passed
```

## Candidate rollouts

| Run | Change | Strict success | Best post-step tip s | r | theta | module s | Module/consistency | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| v709 | `target_tip_stabilize`, 40/10 held-out, full-rate video | false | 0.393 mm | 0.370 mm | 0.0606 rad | -23.2 mm | consistency false | reject: lateral bypass/module lag |
| v710 | `cheatcode_transform` before translation sign fix | false | 1.96 mm | 286 mm | 0.127 rad | -21.6 mm | consistency false | reject: IK sign/frame bug |
| v711 | sign-fixed `cheatcode_transform`, rotation sign -1 | false | near entrance | 0.674 mm | 0.242 rad | -23.4 mm | consistency false | reject: orientation diverges |
| v712 | sign-fixed `cheatcode_transform`, rotation sign +1 | false | 25.13 mm | 0.164 mm | 0.0450 rad | 1.50 mm | consistency false | promote partial: deep tip progress |
| v713 | v712 + final-orientation axial trickle | false | 27.05 mm | 0.152 mm | 0.0468 rad | 3.42 mm | consistency false | promote partial: best depth so far |
| v714 | module-body translation from start | false | -0.38 mm | 1.43 mm | 0.0490 rad | -24.0 mm | consistency false | reject: cannot enter from far start |
| v715 | faster final-orientation axial step | false | 25.00 mm | 0.178 mm | 0.0455 rad | 1.36 mm | consistency false, 20.8 mm short | reject: no depth gain over v713 |
| v716 | staged module translation after 20 mm tip depth, no latch | false | 20.67 mm | 0.227 mm | 0.0541 rad | -2.96 mm | consistency false | reject: switch deactivated after retreat |
| v717 | staged module translation latched after 18 mm tip depth | false | 27.33 mm | 0.426 mm | 0.0439 rad | 3.70 mm | consistency false, 18.5 mm short | reject: latched module phase still retreats |
| v718 | v717 but rotation disabled globally | false | 2.55 mm | 0.320 mm | 0.369 rad | -19.5 mm | consistency false | reject: approach needs rotation |
| v719 | v717 with rotation zeroed only after deep/module stage activates | false | 27.42 mm | 0.280 mm | 0.0452 rad | 3.78 mm | consistency false, 18.4 mm short | reject: best partial, still retreats |
| v720 | v719 with smaller axial/translation micro-steps | false | -8.63 mm | 1.07 mm | 0.0495 rad | -32.3 mm | consistency false | reject: stalled outside gate |
| v721 | v719 with fast approach and deep-only 0.35 mm axial step | false | 28.50 mm | 0.290 mm | 0.0420 rad | 4.87 mm | consistency false, 17.3 mm short | reject: best tip depth, still module-lag/retreat |
| v722 | v721 with deep-only 0.20 mm axial step, metrics only | false | 27.77 mm | 0.301 mm | 0.0443 rad | 4.13 mm | consistency false, 18.0 mm short | reject: smaller deep step plateaus below v721 |
| v724 | standalone wrist/contact probe from 27.8 mm start, target-tip goal full depth | false | 25.18 mm | 0.223 mm | 0.0379 rad | near entrance | consistency ~0, force ~66k final | reject: direct full-depth tip command hits contact/retreat |
| v725 | v724 with `body_sdf_collision` disabled | false | 35.40 mm | 0.387 mm | 0.0393 rad | 11.77 mm | consistency ~0, force ~67k final | diagnostic: disabling body mesh helps but still not strict |
| v726 | v724 with broad module collision disable regex | failed before rollout | n/a | n/a | n/a | n/a | reset backend failure | reject: broad collision toggle invalidates reset |
| v727 | 40/10 cheatcode with `body_sdf_collision` replaced by SDF body boxes | false | 15.87 mm | 0.228 mm | 0.0472 rad | -7.77 mm | consistency false, 29.9 mm short | reject: box replacement stalls earlier than v721 |
| v737 | 40/10 cheatcode with Gazebo-active SDF box replacement | false | 18.11 mm | 0.754 mm | 0.1183 rad | -5.38 mm | metrics saved; checkpoint save failed due full Docker storage | reject: active boxes worsen theta/contact and do not recover module following |
| v738 | 40/10 cheatcode, converted mesh retained, SFP/NIC cage contact offset `0.1 mm`, final checkpoint disabled | false | 23.95 mm | 0.446 mm | 0.0429 rad | 0.32 mm | force proxy much lower, still 21.85 mm short | reject: lower contact offset helps force but not full insertion |

## Current diagnosis

The strongest Isaac cheatcode variant can align laterally and insert the semantic tip about 28.5 mm, but it does not achieve full target depth around 45.8-46.9 mm. The post-step checker remains strict-success false because the tip is still about 17.3 mm short at the best frame, theta remains above the 0.030 rad strict threshold, and `sfp_module_link` consistency remains false. v722 confirmed that reducing only the deep-stage axial step to 0.20 mm does not solve this; it plateaued below v721 at 27.77 mm.

The v724 standalone wrist/contact probe started from a synthetic partial-depth state near the v721/v722 ceiling and directly commanded tiny target-tip motion toward full depth. It did not recover deeper insertion: best tip depth was only 25.18 mm, final tip depth was 22.95 mm, final module depth was -0.68 mm, consistency was effectively zero, and the wrist contact force proxy rose to about 66k. This is evidence that the current Isaac controller/contact setup resists full-depth module-following insertion around this depth, even without the SERL policy or reward in the loop.

The v725 collision ablation disabled only `/World/envs/env_0/Robot/cable/sfp_module/sfp_module_link/collisions/body_sdf_collision`. That improved best/final tip depth to 35.40 mm and module depth to 11.77 mm, so the converted body SDF collision mesh is part of the blocker. It still did not reach strict success: the run remained about 10.4 mm short, theta stayed at 0.0393 rad, module lateral error was 1.25 mm, consistency remained near zero, and the force proxy stayed around 67k. A broad attempt to disable all module collisions (v726) failed during reset, so that path is not a valid success criterion.

The v727 real 40/10 cheatcode rollout tested the non-destructive replacement path, `--replace_sfp_body_sdf_collision_with_sdf_boxes`. It did not transfer the v725 diagnostic improvement into the full guide rollout. Instead, it stalled before deep-stage activation at 15.87 mm best tip depth, with module depth still negative and theta around 0.047 rad. This means the next fix should not simply enable body-box replacement globally; it needs a more careful Gazebo-vs-Isaac collision/contact audit, likely including exact collider extents/poses and port/cage collision interaction.

The v737 rollout then tested the new `--replace_sfp_module_sdf_collision_with_active_sdf_boxes` flag, which disables the
converted `body_sdf_collision` mesh and creates the SFP module box colliders that remain active after applying the
`sfp_sc_cable/model.sdf` removals. It reached step 420 and preserved metrics, but failed while writing
`checkpoint_latest.pt` because Docker storage was full. The best post-step row was worse than v721: tip `s=18.11 mm`,
tip `r=0.754 mm`, theta `0.1183 rad`, module `s=-5.38 mm`, module `r=2.20 mm`, remaining axial error `27.69 mm`,
strict success false/not present. Therefore the active SDF box replacement is rejected as a teacher fix.

The v738 contact-parameter probe kept the original converted collision mesh, tuned matching `body_sdf_collision` and
NIC cage prims to `contactOffset=0.0001 m`, `restOffset=0.0`, and disabled final checkpoint writing with
`--no-save_final_checkpoint`. It finished cleanly and preserved metrics, but did not beat v721. Best row: tip
`s=23.95 mm`, tip `r=0.446 mm`, theta `0.0429 rad`, module `s=0.32 mm`, module `r=0.87 mm`, remaining axial error
`21.85 mm`, force proxy `2.18`. The reduced force proxy shows contact parameters matter, but the module still does not
follow to full depth.

An offline SDF collision audit was added and run:

```text
aic_utils/aic_isaac/scripts/audit_sfp_nic_collision_geometry.py
outputs/agentic_reward_curriculum_20260529/collision_audits/sfp_nic_collision_geometry_audit.json
outputs/agentic_reward_curriculum_20260529/collision_audits/sfp_nic_collision_geometry_audit.md
```

The audit parses `aic_assets/models/SFP Module/model.sdf`, `aic_assets/models/sfp_sc_cable/model.sdf`, and
`aic_assets/models/NIC Card/model.sdf`, then compares the Gazebo-active SFP colliders with the NIC cage dimensions.
It found:

- Gazebo-active SFP body AABB: `13.750 x 47.300 x 8.452 mm`.
- NIC port opening: `14.000 x 8.949 mm`.
- NIC cage depth: `48.720 mm`.
- `sfp_sc_cable/model.sdf` removes 16 raw SFP module port/latch/head colliders; the Gazebo-active set is not equivalent
  to either the current body-only runtime replacement or the all-box replacement.
- v727 created only four `body_collider_box*` replacement boxes.

This makes the contact sensitivity concrete: the body shell has only about `0.25 mm` nominal width clearance and about
`0.50 mm` nominal height clearance, while the body length is only about `1.42 mm` shorter than cage depth. Small
Isaac/Gazebo collider pose, rotation, margin, or contact-model differences can therefore prevent full-depth insertion
even if the semantic guide is correct.

The current blocker is not the old 8 mm success-definition issue; the strict target depth in the metrics is about 45.8 mm. The remaining blocker is controller/teacher behavior: the tip can be driven into the entrance, but the module/body does not follow to the calibrated seated-depth relationship. A module-body guide from the start is too hard for the 40 mm / 10 mm start, so the next tested fix is staged tip approach followed by a latched module-following phase.

## Next test

The staged module translation variants did not produce strict insertion. v717/v719 show that the controller can briefly reach about 27.4 mm tip depth with low lateral error, but then retreats while module consistency remains false. v718 shows that rotation cannot be disabled globally because approach fails without it.

Current best partial run:

```text
outputs/agentic_reward_curriculum_20260529/policy_train_runs_tmp/2026-05-30_14-46-14_train_v721_isaac_cheatcode_transform_deepmodule_deepax035_zero_deep_rot_heldout40x10_fullfps_noreset
```

Host copy while `/data1` is full:

```text
/tmp/aic_agentic_reward_curriculum_20260529/policy_train_runs/2026-05-30_14-46-14_train_v721_isaac_cheatcode_transform_deepmodule_deepax035_zero_deep_rot_heldout40x10_fullfps_noreset
```

Best v721 post-step metrics:

```json
{
  "s": 0.02849951758980751,
  "target_depth": 0.04579966142773628,
  "axial_error": 0.017300143837928772,
  "r": 0.0002895357320085168,
  "theta": 0.0420185886323452,
  "module_s": 0.004866708070039749,
  "module_r": 0.0006150726694613695,
  "consistency_gate": 0.0,
  "strict_success": false
}
```

Artifact note: `/data1` became full during v719. The incomplete v719 replay temp file was removed; metrics, checkpoint, step images, summaries, and manually encoded center/left/right videos are preserved. v720/v721/v722 were routed through `outputs/agentic_reward_curriculum_20260529/policy_train_runs_tmp`, a symlink to host `/tmp`, and replay saving was disabled. v721 still stopped early because Docker's internal overlay filesystem was also full while saving step images. Its copied host `/tmp` run preserves metrics through step 307 and manually encoded partial center/left/right videos. v722 was metrics-only to avoid further image/video storage pressure.

Additional scratch artifacts copied to host `/tmp`:

```text
/tmp/aic_agentic_reward_curriculum_20260529/policy_train_runs/2026-05-30_14-46-14_train_v721_isaac_cheatcode_transform_deepmodule_deepax035_zero_deep_rot_heldout40x10_fullfps_noreset
/tmp/aic_agentic_reward_curriculum_20260529/policy_train_runs/2026-05-30_14-58-24_train_v722_isaac_cheatcode_transform_deepmodule_deepax020_zero_deep_rot_metrics_only
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_15-33-06_v724_partial28_targettip_goalfull_metrics
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_15-44-54_v725_partial28_targettip_goalfull_disable_body_sdf
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_15-55-03_v726_partial28_targettip_goalfull_disable_module_collisions
/tmp/aic_agentic_reward_curriculum_20260529/policy_train_runs/2026-05-30_15-58-10_train_v727_isaac_cheatcode_transform_body_sdf_boxes_deepax035_metrics_only
```

## Next test

Do not continue long image-heavy runs until disk space is available or new runs are explicitly routed to a mounted scratch path that is visible inside Docker. The next smallest controller test is not smaller global steps; v720 shows that stalls before the gate. v721 shows that a deep-only axial limiter improves tip depth slightly but does not solve module-following insertion, and v722 shows that reducing the deep-stage axial step further is not sufficient.

A better next test is either:

1. a no-image metrics-only sweep around the v721 controller to isolate the contact/module-lag limit without filling disk, or
2. a controller/contact diagnostic at the v721 best frame that commands module-body axial motion while measuring realized tip/module deltas and force.

```text
--target_action_guide_deep_translation_body_name sfp_module_link
--target_action_guide_deep_translation_activation_depth_m 0.018
--target_action_guide_zero_rotation_when_deep_translation_active
--target_action_guide_step_size 0.0012
--target_action_guide_axial_step_size 0.0012
deep-only axial clip: {0.0002, 0.00035, 0.0005}
```

Expected outcome: determine whether the retreat after 28.5 mm is caused by excessive axial step/contact, or whether the Isaac wrist IK/contact model cannot advance the module body beyond this partial insertion without a different controller/contact representation.

## Addendum - Isaac-stage collision prim audit v731-v734

I added an Isaac-stage prim logger:

```text
aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/audit_isaac_collision_prims.py
```

It instantiates the AIC Isaac task, optionally applies the runtime SDF-body-box replacement, resets the same held-out
40 mm axial / 10 mm lateral episode, then writes command, config, git status/diff, semantic reset geometry, USD
collision prim transforms, world AABBs where available, and transform-derived collision slab dimensions.

Artifact folders:

```text
/tmp/aic_agentic_reward_curriculum_20260529/collision_prim_audits/2026-05-30_16-44-02_v733_baseline_obb_heldout40x10
/tmp/aic_agentic_reward_curriculum_20260529/collision_prim_audits/2026-05-30_16-44-37_v734_body_sdf_boxes_obb_heldout40x10
```

Both reset to the intended held-out pose:

| body | signed depth `s` | lateral `r` | target depth | axial error |
|---|---:|---:|---:|---:|
| `sfp_tip_link` | `-40.051 mm` | `10.049 mm` | `45.800 mm` | `85.851 mm` |
| `sfp_module_link` | `-63.686 mm` | `9.693 mm` | `45.800 mm` | `109.486 mm` |

Baseline v733 confirms the converted Isaac SFP collision mesh is enabled:

| prim | enabled | measured size |
|---|---:|---|
| `body_sdf_collision` | true | world AABB `13.751 x 47.704 x 21.912 mm` |

Replacement v734 confirms the runtime patch disables that mesh and creates the four Gazebo SDF body boxes:

| prim | enabled | transform-derived OBB |
|---|---:|---|
| `body_sdf_collision` | false | old mesh AABB still present but disabled |
| `runtime_sdf_body_collider_box` | true | `13.750 x 47.300 x 0.789 mm` |
| `runtime_sdf_body_collider_box_001` | true | `13.750 x 47.300 x 0.178 mm` |
| `runtime_sdf_body_collider_box_002` | true | `0.574 x 47.300 x 8.450 mm` |
| `runtime_sdf_body_collider_box_003` | true | `0.811 x 47.300 x 8.450 mm` |

The same audit shows the Isaac NIC cage collision slabs are enabled and match the half-extents encoded in the converted
scene:

| prim | enabled | transform-derived slab size |
|---|---:|---|
| `cage_p0_bottom` / `cage_p1_bottom` | true | `8.112 x 24.360 x 0.527 mm` |
| `cage_p0_top` / `cage_p1_top` | true | `8.112 x 24.360 x 0.461 mm` |
| `cage_p0_wall_left` / `cage_p1_wall_left` | true | `0.549 x 24.360 x 5.542 mm` |
| `cage_p0_wall_right` / `cage_p1_wall_right` | true | `0.562 x 24.360 x 5.561 mm` |
| `cage_p0_front` / `cage_p1_front` | true | `8.112 x 0.125 x 6.849 mm` |

This is consistent with the MuJoCo/Gazebo-style box half-size `0.02436 m` entries, but it also explains why the
visual/evaluator target depth must not be inferred from any one slab row. The official SDF audit still reports full cage
depth `48.720 mm`, while the strict Isaac semantic target for this held-out episode is `45.800 mm`.

Current interpretation:

- The old 8 mm target-depth issue is fixed; current strict metrics use a full-depth target near `45.8 mm`.
- The Isaac reset for 40/10 is correct and pre-contact.
- The runtime replacement does what it claims at the USD prim level, but v727 showed it does not improve the full guide
  rollout. It disables the converted mesh and adds only the four SFP body boxes; it does not yet reproduce the complete
  Gazebo-active SFP collision set and contact behavior.
- The remaining blocker is therefore a collision/contact-controller realization mismatch around a very tight mechanical
  fit: roughly `0.25 mm` nominal width clearance and `0.50 mm` nominal height clearance.

Next smallest fix:

1. Stop using target-tip-only partial-depth probes as candidate fixes. v739 shows that they preserve the same
   false-positive risk as the old 8 mm criterion: the semantic tip can remain partially inserted while the module body
   fails to follow.
2. Add a module-body target diagnostic that commands wrist IK toward `sfp_module_link` axial progress while still
   logging semantic tip `s/r/theta`, module `s/r`, realized body motion, and force.
3. Promote only if post-step `s`, `r`, `theta`, and `sfp_module_link` consistency improve without lateral bypass.

## Addendum - Partial-depth contact-offset probe v739

Run:

```text
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_17-10-22_v739_partial28_targettip_goalfull_contact_offset_0p1mm_metrics
```

This diagnostic started near the v721 partial-depth regime with tip `s ~= 27.8 mm`, low lateral error, and
`contactOffset=0.0001 m` on `body_sdf_collision` plus the NIC cage. It used the existing `target_tip_stabilize` wrist
IK probe toward the full `45.800 mm` target.

Best post-step row:

| metric | value |
|---|---:|
| best step | `5` |
| tip `s` | `25.500 mm` |
| tip `r` | `0.272 mm` |
| theta | `0.0378 rad` |
| target depth | `45.800 mm` |
| remaining axial error | `20.300 mm` |
| module `s` | `1.866 mm` |
| module `r` | `0.628 mm` |

## Addendum - Module-body target diagnostics v821-v824

I added three off-by-default flags to the standalone wrist/contact diagnostic:

```text
--target_module_stabilize_tip_lateral_step_m
--target_module_stabilize_tip_lateral_gate_m
--target_module_stabilize_tip_theta_gate_rad
```

These preserve the old `target_module_stabilize` behavior by default, but allow a diagnostic mode where module-body
axial motion is combined with semantic-tip lateral correction and blocked when tip `r/theta` are outside gates. This is
intended to test whether a module-following teacher can be realized without relying on tip-depth-only false positives.

Validation:

```text
.pixi/envs/default/bin/python -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py
```

Artifacts are under host scratch:

```text
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_19-30-28_v821_partial28_targetmodule_goal22p3_metrics
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_19-32-00_v822_partial28_lowtheta_targetmodule_goal22p3_step50um_metrics
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_19-34-06_v823_partial28_targetmodule_tipguard_step100um_metrics
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_19-35-38_v824_partial28_targetmodule_tipguard_absik_metrics
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_19-38-21_v825_partial28_targetmodule_tipguard_highsolver_metrics
```

Results:

| run | diagnostic | strict | best tip `s` | best tip `r` | theta | best module `s` | module `r` | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v821 | relative IK, target `sfp_module_link` to `22.3 mm` module depth | false | `28.28 mm` | `1.51 mm` | `0.0597` | `4.67 mm` | `2.52 mm` | reject: immediate high-contact backout |
| v822 | v821 with low-theta reset rotation and `50 um` steps | false | `26.68 mm` | `5.32 mm` | `0.0347` | `3.04 mm` | `4.62 mm` | reject: orientation repair creates lateral error |
| v823 | relative IK, module axial + tip lateral guard, `100 um` steps | false | `25.93 mm` | `0.21 mm` | `0.0418` | `2.30 mm` | `0.95 mm` | reject: lateral held, but module/tip retreat |
| v824 | v823 in absolute IK target-pose mode | false | `25.87 mm` | `191.31 mm` | `0.0630` | `2.27 mm` | `189.82 mm` | reject: absolute IK causes lateral realization failure |
| v825 | v823 with solver iterations `128/64` | false | `25.36 mm` | `0.55 mm` | `0.0275` | `1.72 mm` | `1.18 mm` | reject: higher solver iterations increase contact instability |

The important negative result is v823: once semantic-tip lateral error is guarded, the controller no longer makes
module-depth progress from the partial-depth ceiling. The final row retreats to tip `s=20.12 mm` and module
`s=-3.50 mm`. v824 shows that switching this diagnostic to absolute wrist IK does not fix the issue; it creates a large
lateral failure. v825 shows that simply increasing robot articulation solver iterations does not fix it either; the run
retreated to tip `s=0.54 mm`, module `s=-22.80 mm`, and force proxy `~5.8e5`. These traces should not be used as
HIL-SERL expert data.

## Addendum - SFP clearance-collider isolation v827-v829

The module-target diagnostics above suggested the SFP/NIC collision pair is still the limiting path, so I tested the
existing off-by-default SFP clearance-box replacement:

```text
--replace_sfp_body_sdf_collision_with_clearance_box
--sfp_clearance_box_size_m 0.0132 0.0470 0.0080
--sfp_clearance_box_translation_m 0.0 0.001 0.0
```

This disables the converted `body_sdf_collision` mesh and adds a single smaller box around the SFP body. It is a
diagnostic collision modification, not a physical success criterion.

Artifacts:

```text
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_19-46-35_v827_partial28_targetmodule_tipguard_sfpclearancebox_metrics
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_19-47-59_v828_partial28_targetmodule_tipguard_sfpclearancebox_long_metrics
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_19-50-25_v829_partial28_targetmodule_tipguard_sfpclearancebox_tinyoffset_long_metrics
```

Results:

| run | change | strict | best tip `s` | best `r` | theta | best module `s` | module `r` | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v827 | clearance box, 120 probe steps | false | `32.00 mm` | `0.671 mm` | `0.0441` | `8.37 mm` | `1.04 mm` | diagnostic improvement, still not strict |
| v828 | v827 extended to 360 steps | false | `33.57 mm` | `0.648 mm` | `0.0440` | `9.94 mm` | `1.07 mm` | plateaus short of full depth |
| v829 | v828 plus tiny `5 um` contact offset on runtime box and cage | false | `42.38 mm` | `29.90 mm` | `0.0784` | `18.80 mm` | `28.72 mm` | reject: lateral bypass false positive |

v827/v828 are the first controller/contact diagnostics in this series to improve tip and module depth together while
keeping lateral error near the port. They still miss strict success by a wide margin: best v828 is about `12.2 mm`
short of the `45.8 mm` run target and theta remains above the strict `0.030 rad` threshold. v829 shows that making
contact margins too permissive can produce much deeper raw tip/module `s`, but only by exiting the valid insertion
corridor. That is exactly the false-positive mode the strict checker is meant to reject.

Current implication: the next code path should not train on v827-v829 traces. Instead, build a physically grounded SFP
collision representation between the converted mesh and the overly permissive clearance box, then validate it with the
same module-target and visual checks.

## Addendum - Shrunk Gazebo body-box shell v830-v832

I added a more physically bounded diagnostic replacement:

```text
--replace_sfp_body_sdf_collision_with_shrunk_sdf_boxes
--sfp_shrunk_box_margin_m X 0.0 X
```

This keeps the four Gazebo `body_collider_box*` poses and shapes, but shrinks each box in width/height by a controlled
margin. It sits between the original converted mesh / exact Gazebo body boxes and the overly permissive single
clearance box. Defaults are unchanged and the mode is diagnostic-only.

Validation:

```text
.pixi/envs/default/bin/python -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py
.pixi/envs/default/bin/python -m pytest -q \
  aic_utils/aic_isaac/test/test_insertion_reward_geometry.py \
  aic_utils/aic_isaac/test/test_randomized_near_gate_curriculum.py
32 passed
```

Artifacts:

```text
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_19-54-24_v830_partial28_targetmodule_tipguard_shrunkbodyboxes_metrics
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_19-56-09_v831_partial28_targetmodule_tipguard_shrunkbodyboxes_margin075um_metrics
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_19-57-56_v832_partial28_targetmodule_tipguard_shrunkbodyboxes_margin225um_metrics
```

Results:

| run | shrink margin x/z | strict | best tip `s` | best `r` | theta | best module `s` | module `r` | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| v831 | `0.075 mm` | false | `32.91 mm` | `0.606 mm` | `0.0376` | `9.28 mm` | `0.974 mm` | reject: below v830 |
| v830 | `0.150 mm` | false | `36.63 mm` | `0.608 mm` | `0.0393` | `13.00 mm` | `1.046 mm` | best bounded diagnostic |
| v832 | `0.225 mm` | false | `33.32 mm` | `0.624 mm` | `0.0379` | `9.68 mm` | `0.956 mm` | reject: below v830 |

v830 is now the best physically bounded contact diagnostic: it improves both tip and module depth without the large
lateral bypass seen in v829. It is still not strict success. It remains about `9.2 mm` short of the `45.8 mm` target,
theta is above `0.030 rad`, and module consistency is still effectively false. The final row also retreats to tip
`s=27.86 mm`, module `s=4.23 mm`.

Next recommended experiment: keep v830's `0.150 mm` shrunk multi-box shell and add a final-window orientation/lateral
trim or friction/contact-material audit. Do not use v830 as HIL-SERL expert data until a post-step run reaches strict
module-consistent full depth.

## Addendum - Bounded module-target orientation trim v833-v836

I added optional orientation trim controls to the diagnostic `target_module_stabilize` probe:

```text
--target_module_stabilize_orientation_step_rad
--target_module_stabilize_orientation_lateral_gate_m
--target_module_stabilize_orientation_activation_depth_m
--target_module_stabilize_orientation_error_threshold_rad
--target_module_stabilize_rotation_compensation_clip_m
```

Defaults keep the old behavior. The trim uses the same semantic-tip orientation probe and rotation-induced tip-sweep
compensation as `target_tip_stabilize`, but only when the tip is already deep and laterally gated.

Validation:

```text
.pixi/envs/default/bin/python -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py
.pixi/envs/default/bin/python -m pytest -q \
  aic_utils/aic_isaac/test/test_insertion_reward_geometry.py \
  aic_utils/aic_isaac/test/test_randomized_near_gate_curriculum.py
32 passed
```

Artifacts:

```text
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_20-02-14_v833_shrunk150_module_tipguard_orienttrim1mrad_metrics
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_20-04-09_v834_shrunk150_seed830_orienttrim1mrad_metrics
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_20-06-21_v835_shrunk150_seed830_orienttrim0p5mrad_metrics
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_20-08-52_v836_shrunk150_seed830_orienttrim1mrad_lateral500um_metrics
```

Results:

| run | change | strict | best useful row |
|---|---|---:|---|
| v833 | 1 mrad trim, different seed | false | trim never activated; later lateral divergence, reject |
| v834 | v830 seed, 1 mrad trim, `0.65 mm` lateral gate | false | step 128: tip `s=49.53 mm`, `r=0.625 mm`, theta `0.0368`, module `s=25.89 mm`, module `r=1.27 mm`, consistency `0.072` |
| v835 | v830 seed, 0.5 mrad trim | false | best strict-ish: tip `s=39.65 mm`, `r=0.517 mm`, theta `0.0319`, module `s=16.01 mm`, module `r=1.26 mm` |
| v836 | v830 seed, 1 mrad trim, stricter `0.50 mm` lateral gate | false | best: tip `s=37.87 mm`, `r=0.346 mm`, theta `0.0342`, module `s=14.24 mm`, module `r=1.09 mm` |

Interpretation: v834 is the first diagnostic in this series to reach full/over-full tip depth while preserving roughly
sub-millimeter lateral error, but it is not strict success. It still fails lateral threshold, theta threshold, and
module consistency, and final state retreats. Tightening the gate or reducing trim step preserves lateral/theta better
but loses deep axial/module progress. This is a useful controller/contact clue, not a teacher trajectory.

Next experiment should combine v834's depth-enabling behavior with stricter module-lateral/semantic consistency control,
or inspect contact material/friction on the v830 shrunk shell. Do not train on v834 as success data.

## Addendum - Secondary module-lateral trim v837-v838

I added an optional secondary module-lateral trim to `target_module_stabilize`:

```text
--target_module_stabilize_secondary_module_lateral_step_m
--target_module_stabilize_secondary_module_lateral_activation_depth_m
--target_module_stabilize_secondary_module_lateral_threshold_m
```

It defaults to zero and is intended to add a very small module-body lateral correction after the semantic tip is already
deep, without replacing the tip centerline correction.

Validation:

```text
.pixi/envs/default/bin/python -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py
.pixi/envs/default/bin/python -m pytest -q \
  aic_utils/aic_isaac/test/test_insertion_reward_geometry.py \
  aic_utils/aic_isaac/test/test_randomized_near_gate_curriculum.py
32 passed
```

Artifacts:

```text
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_20-13-44_v837_shrunk150_seed830_orienttrim1mrad_modulelat20um_metrics
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_20-16-06_v838_shrunk150_seed830_orienttrim1mrad_modulelat10um_metrics
```

Results:

| run | change | strict | best useful row |
|---|---|---:|---|
| v837 | v834 + `20 um` secondary module-lateral trim | false | best strict-ish row: tip `s=42.35 mm`, `r=0.435 mm`, theta `0.0328`, module `s=18.71 mm`, module `r=1.14 mm`, consistency `0.107` |
| v838 | v834 + `10 um` secondary module-lateral trim | false | raw depth rose to tip `s=65.71 mm`, module `s=42.33 mm`, but theta `0.150` and consistency near zero; best strict-ish row only tip `s=35.38 mm` |

v837 improves the consistency/centering tradeoff relative to v836, but loses v834's near-full depth. v838 is rejected
as another depth-only false-positive mode. No secondary module-lateral trim tested so far achieves strict success.

Current best bounded diagnostics:

- Deepest near-strict row: v834 step 128, tip `s=49.53 mm`, `r=0.625 mm`, theta `0.0368`, module `s=25.89 mm`,
  consistency `0.072`, strict false.
- Best module-lateral/consistency tradeoff with reasonable depth: v837 step 180, tip `s=42.35 mm`, `r=0.435 mm`,
  theta `0.0328`, module `s=18.71 mm`, consistency `0.107`, strict false.

Next recommended experiment: inspect/tune contact material or friction for the v830 shrunk shell and NIC cage, because
the controller-side lateral/orientation trims show a stable tradeoff but do not satisfy all strict gates together.
| module theta | `1.563 rad` |
| force proxy | `66667.5` |
| strict success | `false` |

Final row: tip `s=22.540 mm`, module `s=-1.089 mm`, force proxy `66869.4`, strict success `false`.

Decision: reject. Lower contact offset plus target-tip stabilization does not recover the v721 depth ceiling and still
leaves the module body far behind the tip. The next diagnostic must target and measure module-body progress directly.

## Addendum - Module-body target probes v740-v742

I added diagnostic-only support to `wrist_contact_realization.py`:

```text
--probe target_module_stabilize
--target_module_stabilize_body
--target_module_stabilize_goal_depth_m
--target_module_stabilize_axial_step_m
--target_module_stabilize_lateral_step_m
--fix_isaac_ik_z_sign / --no-fix_isaac_ik_z_sign
```

Validation:

```text
python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py
```

Runs:

```text
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_17-16-01_v740_partial28_targetmodule_goalfull_contact_offset_0p1mm_metrics
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_17-18-14_v741_partial28_targetmodule_goalfull_zsign_contact_offset_0p1mm_metrics
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_17-19-54_v742_partial28_targetmodule_goalfull_ax035_contact_offset_0p1mm_metrics
```

Summary:

| run | change | best tip `s` | final tip `s` | best module `s` | final module `s` | force proxy | strict | decision |
|---|---|---:|---:|---:|---:|---:|---|---|
| v740 | target module, 80 um axial step | `27.842 mm` | `27.842 mm` | `4.207 mm` | `4.207 mm` | `67105.8` | false | reject |
| v741 | v740 plus root-frame Z sign flip | `27.851 mm` | `27.851 mm` | `4.216 mm` | `4.216 mm` | `66647.3` | false | reject |
| v742 | target module, 350 um axial step | `23.891 mm` | `16.646 mm` | `0.258 mm` | `-6.963 mm` | `67779.1` | false | reject |

Detailed v740/v741 rows show the requested module world delta is inward, but the early realized module motion is
outward along the cage axis, and the state carries a very large contact-force proxy. Increasing the commanded axial
step in v742 worsens this into clear ejection rather than overcoming static contact.

Decision: reject all three as teacher fixes. The next smallest useful test is not more axial authority. It is a
reset/geometry/contact consistency diagnostic that starts from shallower, physically plausible module-following states
and checks whether the partial-depth reset itself is already in an overlapped/contact-ejection configuration.

## Addendum - Zero-action settle and shallow approach probes v743-v745

Zero-action settle runs:

```text
/tmp/aic_agentic_reward_curriculum_20260529/reset_settle_runs/2026-05-30_17-22-08_v743_settle_tip10mm_lat03_zero_action
/tmp/aic_agentic_reward_curriculum_20260529/reset_settle_runs/2026-05-30_17-23-14_v744_settle_tip28mm_lat03_zero_action
```

Summary:

| run | reset tip `s` | first-step tip `s` | final tip `s` | reset module `s` | final module `s` | note |
|---|---:|---:|---:|---:|---:|---|
| v743, tip 10 mm | `10.024 mm` | `10.030 mm` | `12.015 mm` | `-13.611 mm` | `-11.617 mm` | shallow positive reset is dynamically stable enough |
| v744, tip 27.8 mm | `27.807 mm` | `25.756 mm` | `25.205 mm` | `4.172 mm` | `1.572 mm` | deep partial reset immediately relaxes outward |

The force-proxy channel is high in both cases, including the benign 10 mm settle, so the current contact sensor should
not be used alone as the failure classifier. The semantic geometry is more reliable here.

Approach-from-shallow run:

```text
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_17-24-58_v745_tip10_targettip_goalfull_ax035_metrics
```

v745 starts from tip `s=10 mm` and runs target-tip stabilization toward the full `45.800 mm` target. Best row: tip
`s=26.592 mm`, `r=0.306 mm`, theta `0.0385 rad`, module `s=2.959 mm`, module `r=0.630 mm`, strict success `false`.
Final row retreats to tip `s=15.400 mm`, theta `0.0708 rad`, module `s=-8.190 mm`.

Decision: reject as a full-insertion teacher, but keep as evidence. The Isaac wrist/guide path reproducibly reaches a
`26-28 mm` tip-depth ceiling from both a 40/10 rollout and a shallow 10 mm reset, then fails to carry the module body
deeper. This is a controller/contact realization ceiling, not a random reset artifact.

## Addendum - Gazebo-speed slow insertion probe v746

The official Gazebo teacher tests document a final insertion speed near `0.0013 m/s` and a long `0.070 / 0.0013 s`
final insertion segment. I tested whether the Isaac partial-depth ceiling was caused by the faster v745 diagnostic
(`0.00035 m` per 50 ms, about `0.007 m/s`).

Run:

```text
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_17-28-18_v746_tip10_targettip_goalfull_gazebo_speed_ax065_metrics
```

Configuration: tip `s=10 mm` reset, target-tip stabilization toward the full `45.800 mm` target, axial step
`0.000065 m` per 50 ms, 620 probe steps.

Result: best tip `s=22.198 mm`, `r=0.308 mm`, theta `0.0428 rad`, module `s=-1.430 mm`, strict success `false`.
Final tip `s=20.984 mm`, module `s=-2.641 mm`.

Decision: reject. Matching the Gazebo nominal insertion speed alone makes Isaac shallower, not deeper. The remaining
Gazebo/Isaac gap is more likely the exact-position pinned-XY insertion target, lateral servo/z-gate behavior, or the
controller/contact backend, not just axial velocity.

## Addendum - Absolute-pose IK diagnostics v747-v748

I added diagnostic-only absolute IK support to `wrist_contact_realization.py`:

```text
--absolute_ik_target_pose
--absolute_ik_pin_reset_orientation
```

This switches the Isaac Lab Differential IK controller to `use_relative_mode=False` only for the standalone diagnostic,
sets the action scale to `1.0`, and sends root-frame absolute wrist target poses. Defaults for training/eval remain
relative IK.

Validation:

```text
python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py
```

Runs:

```text
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_17-33-18_v747_tip10_targettip_goalfull_absoluteik_ax035_metrics
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_17-35-27_v748_tip10_targettip_goalfull_absoluteik_pinquat_ax035_metrics
```

Summary:

| run | mode | best tip `s` | best `r` | theta | best module `s` | final tip `s` | strict |
|---|---|---:|---:|---:|---:|---:|---|
| v747 | absolute IK, current orientation | `20.312 mm` | `0.324 mm` | `0.0473 rad` | `-3.311 mm` | `12.600 mm` | false |
| v748 | absolute IK, pinned reset orientation | `22.648 mm` | `0.244 mm` | `0.0418 rad` | `-0.982 mm` | `14.245 mm` | false |

Decision: reject both. Absolute-pose IK in the standalone Isaac diagnostic does not reproduce Gazebo insertion and is
worse than the best relative-IK guide ceiling. The gap is not solved by simply switching the controller command mode.

## Addendum - Gazebo/Isaac trace comparison and pinned-wrist probes v749-v752

I added a reusable trace comparison script:

```text
scripts/compare_gazebo_isaac_mid_insertion.py
```

Validation:

```text
python -m py_compile scripts/compare_gazebo_isaac_mid_insertion.py \
  aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py
```

Comparison outputs:

```text
/tmp/aic_agentic_reward_curriculum_20260529/cheatcode_trace_compare/2026-05-30_17-39-40_v749_gazebo_nominal_vs_isaac_middepth
/tmp/aic_agentic_reward_curriculum_20260529/cheatcode_trace_compare/2026-05-30_17-43-23_v752_gazebo_nominal_vs_pinned_isaac
```

The successful Gazebo replay used:

```text
outputs/dev/expert_debug/nominal_clean_align_v4_20260504T192000Z/replay_attempts/attempt_000002_candidate_01
```

Gazebo final-insertion command shape:

| metric | value |
|---|---:|
| final insertion commands | `65` |
| target-Z travel | `20.735 mm` |
| mean target-Z step | `0.324 mm` |
| max final XY target drift | `0.000 mm` |
| max final orientation target drift | `0.000000 rad` |
| success event time | `22.06 s` |

This confirms Gazebo's successful final phase is an absolute Cartesian sequence with pinned TCP XY and pinned
orientation while monotonically descending.

I then added a diagnostic-only Isaac probe:

```text
--probe pinned_wrist_axis_descent
--pinned_wrist_axis_descent_step_m
--pinned_wrist_axis_descent_distance_m
```

Runs:

```text
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_17-40-56_v750_tip10_pinned_wrist_axis_descent_absik_pinquat_gazebostep_metrics
/tmp/aic_agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_17-42-20_v751_tip10_pinned_wrist_axis_descent_relik_gazebostep_metrics
```

Summary:

| run | mode | best tip `s` | best `r` | theta | best module `s` | final tip `s` | strict | decision |
|---|---|---:|---:|---:|---:|---:|---|---|
| v750 | pinned wrist, absolute IK, pinned quat | `28.116 mm` | `0.326 mm` | `0.0375 rad` | `4.482 mm` | `16.566 mm` | false | reject |
| v751 | pinned wrist, relative IK | `26.015 mm` | `197.165 mm` | `0.4378 rad` | `4.595 mm` | `-663.164 mm` | false | reject |

Decision: reject both as full-insertion teachers. v750 is useful evidence because it reproduces the `~28 mm` ceiling
from a Gazebo-shaped pinned trajectory, but the module still only reaches `4.48 mm` and the rollout ejects by the end.
v751 shows the same pinned path through relative IK is unstable and creates a large lateral sweep.

Updated recommendation: do not train HIL-SERL from these Isaac teacher traces yet. The Isaac teacher still lacks a
module-following final insertion trajectory. The next bounded fix should target the module/asset/contact representation
or an explicit module-body constrained reset/trajectory source, not reward-only tuning or larger axial action.

## Addendum - Full-depth reset settle probe v753

I tested whether Isaac can directly reset to a nominal full-depth semantic-tip state and remain strict-stable under
zero action:

```text
/tmp/aic_agentic_reward_curriculum_20260529/reset_settle_runs/2026-05-30_17-44-41_v753_settle_tip45p8mm_lat03_zero_action
```

Output summary:

```text
/tmp/aic_agentic_reward_curriculum_20260529/reset_settle_runs/2026-05-30_17-44-41_v753_settle_tip45p8mm_lat03_zero_action/summary_metrics.json
```

| row | tip `s` | tip `r` | tip theta | module `s` | module `r` | module theta | consistency | force proxy | strict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| reset | `45.826 mm` | `0.323 mm` | `0.0356 rad` | `22.191 mm` | `0.788 mm` | `1.5594 rad` | `0.856` | `74654.5` | false |
| first post-step | `49.395 mm` | `0.830 mm` | `0.0261 rad` | `25.753 mm` | `1.393 mm` | `1.5687 rad` | `0.0784` | `68830.8` | false |
| best tip row | `50.579 mm` | `0.662 mm` | `0.0199 rad` | n/a | n/a | n/a | `0.0260` | n/a | false |
| final | `67.112 mm` | `2.775 mm` | `0.1052 rad` | `43.592 mm` | `4.322 mm` | `1.5815 rad` | `~0` | `73990.1` | false |

Interpretation: Isaac can place the semantic tip near or beyond the full-depth threshold, but the module body is not
in a strict-valid pose. The module starts about `23.6 mm` behind the tip, with about `1.56 rad` module orientation error,
then settles into a high-contact, low-consistency state even with zero commanded motion. This rules out using
tip-depth-only full-depth resets as expert trajectories and strengthens the current blocker: the immediate issue is
module/body representability and contact consistency, not a missing reward term.

## Addendum - SFP collision conversion audit and replacement probes v754-v763

I audited the source Gazebo SDF assets and Isaac-stage collision prims:

```text
/tmp/aic_agentic_reward_curriculum_20260529/collision_audits/v754_sdf_collision_audit
/tmp/aic_agentic_reward_curriculum_20260529/collision_prim_audits/2026-05-30_17-49-47_v756_default_heldout40x10_collision_prims
/tmp/aic_agentic_reward_curriculum_20260529/collision_prim_audits/2026-05-30_17-50-52_v757_active_sdf_box_replacement_collision_prims
```

Findings:

- Local `SFP Module/model.sdf`, `NIC Card/model.sdf`, and `sfp_sc_cable/model.sdf` match upstream `intrinsic-dev/aic`
  `main` exactly.
- Gazebo SDF dimensions are plausible: active SFP body shell AABB is `13.750 x 47.300 x 8.452 mm`; NIC opening is
  `14.000 x 8.949 mm`, cage depth `48.720 mm`.
- Isaac default has enabled `/sfp_module_link/collisions/body_sdf_collision` as a mesh with AABB about
  `13.75 x 47.70 x 21.91 mm`, much thicker than the Gazebo body shell.

I added diagnostic-only collision replacement flags to the standalone wrist-contact probe:

```text
--replace_sfp_body_sdf_collision_with_sdf_boxes
--replace_sfp_module_sdf_collision_with_active_sdf_boxes
```

Validation:

```text
.pixi/envs/default/bin/python -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py
```

Results:

| run | test | best tip `s` | best `r` | theta | consistency | final tip `s` | strict | decision |
|---|---|---:|---:|---:|---:|---:|---|---|
| v753 | full-depth reset, default mesh, zero action | `50.579 mm` | `0.662 mm` | `0.0199` | `0.026` | `67.112 mm` | false | reject: ejects/over-inserts |
| v758 | full-depth reset, body mesh disabled, zero action | `49.717 mm` | `0.359 mm` | `0.0359` | `0.075` | `49.864 mm` | false | diagnostic only |
| v760 | full-depth reset, body mesh -> 4 SDF body boxes, zero action | `48.144 mm` | `0.337 mm` | `0.0375` | `0.297` | `48.249 mm` | false | closer but not strict |
| v761 | full-depth reset, 4 SDF body boxes, gripper pose hold | `47.965 mm` | `0.390 mm` | `0.0392` | `0.313` | `47.390 mm` | false | closer but orientation/consistency fail |
| v750 | 10 mm start, pinned Gazebo-shaped descent, default mesh | `28.116 mm` | `0.326 mm` | `0.0375` | `~0` | `16.566 mm` | false | prior ceiling |
| v762 | same descent, 4 SDF body boxes | `16.098 mm` | `0.281 mm` | `0.0371` | `0` | `16.742 mm` | false | worse |
| v763 | same descent, Gazebo-active module boxes | `16.227 mm` | `0.274 mm` | `0.0370` | `0` | `17.488 mm` | false | worse |

Interpretation: the default converted Isaac collision mesh is too thick and contributes to full-depth instability, but
naively replacing it with Gazebo SDF boxes changes contact enough to block the 40/10 pinned descent earlier. The best
current evidence points to an Isaac collision/contact realization mismatch, not a reward or architecture bottleneck.

Contact-offset follow-up:

| run | tuned prims | matched | final `s/r/theta` | consistency | strict |
|---|---|---:|---|---:|---|
| v764 | `body_sdf_collision` | 1 | `36.991 mm / 15.588 mm / 0.1642` | `~0` | false |
| v765 | `body_sdf_collision`, `cage_p0_*` | 6 | `46.395 mm / 22.765 mm / 0.2101` | `~0` | false |

Both used `contact_offset=0.00002 m`, `rest_offset=0.0`; both were rejected. Lower contact offsets do not by themselves
make the full-depth reset stable.

## Addendum - High-solver full-depth representability sweep v766-v775

I tested whether the full-depth instability is primarily a solver-stiffness issue by enabling high articulation solver
iterations and external-force application every physics iteration:

```text
AIC_ISAAC_ENABLE_EXTERNAL_FORCES_EVERY_ITERATION=1
AIC_ISAAC_SOLVER_POSITION_ITERATIONS=64
AIC_ISAAC_SOLVER_VELOCITY_ITERATIONS=32
```

The sweep used zero commanded tip motion from direct full-depth semantic-tip resets. It is a representability diagnostic,
not a policy success test.

| run | variant | best tip `s` | best `r` | theta | consistency | force proxy | final `s/r/theta` | strict |
|---|---|---:|---:|---:|---:|---:|---|---|
| v766 | `s=45.8 mm`, `r=0.3 mm` | `46.592 mm` | `0.550 mm` | `0.03390` | `0.734` | `271717` | `49.531 / 8.034 / 0.06766` | false |
| v769 | `s=45.8 mm`, `r=0.3 mm`, rotvec `+0.002 x` | `42.963 mm` | `3.764 mm` | `0.04251` | `0.034` | `282709` | `45.783 / 10.394 / 0.17875` | false |
| v770 | `s=45.8 mm`, `r=0.1 mm` | `49.179 mm` | `0.123 mm` | `0.02847` | `0.145` | `275512` | `46.472 / 15.938 / 0.13902` | false |
| v771 | `s=45.8 mm`, `r=0.5 mm` | `51.392 mm` | `0.943 mm` | `0.02870` | `0.0039` | `275567` | `55.925 / 8.653 / 0.17340` | false |
| v772 | `s=45.5 mm`, `r=0.3 mm` | `47.927 mm` | `1.674 mm` | `0.04201` | `0.0816` | `274616` | `49.229 / 3.474 / 0.12527` | false |
| v773 | `s=46.0 mm`, `r=0.3 mm` | `38.593 mm` | `0.455 mm` | `0.04045` | `0.00016` | `268975` | `8.397 / 13.255 / 0.35676` | false |
| v774 | `s=45.8 mm`, `r=0.3 mm`, rotvec `-0.001 x` | `49.857 mm` | `1.064 mm` | `0.03209` | `0.037` | `271893` | `47.672 / 8.481 / 0.13842` | false |
| v775 | `s=45.8 mm`, `r=0.3 mm`, rotvec `+0.001 y` | `49.728 mm` | `0.399 mm` | `0.02579` | `0.084` | `272934` | `48.584 / 4.666 / 0.07208` | false |

Closest but invalid case: v770 briefly satisfies tip-only depth, lateral, and orientation (`s=49.179 mm`, `r=0.123 mm`,
`theta=0.02847`), but module consistency is only `0.145` and the force proxy is `275k`. It then ejects laterally. This
is a tip-depth false positive and must not be used as a success or expert trajectory.

Conclusion: higher solver iterations reduce neither the high-contact artifact nor the module/body mismatch enough to
produce a strict full-depth Isaac state. The current blocker remains **Isaac SFP/NIC contact and module-body realization
at full depth**, not reward, curriculum, or model architecture.

Additional bounded checks:

| run | variant | best tip `s` | best `r` | theta | consistency | force proxy | strict |
|---|---|---:|---:|---:|---:|---:|---|
| v776 | `s=45.2 mm`, `r=0.1 mm` | `42.276 mm` | `0.155 mm` | `0.02323` | `0.123` | `275293` | false |
| v777 | `s=45.4 mm`, `r=0.1 mm` | `45.358 mm` | `5.109 mm` | `0.11079` | `0.163` | `291747` | false |
| v778 | `s=45.6 mm`, `r=0.1 mm` | `33.898 mm` | `0.191 mm` | `0.03810` | `~0` | `273253` | false |
| v779 | `s=45.6 mm`, `r=0.1 mm`, rotvec `+0.001 y` | `48.252 mm` | `1.210 mm` | `0.02800` | `0.352` | `275455` | false |
| v780 | `s=45.6 mm`, `r=0.1 mm`, rotvec `-0.001 x` | `47.754 mm` | `1.679 mm` | `0.03450` | `0.115` | `269709` | false |
| v781 | `s=45.8 mm`, `r=0.1 mm`, rotvec `+0.0005 y` | `46.201 mm` | `3.016 mm` | `0.04287` | `0.330` | `275644` | false |
| v782 | Gazebo body boxes, `5 um` contact offset | `48.933 mm` | `4.627 mm` | `0.03080` | `0.0046` | `271307` | false |
| v783 | Gazebo-active module boxes, `5 um` contact offset | `49.374 mm` | `1.220 mm` | `0.01687` | `0.074` | `279843` | false |

The SDF-box replacement with tiny contact offsets also fails. It confirms that the converted default mesh is suspicious,
but simply swapping in Gazebo boxes at runtime is not enough to reproduce Gazebo's stable full-depth contact behavior in
Isaac.

## Addendum - Official NVIDIA asset-pack comparison v784-v785

I downloaded the official Isaac asset pack referenced by upstream `aic_utils/aic_isaac/README.md`:

```text
https://developer.nvidia.com/downloads/Omniverse/learning/Events/Hackathons/Intrinsic_assets.zip
```

Checksum comparison:

```text
official zip sha256: 24f6c75d7d3f09599382557ae8307171becc7c42d6ad9a84371f8ba437fa1a39
official aic_unified_robot_cable_sdf.usd: bded9bc72152f16d33a5d2dbdb900d82ad451ed83e82e77de616dab0d2e67c5a
local active aic_unified_robot_cable_sdf.usd: 9c953d4776dba0cc8b2fed7942330f7b551eb7aeb2eef365c951b2479c5a673d
local aic_unified_robot_cable_sdf.usd.bak: bded9bc72152f16d33a5d2dbdb900d82ad451ed83e82e77de616dab0d2e67c5a
```

So the local `.bak` matches the official robot USD, while the active local `.usd` has been edited. I added a
config-driven override:

```text
AIC_ISAAC_ROBOT_USD_PATH=/path/to/aic_unified_robot_cable_sdf.usd
```

This preserves the current default path and lets diagnostics test the official asset without overwriting local files.

Official-asset collision audit:

```text
/tmp/aic_agentic_reward_curriculum_20260529/collision_prim_audits/2026-05-30_18-27-52_v784_official_robot_usd_collision_prims
```

The official robot USD still has the enabled SFP module `body_sdf_collision` mesh with AABB about
`13.751 x 47.704 x 21.912 mm`, the same problematic thickness seen with the local active USD.

Official-asset full-depth reset probe:

```text
/tmp/aic_agentic_reward_curriculum_20260529/reset_settle_runs/2026-05-30_18-29-00_v785_official_robot_usd_settle_tip45p8_lat03_solver64x32
```

| run | first-step tip `s` | `r` | theta | consistency | force proxy | strict |
|---|---:|---:|---:|---:|---:|---|
| v785 | `49.573 mm` | `0.575 mm` | `0.03037` | `0.075` | `273184` | false |

Conclusion: the local active USD differs from the official asset, but the full-depth blocker reproduces with the
official robot USD. The issue is therefore not just an accidental local camera/asset edit; it is a Gazebo-to-Isaac
physics/collision mismatch in the official Isaac robot/cable asset path.

## Addendum - Single clearance-box isolation v786

I added another diagnostic-only replacement mode:

```text
--replace_sfp_body_sdf_collision_with_clearance_box
--sfp_clearance_box_size_m 0.0135 0.0470 0.0082
--sfp_clearance_box_translation_m 0.0 0.001 0.0
```

This disables the converted `body_sdf_collision` mesh and replaces it with one configurable box approximating the SFP
envelope, then tunes the runtime box and cage contact offsets to `5 um`.

Result:

```text
/tmp/aic_agentic_reward_curriculum_20260529/reset_settle_runs/2026-05-30_18-31-30_v786_clearance_box_tinyoffset_settle_tip45p8_lat03_solver64x32
```

| run | best tip `s` | best `r` | theta | consistency | force proxy | strict |
|---|---:|---:|---:|---:|---:|---|
| v786 | `50.299 mm` | `0.860 mm` | `0.08207` | `0.0175` | `275670` | false |

This replacement also fails. The blocker is broader than a single thick converted mesh; the Isaac contact realization for
the SFP module and NIC cage does not currently produce a stable, module-consistent full-depth seated state.

## Addendum - Contact-pair isolation v787-v794

I isolated the SFP body and NIC cage collisions by disabling selected collision prims at runtime under the same
high-solver full-depth reset diagnostic. These runs are not valid insertion successes because they alter physical
collisions; they identify which contact pair blocks representability.

| run | disabled collisions | best tip `s` | best `r` | theta | consistency | force proxy | strict |
|---|---|---:|---:|---:|---:|---:|---|
| v787 | SFP `body_sdf_collision` | `49.734 mm` | `0.478 mm` | `0.03632` | `0.080` | `273311` | false |
| v788 | NIC `cage_p0_*` | `45.953 mm` | `0.639 mm` | `0.03671` | `0.924` | `273228` | false |
| v789 | SFP body + NIC cage | `45.953 mm` | `0.639 mm` | `0.03671` | `0.924` | `273227` | false |
| v790 | all NIC collisions | `45.953 mm` | `0.639 mm` | `0.03671` | `0.924` | `273228` | false |
| v791 | NIC cage, reset `r=0.1 mm` | `45.876 mm` | `0.468 mm` | `0.03666` | `0.956` | `273162` | false |
| v792 | NIC cage, reset `r=0.0 mm` | `45.906 mm` | `0.482 mm` | `0.03683` | `0.960` | `273227` | false |
| v793 | NIC cage, reset `r=0.1 mm`, rotvec `+0.001 y` | `45.905 mm` | `0.745 mm` | `0.03655` | `0.988` | `273052` | false |
| v794 | NIC cage, reset `r=0.1 mm`, rotvec `-0.001 x` | `45.933 mm` | `0.495 mm` | `0.03704` | `0.960` | `273206` | false |

Interpretation:

- Disabling only the SFP body collision does not recover module consistency.
- Disabling the NIC cage collisions immediately recovers module consistency (`~0.92-0.99`) and full-depth axial geometry,
  while the remaining miss is mostly semantic-tip orientation around `0.037 rad` and small lateral drift.
- Disabling all NIC collisions gives the same result as disabling the cage only, so the cage collision group is the
  relevant NIC-side blocker.

This narrows the blocker: **the NIC cage collision representation blocks stable full-depth module consistency in Isaac**.
The next physical fix should rebuild or patch the NIC cage collision representation, not keep tuning reward/training.

## Addendum - NIC cage p0 SDF-box replacement v802-v806

I added a diagnostic-only NIC cage replacement path:

```text
--replace_nic_cage_p0_with_sdf_boxes
--nic_card_sdf aic/aic_assets/models/NIC Card/model.sdf
```

The first two attempts (v802-v804) failed before reset because the SDF collision names contained hyphens that are invalid
in USD prim paths. I fixed the diagnostic by sanitizing generated prim names. The successful run is:

```text
/tmp/aic_agentic_reward_curriculum_20260529/reset_settle_runs/2026-05-30_18-50-27_v806_replace_nic_cage_p0_sdf_boxes_tip45p8_lat03_solver64x32
```

It disabled five `cage_p0_*` mesh collisions and created five runtime SDF-derived cube colliders.

| run | best tip `s` | best `r` | theta | consistency | strict |
|---|---:|---:|---:|---:|---|
| v806 | `45.913 mm` | `0.629 mm` | `0.03671` | `0.945` | false |

The replacement behaves like disabling the cage meshes: module consistency recovers, but strict success still fails on
lateral error and semantic-tip orientation. This is useful but not a physical success. It suggests the converted
`cage_p0_*` mesh contact is the immediate cause of module-consistency collapse, while the SDF-box replacement is either
too permissive, misregistered, or still missing the final orientation/centering needed for strict seating.

## Addendum - NIC cage registration and aligned-cube probes v807-v813

I extended `audit_isaac_collision_prims.py` so it can apply the NIC cage replacement modes and report cage-collider
centers relative to the evaluator entrance and insertion axis.

Validation:

```text
.pixi/envs/default/bin/python -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/audit_isaac_collision_prims.py \
  aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py

docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p -m pytest -q aic/aic_utils/aic_isaac/test/test_insertion_reward_geometry.py'
31 passed
```

Registration audits:

```text
/tmp/aic_agentic_reward_curriculum_20260529/collision_prim_audits/2026-05-30_18-58-38_v809_baseline_registration_fallback_heldout40x10
/tmp/aic_agentic_reward_curriculum_20260529/collision_prim_audits/2026-05-30_18-59-15_v810_nic_cage_p0_sdf_boxes_registration_fallback_heldout40x10
/tmp/aic_agentic_reward_curriculum_20260529/collision_prim_audits/2026-05-30_19-02-09_v811_nic_cage_p0_aligned_cubes_registration_heldout40x10
```

The v810 SDF-derived runtime boxes are misregistered laterally relative to the original Isaac cage:

| cage representation | wall/top/bottom center lateral from entrance | front center lateral from entrance | decision |
|---|---:|---:|---|
| original Isaac `cage_p0_*` meshes | `5.19-7.82 mm` | `1.13 mm` | baseline |
| SDF-derived runtime boxes | `15.64-30.75 mm` | `23.21 mm` | reject as misregistered |
| aligned runtime cubes from original transforms | `5.19-7.82 mm` | `1.13 mm` | use only as mesh-vs-box diagnostic |

I added another off-by-default diagnostic flag:

```text
--replace_nic_cage_p0_with_aligned_cubes
```

It disables each original `cage_p0_*` mesh collision and creates a USD cube using the original prim's local transform.
This preserves registration while testing whether mesh contact, rather than cage placement, is the blocker.

Full-depth settle probes:

| run | collision mode | best tip `s` | best `r` | theta | module consistency | strict | decision |
|---|---|---:|---:|---:|---:|---|---|
| v812 | aligned cage cubes, `r=0.3 mm` reset | `45.913 mm` | `0.629 mm` | `0.03671` | `0.945` | false | not strict |
| v813 | aligned cage cubes, `r=0.1 mm`, target tip quat override | `45.941 mm` | `0.505 mm` | `0.03686` | `0.960` | false | closest aligned-cube state, still not strict |

Conclusion: the earlier SDF-box replacement was indeed misregistered, so it should not be used as a teacher fix.
The aligned-cube replacement keeps the cage pose correct and recovers module consistency, but still cannot satisfy the
strict checker because semantic-tip theta remains around `0.037 rad` and lateral error remains marginal. This keeps the
blocker in the collision/contact/semantic reset-realization category. It is still not appropriate to start HIL-SERL
from these traces because they are diagnostic collision modifications, not a physically validated Isaac teacher policy.

## Addendum - Axis-aligned full-depth representability probes v815-v817

I reconstructed the next diagnostic from the available `wrist_contact_realization.py` flags because the v811-v813
folders were not present under the host `/tmp` or repo `outputs` paths. The runs were executed in Docker and copied back
to host scratch:

```text
/tmp/aic_agentic_reward_curriculum_20260529/reset_settle_runs/2026-05-30_19-15-28_v815_aligned_cubes_targettipquat_lat0000um_zeroaction
/tmp/aic_agentic_reward_curriculum_20260529/reset_settle_runs/2026-05-30_19-16-09_v815_aligned_cubes_targettipquat_lat0050um_zeroaction
/tmp/aic_agentic_reward_curriculum_20260529/reset_settle_runs/2026-05-30_19-16-48_v815_aligned_cubes_targettipquat_lat0100um_zeroaction
/tmp/aic_agentic_reward_curriculum_20260529/reset_settle_runs/2026-05-30_19-20-34_v816_aligned_cubes_axisrot_scalep0p5_lat0_zeroaction
/tmp/aic_agentic_reward_curriculum_20260529/reset_settle_runs/2026-05-30_19-21-14_v816_aligned_cubes_axisrot_scalep1p0_lat0_zeroaction
/tmp/aic_agentic_reward_curriculum_20260529/reset_settle_runs/2026-05-30_19-21-56_v816_aligned_cubes_axisrot_scalem0p5_lat0_zeroaction
/tmp/aic_agentic_reward_curriculum_20260529/reset_settle_runs/2026-05-30_19-23-22_v817_aligned_cubes_axisrot_scalep1p0_posehold_lat0
```

v815 tried `--override_start_tip_orientation_wxyz` with tip-preserving reset-body derivation. This did not fix the
strict axis-mode orientation, because the checker uses the semantic `sfp_tip_link` axis with the local tip axis flipped.
Best v815 rows stayed at theta `~0.0373 rad`, `r=0.698-0.736 mm`, consistency `0.973-0.986`, and strict success false.

I then computed the world-frame rotation needed to align the post-reset semantic tip axis with the insertion axis:

```text
rotvec ~= [0.012149392, 0.033460875, 0.000423046] rad
```

v816 applied this as `--override_start_orientation_rotvec_world` while preserving the reset tip position. The scale `1.0`
case created a strict-like reset state:

| run | reset tip `s` | reset `r` | reset theta | reset consistency | first/best post-step outcome |
|---|---:|---:|---:|---:|---|
| v816 scale `1.0` | `45.807 mm` | `0.011 mm` | `0.00000` | `0.99996` | ejected laterally; best row `s=46.517 mm`, `r=7.824 mm`, theta `0.03759`, consistency `~0`, strict false |

This is an important negative result: Isaac can be initialized into a full-depth strict-like semantic pose when the
diagnostic aligned-cube cage replacement is active, but the pose is not dynamically stable after stepping physics.

v817 added a gripper `pose_hold` controller from the same strict-like reset. It also failed immediately:

| run | reset tip `s` | reset `r` | reset theta | reset consistency | best post-step row |
|---|---:|---:|---:|---:|---|
| v817 pose hold | `45.804 mm` | `0.006 mm` | `0.00000` | `0.99999` | step 1: `s=46.494 mm`, `r=7.529 mm`, theta `0.03817`, consistency `~0`, strict false |

Decision: reject v815-v817 as success or teacher data. These are diagnostic collision-modified representability tests,
and even they cannot maintain a strict full-depth post-step state. This strengthens the blocker: the missing piece is
not reward, model architecture, or an offline imitation source. It is a full-depth Isaac contact/asset/controller
realization problem around the NIC cage/SFP module pair. HIL-SERL should remain blocked until a physically valid Isaac
teacher can produce post-step stable, module-consistent insertion.

I also ran v818 with the same strict-like reset plus tiny contact offsets on the SFP body and runtime aligned cage cubes:

```text
/tmp/aic_agentic_reward_curriculum_20260529/reset_settle_runs/2026-05-30_19-25-44_v818_aligned_cubes_axisrot_scalep1p0_tinyoffset_zeroaction
```

Reset was again strict-like (`s=45.848 mm`, `r=0.076 mm`, theta `0.00000`, consistency `0.998`), but no post-step row
was strict. Best row by strict-priority score was step 29: `s=46.557 mm`, `r=7.742 mm`, theta `0.04213`, consistency
`~0`, force proxy `275234`. Tiny contact offsets therefore do not prevent the immediate lateral ejection/collapse.

## Addendum - Runtime contact-material tuning v839-v840

The current best bounded contact shell from v830/v834 was also tested with runtime-only physics material binding. The
new diagnostic flags remain off by default:

```text
--collision_material_tune_prim_regex
--collision_static_friction
--collision_dynamic_friction
--collision_restitution
```

Validation:

```text
.pixi/envs/default/bin/python -m py_compile \
  aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py

.pixi/envs/default/bin/python -m pytest -q \
  aic_utils/aic_isaac/test/test_insertion_reward_geometry.py \
  aic_utils/aic_isaac/test/test_randomized_near_gate_curriculum.py
32 passed
```

Metrics-only summaries were copied to:

```text
outputs/agentic_reward_curriculum_20260529/contact_material_diagnostics_20260530/
```

| run | material bracket | best tip `s` | best `r` | theta | module `s` | module `r` | strict | decision |
|---|---|---:|---:|---:|---:|---:|---|---|
| v839 | low friction `0.02/0.02` | `39.55 mm` | `0.506 mm` | `0.0327` | `15.91 mm` | `0.919 mm` | false | reject |
| v840 | high friction `0.5/0.5` | `49.53 mm` | `0.625 mm` | `0.0368` | `25.89 mm` | `1.27 mm` | false | not a promotion |

Low friction reduced useful depth. High friction reproduced the v834 near-full-depth tradeoff but did not satisfy strict
lateral/theta/module-consistency gates. The issue is therefore not fixed by simple material friction tuning. The next
cheatcode-fix step should instrument the v834/v837 transition window with relative tip/module/wrist/entrance transforms
to separate a semantic-frame/evaluator mismatch from a true contact-driven module-body mismatch.

## Addendum - Axial step bracket v841-v842

The per-step relative transforms already logged by `wrist_contact_realization.py` show the
`sfp_module_link -> sfp_tip_link` transform remains rigid through the v840/v834 transition. That makes a semantic
tip/body detachment bug unlikely. I then tested whether reducing the bounded axial command avoids the v834 over-depth
transition while retaining progress:

| run | axial command | best tip `s` | best `r` | theta | module `s` | module `r` | strict | decision |
|---|---:|---:|---:|---:|---:|---:|---|---|
| v841 | `50 um` | `41.68 mm` | `0.430 mm` | `0.0329` | `18.05 mm` | `1.04 mm` | false | stalls shallow |
| v842 | `75 um` | `35.46 mm` strict-ish / `36.84 mm` deepest | `0.307 mm` / `1.160 mm` | `0.0356` / `0.0607` | `11.82 mm` / `13.23 mm` | `1.04 mm` / `1.85 mm` | false | worse |
| v840/v834 | `100 um` | `49.53 mm` | `0.625 mm` | `0.0368` | `25.89 mm` | `1.27 mm` | false | deepest, still not strict |

The axial-step bracket did not reveal a controllable strict window. Smaller commands avoid the deepest snap-through but
lose too much depth and still miss theta/module consistency. The next fix should target contact/controller stability or
collision geometry around the entrance rather than simply adjusting axial step size.

## Addendum - Early orientation trim v843-v845

I tested whether v834/v840's semantic-tip orientation trim was activating too late by moving activation earlier to
`s=20 mm`:

| run | orientation trim | best near-target row | strict | decision |
|---|---:|---|---|---|
| v843 | `0.5 mrad` | tip `s=45.76 mm`, `r=0.222 mm`, theta `0.0371`, module `s=22.12 mm`, module `r=0.721 mm`, consistency `0.878` | false | closest balanced row |
| v844 | `1.0 mrad` | tip `s=45.79 mm`, `r=0.470 mm`, theta `0.0350`, module `s=22.16 mm`, module `r=1.00 mm`, consistency `0.778` | false | theta improves but module gate worsens |
| v845 | `2.0 mrad` | tip `s=49.87 mm`, `r=0.715 mm`, theta `0.0398`, consistency `0.063` | false | over-rotates/over-inserts |

v843 is now the closest balanced diagnostic: full axial depth is reached with low tip lateral error and high module
consistency, but semantic-tip theta remains about `0.037 rad`, above the strict `<0.030 rad` threshold. v844 and v845
confirm that simply increasing rotation authority is not the fix; it trades theta for module/lateral consistency and
eventually returns to a false-positive depth mode.

## Addendum - v846 visual diagnostic for closest balanced setting

I reran v843 with center/left/right video enabled and copied the small artifacts to:

```text
outputs/agentic_reward_curriculum_20260529/contact_visual_diagnostics_20260530/v846_v843_close_balanced_visual/
```

Saved artifacts include separate center/left/right MP4s, selected snapshots, command/config, and
`wrist_contact_summary.json`. The best row reproduced the closest balanced metric state:

```text
step 184: strict=false, tip s=45.853 mm, r=0.223 mm, theta=0.03711 rad, consistency=0.870
```

The visual check does not override the strict failure. The plug is near/inside the entrance region but still visibly
tilted, and the center/left/right camera geometry is partly occluded. This remains a near-success diagnostic, not a
valid insertion demonstration.

## Addendum - Local SFP shell margin bracket v847-v848

I tested whether the v843 theta floor was caused by the exact SFP shrunk-box margin:

| run | x/z shrink margin | best near-target row | strict | decision |
|---|---:|---|---|---|
| v847 | `0.125 mm` | tip `s=38.90 mm`, `r=0.577 mm`, theta `0.0367`, module `s=15.27 mm`, consistency `0.00038` | false | stalls shallow |
| v843 | `0.150 mm` | tip `s=45.76 mm`, `r=0.222 mm`, theta `0.0371`, module `s=22.12 mm`, consistency `0.878` | false | current closest |
| v848 | `0.175 mm` | tip `s=32.93 mm`, `r=0.649 mm`, theta `0.0380`, module `s=9.30 mm`, consistency `~0` | false | worse |

Nearby shell margins do not fix strict insertion. The `0.150 mm` margin is still the best bounded diagnostic setting,
but it leaves a persistent semantic-tip theta error above threshold.

## Addendum - Close-balanced module teacher probes v857-v861

After v846/v843 became the closest visual candidate, I ran a bounded set of follow-ups around the same module-body
teacher setup. The baseline targets `sfp_module_link` to the expected seated module depth while keeping semantic-tip
lateral correction active and using shrunk runtime SDF body boxes with small contact offset.

| run | change from v846 | best post-step result | decision |
|---|---|---|---|
| v857 | same as v858 but without `--enable_cameras` | failed before rollout because Isaac camera sensors require `--enable_cameras` even when image logging is disabled | infrastructure failure |
| v858 | stricter orientation trim: threshold `0.030 rad`, step `0.00025 rad` | best balanced row `s=40.12 mm`, `r=0.323 mm`, theta `0.03627 rad`, consistency `0.004`; later over-inserted tip with lateral sweep | reject |
| v859 | v846 controller plus low-friction material on runtime SDF shell and NIC cage | best `s=38.73 mm`, `r=0.666 mm`, theta `0.03673 rad`, consistency `0.00025`; final lateral sweep | reject |
| v860 | v846 controller with tiny late trim: threshold `0.034 rad`, step `0.00010 rad` | best `s=45.96 mm`, `r=0.240 mm`, theta `0.03524 rad`, module `s=22.28 mm`, module `r=0.606 mm`, consistency `0.910` | closest metrics-only teacher candidate, not success |
| v861 | ultra-tiny trim: threshold `0.032 rad`, step `0.00005 rad` | best summary `s=35.05 mm`, `r=0.670 mm`, theta `0.03937 rad`, consistency near zero | reject |

v860 improves the prior close-balanced candidate: compared with v846/v843, it keeps full-depth tip insertion and module
consistency while reducing theta from about `0.0371 rad` to about `0.0352 rad`. It still fails strict success because
theta remains above the existing `0.030 rad` threshold, and the final state eventually retreats. v858 and v861 show that
more aggressive or lower-threshold orientation trimming is not monotonic; it can destroy the module-consistent seated
state.

Current recommendation: preserve v860 as the closest Isaac teacher/contact configuration for future near-success
imitation data, labeled `near_success_orientation_blocked`. Do not use low-friction material tuning or strict-threshold
continuous orientation trim as the next default. The next code-side improvement should make final orientation
correction module-consistency-aware: apply rotation only when the module is near its expected seated depth/lateral band,
reject predicted or realized module-consistency loss, and capture/stop at the best strict-like frame rather than
continuing into retreat.

## Addendum - Module-consistency-gated orientation trim v862-v863

I added an off-by-default diagnostic flag:

```text
--target_module_stabilize_orientation_min_module_consistency
```

When finite, `target_module_stabilize` disables semantic-tip orientation trim unless the current module-consistency
score is above the configured threshold. This keeps old diagnostic commands backward-compatible and tests whether final
theta correction can be made safer around the v860 near-seated state.

Validation:

```text
.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py
.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_randomized_near_gate_curriculum.py
32 passed
```

| run | module-consistency trim gate | best post-step result | decision |
|---|---:|---|---|
| v862 | `0.85` | `s=46.39 mm`, `r=0.349 mm`, theta `0.03446 rad`, module `s=22.75 mm`, module `r=0.946 mm`, consistency `0.756`; orientation trim inactive at best frame because pre-step consistency was below gate | reject; lower theta but worse module consistency than v860 |
| v863 | `0.65` | summary best `s=45.84 mm`, `r=0.273 mm`, theta `0.04120 rad`, consistency `0.880` | reject; reproduces depth/consistency but theta regresses |

The consistency gate is useful as a safety mechanism but did not solve strict insertion in this sweep. v860 remains the
closest metrics-only candidate: `s=45.96 mm`, `r=0.240 mm`, theta `0.03524 rad`, consistency `0.910`. The remaining
gap is still semantic-tip theta. The next controller-side idea should change the orientation correction strategy itself
or capture an expert trajectory at the best v860-like frame; simply gating the existing trim is insufficient.

## Addendum - Near-success capture stop v864-v865

I added an off-by-default capture option to `wrist_contact_realization.py`:

```text
--stop_on_near_success_capture
--near_success_capture_min_s_m
--near_success_capture_max_r_m
--near_success_capture_max_theta_rad
--near_success_capture_min_module_consistency
```

This stops the diagnostic after the first configured near-success teacher frame and records the thresholds in
`wrist_contact_summary.json`. It does not modify strict success checks and does not claim success.

| run | setup | result | decision |
|---|---|---|---|
| v864 | v860-style command, new seed `864`, capture threshold theta `0.036` | no capture; best `s=45.79 mm`, `r=0.224 mm`, theta `0.03956`, consistency `0.867`; final retreated | reject for teacher promotion, videos preserved |
| v865 | exact v860 seed with capture threshold theta `0.036` | stopped at step `199`: `s=45.19 mm`, `r=0.240 mm`, theta `0.03521`, consistency `0.860`, module `s=21.55 mm`, module `r=0.604 mm`; strict false | accepted as near-success teacher artifact only |

v865 saved separate center/left/right videos:

```text
outputs/agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_21-34-44_v865_v860_seed_repro_near_success_capture_video/env0000_center_full_episode_20fps_quality.mp4
outputs/agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_21-34-44_v865_v860_seed_repro_near_success_capture_video/env0000_left_full_episode_20fps_quality.mp4
outputs/agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_21-34-44_v865_v860_seed_repro_near_success_capture_video/env0000_right_full_episode_20fps_quality.mp4
```

This is useful for imitation/HIL-SERL as a near-seated module-following trajectory, but it is not a strict success
demonstration. The strict blocker remains semantic-tip theta: `0.03521 rad` versus the existing `<0.030 rad` criterion.

## Addendum - Teacher residual extraction for HIL-SERL

I patched `aic_utils/aic_isaac/scripts/extract_privileged_residual_targets.py` so it reads module diagnostics from both
legacy `post_step_all_body_insertion_geometry.sfp_module_link` rows and the newer contact-diagnostic
`post_step_module_geometry` rows. Without this, v865 module `s/r` appeared as `NaN` in the extracted teacher audit even
though the run had valid module diagnostics.

Validation:

```text
.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/scripts/extract_privileged_residual_targets.py
.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_randomized_near_gate_curriculum.py
32 passed
```

v865 extraction output:

```text
outputs/agentic_reward_curriculum_20260529/expert_trajectories_20260530/near_success_v865/
```

Label counts:

| label | rows |
|---|---:|
| `tip_depth_false_positive` | 177 |
| `centered_high_theta_module_near` | 17 |
| `near_full_orientation_blocked` | 6 |

I also extracted the older replay-oriented teacher runs v645/v648/v649:

```text
outputs/agentic_reward_curriculum_20260529/expert_trajectories_20260530/teacher_replay_v645_v648_v649/
```

Those runs are rejected as positive imitation data because the combined labels are dominated by `tip_depth_false_positive`
and `contact_spike` rows. The detailed HIL-SERL decision is in
`docs/hil_serl_from_cheatcode_experts_results_20260530.md`.

## Addendum - Final orientation follow-ups v866-v867

I added two off-by-default flags to make the `target_module_stabilize` final orientation probe aware of predicted module
lateral sweep:

```text
--target_module_stabilize_orientation_module_lateral_penalty
--target_module_stabilize_orientation_module_lateral_margin_m
```

These flags only affect the diagnostic orientation-axis selection when explicitly enabled. They do not change ACT,
SERL, Gazebo, or runtime policy defaults.

Validation:

```text
.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py
.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_randomized_near_gate_curriculum.py
32 passed
```

| run | setup | best strict-like result | decision |
|---|---|---|---|
| v866 | v860 plus module-lateral-sweep orientation penalty `500.0`, margin `0.020 mm` | `s=45.884 mm`, `r=0.240 mm`, theta `0.03677 rad`, consistency `0.898`; alternate depth-ranked row reached `s=46.385 mm` but theta `0.03703`, consistency `0.849` | reject; worsens theta/module tradeoff |
| v867 | v860 tiny trim, strict activation threshold `0.030 rad`, no module-sweep penalty | reproduces v860-like row: `s=45.834 mm`, `r=0.240 mm`, theta `0.03523 rad`, consistency `0.912` | reject; no improvement over v860 |
| v868 | v860 tiny trim but quaternion-mode orientation selection | best summary row `s=43.152 mm`, `r=0.526 mm`, theta `0.02407 rad`, consistency `0.244`; final consistency falls to zero | reject; theta improves only by losing module-following insertion |

Conclusion: the current orientation probe is realization-limited around theta `0.035 rad`. Gating, predicted module sweep
penalty, lower activation threshold, and full-quat selection do not get strict full insertion while preserving module
following. Quat selection confirms the known tradeoff: it can reduce reported theta, but it sacrifices full axial/module
consistency. The next bounded controller test should use a staged strategy: first preserve v860 module-following depth,
then run a separate final-window orientation-only micro-recovery from the captured near-success state with axial motion
zeroed and module-consistency degradation as a hard stop.

## Addendum - Staged pose-hold orientation recovery v869-v871

I tested the staged strategy above by reconstructing reset episodes from the v865 captured near-success frame and then
running the existing `pose_hold_orientation_servo_best` diagnostic. These runs are metrics-only and do not claim
success.

Generated reset folders:

```text
outputs/agentic_reward_curriculum_20260529/generated_episode_configs/v869_posehold_from_v865_step199
outputs/agentic_reward_curriculum_20260529/generated_episode_configs/v870_posehold_from_v865_step199_preserve_lateral
```

Run folders:

```text
outputs/agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_21-53-07_v869_posehold_orientation_from_v865_metrics
outputs/agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_21-54-55_v870_posehold_orientation_from_v865_preserve_lateral_metrics
outputs/agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_21-56-35_v871_posehold_orientation_from_v865_tiny_step_metrics
```

| run | setup | best relevant result | decision |
|---|---|---|---|
| v869 | reset from v865 step 199 but target lateral forced to zero | reset did not reproduce v865: first row `s=45.966 mm`, `r=1.654 mm`, theta `0.03556`, consistency `0.379`; orientation recovery inactive | reject reset reconstruction |
| v870 | preserve v865 lateral offset signs, 2 envs, pose-hold orientation step `0.00010 rad` | env0 starts close to v865: `s=45.242 mm`, `r=0.204 mm`, theta `0.03522`, consistency `0.861`; best env0 score `s=45.371 mm`, `r=0.234 mm`, theta `0.03534`, consistency `0.873`; summary best reached theta `0.03479` only with consistency `0.797` | reject; slight theta gain costs module consistency and remains above strict |
| v871 | same reset, tighter gates, step `0.00005 rad`, higher lateral penalty | env0 best `s=45.517 mm`, `r=0.233 mm`, theta `0.03555`, consistency `0.887`; summary best `s=45.843 mm`, `r=0.278 mm`, theta `0.03534`, consistency `0.826` | reject; preserves consistency better but no theta improvement |

Conclusion: a separate pose-hold orientation recovery reproduces the same semantic-tip orientation floor around
`0.035 rad`. Smaller steps and stricter gates preserve module consistency but do not reduce theta. The remaining
blocker is not just reward, curriculum, or final-window activation timing; it is the realized wrist/cable orientation
authority near the seated contact state. The next code change should instrument and alter the orientation realization
model itself, for example by logging predicted-vs-realized tip/module orientation deltas per candidate axis and then
choosing a candidate based on realized calibration rather than one-step geometric prediction.

## Addendum - Realized final-window axis calibration v872-v874

I ran short final-window axis/sign probes from the v870 reset to measure realized orientation effects instead of relying
only on the one-step geometric predictor.

Run families:

```text
outputs/agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_*_v872_posehold_axis_*_from_v865_metrics
outputs/agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_*_v873_posehold_axis_*_forced_from_v865_metrics
outputs/agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_*_v874_pure_axis_*_from_v865_metrics
```

v872 kept the normal pose-hold activation gates. It did not apply fixed rotations: all six axis/sign runs were
identical, with `pose_hold_fixed_rotation_active_by_env=false`. The internal gate was stricter than the logged
near-success row, so v872 is a gating diagnostic rather than an axis calibration.

v873 forced pose-hold rotation active. The first active step reduced env0 theta from `0.03523` to `0.03284`, but it
also over-inserted the tip to `49.512 mm`, increased `r` to `0.804 mm`, and collapsed consistency to `0.090`. All
axis/sign labels shared this first-step behavior because pose-hold correction dominated the applied motion. This shows
that pose-hold plus rotation can transiently lower theta, but not while preserving strict module-following insertion.

v874 removed pose-hold translation and tested pure one-step rotations:

| axis/sign | env | theta delta | post-step theta | post-step s mm | post-step r mm | post-step consistency | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `+z` | 1 | `-0.00142` | `0.03379` | `45.224` | `0.305` | `0.709` | improves theta slightly, still non-strict and low consistency |
| `+y` | 1 | `-0.00133` | `0.03389` | `45.228` | `0.262` | `0.724` | improves theta slightly, still non-strict and low consistency |
| `-x` | 0 | `+0.00502` | `0.04027` | `49.576` | `0.300` | `0.078` | reject |
| `+x` | 0 | `+0.00671` | `0.04195` | `50.901` | `1.159` | `0.013` | reject |
| `+z` | 0 | `+0.00717` | `0.04242` | `50.486` | `0.945` | `0.025` | reject |
| `-y` | 0 | `+0.00728` | `0.04252` | `49.983` | `0.690` | `0.046` | reject |
| `+y` | 0 | `+0.00757` | `0.04282` | `49.994` | `0.604` | `0.046` | reject |
| `-z` | 0 | `+0.02206` | `0.05731` | `50.902` | `1.385` | `0.012` | reject |

Strict success remained false for all runs.

Interpretation:

- The only realized pure rotations that reduce theta do so by about `0.0013-0.0014 rad`, leaving theta around
  `0.0338 rad`, still above the `<0.030 rad` strict threshold.
- The best module-consistent env0 state is fragile: all pure rotation axes either worsen theta or destroy module
  consistency via over-insertion/contact response.
- The v860/v865 orientation gap is therefore not solved by choosing a different global axis/sign from the current
  wrist action interface.

Next recommendation: stop scalar reward/axis-selection tuning for this branch. The next code-side experiment should
change the actuator realization pathway, for example a two-body constrained micro-adjustment that commands wrist motion
to reduce semantic-tip theta while explicitly holding both semantic-tip lateral position and module axial/lateral
position, or a reset/contact asset fix if such constrained motion is not physically realizable in Isaac.

## Addendum - Two-body constrained rotation compensation v875-v876

I added an off-by-default diagnostic probe:

```text
--probe pose_hold_constrained_rotation_axis
--pose_hold_constrained_tip_weight
--pose_hold_constrained_module_weight
--pose_hold_constrained_compensation_clip_m
```

The probe computes the predicted rotation-induced sweep of both `sfp_tip_link` and `sfp_module_link` around
`wrist_3_link`, then applies one shared translation compensation before sending the wrist IK command. Defaults keep old
paths unchanged.

Validation:

```text
.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py
.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_randomized_near_gate_curriculum.py
32 passed
```

v875 first tried the constrained probe with the default pose-hold start gate. The command became active only after an
initial passive step had already degraded the state, so v875 is rejected as a timing diagnostic.

v876 allowed activation on the first probe step and tested the two previously promising pure-axis signs, `+y` and `+z`:

| run | env | start s/r/theta/cons | after one constrained step | decision |
|---|---:|---|---|---|
| v876 `+y` | 0 | `45.285 mm / 0.221 mm / 0.035214 / 0.868` | `45.575 mm / 0.099 mm / 0.035217 / 0.867` | preserves module consistency and lateral alignment, but theta unchanged |
| v876 `+z` | 0 | `45.285 mm / 0.221 mm / 0.035214 / 0.868` | `45.284 mm / 0.234 mm / 0.035472 / 0.872` | preserves module consistency, theta worsens slightly |
| v876 `+y` | 1 | `45.807 mm / 1.323 mm / 0.035783 / 0.452` | `49.958 mm / 1.620 mm / 0.042580 / 0.026` | reject; bad starting lateral state collapses |
| v876 `+z` | 1 | `45.807 mm / 1.323 mm / 0.035783 / 0.452` | `49.564 mm / 1.585 mm / 0.035916 / 0.042` | reject; bad starting lateral state collapses |

Interpretation:

- The two-body compensation successfully avoids the v873/v874 module-consistency collapse on the good env0 corridor.
- It still does not reduce semantic-tip theta below the current floor; `+y` is effectively neutral and `+z` is worse.
- This rules out a simple shared translation compensation as the final missing piece.

Next recommendation: the remaining bounded controller experiment should change orientation actuation more fundamentally:
test a very small relative cable/module shape adjustment or alternate wrist orientation frame while holding tip/module
geometry, or audit whether the SFP cable articulation/contact model is physically preventing the remaining `~0.005 rad`
semantic-tip orientation correction. More reward, visual backbone, or global axis-sign tuning is not supported by the
current evidence.

## Addendum - Collider transfer and consistency audit v877-v879

I added an off-by-default diagnostic flag to the Isaac collision prim audit so it can recreate the same shrunk SFP
body-box collider mode used by v865/v876:

```text
--replace_sfp_body_sdf_collision_with_shrunk_sdf_boxes
--sfp_shrunk_box_margin_m 0.00015 0.0 0.00015
```

Validation:

```text
.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/audit_isaac_collision_prims.py
```

Offline SDF audit:

```text
outputs/agentic_reward_curriculum_20260529/collision_audits/sfp_nic_collision_geometry_20260530_v877/
```

Key dimensions:

- Gazebo-active SFP body AABB: `13.750 x 47.300 x 8.452 mm`.
- NIC port opening: `14.000 mm x 8.949 mm`.
- NIC cage depth: `48.720 mm`.
- v865/v876 used `shrunk_body_boxes`, four body-shell cubes, not the full Gazebo-active SFP module collider set.

Isaac-stage prim audit:

```text
outputs/agentic_reward_curriculum_20260529/collision_prim_audits/2026-05-30_22-21-59_v878_v870_shrunk_collision_prim_audit/
```

Reset geometry for the preserved v865-like near-success state:

| body | s | r | target depth | raw axial error |
|---|---:|---:|---:|---:|
| `sfp_tip_link` | `45.464 mm` | `0.414 mm` | `45.800 mm` | `0.335 mm` |
| `sfp_module_link` | `21.829 mm` | `0.796 mm` | `45.800 mm` | `23.971 mm` |

The module row is not evidence that the strict checker incorrectly requires `sfp_module_link` to reach the same
absolute target depth as the tip. The strict consistency code stores the reset-time primary-tip-to-module axial gap and
checks the module against `target_depth - reference_gap`. The raw module geometry report remains useful for diagnostics,
but the consistency gate is offset-aware.

v879 tested the direct Gazebo-collider-transfer hypothesis by replacing the converted Isaac SFP collision mesh with the
Gazebo-active SFP module boxes:

```text
outputs/agentic_reward_curriculum_20260529/contact_realization_runs/2026-05-30_22-23-04_v879_v865_gazebo_active_sfp_colliders_metrics/
```

Result:

| run | SFP collider mode | best s | best r | theta | consistency | strict_success | decision |
|---|---|---:|---:|---:|---:|---|---|
| v865 | shrunk body boxes | `45.190 mm` | `0.240 mm` | `0.03521` | `0.860` | false | current near-success artifact |
| v879 | Gazebo-active SFP boxes | `34.294 mm` | `0.697 mm` | `0.04005` | `~0.000` | false | reject |

v879 hit large contact force and regressed to a mid-depth tip-only state. This rejects the naive fix of simply restoring
all Gazebo-active SFP detail colliders in Isaac. The next bounded fix should instead target contact/controller
realization at the near-success state: maintain the permissive shrunk body-shell collider for teacher diagnostics, but
try a very small compliant or staged module-following motion model that can reduce semantic-tip theta without pushing
the tip past the target or collapsing module consistency.
