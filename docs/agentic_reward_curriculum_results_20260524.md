# Agentic Reward/Curriculum Results 2026-05-24

## Status

2026-05-24 correction: the previous `8 mm` strict-depth target was too shallow
relative to the Gazebo SFP port frames. The official Gazebo SDF defines
`sfp_port_*_link_entrance` at `(0, 0, -0.0458)` relative to `sfp_port_*_link`,
and the cage collision bodies are `0.04872 m` deep. Isaac full insertion should
therefore target about `45.8 mm` from the entrance frame, plus any configured
outward entrance offset, not `8 mm`.

The former strict final-window run is now classified as shallow insertion:

- `outputs/agentic_reward_curriculum_20260524_controller_contact/runs/2026-05-24_09-23-09_strict_candidate_x015_zneg024_ax60_images`
- First strict post-step row: step 23 env0, `s=0.008056 m`, `r=0.000226 m`, `theta=0.029892 rad`, module consistency `0.806509`, `strict_success=true`.
- Evidence: `metrics.jsonl`, `strict_metrics.csv`, `strict_success.json`, `run_config.json`, `command.txt`, `git_status.txt`, `git_diff.patch`, center/left/right step images, and center/left/right MP4 clips.
- Corrected interpretation: this is a focused shallow scripted wrist-IK diagnostic derived from the best checkpoint trajectory. It is not full Gazebo-like SFP seating.

The strongest policy-derived near-strict candidate remains:

- `outputs/agentic_reward_curriculum_20260524_final_window_probe/runs/2026-05-24_04-52-05_progress7_fixed_world_xpos_depthhold_tight_step2_guideonly`
- Best near-strict post-step row: step 149 env1, `s=0.008019 m`, `r=0.000262 m`, `theta=0.036075 rad`, module consistency `0.911603`.
- Failure label: `near_success_orientation_blocked`.
- Strict failure reason: semantic tip orientation is above the strict target (`theta <= 0.030 rad`).

## 2026-05-25 Corrected Full-Depth Near-Final Trajectory Result

The corrected full-depth checker has now produced a strict success from a
near-final insertion trajectory, not only from a seated reset.

Strict evidence run:

- `outputs/agentic_reward_curriculum_20260524_depth_correction/runs/2026-05-25_12-40-17_full_depth_insertion_local43_goal0p0456_targettip_no_rot_final4_images_env40`
- Summary: privileged target-tip servo from a local final-window start,
  configured depth `43.5 mm`, lateral `+0.25 mm`, x-rot `0.0 rad`.
- Metrics/evidence: `metrics.jsonl`, `wrist_contact_summary.json`,
  `strict_success_rows.json`, `strict_success_rows.csv`, `run_config.json`,
  `command.txt`, `git_status.txt`, `git_diff.patch`, `agent_decision.json`,
  selected center/left/right images, and center/left/right MP4 clips.
- Higher-quality full-episode video evidence was regenerated from the same
  60-env strict trajectory with diagnostic-only `448x448` cameras, Isaac
  `quality` rendering, frame 0 plus every control step saved, and 20 fps
  encoding:
  `outputs/agentic_reward_curriculum_20260524_depth_correction/runs/2026-05-25_12-51-10_full_depth_insertion_local43_goal0p0456_targettip_no_rot_quality448_fullvideo_env40/env0040_three_camera_full_episode_20fps_quality448.mp4`.

Strict final row:

| step | env | post-step s/r/theta | module consistency | strict_success | decision |
| ---: | ---: | --- | ---: | :---: | --- |
| 4 | 40 | `s=45.990 mm`, `r=0.454 mm`, `theta=0.02947 rad` | `0.844` | true | promote |

Key tuning observation:

- Goal depths `45.0`, `45.2`, and `45.4 mm` did not produce strict success in
  the local final-window sweep.
- `45.6 mm` produced a strict row because it still cleared the corrected
  full-depth depth gate while avoiding the module-consistency drop seen when
  the servo over-drove toward `45.8 mm` and beyond.

Interpretation:

This is a real corrected-depth strict insertion under the current evaluator:
post-step `sfp_tip_link` depth, lateral error, semantic tip orientation, and
module/body consistency all satisfy the strict gates, and visual evidence is
saved. It remains a privileged diagnostic/servo result from a near-final start,
not a learned ACT/SERL policy success and not a robust randomized far-start
result.

The full-episode video appears to begin already inserted because the successful
run deliberately starts from a local final-window reset: frame 0 has
`s=43.501 mm`, `r=0.249 mm`, `theta=0.03855`, and module consistency `0.379`.
The final strict frame reaches `s=45.990 mm`, `r=0.454 mm`, `theta=0.02947`,
and consistency `0.844`. A video from a non-contact entrance approach remains a
separate experiment; previous broader-start attempts failed through
contact/controller realization and module-consistency loss.

Entrance-start diagnostic:

- `outputs/agentic_reward_curriculum_20260524_depth_correction/runs/2026-05-25_13-00-45_precontact_sneg2_goal0p0456_quality448_video`
- Start frame: `s=-1.990 mm`, `r=0.258 mm`, `theta=0.03855`, consistency `0.000`.
- Final frame: `s=0.310 mm`, `r=0.934 mm`, `theta=0.05296`, consistency `0.000`.
- Videos: `env0000_center_full_episode_20fps_quality.mp4`,
  `env0000_left_full_episode_20fps_quality.mp4`,
  `env0000_right_full_episode_20fps_quality.mp4`.
- Interpretation: the current target-tip servo can create a non-contact
  entrance-crossing video, but it does not solve full insertion from this start;
  lateral error grows and module consistency remains zero.

Next recommended step:

Port the `45.6 mm` final-window target-tip recipe into the config-driven
train/eval guard path, then run short online SERL/eval from the best checkpoint
with corrected full-depth strict metrics. The training objective should be to
learn or preserve this final-window retention behavior without allowing
tip-depth-only false positives.

## 2026-05-25 Non-Contact Policy/Guide Starts

Generated corrected full-depth non-contact episode configs with
`seated_depth_m: 0.0458`:

- `outputs/agentic_reward_curriculum_20260524_depth_correction/generated_episode_configs/full_depth_start_10mm_axial_10mm_lateral`
- `outputs/agentic_reward_curriculum_20260524_depth_correction/generated_episode_configs/full_depth_start_40mm_axial_10mm_lateral`

The old curriculum values were indeed `10/20/30/40 mm` axial outside the
entrance crossed with `4/6/8/10 mm` lateral. The previous strict full-depth
video did not use those starts; it started already in the final window at
`s=43.5 mm`, `r=0.25 mm`.

Policy-video saving was added to the SERL runner: when `--save_step_images` is
enabled, `--save_videos` now defaults on and writes separate center/left/right
MP4s from the saved step frames. The Isaac policy cameras are `224x224`; the
diagnostic-only camera runner can still produce `448x448` videos.

| Run | Start / mode | Best post-step row | Video evidence | Failure label |
| --- | --- | --- | --- | --- |
| `2026-05-25_13-10-25_full_policy_start10x10_progress10_eval_video` | policy eval from `10 mm` axial / `10 mm` lateral outside gate | step220? final step180 `s=21.516 mm`, `r=15.413 mm`, `theta=0.15244`, target `45.800 mm` | center/left/right MP4s plus first/last center frames | lateral_bypass / tip-depth false-positive risk |
| `2026-05-25_13-15-23_full_policy_start40x10_progress10_eval_video` | policy eval from `40 mm` axial / `10 mm` lateral outside gate | final step220 `s=-24.658 mm`, `r=13.173 mm`, `theta=0.18435` | center/left/right MP4s plus first/last center frames | no_axial_progress / lateral-orientation blocked |
| `2026-05-25_13-25-59_train_full_depth_start40x10_progress10_sparse_updates` | short sparse online SERL from 40/10, no image logging during training | checkpoint saved after 30 sparse updates | metrics/config/checkpoint only | training candidate |
| `2026-05-25_13-33-08_eval_trained_sparse_start40x10_video` | eval of sparse-trained checkpoint from 40/10 | final step220 `s=-18.304 mm`, `r=1.256 mm`, `theta=0.09749` | center/left/right MP4s plus first/last center frames | no_axial_progress / orientation_blocked |
| `2026-05-25_13-49-54_guide_probe_start10x10_strong_lateral` | guided/guarded probe, stronger lateral correction | final step180 `s=-0.156 mm`, `r=0.162 mm`, `theta=0.06449` | metrics only | near_gate_orientation_blocked |
| `2026-05-25_14-02-05_guide_probe_start10x10_finalorient_preentry` | pre-entry final-orientation window enabled | best theta step233 `theta=0.06385`, `s=-0.769 mm`, `r=0.810 mm` | metrics only | orientation_realization_plateau |
| `2026-05-25_14-09-18_guide_probe_start10x10_finalorient_preentry_signneg` | sign check for orientation refinement | best theta remained initial `0.06969`; later theta worsened to `0.16625` | metrics only | wrong_rotation_sign / orientation worsened |

Current interpretation:

- The corrected non-contact full-policy videos are saved, but neither is a
  strict success.
- Sparse online SERL from the 40/10 start improved approach geometry from
  `s=-24.7 mm`, `r=13.2 mm` to `s=-18.3 mm`, `r=1.26 mm`, but it still did not
  reach the entrance and theta remained too large.
- The strongest guided approach from 10/10 can center the semantic tip at the
  entrance, but semantic tip orientation plateaus near `0.064 rad`. Because the
  strict orientation gate is `0.030 rad`, axial insertion remains correctly
  blocked.
- The immediate blocker for a full trajectory from non-contact starts is now
  pre-entry semantic orientation realization, not depth definition or lateral
  correction.

Validation after SERL video encoding change:

- `.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
- `docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p -m pytest -q aic/aic_utils/aic_isaac/test/test_insertion_reward_geometry.py'`
- Latest result: `29 passed in 4.15s`.

## 2026-05-24 Corrected Full-Depth Seated Result

The corrected full-depth checker has now produced reproducible strict seated
success in Isaac at the Gazebo-derived SFP port depth. This is not a reward-only
claim and not the old `8 mm` shallow criterion.

Strict evidence run:

- `outputs/agentic_reward_curriculum_20260524_depth_correction/runs/2026-05-24_13-00-03_full_depth_reset_sweep_depth_lat_x64_posehold_images_env15_61`
- Summary: `strict_success=true`
- Metrics/evidence: `metrics.jsonl`, `wrist_contact_summary.json`,
  `strict_success_rows.json`, `strict_success_rows.csv`, `run_config.json`,
  `command.txt`, `git_status.txt`, `git_diff.patch`, and selected
  center/left/right images for env15 and env61.

Strict rows:

| step | env | configured start | post-step s/r/theta | module consistency | decision |
| ---: | ---: | --- | --- | ---: | --- |
| 1 | 15 | depth `45.8 mm`, lateral `+0.2 mm`, x-rot `+0.010 rad` | `s=45.636 mm`, `r=0.335 mm`, `theta=0.02751 rad` | `0.929` | strict seated success |
| 3 | 61 | depth `47.0 mm`, lateral `+0.2 mm`, x-rot `0.000 rad` | `s=46.864 mm`, `r=0.330 mm`, `theta=0.02842 rad` | `0.812` | strict seated success |

Why it is real full-depth seating under the current checker:

- The post-step `sfp_tip_link` signed depth satisfies the corrected target of
  about `45.8 mm`, not the superseded `8 mm` shallow target.
- Lateral error is below the strict `0.5 mm` gate.
- Semantic tip orientation is below the strict `0.030 rad` gate.
- Module/body consistency is above the strict `0.80` gate.
- Center/left/right images were saved for the successful envs and are
  consistent with a seated SFP body rather than lateral bypass.

Remaining limitation:

This is a seated reset/pose-hold diagnostic. It demonstrates that the corrected
full-depth strict state is reachable and visually plausible in Isaac. It does
not yet demonstrate a learned ACT/SERL policy or a robust trajectory from a
farther start.

Follow-up trajectory attempts:

| Run | Key change | Best post-step row | Failure label | Decision |
| --- | --- | --- | --- | --- |
| `full_depth_insertion_start_sweep_x16_targettip_no_rot` | target-tip servo from starts at `30/35/40/43 mm` | env14 step1 `s=46.822 mm`, `r=0.289 mm`, `theta=0.02821`, consistency `0.768` | near_success_module_consistency_blocked | tune local final-window |
| `full_depth_insertion_local43_sweep60_targettip_no_rot` | local sweep around `42.5-44.0 mm` starts | env37 step1 `s=46.498 mm`, `r=0.414 mm`, `theta=0.02957`, consistency `0.770` | near_success_module_consistency_blocked | reject as not strict |
| `full_depth_insertion_local43_goal0p0456_targettip_no_rot_final4_images_env40` | local `43.5 mm` start, target-tip goal `45.6 mm`, final frame saved | env40 step4 `s=45.990 mm`, `r=0.454 mm`, `theta=0.02947`, consistency `0.844` | strict_success | promote |

Interpretation:

The final seated state is reachable, but the shallow-start servo tends to move
the tip to full depth while leaving the module consistency gate just below
threshold (`~0.77`). This is no longer a depth-definition problem; it is a
final-window module/body consistency and contact-realization problem.

Validation after the selected-image diagnostic change:

- `python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py`
- `docker exec isaac-lab-base bash -lc 'cd /workspace/isaaclab && ./isaaclab.sh -p -m pytest -q aic/aic_utils/aic_isaac/test/test_insertion_reward_geometry.py'`
- Latest result: `29 passed in 4.10s`.

## 2026-05-24 Retention/Contact Probes

| Run | Key change | Best post-step row | Failure label | Decision |
| --- | --- | --- | --- | --- |
| `progress7_xpos_retention_comp_sweepguard_guideonly` | retention + compensation + sweep guard | `s=0.003606`, `r=0.000153`, `theta=0.0807`, consistency `0.0163` | no axial progress / orientation blocked | reject |
| `progress7_fixed_world_xneg_depthhold_tight_step2_guideonly` | fixed-world final trim, x negative | `s=0.008447`, `r=0.000264`, `theta=0.036350`, consistency `0.88136` | near_success_orientation_blocked | reject |
| `progress7_fixed_world_xneg_forced_depthhold_tight_step2_guideonly` | forced x-negative final trim | `s=0.008131`, `r=0.000251`, `theta=0.038992`, consistency `0.87945` | orientation worsened | reject |
| `progress7_goodorient_sneg2_generated_episode_guideonly` | generated good-orientation `s=-2 mm` reset | pre-step good, step1 post `r=0.017/0.016 m` | controller/contact lateral sweep | reject |
| `progress7_goodorient_sneg2_generated_episode_depen005_guideonly` | depenetration cap 0.05 | step1 post `r=0.0014/0.0018 m`, later blowout | controller/contact lateral sweep | reject |
| `progress7_goodorient_sneg2_depen005_fullguide_axialmicro_nolateral` | full guide, axial micro, no lateral | step160 env0 `s=0.011702`, `r=0.018556`, `theta=0.068399`, consistency `~0` | tip_depth_false_positive / lateral bypass | reject |
| `progress7_goodorient_sneg2_depen005_fullguide_settle8_axialmicro_nolateral` | initial zero-action settle | step1 still jumped to `r=0.00143/0.00179 m`; step160 `r=0.01895` | contact/controller sweep under zero command | reject |
| `progress7_goodorient_sneg4_depen001_fullguide_lat20um_ax20um` | first `s=-4 mm` generated reset, 20 um lateral/axial | best near row step16 env0 `s=-0.001320`, `r=0.000409`, `theta=0.011167`, consistency `~1e-6` | no axial progress / module consistency absent | reject |
| `progress7_goodorient_sneg4_depen001_fullguide_realizedr_recovery` | realized-r backoff recovery | recovery fired, but `r` still grew to >15 mm | controller realization mismatch | reject |
| `progress7_goodorient_sneg4_shifted_depen001_settle12_ax20um_nolateral` | corrected shifted `s=-4 mm` reset preserving good `s=-2 mm` geometry | step1 post `r=0.00030/0.00040`, but step10 `r=0.0030/0.0038` under zero settle | contact/controller sweep under zero command | reject |

## Implemented Hooks

- Added `--insertion_action_guard_final_fixed_world_rotation_force` for strict-gated diagnostic final orientation trim.
- Added `--insertion_action_guard_settle_steps` plus depth/lateral/theta gates for initial zero-action settling.
- Added `--insertion_action_guard_realized_r_recovery` plus margin/backoff/depth gates for recovery when realized lateral error worsens.
- All new flags default to the previous behavior.

Validation:

- `.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
- `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py`
- Latest result: `30 passed`.

## Current Blocker

The strongest current evidence is that the simulator/controller contact realization near the SFP cage entrance is unstable:

- A clean shifted `s=-4 mm` reset starts with strict-compatible lateral/orientation geometry.
- With the settle guard active, commanded translation is zero at the first steps.
- Despite that, post-step lateral error grows from sub-millimeter to several millimeters by step 10.
- Backoff commands during realized-r recovery also do not recover the tip; in some runs `s` continues forward or `r` grows while outward/backoff commands are logged.

This makes reward-only or curriculum-only tuning unlikely to produce strict insertion until the near-entrance controller/contact realization is fixed.

## Next Recommended Step

Add a controller diagnostic that logs commanded wrist/TCP delta versus realized `sfp_tip_link` delta for zero-action, axial-only, and lateral-only commands from the corrected shifted `s=-4 mm` reset. If zero-action drift reproduces there, fix reset/contact settling or physics properties before more reward/SERL runs. If zero-action is stable in the diagnostic but unstable in SERL, inspect the SERL action conversion/action-manager sync path.

## 2026-05-24 Controller Sign and Depth-Servo Probes

| Run | Key change | Best post-step row | Failure label | Decision |
| --- | --- | --- | --- | --- |
| `shifted_sneg4_posehold_zero_probe` | standalone wrist/contact diagnostic, corrected `s=-4 mm`, zero action | step1 env1 `s=-0.003705`, `r=0.000402`, `theta=0.000846`, consistency `~0`; by step80 env0 `s=0.00225`, `r=0.02155` | controller/contact drift under zero command | reject |
| `depth_sweep_sneg8_corrected_posehold_zero` | corrected `s=-8 mm`, zero action | step1 env1 `s=-0.007708`, `r=0.000399`, `theta=0.000846`; by step50 env1 `r=0.01816` | controller/contact drift under zero command | reject |
| `depth_sweep_sneg8_posehold_active_noxyfix` | explicit wrist pose hold with `--no-fix_isaac_ik_xy_sign` | step20 env0 `s=-0.0078`, `r=0.0003`, `theta=0.0016` | stable hold, no insertion | promote as controller diagnostic clue |
| `progress7_sneg8_noxyfix_fullguide_ax20_lat20` | SERL guide-only from corrected `s=-8 mm`, no IK XY sign fix | step160 env1 `s=0.008547`, `r=0.017511`, `theta=0.05419`, consistency `~0` | tip_depth_false_positive / lateral bypass | reject |
| `sneg8_target_depth8mm_noxyfix_lat300` | standalone semantic tip servo toward `s=8 mm`, 300 um lateral, 20 um axial | best env1 `s=0.000072`, `r=0.003309`, `theta=0.05072`, consistency `1.5e-5` | no axial progress past entrance / orientation residual | reject |
| `sneg8_target_depth8mm_noxyfix_lat300_ax100` | same with 100 um axial | best env1 `s=0.000173`, `r=0.003704`, `theta=0.05334`, consistency `1.2e-5` | no axial progress past entrance / orientation residual | reject |
| `progress7_fixed_world_xpos_depthhold_tight_step2_noxyfix` | strongest prior final-window candidate with no IK XY sign fix | step160 env1 `s=0.007122`, `r=0.000137`, `theta=0.03856`, consistency `0.74749` | near_success_orientation_and_consistency_blocked | reject |

Additional implementation notes:

- Fixed the standalone diagnostic episode override generator so it preserves the actual source reset-body/reference offset instead of a stale `reset_body_offset_from_reference_world` field.
- Added `--target_tip_stabilize_goal_depth_m` to the standalone wrist/contact diagnostic so `target_tip_stabilize` can servo toward the strict seated centerline instead of holding the initial reset tip pose.
- Validation after the diagnostic changes:
  - `.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py`
  - `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py`

Current interpretation:

- The Isaac IK XY sign fix is context-dependent. Disabling it makes explicit wrist pose hold stable in the corrected `s=-8 mm` diagnostic, but it worsens the strongest final-window candidate and does not prevent guide-only lateral bypass.
- The remaining blocker is still controller/contact realization at the cage entrance. The best final-window candidate is orientation-limited (`theta ~= 0.036 rad`), while broader start-depth probes either drift laterally under zero command or stall around the entrance with poor module consistency.

Next recommended step:

Implement a train/eval guard mode equivalent to the diagnostic seated-depth semantic servo: target a configurable semantic tip depth, use corrected per-command action-sign selection only when realization diagnostics show it helps, maintain high lateral correction, and block axial advance whenever post-step `r/theta/module consistency` regress. Then run it from the strongest final-window checkpoint and the corrected `s=-8 mm` reset.

## 2026-05-24 Train/Eval Target-Tip Servo Guard

Implemented opt-in guard flags in `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`:

- `--insertion_action_guard_target_tip_servo`
- `--insertion_action_guard_target_tip_servo_goal_depth_m`
- `--insertion_action_guard_target_tip_servo_lateral_step_m`
- `--insertion_action_guard_target_tip_servo_axial_step_m`
- `--insertion_action_guard_target_tip_servo_axial_lateral_gate_m`
- `--insertion_action_guard_target_tip_servo_axial_theta_gate_rad`
- `--insertion_action_guard_target_tip_servo_min_consistency`

The guard keeps semantic lateral correction active continuously, but only allows positive axial motion when lateral error, semantic tip orientation, and module consistency satisfy the configured gates. Defaults preserve previous behavior.

Validation:

- `.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py`
- `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py`
- Latest result: `30 passed`.

| Run | Key change | Best post-step row | Failure label | Decision |
| --- | --- | --- | --- | --- |
| `progress7_sneg8_noxyfix_targettipservo_lat300_ax20` | train guard target-tip servo, 300 um lateral, 20 um axial, global translation clip 200 um | step120 env1 `s=-0.000082`, `r=0.005901`, `theta=0.04697`, consistency `9.8e-8` | no axial progress past entrance / lateral-orientation blocked | reject |
| `progress7_sneg8_noxyfix_targettipservo_lat300_ax20_clip500` | same, translation clip raised to 500 um | step119 env1 `s=-0.000014`, `r=0.003198`, `theta=0.04986`, consistency `1.3e-5` | no axial progress past entrance / lateral-orientation blocked | reject |
| `progress7_xpos_force_targettiphold_finalwindow` | strongest final-window command plus forced fixed-world orientation trim and target-tip servo hold | step91 env1 `s=-0.000673`, `r=0.001087`, `theta=0.03777`, consistency `5.2e-6` | reset/curriculum mismatch; servo held near entrance and prevented previous insertion progress | reject |

Interpretation:

- The new guard prevented the earlier large tip-depth false positive: once `r` exceeded the axial gate, `insertion_action_guard_target_tip_servo_axial_gate_fraction` dropped to `0.0` and the guard stopped positive axial commands.
- Raising the global translation clip let the 300 um lateral servo reach the controller, but the contact/controller dynamics still drove `r` to several millimeters and semantic orientation to `~0.05 rad` near the entrance.
- The current best strict-adjacent result remains the 04:52 final-window run with `s=0.008019`, `r=0.000262`, `theta=0.036075`, consistency `0.911603`.

Next recommended step:

Focus on final-window orientation/module recovery rather than farther-out servo starts. The broad-start controller/contact path is still blocked at the entrance, while the best final-window path is already depth/lateral/consistency close and needs about `0.006 rad` semantic orientation improvement without losing module consistency. The target-tip servo should not be applied to the mixed far-start curriculum as a blanket override; it stalls the previous insertion behavior near the entrance. The next bounded experiment should generate a final-window-only episode config from the best near-strict row, then run very small fixed-world orientation trim sweeps with depth/lateral hold and strict predicted-r rejection.

## 2026-05-24 Strict Final-Window Diagnostic Success

Correction: this section is retained as historical evidence for a shallow
`8 mm` target. After the Gazebo/asset audit, it should not be cited as full
SFP-to-NIC insertion.

Implementation update:

- Added standalone diagnostic probe `--probe target_tip_fixed_rotation_axis`.
- It combines target-tip depth/lateral stabilization, bounded fixed-axis rotation, and tip-sweep compensation.
- This is diagnostic-only; it does not alter ACT, SERL, Gazebo, or runtime policy paths.

Focused final-window reset:

- Generated from pose-logged env1 of `2026-05-24_08-47-46_progress7_xpos_depthhold_step2_poseenvlog`.
- Correct source episode: `episode_000002` from the interleaved curriculum. Reusing the env0 episode template produced an invalid `r ~= 83 mm` reset and was rejected.
- Valid generated config: `outputs/agentic_reward_curriculum_20260524_final_window_probe/generated_episode_configs/20260524_finalwindow_env1_poseenvlog_084746_step155_ep2`.

Key search results:

| Run | Key change | Best post-step row | Failure/success label | Decision |
| --- | --- | --- | --- | --- |
| `finalwindow_env1_ep2_posehold_zero` | corrected env1 final-window reset, pose hold | step24 `s=0.007990`, `r=0.000068`, `theta=0.03956`, consistency `0.8027` | near_success_orientation_blocked | promote for tuning |
| `finalwindow_env1_ep2_targettip_depthhold_noorient` | target-tip depth hold, no orientation trim | step23 `s=0.008110`, `r=0.000055`, `theta=0.03949`, consistency `0.8096` | near_success_orientation_blocked | promote as depth/consistency baseline |
| `finalwindow_env1_ep2_reset_rot_x015_zneg010_targettip_depthhold` | partial reset orientation, `rotvec=(0.015,0,-0.010)` | step19 `s=0.008003`, `r=0.000065`, `theta=0.03347`, consistency `0.8760` | near_success_orientation_blocked | promote |
| `finalwindow_env1_ep2_reset_rot_x015_zneg020_targettip_depthhold` | partial reset orientation, `rotvec=(0.015,0,-0.020)` | step19 `s=0.008045`, `r=0.000046`, `theta=0.03076`, consistency `0.8610` | near_success_orientation_blocked | promote |
| `finalwindow_env1_ep2_reset_rot_x015_zneg024_targettip_depthhold` | partial reset orientation, `rotvec=(0.015,0,-0.024)`, 20 um axial | step25 `s=0.007741`, `r=0.000161`, `theta=0.03002`, consistency `0.8215` | depth just short / theta just high | tune axial |
| `finalwindow_env1_ep2_reset_rot_x015_zneg024_targettip_ax0p00006` | same reset, target-tip axial step 60 um | step23 `s=0.008056`, `r=0.000226`, `theta=0.02989`, consistency `0.8065`, `strict_success=true` | strict_success | promote |
| `strict_candidate_x015_zneg024_ax60_images` | exact rerun with camera logging | step23 `s=0.008056`, `r=0.000226`, `theta=0.02989`, consistency `0.8065`, `strict_success=true` | strict_success_reproduced | preserve |

Strict evidence folder:

- `outputs/agentic_reward_curriculum_20260524_controller_contact/runs/2026-05-24_09-23-09_strict_candidate_x015_zneg024_ax60_images`
- Metrics: `metrics.jsonl`, `strict_metrics.csv`, `strict_success.json`.
- Reproducibility: `run_config.json`, `command.txt`, `git_status.txt`, `git_diff.patch`.
- Visual evidence: `step_images/step_000023/env_0000_center_camera.png`, `env_0000_left_camera.png`, `env_0000_right_camera.png`.
- Videos: `center_camera_insertion.mp4`, `left_camera_insertion.mp4`, `right_camera_insertion.mp4`.

Why this is not a tip-depth false positive:

- Strict success is from post-step geometry, not reward return.
- The strict row satisfies all four numerical gates simultaneously: seated depth, sub-0.5 mm lateral error, semantic tip orientation below 0.030 rad, and module consistency above 0.80.
- The module consistency gate is positive at the strict row (`0.8065`), unlike earlier lateral-bypass runs where positive `s` appeared with near-zero consistency.
- Center, left, and right images at step 23 show the module aligned with and entering the cage rather than bypassing laterally.

Validation after the diagnostic probe change:

- `.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
- `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py`
- Latest result: `30 passed in 12.95s`.

Next recommended step:

Port the successful final-window recipe into the SERL train/eval guard path as a config-driven final-window reset-orientation trim plus target-tip depth hold, then evaluate it from the best checkpoint and at least a small randomized near-window set. The current success is a strong proof that strict geometry is achievable in Isaac, but it is not yet a learned-policy or randomized-start result.

## 2026-05-25 Non-Contact Full-Depth Policy/Guide Attempts

Corrected full-depth non-contact starts were evaluated from the current best policy checkpoint and from guided probes. These starts use the existing outside-gate convention: axial distance is distance before the entrance, not positive insertion depth. The historical curriculum values were `10/20/30/40 mm` axial outside the gate with lateral offsets up to `10 mm`.

Policy video evidence already saved:

- `outputs/agentic_reward_curriculum_20260524_depth_correction/policy_eval_runs/2026-05-25_13-10-25_full_policy_start10x10_progress10_eval_video/env0000_center_full_episode_20fps_quality.mp4`
- `outputs/agentic_reward_curriculum_20260524_depth_correction/policy_eval_runs/2026-05-25_13-10-25_full_policy_start10x10_progress10_eval_video/env0000_left_full_episode_20fps_quality.mp4`
- `outputs/agentic_reward_curriculum_20260524_depth_correction/policy_eval_runs/2026-05-25_13-10-25_full_policy_start10x10_progress10_eval_video/env0000_right_full_episode_20fps_quality.mp4`
- `outputs/agentic_reward_curriculum_20260524_depth_correction/policy_eval_runs/2026-05-25_13-15-23_full_policy_start40x10_progress10_eval_video/env0000_center_full_episode_20fps_quality.mp4`
- `outputs/agentic_reward_curriculum_20260524_depth_correction/policy_eval_runs/2026-05-25_13-15-23_full_policy_start40x10_progress10_eval_video/env0000_left_full_episode_20fps_quality.mp4`
- `outputs/agentic_reward_curriculum_20260524_depth_correction/policy_eval_runs/2026-05-25_13-15-23_full_policy_start40x10_progress10_eval_video/env0000_right_full_episode_20fps_quality.mp4`

Key results:

| Run | Start | Best/final post-step row | Failure label | Decision |
| --- | --- | --- | --- | --- |
| `full_policy_start10x10_progress10_eval_video` | 10 mm axial / 10 mm lateral, policy only | final `s=21.516 mm`, `r=15.413 mm`, `theta=0.15244` | lateral_bypass / tip-depth false-positive risk | reject |
| `full_policy_start40x10_progress10_eval_video` | 40 mm axial / 10 mm lateral, policy only | best r step53 `s=-35.310 mm`, `r=0.939 mm`, `theta=0.09124`; final `s=-24.658 mm`, `r=13.173 mm`, `theta=0.18435` | no_axial_progress / lateral-orientation blocked | reject |
| `eval_trained_sparse_start40x10_video` | sparse online SERL from 40/10 | final `s=-18.304 mm`, `r=1.256 mm`, `theta=0.09749` | improved but still outside; orientation blocked | reject for strict success |
| `guide_probe_start10x10_strong_lateral` | guided 10/10, strong lateral guard | final `s=-0.156 mm`, `r=0.162 mm`, `theta=0.06449` | near_gate_orientation_blocked | promote as diagnostic baseline |
| `guide_probe_start10x10_orientation_probe_basis` | final orientation basis probe | best theta `0.06401`, final `s=-0.853 mm`, `r=0.846 mm`, `theta=0.06418` | controller realization mismatch | reject |
| `guide_probe_start10x10_guard_fixedworld_best` | guard fixed-world best-axis trim | best theta `0.06378`, final `s=-0.639 mm`, `r=0.613 mm`, `theta=0.06407` | controller realization mismatch | reject |
| `guide_probe_start10x10_rootframe_large_fixedworld_best` | larger 0.060 rad fixed-world trim | best theta `0.05763` at `r=2.411 mm`; best s `-0.171 mm` at `r=0.640 mm` | rotation-induced lateral sweep | reject |
| `guide_probe_start10x10_rootframe_midrot_strictgate` | 0.030 rad trim plus strict predicted-r gate | best s `-0.326 mm`, `r=0.487 mm`, `theta=0.06546`; trim mostly rejected | near_gate_orientation_blocked | reject |
| `guide_probe_start10x10_calibrated_orientation` | naive reset orientation calibration from measured axis | first row `s=-6.219 mm`, `r=33.095 mm`, `theta=0.06312`; final `s=0.357 mm`, `r=15.753 mm` | reset calibration regression / lateral bypass | reject |

Interpretation:

- The videos now start from non-contact outside-gate states. The earlier strict full-depth video started from a final-window reset around `s=43.5 mm`, which is why it looked already inserted.
- The learned policy does not yet perform strict full insertion from 10/10 or 40/10 starts.
- Sparse training from 40/10 improved final approach metrics but remained outside the cage and above orientation threshold.
- Strong guided lateral correction can center from a 10 mm lateral start, but semantic tip orientation stalls around `0.064 rad`.
- Orientation commands are aimed in the right direction: diagnostics predicted theta reductions to roughly `0.060` or lower, but realized semantic theta changed by only a small fraction of the predicted amount.
- Increasing rotation authority can reduce theta slightly, but it creates lateral sweep and loses the strict lateral gate.
- The isolated orientation-calibrated reset was rejected because it preserved the intended YAML relation but not the actual Isaac semantic tip placement; it produced a large lateral reset error. This suggests the next fix should calibrate the actual Isaac gripper/TCP-to-`sfp_tip_link` transform during reset generation, not apply a post-hoc world rotation to stale YAML offsets.

Next recommended step:

Add a small Isaac-side reset calibration diagnostic that solves for the reset TCP pose using measured post-reset `sfp_tip_link` pose, then regenerates the near-gate episode config with both semantic tip position and semantic tip axis constrained. Do this before more SERL training; current failures are controller/reset-realization dominated, so reward-only training is unlikely to produce strict full insertion from 10/10 or 40/10 starts.

### 2026-05-25 reset-settle calibration update

The first measured reset calibration did place the semantic tip correctly in the instantaneous reset diagnostic, but the first zero/hold environment step produced a large deterministic settle:

- `resetcheck_10x10_wrist_tip_calibrated`: reset diagnostic `s=-9.455 mm`, `r=10.256 mm`, `theta=0.00035`; first post-step `s=-6.502 mm`, `r=33.222 mm`, `theta=0.06345`, tip moved `23.836 mm`.
- A settle-compensated wrist reset config was generated at `outputs/agentic_reward_curriculum_20260524_depth_correction/generated_episode_configs/full_depth_start_10mm_axial_10mm_lateral_wrist_tip_calibrated_settle1`.
- `resetcheck_10x10_wrist_tip_calibrated_settle1`: reset diagnostic `s=-12.489 mm`, `r=15.118 mm`, `theta=0.0`; first post-step `s=-9.983 mm`, `r=8.561 mm`, `theta=0.06169`.

Guided probes from the settle-compensated non-contact start:

| Run | Key change | Best/final post-step row | Failure label | Decision |
| --- | --- | --- | --- | --- |
| `guide_probe_start10x10_wrist_settle1` | settle-compensated wrist reset, lateral/target-tip guard | final `s=-0.157 mm`, `r=0.145 mm`, `theta=0.06529`; best r step106 `r=0.068 mm`, `theta=0.06501` | near_gate_orientation_blocked | promote as current non-contact baseline |
| `guide_probe_start10x10_wrist_settle1_preentry_orient` | pre-entry final-orientation trim | best s `-0.172 mm`, `r=0.691 mm`, `theta=0.06453`; final `s=-0.503 mm`, `r=0.684 mm`, `theta=0.06409` | orientation_plateau | reject |
| `guide_probe_start10x10_wrist_settle1_guard_fixedworld_best` | guard fixed-world best-axis micro-rotation | final `s=-0.451 mm`, `r=0.570 mm`, `theta=0.06470`; fixed-world rotation active on 112/240 steps | controller_realization_mismatch | reject |
| `diagnostic_direct_tip_ik_wristframe_start10x10` | `AIC_ISAAC_IK_BODY_NAME=sfp_tip_link`, wrist-frame commands | final `s=-0.695 mm`, `r=1.781 mm`, `theta=0.06785` | direct_tip_ik_unstable_or_ineffective | reject |
| `diagnostic_direct_tip_ik_tipframe_start10x10` | `AIC_ISAAC_IK_BODY_NAME=sfp_tip_link`, tip-frame commands | final `s=-0.757 mm`, `r=1.414 mm`, `theta=0.06446` | direct_tip_ik_unstable_or_ineffective | reject |

Code validation after allowing diagnostic body-name action frames:

- `.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`

Interpretation:

- The old 8 mm/shallow checkpoint is still useful as initialization for approach behavior, but it is not a full-insertion policy and cannot be used to claim success under the corrected 45.8 mm seated-depth criterion.
- From the compensated 10 mm axial / 10 mm lateral non-contact start, the current guard can reliably drive to the entrance and center laterally, but it cannot reduce semantic tip orientation below roughly `0.062 rad`.
- The axial guard is doing the correct thing: it refuses forward insertion because `theta > 0.030 rad`. No run in this section has strict success.
- Direct `sfp_tip_link` IK was tested only as a diagnostic and did not solve the orientation plateau, so it should not be promoted to training.

Next recommended code change:

Add a controller-level semantic tip orientation diagnostic/guard that commands bounded orientation refinements while explicitly measuring realized semantic axis change and induced lateral sweep per step. If the realized semantic axis remains pinned near `0.062 rad` under controlled rotations, stop reward/curriculum tuning and investigate asset joint/contact constraints around the cable/module assembly.

### 2026-05-25 front-entry SFP geometry correction

A separate depth/geometry audit found that the previous Isaac SFP episode generator treated the back of the SFP cage as `entrance_pose_world` and used an insertion axis pointing back toward the front opening. That explains why some videos appeared to start already inserted: a nominal `start_near_gate` outside-gate state was outside only relative to the reversed semantic frame, not relative to the physical port mouth.

Code changes:

- `aic_utils/aic_isaac/scripts/isaac_episode_configs.py`
  - Added `SFP_PORT_CAGE_DEPTH_M = 0.04872` from the Gazebo/official SDF cage collision length.
  - Changed SFP semantic entrance to the front mouth: local port offset `(0, 0, +0.002834)`.
  - Changed SFP insertion axis to point into the cage: local `(0, 0, -1)`.
  - Default SFP target without explicit `seated_depth_m` now uses the full cage depth.
- `aic_utils/aic_isaac/scripts/run_one_day_insertion_pipeline.py`
  - Updated generated full-depth SFP target from `45.8 mm` with a back-frame offset to `48.72 mm` from the corrected front mouth.
- `aic_utils/aic_isaac/test/test_isaac_online_serl.py`
  - Updated the SFP entrance/target depth expectation to `0.04872 m`.

Validation:

- `.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/scripts/isaac_episode_configs.py aic_utils/aic_isaac/scripts/run_one_day_insertion_pipeline.py`
- `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_isaac_online_serl.py::test_sfp_entrance_axis_offset_shifts_semantic_gate aic_utils/aic_isaac/test/test_isaac_online_serl.py::test_sfp_seated_depth_override_places_target_along_entrance_axis aic_utils/aic_isaac/test/test_isaac_online_serl.py::test_near_gate_distances_are_to_semantic_tip_center_with_body_offset`
- Result: `3 passed`.

Front-entry reset diagnostics:

| Run | Key result | Decision |
| --- | --- | --- |
| `resetcheck_front_entry_direct_tip` | reset tip `s=-10.167 mm`, `r=9.779 mm`, but `theta=pi`; module is ahead of tip, meaning plug orientation is reversed for the corrected axis | reject |
| `resetcheck_front_entry_orientation_sweep` | `pre_ry_pi` candidate gives reset tip `s=-10.303 mm`, `r=10.358 mm`, `theta=0.02523`, and module behind tip by `23.642 mm`; direct-tip reset then explodes under physics | promote orientation only |
| `resetcheck_front_entry_10x10_wrist_prery` | wrist-calibrated from old transform lands in high-force penetration: post-step `s=165.763 mm`, `r=317.184 mm`, `theta=1.81763`, contact force `~2.60e6` | reject |
| `resetcheck_front_entry_10x10_wrist_prery_from_sweep` | wrist-calibrated from the direct-tip sweep still lands in high-force penetration: post-step `s=173.767 mm`, `r=294.391 mm`, `theta=1.75139`, contact force `~2.23e6` | reject |

Interpretation:

- The user concern was correct: for Gazebo-like full insertion, the physical SFP cage depth is `48.72 mm`, and the prior Isaac semantic frame was reversed/back-referenced.
- The old `8 mm` and later `45.8 mm` results should be treated as superseded under the corrected front-entry frame.
- The immediate blocker is now reset/controller feasibility with the corrected frame. A direct semantic tip reset can describe the desired outside start and orientation, but direct-tip IK is unstable and violates the prior constraint. Wrist IK calibration using the old transform does not yet realize the corrected semantic pose and produces high-force penetration.

Next recommended code change:

Implement a wrist-reset solver that derives the wrist pose from the desired `sfp_tip_link` pose using the current measured `wrist_3_link -> sfp_tip_link` transform at the same orientation family, then validates with a zero-action settle step before any policy/guide rollout. Do not resume reward training until the corrected front-entry `10 mm axial / 10 mm lateral` reset has low contact force, post-step `s<0`, `r≈10 mm`, and `theta<0.030 rad`.

### 2026-05-26 corrected-depth warm-start probes

Follow-up validation supersedes the intermediate front-entry-axis note above. The current working SFP geometry in `isaac_episode_configs.py` is:

- `SFP_PORT_ENTRANCE_LOCAL = (0.0, 0.0, -0.0458)`
- `SFP_PORT_INSERTION_AXIS_LOCAL = (0.0, 0.0, 1.0)`
- `SFP_PORT_CAGE_DEPTH_M = 0.04872`
- semantic tip local offset `(0.0, -0.02365, 0.0)`

The previous shallow/8 mm policy is useful only as a warm start. It does start from a non-contact corrected reset in the stable `40 mm axial / 10 mm lateral` old-offset config, but it is not a full insertion policy under the corrected full-depth criterion.

New guard patch:

- Added optional measured-depth target-tip servo flags in `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`:
  - `--insertion_action_guard_target_tip_servo_stable_steps`
  - `--insertion_action_guard_target_tip_servo_realized_depth_limit_m`
  - `--insertion_action_guard_target_tip_servo_realized_depth_backoff_m`
- Purpose: prevent apparent insertion from lateral/contact/controller coupling when the axial gate did not actually command inward motion.

Validation:

- `.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
- `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_isaac_online_serl.py::test_sfp_entrance_axis_offset_shifts_semantic_gate aic_utils/aic_isaac/test/test_isaac_online_serl.py::test_sfp_seated_depth_override_places_target_along_entrance_axis aic_utils/aic_isaac/test/test_isaac_online_serl.py::test_near_gate_distances_are_to_semantic_tip_center_with_body_offset`
- Result: `32 passed`.

Candidate runs:

| Run | Start/checkpoint | Best strict-relevant metrics | Final metrics | Failure label | Decision |
| --- | --- | --- | --- | --- | --- |
| `guide_probe_40x10_oldoffset_tight_r_microaxial_servo` | 40 mm axial / 10 mm lateral, warm-start checkpoint | best `r=0.056 mm` at `s=-15.716 mm`, `theta=0.07849`; best `s=-1.059 mm`, `r=0.381 mm`, `theta=0.06906` | no strict success, module consistency `0` | orientation_blocked / module_consistency_blocked | reject |
| `guide_probe_40x10_oldoffset_depth_limited_servo` | same, measured-depth limiter | best `s=-19.013 mm`, `r=0.443 mm`, `theta=0.06562` | no strict success, module consistency `0` | orientation_blocked; depth limiter active | reject |
| `guide_probe_start10x10_theta0022_warmstart_guarded` | isolated 10/10 reset from orientation sweep, warm-start checkpoint | best combo `s=-6.115 mm`, `r=0.052 mm`, `theta=0.02993`; best `r=0.015 mm` | final `s=-0.399 mm`, `r=0.066 mm`, `theta=0.04783`, consistency `0` | shallow_entry_orientation_drift | promote for focused shallow-entry tuning |
| `guide_probe_start10x10_theta0022_axial_gate_fast` | same reset, axial gate stable-steps=1 | first axial gate step 65; first positive `s` step 587; best `s=0.449 mm`, `r=0.106 mm`, `theta=0.06175` | final `s=0.449 mm`, consistency `0` | shallow_entry_orientation_drift | reject as strict run, useful evidence |
| `guide_probe_start10x10_theta0005_axial_gate_fast` | lower-theta 10/10 reset, axial gate stable-steps=1 | first axial gate step 100; best combo `s=-2.030 mm`, `r=0.063 mm`, `theta=0.02999`; best `r=0.009 mm` | final `s=-0.220 mm`, `r=0.093 mm`, `theta=0.05767`, consistency `0` | shallow_entry_orientation_drift | reject as strict run |
| `diagnostic_start10x10_theta0005_loose_theta_depth_reach` | lower-theta reset, diagnostic loose theta gate 0.060 | best `s=-0.158 mm`, `r=0.077 mm`, `theta=0.01922`; best `r=0.020 mm`, `theta=0.00413` | no positive `s`; final `theta=0.10515`, consistency `0` | orientation/contact drift prevents depth travel | reject |

Interpretation:

- Yes, the old 8 mm-capable model/checkpoint can be used as a warm start, and it can start from non-contact resets.
- Under the corrected full-depth success definition, it only gets us to approach/centering. It has not demonstrated full insertion.
- The best current non-contact candidate is the 10 mm axial / 10 mm lateral low-theta reset plus warm-start checkpoint. It reaches strict lateral/orientation gates before insertion, but contact/controller dynamics increase semantic tip orientation during shallow entry.
- No run has strict success. Positive tip depth remains a false-positive risk because module consistency stays `0` and consistency axial error remains about `45-46 mm`.

Next recommended code change:

Add a measured semantic-orientation hold/recovery guard for the shallow-entry window: when `r < 0.1 mm` and `theta < 0.03`, freeze or back off axial motion as soon as measured `theta` increases, then test bounded micro-rotations with realized semantic-axis feedback before allowing further depth. If the measured semantic axis cannot be held under contact, this becomes a controller/asset/contact blocker rather than a reward issue.

### 2026-05-26 shallow-entry orientation recovery and full-depth diagnostics

Code change:

- Added optional target-tip servo orientation recovery flags in `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`:
  - `--insertion_action_guard_target_tip_servo_orientation_recovery`
  - `--insertion_action_guard_target_tip_servo_orientation_recovery_activation_depth_m`
  - `--insertion_action_guard_target_tip_servo_orientation_recovery_lateral_gate_m`
  - `--insertion_action_guard_target_tip_servo_orientation_recovery_theta_limit_rad`
  - `--insertion_action_guard_target_tip_servo_orientation_recovery_worsen_margin_rad`
  - `--insertion_action_guard_target_tip_servo_orientation_recovery_backoff_m`
- The guard records measured semantic theta deltas and prevents stable axial release while the shallow-entry theta recovery condition is active.

Validation:

- `.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
- `.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_isaac_online_serl.py::test_sfp_entrance_axis_offset_shifts_semantic_gate aic_utils/aic_isaac/test/test_isaac_online_serl.py::test_sfp_seated_depth_override_places_target_along_entrance_axis aic_utils/aic_isaac/test/test_isaac_online_serl.py::test_near_gate_distances_are_to_semantic_tip_center_with_body_offset`
- Result: `32 passed`.

Candidate runs:

| Run | Start/checkpoint | Best strict-relevant metrics | Final metrics | Failure label | Decision |
| --- | --- | --- | --- | --- | --- |
| `guide_probe_start10x10_theta0005_orientation_recovery` | non-contact 10 mm axial / 10 mm lateral low-theta reset, warm-start checkpoint | best combo `s=-8.247 mm`, `r=0.059 mm`, axis `theta=0.00742`; best strict r/theta depth `s=-0.530 mm`, `r=0.017 mm`, axis `theta=0.02702`; first positive `s=0.004 mm` had `r=1.196 mm` | final `s=-0.965 mm`, `r=7.859 mm`, axis `theta=0.13520`, consistency `0` | shallow lateral bypass after theta recovery | reject |
| `guide_probe_start10x10_theta0005_final_orientation_probe` | same, final orientation probe-basis guide | best `s=-0.480 mm`, `r=0.017 mm`, axis `theta=0.01987`, consistency `0` | final `s=-0.605 mm`, `r=0.109 mm`, axis `theta=0.02006`, consistency `0` | hover just outside entrance; no axial progress | promote as stable pre-entry evidence |
| `guide_probe_start10x10_theta0005_probe_release_axial40um` | same, released axial gate at 40 um | best `s=-0.488 mm`, `r=0.040 mm`, axis `theta=0.02584`; guarded axial command was active but realized tip-depth ratio collapsed near contact | final `s=-0.535 mm`, `r=0.175 mm`, axis `theta=0.02612`, consistency `0` | controller/contact realization mismatch at entrance | reject |
| `guide_probe_start10x10_fullquat_final_orient_axial40um` | same, full-quaternion semantic orientation | best `s=-0.232 mm`, `r=0.343 mm`, full-quat `theta=0.08231`, consistency `0` | final `s=-0.387 mm`, `r=0.366 mm`, full-quat `theta=0.08169`, consistency `0` | full-quat orientation residual | reject |
| `guide_probe_start10x10_fullquat_strong_final_orient_axial40um` | same, stronger full-quat final rotation | best combo `s=-0.376 mm`, `r=0.060 mm`, full-quat `theta=0.08160`; best theta `0.08089` | final `s=-0.499 mm`, `r=0.770 mm`, full-quat `theta=0.08193`, consistency `0` | full-quat orientation residual; stronger rotation did not solve | reject |
| `diagnostic_inside5mm_fullquat_seating_axial80um` | diagnostic only, requested inside-start `s=+5 mm`, full-quat reset | post-reset immediately settled/ejected to `s=-0.853 mm`; best `s=0.487 mm`, `r=0.378 mm`, full-quat `theta=0.06698` | final `s=0.486 mm`, `r=0.378 mm`, full-quat `theta=0.06698`, consistency `0` | inside reset/contact instability; not a valid success start | reject |
| `train_full_depth_fullquat_neargate_residual_3k` | short online residual training from warm-start checkpoint; stopped after saved step-1000 checkpoint | training reward improved from about `-5.85` to `-4.66`, but geometry remained poor; best training `s=-0.521 mm`, `r=0.177 mm`, full-quat `theta=0.30950` | latest training metric `s=-0.594 mm`, `r=0.135 mm`, full-quat `theta=0.27459`, consistency `0` | unstable/ineffective learning for final orientation | reject |
| `eval_train_fullquat_residual_1k_video` | frozen eval of the step-1000 residual checkpoint | best `s=-0.265 mm`, `r=1.733 mm`, full-quat `theta=0.16014`; best theta `0.12783`; no axial gate | final `s=-0.265 mm`, `r=1.733 mm`, full-quat `theta=0.16014`, consistency `0` | trained residual worsened strict metrics | reject |

Video evidence:

- `outputs/agentic_reward_curriculum_20260524_depth_correction/guide_probe_runs/2026-05-26_02-45-12_eval_train_fullquat_residual_1k_video/env0000_center_full_episode_20fps_quality.mp4`
- `outputs/agentic_reward_curriculum_20260524_depth_correction/guide_probe_runs/2026-05-26_02-45-12_eval_train_fullquat_residual_1k_video/env0000_left_full_episode_20fps_quality.mp4`
- `outputs/agentic_reward_curriculum_20260524_depth_correction/guide_probe_runs/2026-05-26_02-45-12_eval_train_fullquat_residual_1k_video/env0000_right_full_episode_20fps_quality.mp4`

Interpretation:

- Axis-only orientation can create a near-entry false positive: `r` and axis theta become strict near `s=-0.5 mm`, but the plug does not enter and module consistency stays `0`.
- Full-quaternion semantic orientation exposes a residual around `0.08 rad` even when axis theta looks good. This likely explains why the rectangular plug cannot pass the port mouth.
- The guarded axial servo commands positive depth at the entrance, but realized tip-depth projection collapses under contact and force rises. This is now stronger evidence for a controller/contact/orientation-realization blocker than a reward-only blocker.
- The short residual SERL run improved reward but worsened strict geometry in held-out eval, so reward return remains explicitly rejected as success evidence.

Next recommended code change:

Add a small controller diagnostic that commands full-quaternion semantic tip alignment in free space and at the entrance while logging realized `wrist_3_link -> sfp_tip_link` transform changes, contact force, and full-quat theta. If full-quat theta cannot be driven below `0.03 rad` without contact-induced lateral sweep using wrist IK, stop reward tuning and fix the wrist-to-tip compensation/controller path before further SERL training.
