# Isaac Near-Gate Experiments - 2026-05-15

## Scope

Investigated `start_near_gate` for SFP-to-NIC with axial/lateral distances measured from the port entrance to the semantic end-effector tip center. The key request was axial 6 mm, lateral 6 mm, where axial is along the insertion axis and positive insertion proceeds into the port.

## Code Changes

- Added semantic tip-center handling in `aic_utils/aic_isaac/scripts/isaac_episode_configs.py`:
  - `scene.end_effector_tip.body_name`
  - `scene.end_effector_tip.body_position_offset`
  - per-task defaults: SFP uses `sfp_tip_link`, SC uses `sc_tip_link`
  - reset metadata now records `reference_tip_center_position_world` and the body offset used to place the reset body.
- Updated online SERL wrapper/trainer to use materialized episode tip body/offset when `--target-reward-body auto`.
- Added near-gate IK reset overrides to the SERL launcher/trainer:
  - `--near-gate-reset-max-iterations`
  - `--near-gate-reset-position-tolerance`
  - `--near-gate-reset-orientation-tolerance`
- Fixed direct ACT mode semantics so `act_direct` predicts the full TCP delta action directly instead of adding a residual to ACT.
- Fixed reward audit default target depth to 8 mm so the insertion-geometry audit runs under the helper's valid depth checks.
- Fixed the SFP module-to-tip body offset sign used by semantic episode generation: the measured local offset from `sfp_module_link` to `sfp_tip_link` is approximately `[0.0, +0.02365, 0.0]`, not negative Y.
- Changed reward presets in both Isaac trainer entry points so explicit CLI reward weights override preset defaults. This was needed because an axial/corridor sweep initially logged the preset defaults despite explicit weights on the command line.

## Verification

Formula and launcher tests:

```text
21 passed
```

Reward audit:

```text
outputs/reward_audits/near_gate_corridor_v1_20260515_112904
```

It confirmed the corridor reward peaks at seated, centered depth and penalizes off-axis forward/bypass motion.

## Near-Gate Reset Findings

Materialized geometry for SFP-to-NIC axial 6 mm / lateral 6 mm is semantically correct: the generated tip-center reference is outside the entrance plane by 6 mm and laterally offset by 6 mm.

However, Isaac zero-action audits show the reset is not physically stable. With zero executed TCP action, the tip moves laterally from roughly 0-7 mm to 21-29 mm after the first environment step, and the wrist force is already clipped at 35 N at reset. This happens even when:

- the reset IK tolerance is tightened from 2.0 mm to about 0.06 mm actual final error,
- lateral offset is 0 mm instead of 6 mm,
- lateral side is flipped,
- action guide and action guard are disabled,
- the direct actor outputs zero action.

Representative runs:

| Run | Start Geometry | Result |
| --- | --- | --- |
| `near_gate_zero_direct_noguard_audit_20260515_114130` | sfp_tip_link, 6 mm axial / 6 mm lateral | zero action jumps lateral 7.5 mm -> 21.4 mm |
| `near_gate_zero_direct_tightreset_audit_20260515_114558` | sfp_tip_link, tight reset, 6/6 | starts at s=-5.96 mm, r=6.05 mm, then r=21.8 mm |
| `near_gate_zero_direct_centered_synced_audit_20260515_115124` | sfp_tip_link, tight reset, 6/0 | starts centered r=0.05 mm, then r=23.9 mm |
| `near_gate_tipcenter_module_centered_audit_20260515_115654` | sfp_module_link + tip offset, 6/0 | worse; force remains clipped and signed depth runs away |

Visual inspection of saved frames agreed with the metrics: the plug appears to be in/near contact at reset and then passively ejects/slips on the first physics step, not as a learned-policy action.

## Handoff r105 Restart Probe

After re-reading `docs/isaac_near_gate_handoff_20260515.md`, I recreated the most useful old-server diagnostic template rather than starting with reward learning:

- request: `outputs/analysis/isaac_near_gate_handoff_r105/request.yaml`
- materialized episodes: `outputs/analysis/isaac_near_gate_handoff_r105/episode_configs/episodes/`
- run: `outputs/train/isaac_online_serl_near_gate/audit/2026-05-15_12-03-03_handoff_r105_cheatcode_tip_20260515_120240`
- guide: `cheatcode_transform`, root TCP action frame, collect blend `1.0`, no learning updates
- start: `sfp_tip_link` near gate, axial `0.5 mm`, lateral `0.2 mm`, seated depth `8 mm`
- reset body: `gripper_tcp`, final reset error about `0.19 mm`
- reward preset: `near_gate_corridor_v1`

Artifacts:

- `videos/env0_center_h264.mp4`
- `videos/env0_left_h264.mp4`
- `videos/env0_right_h264.mp4`
- `plots/port_frame_trajectory_env0.png`
- `audit_log.jsonl`
- `diagnostics_summary.json`

Env0 metric summary:

| Body | First s/r/dist | Final s/r/dist | Best lateral | Max signed depth |
| --- | --- | --- | --- | --- |
| `sfp_tip_link` | `-0.282 / 0.540 / 8.299 mm` | `0.159 / 0.871 / 7.889 mm` | step 2, `r=0.245 mm`, `s=0.004 mm` | step 45, `s=0.167 mm`, `r=0.864 mm` |
| `sfp_module_link` | `-23.886 / 2.009 / 31.949 mm` | `-23.451 / 0.807 / 31.461 mm` | step 48, `r=0.799 mm`, `s=-23.446 mm` | step 45, `s=-23.443 mm`, `r=0.807 mm` |
| `gripper_tcp` | `-59.366 / 7.976 / 67.836 mm` | `-58.917 / 6.681 / 67.250 mm` | step 6, `r=6.565 mm`, `s=-58.978 mm` | step 47, `s=-58.908 mm`, `r=6.697 mm` |

Visual inspection of first/last frames agrees with the metrics: the visible module hovers near the gate with slight alignment change, but does not visibly insert. The `sfp_tip_link` metric crosses the entrance plane by only about `0.16 mm`, while `sfp_module_link` remains about `23.5 mm` outside axially. This reinforces the handoff warning that `sfp_tip_link` alone can overstate insertion quality.

The center-camera saved PNGs/videos contain the diagnostic legend only because `--debug-visual-overlays` was enabled. The training/evaluation model path does not receive that overlay: `_raw_camera_images()` reads camera RGB tensors, `_act_obs_from_env()` consumes those tensors for ACT/SERL, and `_save_images()` later copies the tensor to a PIL image and draws the overlay on the saved visualization copy. The left/right saved PNGs are not overlaid.

## Online SERL Reward Iterations

All runs below used the r105 near-gate episodes, ACT TorchScript checkpoint `175000`, `act_direct`, root TCP action frame, `n_action_steps=1`, guide mode `cheatcode_transform`, guide collect steps `100`, and `target_reward_body=sfp_tip_link`.

| Run | Reward / Stability Change | Result |
| --- | --- | --- |
| `online_r105_tip_direct_guide100_bc5_20260515_125356` | baseline `near_gate_corridor_v1`, guide BC weight `5.0` | stable after guide cutoff; final `s=0.599 mm`, `r=0.836 mm`; visually holds near gate with slight forward progress |
| `online_r105_tip_direct_axial05_corr1_20260515_125931` | intended axial `0.50`, corridor `1.00`, but preset bug kept `0.25/0.50` | useful longer baseline; final `s=0.686 mm`, `r=0.745 mm`; no bypass |
| `online_r105_tip_direct_axial05_corr1_fixedpreset_20260515_130457` | actual axial `0.50`, corridor `1.00` after preset fix | best bounded metric result; max `s=0.834 mm`, final `s=0.537 mm`, final `r=0.785 mm`; no off-axis positive axial-reward events |
| `online_r105_tip_direct_axial075_corr05_orient025_20260515_131023` | axial `0.75`, corridor `0.50`, orientation `0.25`, force penalty `0.02`, guide BC `10.0` | rejected; transient max `s=0.760 mm`, then large negative axial event at step 170 and final retreat to `s=-0.137 mm`, `r=1.595 mm` |

H264 videos were generated for all three cameras for the fixed-preset and rejected orientation-heavy runs:

- `outputs/train/isaac_online_serl_near_gate/online/2026-05-15_13-05-19_online_r105_tip_direct_axial05_corr1_fixedpreset_20260515_130457/videos/`
- `outputs/train/isaac_online_serl_near_gate/online/2026-05-15_13-10-46_online_r105_tip_direct_axial075_corr05_orient025_20260515_131023/videos/`

Detailed interpretation:

- The reward geometry is behaving correctly with respect to lateral gating: there were zero cases where positive axial-progress reward was paid while `r > 2.5 mm`.
- Increasing corridor weight improves shallow centered depth, but it also makes the total reward high while the module is still visibly outside the receptacle.
- Visual inspection of step 100/180/220 for the fixed-preset run shows the SFP/module remains at the gate edge; the logged `sfp_tip_link` depth increase from `0.166 mm` to `0.720 mm` is real at the point level but not a visible seated insertion.
- The orientation-heavy variant made reward less semantically useful: orientation dominated early reward, then the policy hit a brittle negative event and retreated.
- The actor after guide cutoff mostly learns a clipped translational push; its rotation action is far smaller than the guide rotation. This is why the visible module can stay angled even while the tip point advances slightly.

## Current Diagnosis

The original insertion failure is a combination of:

- disabled or underweighted insertion corridor rewards in older defaults,
- direct actor mode previously still behaving like a residual-to-ACT adapter,
- and, most importantly for the current near-gate curriculum, reset/contact geometry and target-body semantics.

The current blocker is not reward learning yet. The 6 mm axial reset places the semantic tip point correctly, but the physical SFP/port collision state is invalid or under tension at reset. The handoff r105 restart additionally shows that even a very close `0.5 mm` axial / `0.2 mm` lateral guided probe only moves `sfp_tip_link` slightly across the entrance while the visible module remains outside. This must be resolved before online SERL training can produce meaningful insertion data.

For the r105 close-start online curriculum, the current best short-run setting is `online_r105_tip_direct_axial05_corr1_fixedpreset_20260515_130457`. It is stable and centered, but it is not insertion. The reward issue is now more specific: tip-only corridor depth is too permissive as the main positive insertion signal. The next reward change should either gate corridor reward by orientation or add a second semantic plug-body consistency term before shallow tip depth receives large positive reward.

## Next Recommended Run

Do not start long SERL training until a zero-action near-gate reset holds position for at least 5-10 steps with low initial contact force.

Next concrete experiment:

1. Add an orientation-gated corridor term or a compound insertion diagnostic/reward requiring both `sfp_tip_link` entrance progress and plug-body consistency, so tip-only crossing cannot dominate reward.
2. Add action diagnostics or a guide-loss split for translation vs rotation; the current actor preserves the translational guide but largely drops the rotational component after guide cutoff.
3. Add or inspect collision geometry/contact pair diagnostics for the SFP module and NIC port.
4. Calibrate `entrance_axis_offset_m` outward until centered axial-6mm reset has low force and does not eject under zero action.
5. Once stable, rerun the 6 mm lateral curriculum with the best r105 setting, direct actor mode, actor updates delayed, and the insertion action guard enabled.

## CheatCode Phase Reward v1

After stopping the guide-tuning experiments, I implemented a new `cheatcode_insertion_v1` reward preset that encodes the Gazebo `CheatCode` phase logic directly:

- align laterally and orientationally before insertion,
- keep a safe pre-insertion hover depth while misaligned,
- make inward axial motion negative outside a tight alignment tube,
- reward centered depth only inside the tight tube,
- penalize inside-port lateral/orientation drift and retreat,
- require axial, lateral, orientation, and SFP body-consistency checks for success.

Changed code:

- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/insertion_geometry.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py`
- `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py`
- `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
- `aic_utils/aic_isaac/scripts/train_isaac_online_serl.py`
- `aic_utils/aic_isaac/scripts/audit_insertion_reward_geometry.py`
- `aic_utils/aic_isaac/test/test_insertion_reward_geometry.py`

Formula audit:

- output: `outputs/reward_audits/cheatcode_insertion_v1/`
- plots:
  - `cheatcode_phase_total_theta_0.00.png`
  - `cheatcode_phase_total_theta_0.05.png`
  - `cheatcode_phase_total_theta_0.10.png`
  - `cheatcode_phase_total_theta_0.15.png`
- summary: `outputs/reward_audits/cheatcode_insertion_v1/summary.json`

Key formula checks:

| Theta | 6 mm lateral inward axial reward | 0.5 mm lateral pre-entry inward reward |
| --- | ---: | ---: |
| `0.00 rad` | `-0.500` | `+0.488` |
| `0.05 rad` | `-0.500` | `+0.110` |
| `0.10 rad` | `-0.500` | `-0.500` |
| `0.15 rad` | `-0.500` | `-0.500` |

The first audit with `sigma_theta_insert=0.05` was too strict: exactly `theta=0.05 rad` still made aligned inward motion negative. I changed the default to `0.06 rad`, still within the suggested range, and regenerated the plots above.

Tests:

```text
.pixi/envs/default/bin/python -m pytest \
  aic_utils/aic_isaac/test/test_insertion_reward_geometry.py \
  aic_utils/aic_isaac/test/test_isaac_online_serl.py -q

33 passed
```

Short Isaac no-learning smoke:

- run: `outputs/train/isaac_online_serl_near_gate/audit/2026-05-15_14-38-58_audit_cheatcode_phase_v1_actonly_30_20260515_143835`
- command: `serl/train.py --reward_preset cheatcode_insertion_v1 --steps 30 --updates 30 --update_every_steps 100000 --warmup_steps 30 --actor_update_start_steps 100000 --act_only --act_only_actor_mode act_direct --target_reward_body sfp_tip_link --episode_config_dir outputs/analysis/isaac_near_gate_6mm_orientation_gate/episode_configs/episodes`
- videos:
  - `videos/env0_center_h264.mp4`
  - `videos/env0_left_h264.mp4`
  - `videos/env0_right_h264.mp4`

Smoke metrics:

| Step | Reward | s mm | r mm | theta rad | phase term | success |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `-1.017` | `-5.918` | `5.999` | `0.078` | `-0.812` | false |
| 10 | `-0.874` | `-4.931` | `4.939` | `0.081` | `-0.689` | false |
| 20 | `-0.818` | `-4.245` | `4.522` | `0.083` | `-0.637` | false |
| 30 | `-0.795` | `-3.576` | `3.901` | `0.084` | `-0.620` | false |

Interpretation:

- The runtime reward plumbing works.
- The reward no longer gives a success bonus or positive insertion reward from axial depth alone.
- At the 6 mm start, the phase reward is negative and dominated by the near-misalignment penalty.
- As ACT reduces lateral error, the penalty becomes less negative, which is intended.
- Axial progress remains negative because the tip is still outside the strict alignment tube.

Next recommended short run:

Use `cheatcode_insertion_v1` for an alignment-only diagnostic by explicitly setting axial/corridor terms to zero through the phase reward weights in code or by adding CLI knobs for the subweights, then run 100-300 no-learning or delayed-learning steps. Only turn on learning after the logged `r` and `theta` improve before `s` increases.
