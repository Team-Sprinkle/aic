# Cheatcode Insertion Reward Validation - 2026-05-15

Branch: `feat/hybrid-train`

## Code changes

- Added cheatcode phase subweight CLI knobs in:
  - `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
  - `aic_utils/aic_isaac/scripts/train_isaac_online_serl.py`
- Added `cheatcode_alignment_v1`, which keeps the phase reward active but disables axial/corridor/inside/retreat subterms.
- Added per-run `cheatcode_phase_summary.json` from `metrics.jsonl`.
- Fixed the phase reward hover/retreat escape:
  - `hover_scale`: `0.004 -> 0.002`
  - hover gate now stays active across the near-gate workspace: `hover_gate_start=-0.100`, `hover_gate_scale=0.010`
- Added regression coverage so far-outside retreat is not a reward escape.
- Aligned default `sigma_theta_insert` to `0.06`.

## Verification

```bash
.pixi/envs/default/bin/python -m pytest aic_utils/aic_isaac/test/test_insertion_reward_geometry.py -q
# 21 passed
```

Formula audit:

```bash
.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/audit_insertion_reward_geometry.py \
  --mode cheatcode \
  --run-name cheatcode_insertion_v1_validation_20260515_hoverfix3 \
  --output-root outputs/reward_audits \
  --target-depth-m 0.010 \
  --sigma-r 0.0015 \
  --bypass-penalty-scale 6.0 \
  --grid 121
```

Audit plots:

```text
outputs/reward_audits/cheatcode_insertion_v1_validation_20260515_hoverfix3/
```

Formula audit result:

| theta | r=6 mm inward | r=0.5 mm inward | max total |
| --- | --- | --- | --- |
| 0.00 | negative | positive | 2.650 |
| 0.05 | negative | positive but small | 0.030 |
| 0.10 | negative | negative | -0.336 |
| 0.15 | negative | negative | -0.569 |

This matches the intended phase order: inward motion is only attractive in the tight aligned tube.

## Simulation Runs

No-learning frozen-ACT rollout after hover fix:

```text
outputs/train/isaac_online_serl_near_gate/audit/2026-05-15_15-13-39_audit_cheatcode_phase_v1_actadapter_hoverfix_60_20260515_151317
```

Generated H264 videos:

```text
.../videos/env0_center_camera_h264.mp4
.../videos/env0_left_camera_h264.mp4
.../videos/env0_right_camera_h264.mp4
```

Key metrics:

| run | steps | updates | s range mm | r range mm | theta range rad | positive axial while misaligned | success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ACT adapter hoverfix | 60 | 0 | -31.7 to -5.9 | 6.0 to 24.3 | 0.078 to 0.099 | 0 | no |
| alignment-only before broad hover gate | 300 | 63 | -75.6 to -0.9 | 2.8 to 66.5 | 0.069 to 0.135 | 0 | no |
| alignment-only broad hover gate | 200 | 38 | -40.1 to -0.9 | 2.8 to 35.6 | 0.069 to 0.108 | 0 | no |

Alignment-only rerun path:

```text
outputs/train/isaac_online_serl_near_gate/audit/2026-05-15_15-19-16_alignment_only_cheatcode_v1_hoverfix_200_20260515_151853
```

Generated H264 videos:

```text
.../videos/env0_center_camera_h264.mp4
.../videos/env0_left_camera_h264.mp4
.../videos/env0_right_camera_h264.mp4
```

## Conclusions

- Position and orientation are both used: `s`, `r`, and `theta` are present in diagnostics and reward components.
- The insertion gate is stricter than the pre-insertion gate: `sigma_lat_insert=0.0015 < 0.0025`, `sigma_theta_insert=0.06 < 0.10`.
- Success is not axial-only: no success occurred, and success violation checks stayed false.
- The original phase reward had a retreat loophole: backing far outside could reduce near-gate penalties and improve reward. The broader hover gate fixes this in formula tests and in the 200-step alignment-only rerun.
- Learning is still not ready for axial/corridor terms. The alignment-only run does not reliably reduce `r`/`theta`; it still drifts away, but now the reward correctly gets worse instead of better.
- `pow4` insertion gates are behaving as intended for insertion, but they are very sparse for exploration. The current best next step is not to loosen insertion gates yet; first add a stronger action prior or guide for lateral/orientation alignment.

## Recommended Next Run

Do not run full `cheatcode_insertion_v1` yet. Try alignment-only again with a stronger guide or smaller direct-action exploration:

```bash
--reward_preset cheatcode_alignment_v1 \
--target_action_guide_weight 0.05 \
--target_action_guide_mode cheatcode_transform \
--target_action_guide_step_size 0.0005 \
--target_action_guide_rotation_step_size 0.01 \
--actor_update_start_steps 200 \
--tcp_translation_action_clip 0.0005
```

Proceed to axial/corridor only after `r` and `theta` decrease while `s` stays outside the entrance.
