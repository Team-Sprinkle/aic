# Curriculum Insertion Pipeline - 2026-05-15

Branch: `feat/hybrid-train`

## Code Changes

- Extended `aic_utils/aic_isaac/scripts/run_one_day_insertion_pipeline.py` with `--curriculum-mode staged`.
- Added Stage A-D curriculum episode generation using existing `isaac_episode_configs.py` infrastructure.
- Added per-stage episode validation for signed start depth `s`, lateral error `r`, and target depth.
- Changed experiment launch so each run can use a stage-specific `episode_config_dir`.
- Added Stage A/B axial guards so far-alignment stages stop candidates that drift too close to the entrance.
- Tightened promotion so early-stopped/nonzero-return candidates cannot seed the next stage.

## Validation

Commands run:

```bash
python -m py_compile \
  aic_utils/aic_isaac/scripts/run_one_day_insertion_pipeline.py \
  aic_utils/aic_isaac/scripts/isaac_episode_configs.py \
  aic_utils/aic_isaac/scripts/train_isaac_online_serl.py \
  aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/insertion_geometry.py \
  aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py

.pixi/envs/default/bin/python -m pytest \
  aic_utils/aic_isaac/test/test_insertion_reward_geometry.py -q
```

Result: `21 passed`.

## Curriculum Episode Configs

Generated and validated under `outputs/analysis/curriculum_insertion_20260515/`.

| Stage | Episodes | Expected | Validated signed start |
| --- | --- | --- | --- |
| Stage A | `stage_a_20mm_align/episode_configs/episodes` | 20 mm axial, 4-8 mm lateral | `s=-20.000 mm`, `r=4.000..8.000 mm` |
| Stage B | `stage_b_12mm_approach/episode_configs/episodes` | 12 mm axial, 2-6 mm lateral | `s=-12.000 mm`, `r=2.000..6.000 mm` |
| Stage C | `stage_c_6_8mm_entry/episode_configs/episodes` | 6-8 mm axial, 0-3 mm lateral | `s=-8.000..-6.000 mm`, `r=0.003..3.000 mm` |
| Stage D | `stage_d_3_6mm_final/episode_configs/episodes` | 3-6 mm axial, 0-1.5 mm lateral | `s=-6.000..-3.000 mm`, `r=0.001..1.500 mm` |

The sign convention is correct: all stage starts are outside the port, hence negative `s`.

## Stage A Runs

Main guarded run root:

```text
outputs/one_day_insertion_pipeline/curriculum_20260515_stage_a_guarded
```

Command:

```bash
.pixi/envs/default/bin/python aic_utils/aic_isaac/scripts/run_one_day_insertion_pipeline.py \
  --curriculum-mode staged \
  --output-root outputs/one_day_insertion_pipeline/curriculum_20260515_stage_a_guarded \
  --gpus 0,1,2,3 \
  --max-wall-time-minutes 75 \
  --per-run-max-wall-time-minutes 12 \
  --episodes-per-stage 8 \
  --reuse-existing-stage-configs \
  --skip-preflight \
  --stage1-steps 30 \
  --stage2-wave1-steps 120 \
  --stage2-wave2-steps 240 \
  --stage3-steps 220 \
  --stage5-steps 500
```

Stage A zero-action reset was stable compared with the old 6 mm start:

| Run | Final s mm | Final r mm | Final theta | Force clip frac |
| --- | ---: | ---: | ---: | ---: |
| zero_stability | `-17.04` | `5.61` | `0.0767` | `0.000` |
| zero_tight_reset | `-17.04` | `5.61` | `0.0767` | `0.000` |

Best Stage A guide/alignment results:

| Run | Final s mm | Max s mm | Final r mm | Best r mm | Final theta | Force clip frac | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `guide_w2_from_1` | `-3.30` | `-3.30` | `0.598` | `0.093` | `0.0594` | `0.029` | Best guide-only alignment; outside, close enough for Stage B handoff. |
| `imit_p1_1` | `-4.79` | `-4.79` | `1.095` | `0.093` | `0.0610` | `0.032` | Retains guide behavior but approaches more than ideal for far alignment. |
| `imit_p1_4` | `-17.91` | `-13.60` | `1.955` | `0.321` | `0.0671` | `0.005` | Cleaner far-alignment retention; lower force, weaker final lateral accuracy. |

Rejected / fixed behavior:

- `guide_w2_from_3` had good raw metrics but drifted too close to the entrance (`max_s=-0.223 mm`), so the new Stage A axial guard should reject it.
- The first guarded rerun showed that early-stopped candidates could still be promoted from partial metrics. I patched promotion to require `returncode == 0`; rerun after this patch is still needed before continuing to Stage B.

## Visual Inspection

Best guide-only Stage A videos:

```text
outputs/one_day_insertion_pipeline/curriculum_20260515_stage_a_guarded/runs/2026-05-15_17-57-14_stage_a_guide_w2_from_1_20260515_175649_gpu0/videos/env0_center_camera_h264.mp4
outputs/one_day_insertion_pipeline/curriculum_20260515_stage_a_guarded/runs/2026-05-15_17-57-14_stage_a_guide_w2_from_1_20260515_175649_gpu0/videos/env0_left_camera_h264.mp4
outputs/one_day_insertion_pipeline/curriculum_20260515_stage_a_guarded/runs/2026-05-15_17-57-14_stage_a_guide_w2_from_1_20260515_175649_gpu0/videos/env0_right_camera_h264.mp4
```

Best low-force imitation videos:

```text
outputs/one_day_insertion_pipeline/curriculum_20260515_stage_a_guarded/runs/2026-05-15_17-59-12_stage_a_imit_p1_4_20260515_175848_gpu3/videos/env0_center_camera_h264.mp4
outputs/one_day_insertion_pipeline/curriculum_20260515_stage_a_guarded/runs/2026-05-15_17-59-12_stage_a_imit_p1_4_20260515_175848_gpu3/videos/env0_left_camera_h264.mp4
outputs/one_day_insertion_pipeline/curriculum_20260515_stage_a_guarded/runs/2026-05-15_17-59-12_stage_a_imit_p1_4_20260515_175848_gpu3/videos/env0_right_camera_h264.mp4
```

Visual check of all three cameras for `guide_w2_from_1`: the plug moves into a plausible pre-insertion alignment posture without visible gate contact or insertion. The center frame shows the SFP tip centered relative to the target; side views show the module remains outside.

## Conclusion

The staged curriculum is necessary and useful. Stage A at 20 mm avoids the severe reset/contact instability seen in the 6 mm start and produces real alignment progress.

Current best fallback/reference for the next run:

- Guide-only: `stage_a_guide_w2_from_1`
- Conservative low-force imitation: `stage_a_imit_p1_4`

Before continuing to Stage B, rerun the guarded pipeline after the final promotion patch so early-stopped candidates cannot seed later stages. Recommended next command is the same guarded command above, with `--stage3-steps 220` or lower.
