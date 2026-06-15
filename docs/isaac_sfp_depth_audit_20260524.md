# Isaac SFP Depth Audit 2026-05-24

## Finding

The previous Isaac strict-success depth target of `8 mm` was wrong for full
SFP-to-NIC seating. It only represented shallow tip insertion near the cage
front.

The Gazebo/official asset semantics are:

- `aic_assets/models/NIC Card/model.sdf` defines `sfp_port_0_link` at the SFP
  port frame.
- `sfp_port_0_link_entrance` is at `(0, 0, -0.0458)` relative to
  `sfp_port_0_link`.
- The SFP cage collision bodies have length `0.04872 m`.

So the semantic entrance-to-port-frame depth is `45.8 mm`. With the Isaac
curriculum's existing `entrance_axis_offset_m: -0.0009`, the generated
entrance-to-target depth is approximately `46.7 mm`.

The `48.72 mm` dimension is the collision cage wall length. The repo's semantic
port-link insertion target is slightly shorter at `45.8 mm`; this is the frame
used by Gazebo's `sfp_port_*_link_entrance`.

## What Was Wrong

- `run_one_day_insertion_pipeline.py` generated SFP targets with
  `seated_depth_m: 0.008`.
- Isaac and Gazebo insertion geometry validators rejected target depths above
  `30 mm`, which made the true SFP depth impossible to use.
- The diagnostic strict checker used a hard-coded `8 mm` minimum depth instead
  of the episode's `target_depth`.
- The train/eval target-tip servo defaulted to `8 mm`.

## Changes

- Isaac episode generation now defaults SFP full insertion to the Gazebo port
  frame rather than the shallow 8 mm override.
- `seated_depth_m` remains available as an explicit override for shallow
  curriculum experiments.
- Isaac/Gazebo insertion geometry validators now accept target depths up to
  `60 mm`.
- Diagnostic strict success now requires `axial_depth >= target_depth - 0.5 mm`.
- Gazebo reward default insertion target depth is now `45.8 mm`.
- The train/eval target-tip servo now defaults to the episode target depth when
  no explicit goal depth is provided.
- Offline privileged-servo and reward-funnel audit utilities now default to
  `45.8 mm` instead of `10 mm`.
- The previous `8 mm` best-run document was marked as superseded shallow
  insertion.

## Validation

Generated SFP episode without `seated_depth_m` override:

- `target_depth ~= 0.0467 m` with `entrance_axis_offset_m: -0.0009`.
- Lateral target residual is near zero.

Commands run:

```bash
.pixi/envs/default/bin/python -m py_compile aic_utils/aic_isaac/scripts/isaac_episode_configs.py aic_utils/aic_isaac/scripts/run_one_day_insertion_pipeline.py aic_utils/aic_isaac/scripts/audit_insertion_reward_geometry.py aic_utils/aic_isaac/scripts/agentic_insertion_reward_curriculum_loop.py aic_utils/aic_isaac/scripts/analyze_contact_command_diagnostic.py aic_utils/aic_isaac/scripts/controller_contact_realization_diagnostic.py aic_utils/gazebo_rl/gazebo_rl/insertion_reward.py aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/insertion_geometry.py aic_utils/aic_isaac/aic_isaaclab/scripts/diagnostics/wrist_contact_realization.py aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py
.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_isaac_online_serl.py aic_utils/gazebo_rl/test/test_insertion_reward.py
.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py aic_utils/gazebo_rl/test/test_insertion_reward.py
.pixi/envs/default/bin/python -m pytest -q aic_utils/aic_isaac/test/test_insertion_reward_geometry.py aic_utils/aic_isaac/test/test_isaac_online_serl.py aic_utils/gazebo_rl/test/test_insertion_reward.py aic_utils/aic_isaac/test/test_agent_reward_funnel_scripts.py
```

Results:

- `49 passed in 3.91s`
- `9 passed in 8.30s`
- `51 passed in 11.17s`

## Corrected Full-Depth Isaac Smoke

A corrected full-depth episode was created from the prior best final-window
reset:

```text
outputs/agentic_reward_curriculum_20260524_depth_correction/generated_episode_configs/full_depth_finalwindow_env1_ep2
```

Its target is `45.7997 mm` from the configured entrance, with near-zero lateral
target residual. Two scripted wrist-IK target-tip diagnostics were then run in
Isaac:

```text
outputs/agentic_reward_curriculum_20260524_depth_correction/runs/2026-05-24_11-48-24_full_depth_targettip_smoke_cameras
outputs/agentic_reward_curriculum_20260524_depth_correction/runs/2026-05-24_11-51-08_full_depth_targettip_long_ax120
```

Results:

| run | best s | target s | best r | best theta | module consistency | strict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `full_depth_targettip_smoke_cameras` | `21.36 mm` | `45.80 mm` | `0.240 mm` | `0.04646 rad` | `~0` | false |
| `full_depth_targettip_long_ax120` | `19.60 mm` | `45.80 mm` | `0.182 mm` | `0.04825 rad` | `0` | false |

The longer run regressed after its best point and ended at only `3.96 mm` depth,
`0.644 mm` lateral error, `0.260 rad` orientation error, and zero module
consistency. This is evidence that the previous `8 mm` result was a shallow
insertion artifact and that the corrected full-depth target exposes a real
controller/contact/module-consistency blocker.

## Next Step

The next useful code change is a full-depth-aware retention guard that stops
axial motion when wrist/contact force spikes or module consistency fails, backs
out, and retries with lateral/orientation correction. Reward-only tuning should
not proceed until this corrected-depth controller/contact blocker is addressed.
