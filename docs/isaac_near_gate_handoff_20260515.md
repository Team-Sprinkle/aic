# Isaac Near-Gate Handoff - 2026-05-15

This note summarizes the Isaac `start_near_gate` / near-gate insertion debugging done on this server. Treat these results as environment-specific. The server used for these experiments had visible asset/camera problems: the SFP tip/plug visual was missing or not reliably visible in camera images until asset references were repaired, and some camera views looked wrong. The next server is expected to be based on the latest remote `feat/hybrid-train`, where the cheatcode reportedly works and the tip is visible. Results below are useful mainly as diagnostics and experiment design notes, not as final tuning conclusions.

## Server-Specific Caveats

- The official NVIDIA `Intrinsic_assets.zip` was re-downloaded and installed according to the upstream Isaac README. In this runtime, the downloaded pack still left direct `.glb` visual references for `lc_plug_visual`, `sfp_module_visual`, and `sc_plug_visual` inside `aic_unified_robot_cable_sdf.usd`.
- Isaac Sim on this server failed to open those direct GLB references, producing missing plug visuals. This can make camera-based conclusions unreliable.
- The live asset directory was repaired locally by converting those GLBs to USD and patching the robot USD references. That repair is local/generated and should not be assumed necessary on the next server if assets already render correctly.
- Camera constants appeared unchanged from upstream: `PinholeCameraCfg`, `224x224`, ROS camera offset `(pos=(0,0,0), rot=(1,0,0,0))`, and optical prim paths were the same.
- A branch-local robot reset regression was found: `shoulder_pan_joint` had been flipped from upstream `0.1597` to `-0.1597`. Since cameras are robot-mounted, that can make views look wrong even when camera constants are correct.
- Physics did start. Cable/plug dangling was likely from dynamic cable joints and only the arm joints being actuated, not from a fully paused simulation.

## Near-Gate Experiments Run Here

The experiments focused on whether Isaac could produce physically visible insertion from near-gate starts before doing reward tuning.

- Restored `shoulder_pan_joint` to upstream `0.1597`.
- Added SFP geometry diagnostics and tests around target, entrance, and insertion axis.
- Replaced inconsistent hardcoded SFP seated target constants with a target derived from:
  - SFP port root-local entrance
  - port rotation / insertion axis
  - seated depth
- Added fail-fast validation for target geometry:
  - target must be collinear with entrance and insertion axis
  - seated depth must be within expected bounds
  - lateral error must be near zero
- Added all-body insertion diagnostics for `wrist_3_link`, `gripper_tcp`, `sfp_tip_link`, and `sfp_module_link`.
- Added `cheatcode_tcp` action-guide mode to compare Isaac guide semantics with the Gazebo cheatcode idea.
- Ran short near-gate / cheatcode probes with debug overlays and videos under:
  - `outputs/debug/isaac_cheatcode_fix_20260515/`

## Experiment Setup Details

The concrete settings below are the useful starting point for recreating these probes on a new machine. They are intentionally short-horizon diagnostics, not final training settings.

Primary scripts and code paths:

- Episode materializer: `aic_utils/aic_isaac/scripts/isaac_episode_configs.py`
- Host launcher: `aic_utils/aic_isaac/scripts/train_isaac_online_serl.py`
- Isaac trainer: `aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py`
- Reward code: `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py`
- Reset consumer: `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/events.py`

The run directory was:

```text
outputs/debug/isaac_cheatcode_fix_20260515/
```

The generated diagnostic episode request was:

```yaml
task_family: sfp_to_nic
generation:
  target_accepted_trajectories: 2
  seed: 236
scene:
  target:
    entrance_axis_offset_m: -0.0009
    seated_depth_m: 0.008
  start_near_gate:
    axial_distance_m: 0.0005
    lateral_distance_m: 0.0002
    min_clearance_m: 0.0002
    reset_body_name: gripper_tcp
    reset_body_offset_from_reference_world: [-0.007149, 0.002556, 0.059066]
    reset_body_orientation_wxyz: [0.026548, 0.013188, 0.991236, 0.128732]
  nic_cards:
    count: 1
    target_card: 0
    target_port: sfp_port_0
```

The materialized episodes live at:

```text
outputs/debug/isaac_cheatcode_fix_20260515/episode_configs/episodes/
```

Near-gate reset semantics:

- `start_near_gate.reset_mode` is `body_start_position_world`.
- Reset body is `gripper_tcp`, but the reference reward body for the generated request is `sfp_tip_link`.
- The materializer places the reference body relative to the port entrance, not the seated target.
- For the probe episodes, achieved offsets were approximately `0.5 mm` axial and `0.2 mm` lateral from the entrance gate.
- The gate axis recorded in diagnostics was approximately `[-0.0, 0.012642, -0.99992]`.
- The seated target depth was `0.008 m`; the gate came from `entrance_pose_world`.
- `events.reset_robot_tcp_to_episode_start()` used 6D damped IK with up to `8` iterations. In the r105 diagnostic, reset final error was about `0.64 mm` position and `0.00054 rad` orientation.

## Reward Designs Tried

The first pass used the default target reward weights from the trainer:

```text
target_reward_body=sfp_tip_link
distance_weight=0.25
close_weight=0.35
progress_weight=0.25
orientation_weight=0.10
reaching_weight=0.0
terminal_weight=1.0
lateral_weight=-0.05
motion_projection_weight=0.0
lateral_progress_weight=0.0
axial_progress_weight=0.0
insertion_corridor_weight=0.0
force_delta_penalty_weight=0.3 in the wrapper dry-run plan, later 0.05 in short probes
```

For the actual 2026-05-15 short probes r101-r107, use `--reward-preset near_gate_corridor_v1`. That preset overrides the target reward knobs to:

```text
target_reward_distance_weight=0.02
target_reward_close_weight=0.05
target_reward_progress_weight=0.0
target_reward_lateral_weight=-0.10
target_reward_lateral_progress_weight=0.25
target_reward_axial_progress_weight=0.25
target_reward_insertion_corridor_weight=0.50
target_reward_orientation_weight=0.05
target_reward_reaching_weight=0.0
target_reward_terminal_weight=1.0
force_delta_penalty_weight=0.05
```

Other reward geometry/scales in those probes:

```text
target_reward_distance_std=0.02
target_reward_close_sigma=0.006
target_reward_progress_scale=0.003
target_reward_lateral_error_scale=0.006
target_reward_lateral_progress_scale=0.001
target_reward_axial_progress_scale=0.001
target_reward_lateral_gate_sigma=0.004
target_reward_orientation_gate_sigma=0.01
target_reward_orientation_std=0.1
target_reward_insertion_corridor_sigma=0.0025
target_reward_insertion_bypass_penalty_scale=2.0
target_reward_insertion_axis=0
target_success_axial_threshold=0.00025
target_success_lateral_threshold=0.0005
target_success_termination_threshold=0.0005
terminate_on_target_success=true
```

Important Isaac scaling detail: `_configure_task_geometry_rewards()` multiplies CLI reward weights by `1 / env_step_dt`. With `policy_hz=20`, resolved Isaac weights were 20x the CLI values. For example, the r105 resolved weights were:

```text
target_distance_tanh=0.4
target_distance_exp=1.0
target_distance_progress=0.0
target_lateral_error=-2.0
target_lateral_progress=5.0
target_axial_progress=5.0
target_insertion_corridor=10.0
target_orientation_gated_exp=1.0
target_reaching_bonus=0.0
target_success_once_bonus=20.0
force_delta_penalty=1.0
```

Command-pose PPO rewards were disabled/zeroed in the SERL trainer for these probes; the active insertion signal was the target-geometry reward stack above plus the standard small regularizers.

## Guide / Cheatcode Settings Tried

All r101-r107 probes used:

```text
act_only=true
act_torchscript=outputs/debug/isaac_cheatcode_fix_20260515/dummy_act/act_ts.pt
act_only_state_dim=82
act_only_single_action_dim=6
act_only_actor_mode=act_adapter
freeze_act=true
policy_hz=20
n_action_steps=1
isaac_action_scale=1.0
tcp_action_frame=root unless noted
enable_cameras=true
enable_contact_sensor=true
debug_diagnostics=true
debug_visual_overlays=true
save_step_images=true
diagnostics_every=1
image_log_every=2
max_logged_image_steps=80
headless=true
rendering_mode=performance
```

The dummy ACT TorchScript produced zero base actions. During guide-collection probes, the executed env action came from the guide path, not from a learned adapter.

Guide settings for the cheatcode probes. The saved r103-r107 configs record this mode as `cheatcode_tcp`; in the current launcher/trainer CLI the same rigid-transform guide is exposed as `cheatcode_transform`.

```text
target_action_guide_mode=cheatcode_tcp / cheatcode_transform
target_action_guide_collect_blend=1.0
target_action_guide_collect_steps=80
target_action_guide_step_size=0.0015
target_action_guide_rotation_step_size=0.02
target_action_guide_axial_step_size=0.0
target_action_guide_lateral_switch_m=0.002
target_action_guide_axial_blend_lateral_m=0.006
target_action_guide_collect_decay=false
target_action_guide_prefix_decay=false
target_action_guide_train_executed=false
target_action_guide_weight=0.0
```

The meaning of `target_action_guide_weight=0.0` is easy to miss: it disables an actor imitation loss toward the guide. It does not disable guide collection. The guide still replaces/blends the action during collection because `target_action_guide_collect_blend=1.0` and `target_action_guide_collect_steps=80`.

Trainer modes tried:

- r101: no guide collection, `target_reward_body=sfp_tip_link`, `fix_isaac_ik_xy_sign=true`, 50 steps, 2 envs.
- r102: same as r101 but `fix_isaac_ik_xy_sign=false`.
- r103: `cheatcode_tcp` / `cheatcode_transform`, root action frame, `target_reward_body=sfp_tip_link`, `fix_isaac_ik_xy_sign=false`, 80 steps, 2 envs.
- r104: same as r103 but `tcp_action_frame=gripper_tcp`.
- r105: `cheatcode_tcp` / `cheatcode_transform`, root action frame, `target_reward_body=sfp_tip_link`, IK-body/action-scale diagnostics enabled, `fix_isaac_ik_xy_sign=false`, 80 steps, 2 envs.
- r106: short asset visual probe, `cheatcode_tcp` / `cheatcode_transform`, `target_reward_body=sfp_tip_link`, 4 steps, 1 env.
- r107: same guide family as r105 but `target_reward_body=sfp_module_link`, `batch_size=4`, `update_every_steps=1`, 80 steps, 2 envs.

r105 is the best diagnostic template from this server. r107 is useful only as a warning that changing the reward/guide body from `sfp_tip_link` to `sfp_module_link` did not automatically produce physical insertion here.

## Starter Commands For A New Machine

First regenerate a tiny pure SFP near-gate episode set:

```bash
mkdir -p outputs/debug/isaac_near_gate_restart/episode_configs
cat > outputs/debug/isaac_near_gate_restart/episode_configs/request.yaml <<'YAML'
task_family: sfp_to_nic
generation:
  target_accepted_trajectories: 2
  seed: 236
scene:
  target:
    entrance_axis_offset_m: -0.0009
    seated_depth_m: 0.008
  start_near_gate:
    axial_distance_m: 0.0005
    lateral_distance_m: 0.0002
    min_clearance_m: 0.0002
    reset_body_name: gripper_tcp
    reset_body_offset_from_reference_world: [-0.007149, 0.002556, 0.059066]
    reset_body_orientation_wxyz: [0.026548, 0.013188, 0.991236, 0.128732]
  nic_cards:
    count: 1
    target_card: 0
    target_port: sfp_port_0
YAML

python3 aic_utils/aic_isaac/scripts/isaac_episode_configs.py \
  --request-yaml outputs/debug/isaac_near_gate_restart/episode_configs/request.yaml \
  --output-dir outputs/debug/isaac_near_gate_restart/episode_configs
```

Then run an r105-style no-learning cheatcode/guide probe. Update `--act-torchscript` to a real ACT TorchScript if you want ACT+guide behavior; keep the dummy/zero ACT only if you are isolating guide geometry.

```bash
python3 aic_utils/aic_isaac/scripts/train_isaac_online_serl.py \
  --act-only \
  --act-torchscript outputs/debug/isaac_cheatcode_fix_20260515/dummy_act/act_ts.pt \
  --episode-config-dir outputs/debug/isaac_near_gate_restart/episode_configs/episodes \
  --output-dir outputs/debug/isaac_near_gate_restart/train_runs \
  --run-name r105_restart_cheatcode_tip \
  --num-envs 2 \
  --steps 80 \
  --update-every-steps 100000 \
  --policy-hz 20 \
  --n-action-steps 1 \
  --isaac-action-scale 1.0 \
  --no-fix-isaac-ik-xy-sign \
  --tcp-action-frame root \
  --reward-preset near_gate_corridor_v1 \
  --target-reward-body sfp_tip_link \
  --target-success-axial-threshold 0.00025 \
  --target-success-lateral-threshold 0.0005 \
  --target-success-termination-threshold 0.0005 \
  --force-delta-penalty-weight 0.05 \
  --target-action-guide-mode cheatcode_transform \
  --target-action-guide-collect-blend 1.0 \
  --target-action-guide-collect-steps 80 \
  --target-action-guide-step-size 0.0015 \
  --target-action-guide-rotation-step-size 0.02 \
  --target-action-guide-lateral-switch-m 0.002 \
  --target-action-guide-axial-blend-lateral-m 0.006 \
  --no-target-action-guide-collect-decay \
  --no-target-action-guide-prefix-decay \
  --no-target-action-guide-train-executed \
  --debug-diagnostics \
  --debug-visual-overlays \
  --diagnostics-every 1 \
  --debug-audit-steps 80 \
  --save-step-images \
  --image-log-every 2 \
  --max-logged-image-steps 80 \
  --enable-contact-sensor \
  --headless \
  --rendering-mode performance
```

If physical video and semantic metrics agree on the new machine, the next online-learning starting point from the 2026-05-14 root-cause work was:

```text
ACT warm start
pure SFP 6 mm near-gate episode shard, not the full mixed 24-episode shard
policy_hz=20
act_only_action_horizon=8
n_action_steps=4
isaac_action_scale=0.5
adapter_delta_clip=0.001
adapter_lr=1e-4
adapter_penalty_weight=1.0
act_preservation_weight=1.0
update_every_steps=4
target reward preset/geometry rechecked against visible insertion before long runs
```

That setting produced the best short online result on this host: last-100 reward `0.0231`, last step reward `0.0957`, last-100 progress `0.0381`, no adapter clipping, but last-500 reward still slightly negative and force spikes persisted. It was not strong enough to justify launching a 6-hour mixed SFP+SC run without a longer single-task confirmation.

Important video artifacts from this server:

- Missing/asset visual probe:
  `/home/ubuntu/ws_aic/src/aic/outputs/debug/isaac_cheatcode_fix_20260515/videos/r106_asset_visual_probe_env0_center.mp4`
- Module-link cheatcode probe:
  `/home/ubuntu/ws_aic/src/aic/outputs/debug/isaac_cheatcode_fix_20260515/videos/r107_module_link_cheatcode_env0_center.mp4`
- Earlier tip-link cheatcode side-by-side:
  `/home/ubuntu/ws_aic/src/aic/outputs/debug/isaac_cheatcode_fix_20260515/videos/r105_fixed_geometry_cheatcode_ikbody_no_xy_sign_env0_env1_center_side_by_side.mp4`

Observed on this server:

- Semantic metrics could show partial insertion or positive SFP tip depth while the camera did not show convincing physical insertion.
- With `target_reward_body=sfp_tip_link`, diagnostics could show the tip near/inside the target plane while the module body was still visibly outside. This suggests that tip-only metrics can overstate insertion if the visible module/plug body does not follow.
- Switching the guide to a module-link target did not fix insertion here; in one run it drove the module farther away.
- The global Isaac IK x/y sign fix was not sufficient by itself.
- Because plug visuals were missing for part of the debugging, any purely visual conclusion from these runs should be rechecked on the new server.

## What To Reuse On The New Server

If the new server has the latest branch, visible plug/tip, and working cheatcode, do not start by applying the local asset repair or old visual conclusions. Instead, reuse the diagnostics pattern:

1. Confirm camera and asset sanity first.
2. Run one no-learning near-gate cheatcode probe with overlays and save video.
3. Compare physical video against semantic diagnostics for:
   - `sfp_tip_link`
   - `sfp_module_link`
   - port entrance
   - seated target
   - insertion axis
4. Only tune insertion behavior after physical and semantic diagnostics agree.
5. Prefer minimal changes per iteration, with a short run and video after each change.

Most useful checks for the new server:

- Is the visible SFP module aligned with the port mouth at reset?
- Does the cheatcode move the visible module toward the port, not just the semantic tip?
- Do lateral error, axial depth, and orientation error improve monotonically during insertion?
- Does any positive insertion depth correspond to visible physical insertion?
- Does cable dynamics pull the module away after reset or during the first few sim steps?
- Does `sfp_tip_link` differ from `sfp_module_link` enough that the reward body should be changed or supplemented?

## Suggested Prompt For The Next Server

Use a prompt like this for autonomous iterative work:

```text
We are on the new server with the latest remote feat/hybrid-train where the Isaac plug/tip is visible and the cheatcode works. Do not reuse conclusions from the older server unless revalidated here.

Goal: improve SFP-to-NIC near-gate insertion in Isaac, not reward-tune blindly.

Work iteratively and autonomously:
1. Run a short near-gate cheatcode/guide probe with cameras, overlays, diagnostics, and saved video.
2. Inspect video and diagnostics together. Check whether visible SFP module insertion matches semantic metrics for sfp_tip_link, sfp_module_link, entrance, target, insertion axis, lateral error, axial depth, and orientation.
3. If physical and semantic metrics disagree, fix frames/geometry/target body/reset semantics before changing rewards.
4. If they agree but insertion fails, make one minimal change to guide/reset/action-frame/insertion-axis behavior.
5. Re-run the same short probe, compare against the previous run, and repeat.
6. Save every run's video path, diagnostics summary, exact command, and a concise explanation of what changed and whether it helped.

Use the Gazebo cheatcode as the semantic reference, but verify the Isaac adaptation body frame, root frame, x/y sign, target body, and insertion axis directly in Isaac. Do not assume tip-depth success is valid unless the video shows the visible plug entering the port.
```

## Recommendation

Share this note with the next server, but frame it as a cautionary handoff. The useful transferable parts are the diagnostic methodology, the suspected shoulder-pan/camera coupling, and the warning that `sfp_tip_link` metrics can disagree with visible module insertion. The actual failed videos from this server should not be used as evidence that the latest remote branch fails, because this server had asset/rendering differences and possibly stale branch state.
