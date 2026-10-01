# MoveIt 2 fixed-scene SC cable-route pilot

Date: 2026-09-29. This record preserves the original three-route comparison and its original conservative gate. **The later low-cross revision and the user's clarified snag criterion supersede its route decision; see the [low-cross continuation](2026-09-29-moveit-low-cross-continuation.md).** The original three rollouts were privileged, route-only teachers on one development scene. No BC/RL update, autonomous actor, or reserved final scene was used.

## Fixed scene and implementation

The scene is the archived seed-51500, five NIC cards all at their leftmost rail positions, SC port 0 `forward_minus10` case from the [±20 mm study](2026-09-29-sc-low-cross-2cm-rescue.md). Its `eval_config.yaml` SHA-256 is `b1087e7a008bf322428c682233e389f0d91669b97d13e9c09b48b256512f8b2b`; each pilot copied the same config, world, and bridge files. The [machine summary](2026-09-29-moveit-sc-pilot-results.json) records all input, plan, scored-bag, analysis, code, and video hashes.

The old `collision_scene.py` approximation put unrotated small NIC boxes at the board origin, and `MoveItPyPlanningBackend` passed an empty scene to `plan()`. The new extractor expands the trial's task-board xacro and included Gazebo SDF files into **240 actual rigid collision shapes**: five board pieces, 195 shapes from five NIC assemblies, and 40 fixture/SC-port shapes. It retains box/cylinder dimensions, local roll/pitch/yaw, world frame, and per-model origins. The default backend now fails closed unless these objects and the recorded UR5e start joints are available, applies them to the MoveIt planning scene, and verifies object registration before planning. It returns the actual joint trajectory, not scripted Cartesian interpolation.

The exact model-frame [Gazebo TF audit](</var/tmp/chmin_aic_20260918_act/moveit_sc_corridor_baseline_20260929/scene_tf_audit.json>) found at most **0.00013 mm** position and **0.000016°** orientation difference for the board and six included models. The recorded first `/joint_states` message was used as the explicit planning start. Its MoveIt FK TCP position matched the archived route start within about **0.007 mm**. A current-time ROS joint-state publisher only let MoveItCpp initialize in the offline host planning process. The plan itself set the start explicitly from the scored bag. Gazebo later ran in a separate rootless official image on GPU 3; its policy received only the saved joint trajectory.

The first offline MoveIt smoke failed because MoveItCpp waited for a recent `/joint_states` message. With the recorded-state publisher, MoveIt generated a six-point small-motion smoke plan and then the real corridor plans. The local MoveIt environment emitted a missing *execution-controller plugin* warning; no MoveIt execution API was used. The official Gazebo policy executed joint targets through its normal controller. One strict-return Gazebo process ended with exit 143 before a score; its partial logs and frames are retained, and the same plan was repeated under a new run ID.

## Matched corridor rollouts

All plans use the same measured start and selected port. MoveIt chose the joint path between a few corridor poses. The sparse baseline has four corridor goals. The second moves the outside lane 20 mm farther from the cards. The strict return adds a separate crossing *beyond* the final card and 20 mm more forward clearance, preventing the planner from taking a direct diagonal from the outer lane to the final standoff. The insertion handoff was disabled, so Tier 3 scores are **not** an insertion comparison.

| Plan | MoveIt trajectory points | Executed joint commands / Cartesian fallback | End TCP error to plan | Peak measured wrist force | Minimum old center-to-main-PCB gap | Exact NIC-box cable clearance diagnostic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sparse baseline | 244 | 520 / 0 | 4.26 mm | 21.84 N | 39.25 mm | sampled surface −2 mm; link 2 near card 4 |
| Outside lane +20 mm | 248 | 529 / 0 | 4.26 mm | 21.66 N | 40.34 mm | sampled surface −2 mm; link 2 near card 4 |
| Strict return, lane +20 mm, forward +20 mm | 265 | 558 / 0 | 4.16 mm | 21.60 N | see [raw geometry](</var/tmp/chmin_aic_20260918_act/moveit_sc_corridor_strict_return_retry1_20260929/geometry/cable_geometry_summary.json>) | sampled surface −2 mm; link 2 near card 4 |

All three official score files report no *scored off-limit* contact or excessive-force event. None inserted, by design; they stopped about 0.16 m plug-to-port distance before the separate insertion stage. The route-only policy's `insert_cable()` return value denotes that it finished its trajectory, **not** a successful insertion.

The new [exact-shape clearance analysis](../../scripts/analyze_cable_clearance_exact.py) checks the 2 mm cable-cylinder radius against all 195 NIC collision shapes, including the protrusions omitted by the earlier main-PCB metric. It samples each cable-link axis at nine points and subtracts half the point spacing for a conservative along-link bound. At 730–748 scored TF times, no more than about 95 ms apart, the first nonpositive bound appeared late in each run: about 34.8, 36.7, and 38.8 s from bag start respectively. A sample later placed link 2's axis inside a card-4 box proxy, yielding −2 mm sampled surface clearance and −5 mm conservative lower bound. These are geometry/TF diagnostics, **not a named Gazebo cable/card contact event**; the official scored contact category does not establish the absence of such contact. Exact model origins/yaws were checked against Gazebo TF, but continuous-time cable clearance and force attribution remain unverified.

I inspected the synchronized overhead and side contact sheets for all three runs. The cable visibly approaches the final card tops and is partly hidden by the arm near the end. The videos and geometry do not support the required claim that the full cable stays beside the row without draping or contact. The measured joint endpoint errors show controller tracking was adequate; missing rigid obstacles are unlikely. The remaining likely issue is cable motion near the final standoff, which MoveIt does not model. No insertion diagnosis is possible here because insertion was disabled.

## Reproduction, videos, and archive

With the rootless socket `DOCKER_HOST=unix:///run/user/1008/docker.sock`, source `.pixi/envs/default/setup.bash`, publish the saved [measured start](../../outputs/experiments/2026-09-29_moveit_gazebo_pilot/measured_start_joint_state.json), and run:

```bash
ROS_DOMAIN_ID=93 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  .pixi/envs/default/bin/python scripts/publish_moveit_joint_state.py \
  outputs/experiments/2026-09-29_moveit_gazebo_pilot/measured_start_joint_state.json
```

In a second shell, after sourcing the same pixi setup, run:

```bash
PYTHONPATH=aic_teacher_official:$PYTHONPATH ROS_DOMAIN_ID=93 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  .pixi/envs/default/bin/python scripts/plan_moveit_sc_fixed_route.py \
  --config /var/tmp/chmin_aic_20260918_act/sc_low_cross_forward_minus10_20260926/eval_config.yaml \
  --joint-state outputs/experiments/2026-09-29_moveit_gazebo_pilot/measured_start_joint_state.json \
  --route-plan /var/tmp/chmin_aic_20260918_act/sc_low_cross_forward_minus10_20260926/route_plan.json \
  --lane-extra-m 0.02 --forward-extra-m 0.02 --strict-return \
  --output-dir outputs/experiments/2026-09-29_moveit_gazebo_pilot/plan_strict_return
AIC_MOVEIT_PILOT_GPU=3 bash scripts/run_moveit_sc_route_pilot.sh \
  outputs/experiments/2026-09-29_moveit_gazebo_pilot/plan_strict_return \
  /var/tmp/chmin_aic_20260918_act/NEW_UNIQUE_AUDIT_DIR
```

The planner requires that separate [recorded-state publisher](../../scripts/publish_moveit_joint_state.py) on ROS domain 93. The runner rejects any trajectory without MoveIt metadata, applied collision objects, an explicit recorded start, and joint waypoints. Its runtime uses the pinned official evaluation image, a local `aic_model`/teacher policy import for version compatibility, and no sudo.

The 20 fps H.264 review recordings combine three wrist cameras with overhead and side views:

- [Sparse baseline](../../outputs/experiments/2026-09-29_moveit_gazebo_pilot/baseline_video/all_views_20fps.mp4)
- [Outside lane +20 mm](../../outputs/experiments/2026-09-29_moveit_gazebo_pilot/lane20_video/all_views_20fps.mp4)
- [Strict return](../../outputs/experiments/2026-09-29_moveit_gazebo_pilot/strict_return_video/all_views_20fps.mp4)

The individual `overhead_20fps.mp4`, `side_20fps.mp4`, `wrist_triptych_20fps.mp4`, contact sheets, and render coverage reports sit beside each combined video. Original scored MCAPs, all frames, exact configs/plans/traces, and the interrupted attempt remain under `/var/tmp/chmin_aic_20260918_act/moveit_sc_corridor_*_20260929/` and were copied to `/home/nas/chmin/aic_archive_20260929/moveit_sc_pilot_raw/`. The [archive manifest](2026-09-29-moveit-sc-pilot-archive.json) verifies scene/plan/trajectory files and all three scored MCAPs by SHA-256.

## Gate and next controlled repair

**At the time of this comparison, Step 2 passed for fixed-scene rigid planning and verified Gazebo joint replay, while Step 3 failed the original zero-clearance route gate.** The later continuation changed the route height and used the clarified criterion of sustained snag or loss of progress. Do not admit these three original routes to transport BC as clean examples or generalize them to other SC settings. Read the later continuation for current gate status.

Only the outside-left topology was executed in this pilot. An outside-right route was not qualified and is not evidence of a clear alternative. The measured arm reached the standoff in all three completed runs, so the trace shows a cable-clearance risk rather than a physical arm stall. The next diagnosis needs a named cable/card contact stream or equally direct high-rate physical measurement before attributing a scored failure to snagging.

A future bounded route revision should log named cable-to-card contacts or an equivalent high-rate collision trace, then test one cable-aware terminal route choice that keeps the gripped cable on the outside of the final card while approaching a standoff. Compare it with this frozen MoveIt control on this scene first; require positive physical clearance, no sustained contact, and repeatable standoff completion before opening new SC starts. Do not infer contact or cable safety from Tier 3 alone.

The independent [perception-label audit](2026-09-29-perception-label-readiness.json) found 1,485 native SC frames in 21 scene groups with no missing images or train/validation scene overlap, but the opening flag is only an in-frame projection. There are no actual selected-port visibility labels, wrong-port negatives, or SFP rows in that source. Training a calibrated two-family visibility gate from these fields would be misleading; obtain actual occlusion/identifiability labels and SFP coverage before the step-6 training gate.
