# Agent Teleop Expert Generator Current Status

Last updated: 2026-05-01

This document summarizes the current state of the VLM/MoveIt/CheatCode expert trajectory generator on `feat/agent-teleop`. It is intended as a handoff for continuing development after the first successful real nominal run.

## Current Status

The nominal live pipeline now completes a real Gazebo/ROS/MoveIt/VLM run and accepts a trajectory above the default score threshold.

Accepted run:

```text
outputs/expert_datasets/nominal_live_full_20260430T231933Z
```

Result:

```text
accepted: 1
score: 95.01970108192813
insertion_event_reached: true
offlimit_contact_count: 0
moveit_success: true
vlm_cable_risk: medium
trajectory_duration_s: 13.09
end_effector_path_length: 0.18 m
average_linear_jerk: 24.56 m/s^3
```

The earlier startup failure was caused by stale ROS/Gazebo/controller processes surviving across runs. The launcher now performs host-side and container-side cleanup before each trial and during teardown.

Live camera images are being captured during the planner pass, not read from old local feedback frames. The accepted run saved live images and validation metadata under:

```text
outputs/expert_datasets/nominal_live_full_20260430T231933Z/planner_attempts/attempt_000001_candidate_00/planner_debug/live_strategy_images/
```

The image validation manifest reported 12 captured images and 8 selected images. The selected frames were valid, non-empty, and not near-constant.

## High-Level Architecture

The pipeline is now:

1. Launch Gazebo/ROS with the official engine config.
2. Launch MoveIt alongside the sim for planner-policy runs.
3. Run `aic_teacher_official.OfficialExpertGeneratorPlanner`.
4. Capture live scene state and camera images.
5. Ask GPT-5-mini for symbolic strategy and cable-risk JSON only.
6. Generate deterministic candidate poses from CheatCode-style target geometry.
7. Use MoveItPy to plan free-space motion to the high pre-insertion pose.
8. Save a `piecewise_trajectory.json` containing MoveIt joint trajectory samples.
9. Postprocess to `smooth_trajectory.json`.
10. Replay with `aic_teacher_official.OfficialTeacherReplay`.
11. During replay, execute MoveIt joint targets first, then switch to online CheatCode-style final insertion.
12. Parse official scoring YAML and accept only trajectories that pass thresholds.

GPT-5-mini does not generate executable waypoints. It only produces scene interpretation and strategy/cable-risk fields.

## Important Entry Points

Main CLI:

```text
scripts/generate_expert_trajectories.py
```

Live planner policy:

```text
aic_teacher_official/aic_teacher_official/OfficialExpertGeneratorPlanner.py
```

MoveIt backend:

```text
aic_teacher_official/aic_teacher_official/expert_generator/moveit_py_backend.py
```

Replay policy:

```text
aic_teacher_official/aic_teacher_official/OfficialTeacherReplay.py
```

Per-trial sim/policy/recorder launcher:

```text
aic_utils/lerobot_robot_aic/scripts/launch_policy_recording_per_trial.sh
```

MoveIt config:

```text
aic_moveit_config/
```

## CLI Flow

`scripts/generate_expert_trajectories.py` builds separate planner and replay configs:

```python
replay_config = OfficialReplayConfig(
    repo_root=REPO_ROOT,
    engine_config=Path(args.config),
    output_dir=output_dir / "replay_attempts",
    action_mode="joint_position_then_cheatcode",
    gazebo_gui=args.gazebo_gui,
    launch_rviz=args.launch_rviz,
    startup_delay_sec=args.startup_delay_sec,
    recorder_drain_sec=args.recorder_drain_sec,
    per_trial_timeout_sec=args.per_trial_timeout_sec,
    sim_distrobox=args.sim_distrobox,
)
planner_config = ExpertPlannerRunConfig(
    repo_root=REPO_ROOT,
    engine_config=Path(args.config),
    output_dir=output_dir / "planner_attempts",
    expert_mode=mode.value,
    strategy_model=args.strategy_model,
    candidates_per_scene=args.candidates_per_scene,
    gazebo_gui=args.gazebo_gui,
    launch_rviz=args.launch_rviz,
    startup_delay_sec=args.startup_delay_sec,
    recorder_drain_sec=args.planner_recorder_drain_sec,
    per_trial_timeout_sec=args.per_trial_timeout_sec,
    sim_distrobox=args.sim_distrobox,
    launch_moveit=args.launch_moveit,
    moveit_launch_file=args.moveit_launch_file,
)
```

The generation loop counts accepted trajectories, not raw attempts:

```python
while accepted < args.target_accepted_trajectories and attempts < args.max_total_attempts:
    candidate_index = attempts % args.candidates_per_scene
    attempts += 1
    planner_result = planner_runner.run_planner(
        attempt_index=attempts,
        candidate_index=candidate_index,
        seed=args.seed + attempts - 1,
    )
    ...
    replay_metrics = replay_runner.replay_and_score(
        smooth,
        attempt_index=attempts,
        candidate_index=candidate_index,
    )
    validation = validator.evaluate({...})
    if validation.accepted:
        metadata_writer.append_episode(...)
        accepted += 1
```

## VLM Role

The VLM schema lives in:

```text
aic_teacher_official/aic_teacher_official/expert_generator/vlm_strategy.py
```

The accepted run produced this strategy:

```json
{
  "mode": "nominal",
  "approach_side": "above_right",
  "cable_risk": "medium",
  "preferred_clearance_m": 0.05,
  "avoid_regions": [
    "left_cable_sweep_zone",
    "task_board_edge_near_nic_card",
    "nic_card_face_and_mounts",
    "sc_mounts_zone"
  ],
  "insertion_strategy": "straight_slow_descent",
  "recovery_allowed": false
}
```

The VLM output is intentionally symbolic. It can influence candidate ordering, clearance, and avoid-region metadata, but it must not produce joint positions, Cartesian waypoints, velocities, or low-level commands.

## MoveIt Integration

MoveIt is required. There is no geometric fallback path.

`MoveItPlanner` fails loudly if no supported MoveIt Python module is available:

```python
if required and backend is None and self.available_backend_name is None:
    raise MoveItUnavailableError(
        "MoveIt is required for expert generation, but no supported Python MoveIt "
        f"module was found. Checked: {', '.join(import_names)}. No geometric fallback is available."
    )
```

The live backend uses MoveItPy:

```python
from moveit.planning import MoveItPy

return MoveItPy(node_name=self.config.node_name, config_dict=self._moveit_config_dict())
```

The current default approach mode is `direct_pre_insert`:

```python
@dataclass(frozen=True)
class MoveItPyBackendConfig:
    ...
    max_velocity_scaling_factor: float = 0.1
    max_acceleration_scaling_factor: float = 0.1
    approach_segment_mode: str = "direct_pre_insert"
```

That means MoveIt plans one free-space segment directly to the high CheatCode-derived pre-insertion pose:

```python
def _segment_specs(
    self,
    candidate_pose: ApproachCandidate,
) -> tuple[tuple[str, Any, PhaseLabel], ...]:
    if self.config.approach_segment_mode == "three_stage":
        return (
            ("safe_lift", candidate_pose.safe_lift_pose, PhaseLabel.APPROACH),
            ("approach_standoff", candidate_pose.approach_standoff_pose, PhaseLabel.ALIGNMENT),
            ("pre_insert", candidate_pose.pre_insert_pose, PhaseLabel.PRE_INSERTION),
        )
    if self.config.approach_segment_mode != "direct_pre_insert":
        raise ValueError(...)
    return (("pre_insert", candidate_pose.pre_insert_pose, PhaseLabel.PRE_INSERTION),)
```

This change fixed the score regression from the three-stage plan. The three-stage plan was valid but added extra stop-and-go motion, longer path length, and higher jerk. The accepted run used one MoveIt segment with 46 MoveIt joint trajectory points.

## Candidate Generation

Candidate poses are generated deterministically from scene state and CheatCode-style target geometry:

```python
cheatcode_pre_insert = _cheatcode_gripper_target(snapshot, z_offset=pre_insert_z_offset_m)
orientation = list(cheatcode_pre_insert.orientation_xyzw)
...
pre_insert = _pose_at(
    cheatcode_pre_insert,
    x=cheatcode_pre_insert.position[0],
    y=cheatcode_pre_insert.position[1],
    z=staging_z,
    orientation_xyzw=orientation,
)
```

The current default `pre_insert_z_offset_m` is `0.20`. This keeps the MoveIt free-space target high enough to avoid board/NIC contact before CheatCode takes over final insertion.

## Replay

Replay uses `joint_position_then_cheatcode`.

The MoveIt portion sends joint targets:

```python
move_robot(
    joint_motion_update=JointMotionUpdate(
        target_state=JointTrajectoryPoint(
            positions=[float(v) for v in target.joint_positions],
            velocities=[float(v) for v in velocities],
        ),
        target_stiffness=[100.0] * n_joints,
        target_damping=[20.0] * n_joints,
        trajectory_generation_mode=TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_POSITION,
        ),
        target_feedforward_torque=[0.0] * n_joints,
    )
)
```

Final insertion is delegated to the online CheatCode-style geometry path. The replay trajectory marks that phase as `final_insertion` and does not use MoveIt for insertion.

## Launcher Fix

The immediate blocker was stale host-visible ROS/Gazebo processes. They caused controller startup errors such as already-loaded controllers and failed broadcaster activation.

The launcher now performs host cleanup:

```bash
cleanup_host_sim_processes() {
  local patterns=(
    "ros2 launch aic_bringup"
    "aic_gz_bringup.launch.py"
    "/opt/ros/kilted/lib/rclcpp_components/component_container"
    "/ws_aic/install/lib/aic_engine/aic_engine"
    "/ws_aic/install/lib/aic_adapter/aic_adapter"
    "/ws_aic/install/lib/controller_manager/ros2_control_node"
    "/opt/ros/kilted/lib/controller_manager/spawner"
    "/opt/ros/kilted/lib/robot_state_publisher/robot_state_publisher"
    "/opt/ros/kilted/lib/topic_tools/relay"
    "/opt/ros/kilted/lib/tf2_ros/static_transform_publisher"
    "/opt/ros/kilted/lib/moveit_ros_move_group/move_group"
    "gz sim"
    "gzserver"
    "ruby.*gz sim"
    "rmw_zenoh_cpp rmw_zenohd"
    "/opt/ros/kilted/lib/rmw_zenoh_cpp/rmw_zenohd"
  )
  ...
}
```

And it calls cleanup before startup and during teardown:

```bash
cleanup_stale_sim_router() {
  echo "  preflight: cleaning stale ROS/Gazebo processes in distrobox '${SIM_DISTROBOX_NAME}'..."
  cleanup_host_sim_processes "preflight"
  cleanup_sim_container_processes "preflight"
  cleanup_host_sim_processes "preflight"
  sleep 2
}
```

The launcher also starts the policy before the recorder because the AIC engine has a finite participant discovery window:

```bash
pixi run env "PYTHONPATH=${WORKSPACE_DIR}/aic_teacher_official:${PYTHONPATH:-}" \
  ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p "policy:=${POLICY_CLASS}"
```

## How To Rerun The Successful Nominal Smoke

Use one accepted trajectory and one max attempt:

```bash
RUN_DIR=outputs/expert_datasets/nominal_live_full_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN_DIR"

pixi run python scripts/generate_expert_trajectories.py \
  --expert-mode nominal \
  --config outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n1__test_n3/engine_config.yaml \
  --target-accepted-trajectories 1 \
  --max-total-attempts 1 \
  --candidates-per-scene 1 \
  --score-threshold 95 \
  --max-offlimit-contacts 0 \
  --require-insertion-event true \
  --rerandomize-scene false \
  --strategy-model gpt-5-mini \
  --output-dir "$RUN_DIR" \
  --gazebo-gui false \
  --launch-rviz false \
  --startup-delay-sec 8 \
  --planner-recorder-drain-sec 120 \
  --recorder-drain-sec 180 \
  --per-trial-timeout-sec 420 \
  --launch-moveit true
```

Expected output layout:

```text
$RUN_DIR/generation_config.json
$RUN_DIR/generation_summary.json
$RUN_DIR/planner_attempts/attempt_000001_candidate_00/
$RUN_DIR/planner_attempts/attempt_000001_candidate_00/piecewise_trajectory.json
$RUN_DIR/planner_attempts/attempt_000001_candidate_00/smooth_trajectory.json
$RUN_DIR/planner_attempts/attempt_000001_candidate_00/planner_debug/
$RUN_DIR/replay_attempts/attempt_000001_candidate_00/
$RUN_DIR/replay_attempts/attempt_000001_candidate_00/results/trial_1_trial_000001/scoring.yaml
$RUN_DIR/accepted_metadata/meta/
```

## Validation Commands Run

These commands passed after the accepted run:

```bash
pixi run python -m pytest aic_teacher_official/test/test_expert_generator.py -q
pixi run python -m pytest aic_teacher_official/test -q
timeout 180 pixi run python -m pytest aic_utils/lerobot_robot_aic/test -q
timeout 180 pixi run python -m pytest aic_model/test -q
git diff --check
```

Results:

```text
aic_teacher_official/test/test_expert_generator.py: 25 passed
aic_teacher_official/test: 56 passed
aic_utils/lerobot_robot_aic/test: 11 passed
aic_model/test: 2 passed
git diff --check: passed
```

## Known Issues And Limitations

MoveIt/Zenoh shutdown still emits noisy SIGINT/atexit panic logs during teardown. The run is valid and teardown now clears stale processes, but this should be cleaned up later if possible.

The accepted run had a short force spike:

```text
Max detected force: 179.53N
Duration above 20N: 0.04 seconds
Penalty applied: no
```

The official scorer did not penalize it because the force spike was below the 1 second threshold. For higher-quality datasets, do not rely only on score; add a stricter `--max-force-threshold` or an impulse/contact-duration metric once available.

Trajectory smoothness is still worse than CheatCode alone:

```text
accepted MoveIt run jerk: 24.56 m/s^3
CheatCode smoke run jerk: 6.93 m/s^3
```

The current approach passes the default threshold, but it is not yet a high-margin expert.

The current MoveIt scene integration is still limited. It validates that MoveIt is used, but more complete collision object modeling from task-board assets/config/TF should be added.

Recovery mode has a modular state machine, but nominal live validation is currently much more mature than recovery dataset generation.

## Recommended Next Steps

1. Add stricter force acceptance for nominal datasets.

   The nominal expert should probably reject the accepted smoke trajectory if the dataset target is "highest-scoring minimal-contact experts", despite the official score passing.

   Suggested initial command change:

   ```bash
   --max-force-threshold 20
   ```

   If this rejects too much, implement a duration-aware force threshold instead of using only max force.

2. Generate a small nominal batch.

   Run 10 accepted trajectories with a reasonable attempt budget:

   ```bash
   pixi run python scripts/generate_expert_trajectories.py \
     --expert-mode nominal \
     --config outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n1__test_n3/engine_config.yaml \
     --target-accepted-trajectories 10 \
     --max-total-attempts 40 \
     --candidates-per-scene 3 \
     --score-threshold 95 \
     --max-offlimit-contacts 0 \
     --require-insertion-event true \
     --rerandomize-scene false \
     --strategy-model gpt-5-mini \
     --output-dir outputs/expert_datasets/nominal_sfp2nic_batch10 \
     --gazebo-gui false \
     --launch-rviz false \
     --startup-delay-sec 8 \
     --planner-recorder-drain-sec 120 \
     --recorder-drain-sec 180 \
     --per-trial-timeout-sec 420 \
     --launch-moveit true
   ```

3. Add score-margin reporting.

   `generation_summary.json` should include category-level scoring values extracted from `scoring.yaml`, not just total score and basic metrics. This will make it obvious whether failures are from duration, path length, jerk, force, or contacts.

4. Improve smoothness before large dataset generation.

   The default direct MoveIt plan passes but still has high jerk. Next options:

   - tune MoveIt time parameterization,
   - lower replay joint stiffness/damping during approach,
   - add a jerk-aware postprocess for joint trajectories,
   - evaluate Servo/Cartesian constrained approach after MoveIt reaches a safe staging pose.

5. Make MoveIt collision scene more complete.

   Build collision objects from task-board/asset config and TF, then add inflated keep-out zones from VLM `avoid_regions`. This should reduce reliance on luck while preserving the rule that MoveIt is required.

6. Run a real recovery-mode smoke.

   Recovery should be tested separately from nominal:

   ```bash
   pixi run python scripts/generate_expert_trajectories.py \
     --expert-mode recovery \
     --config outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n1__test_n3/engine_config.yaml \
     --target-accepted-trajectories 1 \
     --max-total-attempts 5 \
     --candidates-per-scene 1 \
     --score-threshold 90 \
     --ft-soft-threshold 1.0 \
     --ft-hard-threshold 3.0 \
     --backup-distance-m 0.002 \
     --max-retries 3 \
     --probe-pattern small_cross \
     --require-insertion-event true \
     --rerandomize-scene false \
     --strategy-model gpt-5-mini \
     --output-dir outputs/expert_datasets/recovery_smoke \
     --gazebo-gui false \
     --launch-rviz false \
     --launch-moveit true
   ```

7. Add integration tests behind explicit marks.

   Unit tests currently cover parser, MoveIt failure behavior, replay interpolation, dataset metadata, and recovery state transitions. Add opt-in integration tests for:

   - live image capture produces nonblank images,
   - MoveItPy plans at least one free-space segment,
   - replay reaches official scoring output,
   - cleanup leaves no stale `aic_engine`, `component_container`, or `static_transform_publisher` process.

8. Keep GPT-5-mini out of the low-level control loop.

   The current working design supports the key architectural decision: GPT-5-mini is useful for scene/cable-risk interpretation, but executable robot motion should come from deterministic robotics components and be filtered by Gazebo replay.

---

# Update: Final Insertion Debug Loop

Last updated: 2026-05-02

This section supersedes the older CLI examples above where they still mention `--expert-mode`, `--ft-soft-threshold`, or `--ft-hard-threshold`. The preferred interface is now `--nominal`, `--nominalrecovery`, and `--recovery`, with a single `--ft-threshold`.

## Current Approach

The architecture is still:

1. GPT-5-mini provides symbolic strategy and cable-risk assessment only.
2. MoveIt plans rigid obstacle-aware free-space approach.
3. Replay executes the MoveIt approach in joint space.
4. Near the port, `OfficialTeacherReplay` switches to deterministic CheatCode-style geometry.
5. Final insertion remains geometric CheatCode insertion, not MoveIt and not VLM waypoints.

The current default replay sequence is:

```text
MoveIt joint replay
  -> local_preinsert_align
  -> preinsert tracking/settle gate
  -> CheatCode exact-position guarded insertion
```

Important current defaults:

```text
AIC_OFFICIAL_TEACHER_CHEATCODE_Z_MODE=cheatcode_offsets
AIC_OFFICIAL_TEACHER_INSERTION_COMMAND_MODE=exact_position
AIC_OFFICIAL_TEACHER_CHEATCODE_INSERTION_SPEED_MPS=0.0013
AIC_OFFICIAL_TEACHER_PIN_INSERTION_TARGET=false
AIC_OFFICIAL_TEACHER_RECOVERY_BACKOFF_SEC=0.45
AIC_OFFICIAL_TEACHER_FORCE_RELEASE_STAGE_CHECK_SEC=0.10
```

`exact_position` means final insertion sends absolute Cartesian TCP targets in `base_link`, computed from CheatCode geometry. This replaced the previous default of converting the CheatCode target to `gripper/tcp` relative deltas, because live debug showed the relative-delta handoff could produce actual TCP speed spikes far above the commanded insertion speed.

## Live Debug Evidence

All runs below used:

```text
outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n1__test_n3/engine_config.yaml
```

The most useful successful live run was:

```text
outputs/expert_debug/nominal_exact_position_v1
```

Best candidate:

```text
attempt: 3
score: 92.45425262411736
task_score_excluding_tier_1: 91.45425262411736
insertion_event_reached: true
contact_detected: false
offlimit_contact_count: 0
max_tracking_error_m: 0.017845521164983353
max_guarded_insert_speed_mps: 0.023310237596234528
post_insert_max_force_delta_n: 3.405697326123005
```

This was a real insertion success, but it was still rejected by the current validator because:

```text
score threshold in that smoke was stricter than the measured score
guarded insert speed exceeded the 0.02 m/s validation threshold
```

Useful video path:

```text
outputs/expert_debug/nominal_exact_position_v1/replay_attempts/attempt_000003_candidate_02/dataset/videos/observation.images.center_camera/chunk-000/file-000.mp4
```

GPT-5 failure analysis was run on:

```text
outputs/expert_debug/nominal_exact_position_v1/rejected_attempts/attempt_000003/debug
```

Payload confirmed:

```text
observations_sampled: 80
actions_sampled: 80
ft_windows: 80
tracking_error_sampled: 80
sampled images: 8
sample_period_sec: 0.5
code_context included VLM, MoveIt, smoothing, insertion, replay/frame semantics
```

GPT-5 conclusion:

- MoveIt/free-space and VLM strategy were not the primary problem.
- `local_preinsert_align` helped and was not too slow.
- Final insertion succeeded and had no hard contact in the successful attempt.
- The remaining failure was actual guarded-insert speed: commanded `0.0013 m/s`, but measured max was about `0.0233 m/s`.
- There is still no global executable joint retiming across MoveIt approach segments.
- Frame/action logging must stay explicit; exact-position insertion should be logged as absolute Cartesian `base_link`, not relative `gripper/tcp`.

## Experiments Rejected By Live Data

Earlier relative-delta CheatCode insertion:

```text
outputs/expert_debug/nominal_ft5_v1
```

Mechanism:

```text
MoveIt joint-space approach
  -> local_preinsert_align as absolute base_link target
  -> preinsert settle/tracking gate
  -> CheatCode target recomputed in base_link
  -> compute_delta_pose converts target-current into gripper/tcp relative delta
```

Result:

```text
official score: 1
insertion_event_reached: false
contact_detected: true
max_guarded_insert_speed_mps: 0.025781
commanded insertion speed: 0.002 m/s
```

Why it failed:

- The handoff mixed action representations: joint-position MoveIt replay, absolute base-link local align, then gripper/tcp relative deltas.
- GPT-5 analysis found the actual TCP speed could spike about 10x above the commanded speed.
- The final insertion contact happened near the start of descent.
- This is why the default was changed away from relative deltas and toward exact-position base-link targets.

Dynamic exact-position CheatCode insertion at 1.3 mm/s:

```text
outputs/expert_debug/nominal_exact_position_v1
```

Mechanism:

```text
MoveIt joint-space approach
  -> local_preinsert_align as absolute base_link target
  -> preinsert settle/tracking gate
  -> exact-position CheatCode guarded insertion as absolute base_link TCP targets
  -> CheatCode target is recomputed dynamically during descent
```

Result:

```text
best attempt: attempt_000003_candidate_02
score: 92.45425262411736
task_score_excluding_tier_1: 91.45425262411736
insertion_event_reached: true
contact_detected: false
post_insert_max_force_delta_n: 3.405697326123005
max_tracking_error_m: 0.017845521164983353
max_guarded_insert_speed_mps: 0.023310237596234528
```

Why it partly succeeded:

- Removing gripper/tcp relative deltas reduced the handoff/frame ambiguity.
- Dynamic CheatCode target recomputation preserved enough alignment to seat the plug.
- F/T stayed below the 5 N debug threshold after insertion.

Why it was not accepted as clean nominal:

- The official score was below a stricter 95 threshold in that smoke.
- The measured guarded insert speed still exceeded the `0.02 m/s` validation threshold.
- This is currently the best baseline, but it still needs measured-speed-gated advancement.

Pinned XY/orientation insertion:

```text
outputs/expert_debug/nominal_pinned_insert_v1
```

Result:

```text
all 3 attempts failed
max_guarded_insert_speed_mps: about 0.0011 to 0.0013
insertion_event_reached: false for all attempts
contact_detected: true for all attempts
```

Interpretation: pinning the pre-insert XY/orientation through the entire descent makes speed smooth, but it is too rigid and misses insertion in this scene. It remains available only as:

```text
AIC_OFFICIAL_TEACHER_PIN_INSERTION_TARGET=true
```

Why it failed:

- The mechanism intentionally removed dynamic lateral/orientation adjustment during descent.
- That made actual speed excellent, but it removed the small compliance/adaptation that the successful dynamic exact-position run relied on.
- It consistently contacted or missed before insertion.
- Conclusion: use pinning only as an experiment/debug baseline, not as default.

Slower dynamic exact-position insertion at 1.0 mm/s:

```text
outputs/expert_debug/nominal_exact_slow_v1
```

Result:

```text
all 3 attempts failed
insertion_event_reached: false for all attempts
contact_detected: true for all attempts
max_guarded_insert_speed_mps: about 0.022 to 0.026
```

Interpretation: lowering commanded speed alone did not solve contact and sometimes contacted near the start of descent. Do not assume "slower is always safer" for this controller/path.

Why it failed:

- It changed timing but not the underlying target/tracking/contact geometry.
- Actual TCP speed still spiked to about `0.022-0.026 m/s`, so the controller did not obey the lower commanded speed in the way we needed.
- Contact timing got worse in some attempts, with contact near the start of descent.
- Conclusion: speed must be gated from measured actual TCP velocity, not only by lowering commanded insertion speed.

Pre-contact port/lateral alignment experiments:

```text
outputs/expert_debug/nominal_precontact_align_v1
outputs/expert_debug/nominal_precontact_align_v2
```

Mechanism:

```text
before guarded insertion:
  compute plug-tip lateral error relative to port in base_link
  command small absolute base_link lateral offsets
  abort if F/T rises near threshold
```

Results:

```text
v1 max offset: about 1.2 mm
v1 aborted with force_delta_n around 3.0 N

v2 max offset: about 0.25 mm, gain 0.15
v2 still aborted with force_delta_n around 3.5 N
```

Why it failed:

- Even tiny lateral corrections near the port created measurable contact/force buildup.
- This made insertion timing less reliable rather than improving it.
- The phase is now disabled by default:

```text
AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_SEC=0.0
```

Preinsert micro-align in gripper/tcp:

Mechanism:

```text
compute lateral tip error in base_link
rotate correction into gripper/tcp
send small gripper/tcp relative lateral deltas
abort on force rise
```

Why it was rejected:

- It reintroduced the frame/action mode that already caused handoff ambiguity.
- Live tests showed both absolute and relative x/y correction could worsen contact timing.
- It remains disabled by default:

```text
AIC_OFFICIAL_TEACHER_PREINSERT_MICRO_ALIGN_SEC=0.0
```

Live-Z insertion start / skip handoff experiment:

Mechanism:

```text
if current plug-port z offset is higher than the nominal CheatCode start offset:
  start guarded insertion from the measured live z offset
  skip forcing the robot down to the nominal 30 mm start offset first
```

Why it failed:

- It avoided a large handoff jump, but it also made insertion depth and duration very large in some runs.
- A known bad run used roughly:

```text
insertion_start_z_offset: about 0.069 m
insertion_depth_m: about 0.084 m
insertion_duration_sec: about 93 s
```

- The result was slow, misaligned, and not better than the fixed CheatCode start.
- Default now uses the fixed CheatCode-style start offset:

```text
AIC_OFFICIAL_TEACHER_CHEATCODE_START_Z_OFFSET=0.03
AIC_OFFICIAL_TEACHER_SKIP_CHEATCODE_HANDOFF_WHEN_LIVE_OFFSET=false
```

Strict nominal rejection with `--ft-threshold 1.0`:

Mechanism:

```text
nominal mode rejects on F/T threshold violation
no backoff/recovery after contact
```

Result:

- Correct semantically, but too strict for debugging because many official high-score trajectories have brief force/contact spikes that the official scorer does not penalize.
- It is still the right behavior for clean nominal data, but smoke debugging used `--ft-threshold 5.0` to see whether the trajectory could otherwise insert.

Why this matters:

- Do not compare official score alone to nominal dataset quality.
- `score: 92-95` may still include brief force behavior that clean ACT data should reject.
- Use `task_score_excluding_tier_1`, F/T windows, runtime trace, and phase-speed metrics together.

Nominalrecovery staged backoff:

```text
outputs/expert_debug/nominalrecovery_highscore_regression_v1
outputs/expert_debug/nominalrecovery_highscore_regression_v2
```

Mechanism:

```text
on F/T threshold during insertion:
  stop descent
  back off in smooth 5 mm stages
  wait for force release
  only then realign/retry
```

Results:

```text
backoff occurred: true
stages reached: up to about 25-30 mm in tested runs
force_release_before_realign: false
score: 1
```

Why it did not yet succeed:

- The recovery backoff mechanism is visible and smooth, but force did not drop below the threshold before the retry stage.
- This suggests either the backoff axis/baseline is still not ideal, or the force threshold used for debug was too strict relative to the sensor baseline/contact state.
- The newer defaults make backoff checks faster, but recovery still needs a dedicated live loop after the nominal insertion speed gate is fixed.

Score parser fix:

Mechanism:

```text
parse official total score as score
also expose task_score_excluding_tier_1
record tier_1_score, tier_2_score, tier_3_score, tier_3_message
```

Why it succeeded:

- It resolved the confusion that all failures were being parsed incorrectly as score `1`.
- `score: 1` is often the official Tier 1 model-validation credit only, not a parser failure.
- Prior high-score paths were found and used for comparison.

Important high-score historical examples:

```text
outputs/expert_debug/nominalrecovery_repair_v3/... score 95.312878
outputs/expert_debug/nominal_smoke/...candidate_01/... score 95.259664
outputs/expert_debug/nominal_repair_v1/... score 95.248257
outputs/expert_debug/nominalrecovery_repair_v1/... score 95.143273
outputs/expert_debug/nominal_smoke/...candidate_00/... score 95.027456
```

What they showed:

- Fixed CheatCode-style geometry around 30 mm pre-insert offset was more reliable than live-Z depth expansion.
- Some high scores tolerated short force spikes that should still be rejected for clean nominal datasets.
- The useful lesson was not "copy every old setting"; it was that the final insertion/handoff was the important subsystem, not VLM strategy or MoveIt planning.

## Current Code Changes To Preserve

These are intentional:

- `OfficialTeacherReplay` defaults guarded insertion to `exact_position`.
- Default commanded insertion speed is `0.0013 m/s`.
- Fully pinned insertion target is behind `AIC_OFFICIAL_TEACHER_PIN_INSERTION_TARGET=true`, not default.
- Recovery backoff checks are faster so nominalrecovery/recovery can back off promptly after contact.
- Debug/GPT code context describes exact-position/base-link insertion.
- Final-insertion debug action representation is corrected to absolute Cartesian/base-link when `insertion_command_mode=exact_position`.
- `score` parsing preserves the official total and also reports `task_score_excluding_tier_1`; do not treat every `score: 1` as a parser bug.

## Current Known Problems

1. Actual guarded-insert speed can still spike above the commanded speed and above the `0.02 m/s` validation threshold.

   This is the main remaining issue after the exact-position change.

2. Dynamic CheatCode exact-position can insert successfully, but success is not yet high-margin enough for clean nominal ACT data.

3. Fully pinned insertion and blind speed reduction both failed live.

4. MoveIt approach replay is still joint-space, while some smoothing/postprocessing is TCP metadata. Do not claim global executable smoothing unless the joint targets are actually retimed/resampled.

5. The accepted/rejected debug folders are useful, but generation currently writes placeholder GPT analysis files during debug artifact export. For real GPT-5 feedback, explicitly run:

   ```bash
   pixi run python scripts/analyze_expert_trajectory_failure.py \
     --debug-dir <debug-dir> \
     --model gpt-5
   ```

## Recommended Next Step

Do not keep blindly changing insertion speed.

The next targeted fix should be measured-speed-gated advancement during `exact_position` guarded insertion:

1. Keep dynamic CheatCode exact-position targets.
2. Keep commanded speed at `0.0013 m/s` initially.
3. During guarded insertion, measure actual TCP speed each control tick.
4. If actual TCP speed exceeds the speed gate, hold the previous target and do not advance depth.
5. Resume depth advancement only after measured speed falls below the gate.
6. Keep nominal behavior strict: if F/T threshold is exceeded, reject; no post-contact recovery in `--nominal`.
7. In `--nominalrecovery` and `--recovery`, if F/T threshold is exceeded, stop descent immediately and use smooth staged backoff before any realignment.

Suggested implementation location:

```text
aic_teacher_official/aic_teacher_official/OfficialTeacherReplay.py
```

Specifically inside `_run_online_cheatcode_insertion`, in the guarded insertion loop after `target_pose` is computed and before advancing to the next depth step.

Suggested trace fields:

```text
guarded_insert_speed_gate_checked
guarded_insert_speed_gate_held
actual_tcp_speed_mps
speed_threshold_mps
held_depth_step_count
target_z_offset
```

After implementing, rerun:

```bash
pixi run python scripts/generate_expert_trajectories.py \
  --nominal \
  --debug \
  --config outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n1__test_n3/engine_config.yaml \
  --target-accepted-trajectories 1 \
  --max-total-attempts 5 \
  --candidates-per-scene 5 \
  --score-threshold 90 \
  --ft-threshold 5.0 \
  --require-insertion-event true \
  --rerandomize-scene false \
  --strategy-model gpt-5-mini \
  --output-dir outputs/expert_debug/nominal_speed_gate_v1
```

Then run GPT-5 analysis on the best complete debug folder:

```bash
pixi run python scripts/analyze_expert_trajectory_failure.py \
  --debug-dir outputs/expert_debug/nominal_speed_gate_v1/rejected_attempts/attempt_00000X/debug \
  --model gpt-5
```

Compare against `nominal_exact_position_v1`:

```text
score
insertion_event_reached
contact_detected
max_guarded_insert_speed_mps
p95_guarded_insert_speed_mps
max_tracking_error_m
post_insert_force_delta_n
video smoothness
```

## Tests Last Run

After the latest code changes:

```text
pixi run python -m pytest aic_teacher_official/test -q
70 passed

pixi run python -m pytest aic_utils/lerobot_robot_aic/test -q
11 passed

pixi run python -m pytest aic_model/test -q
2 passed

git diff --check
passed
```

## Important Constraint Reminder

- Do not let VLM output low-level executable waypoints.
- Do not model cable dynamics explicitly.
- Do not use MoveIt for final insertion.
- Keep final insertion CheatCode/geometric.
- Keep LeRobot dataset compatibility.
- Keep `--debug` as one flag.
- Keep one `--ft-threshold`.
- Keep `--nominal`, `--nominalrecovery`, and `--recovery`.
- For `--nominal`, pre-contact repair is allowed, but post-contact backoff/recovery is not.
