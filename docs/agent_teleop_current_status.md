# Agent Teleop Expert Generator Current Status

Last updated: 2026-05-02

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

## 2026-05-02 Measured Speed Gate Update

Measured-speed-gated advancement has now been implemented in `OfficialTeacherReplay._run_online_cheatcode_insertion` for `exact_position` guarded insertion.

The new controls are:

```text
AIC_OFFICIAL_TEACHER_GUARDED_INSERT_SPEED_GATE_MPS=0.012
AIC_OFFICIAL_TEACHER_GUARDED_INSERT_SPEED_GATE_MAX_HOLD_SEC=1.20
```

When actual TCP speed is above the gate, replay holds the previous absolute target and does not advance insertion depth. The guarded loop still checks F/T while held, so `--nominal` rejects on confirmed contact and `--nominalrecovery`/`--recovery` can enter the staged backoff path. Runtime traces now include `guarded_insert_speed_gate_hold` and `guarded_insert_speed_gate_advance` with:

```text
guarded_insert_speed_gate_checked
guarded_insert_speed_gate_held
actual_tcp_speed_mps
speed_threshold_mps
held_depth_step_count
target_z_offset
```

Live results so far:

```text
outputs/expert_debug/nominal_speed_gate_v2_20260502T150033Z
```

This first implementation settled after every insertion step. It made the speed profile much cleaner but was too slow for useful nominal data: official score `1`, no insertion event, and a late contact near `z_offset=0.0042 m` after about 186 seconds.

```text
outputs/expert_debug/nominal_speed_gate_v3_20260502T151604Z
```

This corrected the gate to hold only on measured speed spikes. It failed early in nominal at the top of insertion: official score `1`, no insertion event, `max_guarded_insert_speed_mps=0.025878`, and confirmed contact at about `z_offset=0.02999 m`.

```text
outputs/expert_debug/nominalrecovery_speed_gate_v1_20260502T152011Z
```

This is the best current run. It mechanically succeeded and reached insertion, but validation rejected it only for the guarded-insert speed threshold:

```text
score: 92.04770098350596
task_score_excluding_tier_1: 91.04770098350596
insertion_event_reached: true
trajectory_duration_s: 36.22
contact_detected: false
backoff_occurred: false
post_insert_force_delta_n: 3.539299581344664
max_guarded_insert_speed_mps: 0.02221033771474994
p95_guarded_insert_speed_mps: 0.003956474725662291
median_guarded_insert_speed_mps: 0.0014781773340932912
```

The trace had 421 `guarded_insert_speed_gate_advance` events and 12 `guarded_insert_speed_gate_hold` events. Backoff is still available in `--nominalrecovery`/`--recovery`, but this run did not exercise it because no contact was confirmed.

```text
outputs/expert_debug/nominalrecovery_speed_gate_v2_20260502T152433Z
```

This attempted a stricter `0.006 m/s` gate but is inconclusive. The engine timed out waiting for the `aic_model` lifecycle state before a useful runtime trace was produced.

Follow-up preserve-Z runs:

```text
outputs/expert_debug/nominalrecovery_strict_preinsert_speed_gate_v1_20260502T153950Z
```

This used a strict `5 mm` preinsert tracking gate and `0.010 m/s` speed gate. It showed that guarded insertion speed was no longer the limiter (`max_guarded_insert_speed_mps=0.00388`), but the preinsert gate correctly failed at about `16.2 mm` pose error before insertion. GPT-5 analysis confirmed that asking the preinsert gate to solve the axial Z descent was fighting the controller.

```text
outputs/expert_debug/nominalrecovery_preserve_z_gate_v1_20260502T154721Z
```

This added preserve-current-Z behavior for the preinsert align/gate before online insertion. It reached insertion with `score=89.9473`, `max_guarded_insert_speed_mps=0.00684`, no off-limit contacts, and post-insert force delta below the policy threshold. Validation still rejected it before the bookkeeping fix because the handoff gate was repaired with live Z but still recorded as a failed gate, and the official scorer reported a short force spike that explicitly had no penalty applied.

```text
outputs/expert_debug/nominalrecovery_preserve_z_fast_insert_v1_20260502T155554Z
```

This is the current best accepted nominalrecovery run. It kept the strict `5 mm` preinsert gate, preserve-Z preinsert alignment/gating, `0.010 m/s` measured speed gate, and used `AIC_OFFICIAL_TEACHER_CHEATCODE_INSERTION_SPEED_MPS=0.0018`:

```text
accepted: 1
score: 91.74057004193335
task_score_excluding_tier_1: 90.74057004193335
insertion_event_reached: true
trajectory_duration_s: 37.23
contact_detected: false
backoff_occurred: false
max_guarded_insert_speed_mps: 0.00513056762681235
p95_guarded_insert_speed_mps: 0.003838073376777609
median_guarded_insert_speed_mps: 0.001623213218924153
official_max_force_n: 26.0
insertion_force_penalty_applied: false
post_insert_force_delta_n: 1.1542886447916316
```

The official scorer reported the force above 20 N for only `0.02 s`, below its `1.00 s` penalty threshold. The parser now preserves that raw value as `official_max_force_n` but does not reject on `max_force_n` when the official scorer says the penalty was not applied.

## Recovery Backoff Update - 2026-05-02

Backoff is still actively being exercised. The latest recovery work found that the CLI `--backup-distance-m` was only recorded in generation metadata and was not controlling online replay's staged backoff distance. Replay now receives:

```text
AIC_OFFICIAL_TEACHER_RECOVERY_MAX_BACKOFF_DISTANCE_M
AIC_OFFICIAL_TEACHER_RECOVERY_MAX_RETRIES
AIC_OFFICIAL_TEACHER_RECOVERY_RELEASE_FORCE_THRESHOLD_N
```

The new `--recovery-release-force-threshold` CLI flag decouples the force-delta threshold used to trigger contact from the threshold used to decide that a recovery backoff released force. This matters for strict debug runs because a `1.0 N` contact trigger is below the residual/noise level seen after small backoffs. Recovery realign also now preserves current TCP Z by default through `AIC_OFFICIAL_TEACHER_RECOVERY_REALIGN_PRESERVE_CURRENT_Z=true` and uses a slower recovery-specific speed cap, `AIC_OFFICIAL_TEACHER_RECOVERY_REALIGN_SPEED_MPS=0.02`.

Live forced-backoff runs:

```text
outputs/expert_debug/nominalrecovery_forced_backoff_release2_max5mm_v1_20260502T170239Z
```

This proved the CLI/env wiring worked: the backoff stopped at exactly `0.005 m`, and force release passed with a separate `2.0 N` release threshold. It still failed because the post-release realign did not preserve Z and the next gate failed with about `9.17 mm` tracking error and `2.18 N` force delta.

```text
outputs/expert_debug/nominalrecovery_forced_backoff_release2_max5mm_preservez_v2_20260502T170736Z
```

This is the best recovery mechanics run so far. It used `--ft-threshold 1.0`, `--backup-distance-m 0.005`, `--max-retries 2`, `--recovery-release-force-threshold 2.0`, and Z-preserving slower recovery realign. It still rejected because insertion did not complete, but the recovery stages behaved correctly:

```text
backoff_occurred: true
backoff_distance_achieved_m: 0.005
force_release_before_realign: true
recovery_post_realign_gate: passed
retry_tracking_gate: passed
max_guarded_insert_speed_mps: 0.004922984356074967
official_max_force_n: 0.0
```

The retry then contacted again around `1.09 N` under the deliberately strict `1.0 N` threshold and exhausted retries. Increasing only `--max-retries` to `3` repeated the same pattern: recovery and retry gates passed, guarded insertion resumed at low speed, then another small residual force delta exhausted the extra retry.

GPT-5 failure analysis on this run concluded that speed was no longer the root failure (`max_guarded_insert_speed_mps=0.00492`). The remaining issue is that the handoff/tracking gate can fail with large XY/Z tracking error and still start guarded insertion through live-Z repair. The recommended change is to make live-Z repair conditional on XY/yaw being in spec, then re-enable small body/port-frame micro-alignment before descent instead of allowing a Z-only descent from a laterally imperfect pose.

```text
outputs/expert_debug/nominalrecovery_forced_backoff_ft12_release2_max5mm_preservez_v1_20260502T171851Z
```

Raising the contact threshold to `1.2 N` still forced an initial backoff, but the fixed `5 mm` retreat did not satisfy the `2.0 N` release threshold in that run. This suggests the controller/backoff difficulty is not just the threshold value: small absolute-Z retreats can release in one run and not another because Cartesian `MODE_POSITION` targets are filtered through controller dynamics and residual cable/plug loading.

## Current Controller Finding

The speed/backoff difficulty appears partly inherent to `aic_controller`: Cartesian `MODE_POSITION` targets do not directly command TCP velocity. They provide position references to the controller, and the impedance dynamics plus current tracking error determine the actual TCP motion. That makes small commanded deltas and nominal insertion speeds only indirect controls over measured speed, contact timing, and backoff distance.

The practical mitigation is to keep using closed-loop gates around the controller:

1. Measure actual TCP speed and hold insertion depth when it spikes.
2. Measure actual TCP pose before descent and reject or realign if the preinsert pose is not tight enough.
3. Measure force release during recovery backoff instead of assuming commanded backoff distance equals physical separation.
4. Prefer port-frame or measured-pose delta logic for the final insertion path, but still execute it through bounded absolute targets.

## Recommended Next Steps

Do not keep blindly changing insertion speed. The preserve-Z run shows the nominalrecovery path can now produce accepted insertion trajectories with guarded speed below validation limits.

Try these in order, one variation at a time:

1. Re-run `nominalrecovery_preserve_z_fast_insert` for multiple seeds/candidates to check repeatability at score threshold 90.
2. Try the same preserve-Z/speed-gated settings in strict `--nominal`; nominal should still reject rather than recover on confirmed contact.
3. Tighten live-Z repair: only allow it when XY/orientation tracking is already within spec. If the handoff gate fails from a large pose error, run body/port-frame micro-align or reject instead of beginning Z descent.
4. For recovery data, do not keep increasing retries blindly. The forced runs show stable 5 mm recovery mechanics but repeated shallow contacts under strict thresholds. Next try an adaptive backoff distance: start at 5 mm, but if release is not observed or the next retry contacts before meaningful depth progress, increase to 10-15 mm before realign.
5. Add high-rate F/T debug around the handoff and start-of-insertion window so the short official force spikes can be localized instead of inferred from scoring YAML.
6. Longer-term, implement executable joint retiming/cross-fade for the MoveIt-to-Cartesian handoff and consider a port-frame delta-Z insertion formulation so the final descent starts from the measured live preinsert pose rather than from an assumed geometric pose.

## What Works Consistently So Far

Working:

- VLM strategy generation constrained to high-level planning plus MoveIt free-space approach, with CheatCode/geometric final insertion, remains the right architecture.
- Preserve-current-Z preinsert alignment and preinsert gate repair are the most reliable nominalrecovery configuration so far.
- Exact-position CheatCode-offset guarded insertion with measured TCP speed gating works for accepted nominal insertion. The best runs keep guarded speed below validation thresholds and reach insertion without contact.
- Recovery backoff is active and observable. With the latest wiring, `--backup-distance-m`, `--max-retries`, and `--recovery-release-force-threshold` reach online replay; 5 mm backoff plus Z-preserving recovery realign can release force and pass retry gates.
- GPT-5 failure analysis is useful for separating speed failures from geometry/gate failures; the latest analysis identified live-Z repair after failed lateral tracking as the next issue.

Not working reliably:

- Blindly lowering commanded insertion speed. The controller's impedance dynamics mean commanded speed is only indirect; measured TCP speed gates are required.
- Relative gripper/tcp insertion and mixed frame semantics. Earlier runs showed actual TCP speeds far above commanded speed.
- Pinned XY/orientation insertion as a default. It reduced speed but missed insertion in live testing.
- Precontact port align and TCP micro-align as previously configured. They tended to create sub-threshold force buildup or worsen contact timing.
- Increasing retry count alone. Forced-backoff runs with 2 and 3 retries repeated the same shallow contact pattern under strict thresholds.
- Fixed 5 mm backoff as a universal recovery distance. It can release in one run and fail release in another because the Cartesian controller does not make physical separation exactly equal commanded retreat.
- Live-Z repair after a failed handoff gate. It can rescue Z bookkeeping, but if XY/yaw are not already in spec it permits a Z-only descent from a laterally imperfect pose.

## GPT-5 Central-Camera Failure Analysis Update

The latest GPT-5 failure analysis uses only central-camera frames, sampled every `1.5 s`, with the nearest LeRobot observation/action row attached to each frame. The prompt explicitly describes the two-step strategy:

1. First trial: plan/execute segment-wise. Stalls are acceptable if geometry and recovery behavior are useful.
2. Postprocess: remove stall intervals in image-retrieval cadence units, globally smooth except near the last insertion, convert to a policy/action dataset, and replay the policy to save a cleaner final trajectory.

Artifacts:

```text
/home/ubuntu/ws_aic/src/aic/outputs/expert_debug/nominalrecovery_minbackoff15_live_lateral_repair_v1_20260502T191023Z/replay_attempts/attempt_000001_candidate_00/gpt5_replay_analysis/payload.json
/home/ubuntu/ws_aic/src/aic/outputs/expert_debug/nominalrecovery_minbackoff15_live_lateral_repair_v1_20260502T191023Z/replay_attempts/attempt_000001_candidate_00/gpt5_replay_analysis/prompt.md
/home/ubuntu/ws_aic/src/aic/outputs/expert_debug/nominalrecovery_minbackoff15_live_lateral_repair_v1_20260502T191023Z/replay_attempts/attempt_000001_candidate_00/gpt5_replay_analysis/analysis.md
```

GPT-5's core feedback: center-camera frames show the plug hovering offset at the cage mouth, and runtime trace shows `live_z_offset` around `44-46 mm` while lateral offsets can reach `~11 mm`. Recovery backoff succeeds, but the retry returns to the same misaligned preinsert and contacts immediately. This is an online geometry/gating problem; postprocessing cannot turn repeated first-step contacts into a successful insertion.

Ranked directions after GPT-5 review and local trace inspection:

1. **Gate or replace live-Z repair.** Only permit live-Z repair when lateral error, force, and live start height are all within bounds. Disable it by default in nominal.
2. **Try a smooth aligned transport target.** Use plug/port pose to move to an aligned pose closer to the port than the old transport target, then descend. This is the more promising way to make the first trial smooth without relying on postprocessing to fix collisions.
3. **Keep cadence-aware postprocessing.** Remove stalls after replay, in `1.5 s` central-frame cadence units, and protect the final insertion region from over-smoothing.
4. **Improve recovery only after descent is meaningful.** Backoff now works; more retries do not help if every retry starts from the same bad lateral pose.

## Nominalrecovery Smoothing Update

The current nominalrecovery priority is no longer to speed up the live replay trajectory before execution. A live attempt with pre-replay stall compaction and `2x` retiming made the replay controller produce excessive guarded-insert speed and failed validation. The better split is:

1. Keep live replay conservative and controller-gated.
2. Make backoff more deterministic online with immediate force confirmation and a minimum physical backoff distance before release checks.
3. Compact stalls after the real replay is recorded, then train/evaluate policy data from the postprocessed LeRobot dataset.

New controls:

```text
--backup-distance-m 0.015
--min-backoff-distance-m 0.015
--force-confirm-sec 0.0
--compact-stalls false
```

`--min-backoff-distance-m` is now passed to replay as `AIC_OFFICIAL_TEACHER_RECOVERY_MIN_BACKOFF_DISTANCE_M`. `OfficialTeacherReplay` will continue staged Z backoff until at least that distance is reached before it starts checking whether force has released. This addresses the observed failure mode where a `15 mm` max backoff still stopped at the first `5 mm` stage because force release happened early.

One live check with `--backup-distance-m 0.015`, `--min-backoff-distance-m 0.015`, and no pre-replay compaction confirmed the fix:

```text
outputs/expert_debug/nominalrecovery_minbackoff15_nocompact_v1_20260502T185340Z
backoff_occurred: true
backoff_distance_achieved_m: 0.015
force_release_before_realign: true
```

That run still rejected with score `1.0`, no insertion event, and `max_guarded_insert_speed_mps: 0.01293`. The important failure moved downstream: recovery returned with zero force and passed its internal return gates, but the redundant post-recovery retry gate then commanded back to the nominal port-center target, reintroduced about `2.0 N` force, and aborted before a real retry. The current mitigation is `AIC_OFFICIAL_TEACHER_SKIP_RETRY_GATE_AFTER_RECOVERY=true` by default, because `_backoff_and_realign` already checks the return-to-preinsert pose after force release and after realign.

Two follow-up live checks clarified the next bottleneck:

```text
outputs/expert_debug/nominalrecovery_minbackoff15_skipretrygate_v1_20260502T185920Z
```

Skipping the redundant recovery retry gate moved execution forward, but retry `1` failed at the forced retry preinsert micro-align gate. That gate ran even though micro-align and port-align durations were both disabled, and again pulled toward the nominal port-center target. The current mitigation is `AIC_OFFICIAL_TEACHER_SKIP_PREINSERT_GATE_ON_RECOVERY_RETRY=true` by default.

```text
outputs/expert_debug/nominalrecovery_minbackoff15_skip_retry_preinsert_gates_v1_20260502T190348Z
```

Skipping both redundant gates allowed actual retries. The run performed repeated `15 mm` backoffs and exhausted `--max-retries 2`. Backoff and force-release behavior was consistent, but every retry contacted immediately on the first guarded insert command. Measured guarded speed stayed below the configured speed gate (`max_guarded_insert_speed_mps: 0.00935`), so this was no longer a speed problem.

```text
outputs/expert_debug/nominalrecovery_minbackoff15_live_lateral_repair_v1_20260502T191023Z
```

Preserving measured live lateral offset during live-Z repair did not fix the first-command contact. It produced traceable offsets such as:

```text
lateral_offset_base_m: [-0.004825, -0.009798, 0.0]
```

but still contacted immediately and worsened measured guarded speed (`max_guarded_insert_speed_mps: 0.01664`). Treat this branch as not working reliably. The next likely change should be more structural: do not permit live-Z-repaired guarded descent when the retry context is laterally/force suspicious; either regenerate/replan the preinsert target, use a retry-specific insertion target policy that holds measured XY/orientation for the first few millimeters, or reject this recovery sample after recording the successful backoff.

Strict live-Z gating was then implemented:

```text
AIC_OFFICIAL_TEACHER_ENABLE_LIVE_Z_REPAIR=false by default in nominal
AIC_OFFICIAL_TEACHER_LIVE_Z_REPAIR_MAX_START_Z_OFFSET_M=0.035
AIC_OFFICIAL_TEACHER_LIVE_Z_REPAIR_MAX_LATERAL_ERROR_M=0.003
AIC_OFFICIAL_TEACHER_LIVE_Z_REPAIR_MAX_FORCE_DELTA_N=0.7
```

Live checks:

```text
outputs/expert_debug/nominalrecovery_strict_livez_gate_v1_20260502T193933Z
```

This rejected live-Z repair before descent. The trace showed `live_z_offset: 0.045776`, above the `0.035 m` cap, so the run stopped without immediate-contact recovery loops.

```text
outputs/expert_debug/nominal_strict_livez_disabled_v1_20260502T194301Z
```

Nominal rejected at the tracking gate with no live-Z repair, no descent, and no contact. This is safer than producing a bad nominal trajectory, but it is not enough to generate high-scoring nominal data.

Aligned-start experiments:

```text
outputs/expert_debug/nominal_aligned_start_z40_v1_20260502T194635Z
outputs/expert_debug/nominal_aligned_start_z40_gate7_v1_20260502T195006Z
```

Setting `AIC_OFFICIAL_TEACHER_CHEATCODE_START_Z_OFFSET=0.04` improved the initial aligned pose and sometimes passed the first gate, but did not produce insertion. With the strict `5 mm` gate, the handoff missed by about `6.3 mm`, mostly Z. With a `7 mm` gate, the initial gate failed due force (`~1.93 N`) before descent. This suggests the aligned-transport idea is still promising, but it needs an actual smooth target/settle policy and force-quiet check, not just a different z-offset.

For policy smoothing, use the post-replay compactor instead of pre-replay compaction:

```bash
.pixi/envs/default/bin/python scripts/compact_lerobot_stalls.py \
  --input-dataset <attempt>/dataset \
  --output-dataset <attempt>/dataset_compacted_stalls_motion005_v1 \
  --stall-translation-m 0.00005 \
  --stall-action-norm 0.00005 \
  --min-stall-frames 5 \
  --cadence-sec 1.5 \
  --cadence-stall-translation-m 0.001 \
  --cadence-stall-action-norm 0.0001 \
  --no-trim-videos \
  --overwrite
```

The first tuned compaction pass on a forced-backoff nominalrecovery attempt reduced the frame table from `2469` frames to `316` frames and retimed it from about `123.4 s` to `15.75 s`, while recomputing delta-pose actions from consecutive TCP states. Videos are copied unchanged unless `--trim-videos` is explicitly enabled, so this compacted dataset should be used for state/action policy data first.

Additional compaction passes:

```text
nominalrecovery_minbackoff15_nocompact_v1_20260502T185340Z: 927 -> 577 frames
nominalrecovery_minbackoff15_skipretrygate_v1_20260502T185920Z: 925 -> 464 frames
nominalrecovery_minbackoff15_skip_retry_preinsert_gates_v1_20260502T190348Z: 1701 -> 681 frames
nominalrecovery_minbackoff15_live_lateral_repair_v1_20260502T191023Z: 1705 -> 1000 frames
nominalrecovery_minbackoff15_live_lateral_repair_v1_20260502T191023Z with 1.5s cadence windows: 1705 -> 891 frames, 24 cadence windows removed
```

These contain increasingly complete recovery/backoff behavior, but none is an accepted insertion trajectory.

Useful success/debug folders:

Nominal:

```text
/home/ubuntu/ws_aic/src/aic/outputs/expert_debug/nominal_smoke/accepted_dataset_nominal
/home/ubuntu/ws_aic/src/aic/outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n2__act_smoke/accepted_dataset
/home/ubuntu/ws_aic/src/aic/outputs/trajectory_datasets/sfp_to_nic/cheatcode/nic_cards_1/n10__act_smoke/accepted_dataset
```

Nominalrecovery accepted:

```text
/home/ubuntu/ws_aic/src/aic/outputs/expert_debug/nominalrecovery_preserve_z_fast_insert_v1_20260502T155554Z
/home/ubuntu/ws_aic/src/aic/outputs/expert_debug/nominalrecovery_preserve_z_fast_insert_v1_20260502T155554Z/accepted_dataset_nominalrecovery
/home/ubuntu/ws_aic/src/aic/outputs/expert_debug/nominalrecovery_preserve_z_repeat_single_v1_20260502T165045Z
/home/ubuntu/ws_aic/src/aic/outputs/expert_debug/nominalrecovery_preserve_z_repeat_single_v1_20260502T165045Z/accepted_dataset_nominalrecovery
```

Recovery/backoff mechanics, not accepted insertion yet:

```text
/home/ubuntu/ws_aic/src/aic/outputs/expert_debug/nominalrecovery_forced_backoff_release2_max5mm_v1_20260502T170239Z
/home/ubuntu/ws_aic/src/aic/outputs/expert_debug/nominalrecovery_forced_backoff_release2_max5mm_preservez_v2_20260502T170736Z
/home/ubuntu/ws_aic/src/aic/outputs/expert_debug/nominalrecovery_forced_backoff_release2_max5mm_preservez_max3_v1_20260502T171312Z
/home/ubuntu/ws_aic/src/aic/outputs/expert_debug/nominalrecovery_forced_backoff_ft12_release2_max5mm_preservez_v1_20260502T171851Z
```

Postprocessed nominalrecovery policy-data smoothing:

```text
/home/ubuntu/ws_aic/src/aic/outputs/expert_debug/nominalrecovery_forced_backoff_release2_max5mm_preservez_v2_20260502T170736Z/replay_attempts/attempt_000001_candidate_00/dataset_compacted_stalls_motion005_v1
/home/ubuntu/ws_aic/src/aic/outputs/expert_debug/nominalrecovery_minbackoff15_nocompact_v1_20260502T185340Z/replay_attempts/attempt_000001_candidate_00/dataset_compacted_stalls_motion005_v1
/home/ubuntu/ws_aic/src/aic/outputs/expert_debug/nominalrecovery_minbackoff15_skipretrygate_v1_20260502T185920Z/replay_attempts/attempt_000001_candidate_00/dataset_compacted_stalls_motion005_v1
/home/ubuntu/ws_aic/src/aic/outputs/expert_debug/nominalrecovery_minbackoff15_skip_retry_preinsert_gates_v1_20260502T190348Z/replay_attempts/attempt_000001_candidate_00/dataset_compacted_stalls_motion005_v1
/home/ubuntu/ws_aic/src/aic/outputs/expert_debug/nominalrecovery_minbackoff15_live_lateral_repair_v1_20260502T191023Z/replay_attempts/attempt_000001_candidate_00/dataset_compacted_stalls_motion005_v1
/home/ubuntu/ws_aic/src/aic/outputs/expert_debug/nominalrecovery_minbackoff15_live_lateral_repair_v1_20260502T191023Z/replay_attempts/attempt_000001_candidate_00/dataset_compacted_stalls_cadence15_v1
```

## 2026-05-02 Central-Camera GPT-5 Follow-up

Current issue summary:

- VLM/MoveIt transport is no longer the main blocker. The runs generally reach the cage mouth, then fail during final insertion or recovery.
- Live-Z repair is now safer because it is disabled by default in nominal and gated in recovery, but the remaining failure is still geometric: the plug can be slightly low or laterally biased, then exact-position descent scrubs the face/lower lip.
- The inherent controller makes precise sub-millimeter gating difficult. Some runs settle near `0.6-0.8 mm` lateral error and are useful; others settle around `2-3.5 mm` even after several seconds. A new optional `AIC_OFFICIAL_TEACHER_TRACKING_GATE_MAX_LATERAL_ERROR_M` gate prevents descent from those bad residuals, but strict values can make runs reject before recovery starts.
- Postprocessing removes stalls and recomputes actions, but it cannot create insertion progress if the online run never inserts.

New controls added:

```text
AIC_OFFICIAL_TEACHER_MICRO_ALIGN_COMMAND_MODE=base_absolute
AIC_OFFICIAL_TEACHER_PREINSERT_VERTICAL_BIAS_M=<meters>
AIC_OFFICIAL_TEACHER_TRACKING_GATE_MAX_LATERAL_ERROR_M=<meters>
```

The `base_absolute` micro-align mode was added as a diagnostic alternative to relative `gripper/tcp` deltas. The vertical bias adds a base-Z offset to CheatCode preinsert/insertion target generation. The lateral gate is separate from aggregate controller TCP error, so we can reject or recover when the final x/y residual is too large even if controller error passes.

Runs and outcomes:

```text
outputs/expert_debug/nominal_aligned_z45_ft15_microalign_base_inverted_v1_20260502T203855Z
```

Base-frame micro-align with inverted gain improved compared with the positive base-frame sign. It still failed, but reached about `z_offset=0.0394` before contact, versus immediate/shallow contacts in several earlier nominal runs. GPT-5 central-camera analysis still called this a geometric lower-lip/low-start miss, not a timing-only issue.

```text
outputs/expert_debug/nominal_aligned_z45_up12_ft15_v1_20260502T204552Z
outputs/expert_debug/nominal_aligned_z45_up12_strictgate_ft15_v1_20260502T204931Z
```

A `+1.2 mm` vertical bias alone was not reliable. Without a strict lateral gate it contacted almost immediately; with a strict gate it correctly rejected before descent because final lateral error stayed around `1.9 mm`.

```text
outputs/expert_debug/nominalrecovery_z45_up06_microbase_inv_lateral15_ft15_v1_20260502T205422Z
```

Combined `+0.6 mm` vertical bias, inverted base-frame micro-align, `1.5 mm` lateral gate, and `1.5 N` contact threshold. It descended about `2.2 mm`, detected contact, backed off all the way to `30 mm`, but failed force release with the too-strict `0.5 N` release threshold.

```text
outputs/expert_debug/nominalrecovery_z45_up06_microbase_inv_lateral15_release15_ft15_v1_20260502T205818Z
```

Same idea with a relaxed release threshold, but the initial controller settle landed about `3.46 mm` lateral error. The new lateral gate prevented descent, which is the right safety behavior but also shows the current alignment path is not reliably precise enough.

```text
outputs/expert_debug/nominalrecovery_z45_ft15_release15_control_v1_20260502T210206Z
```

Best current recovery-mechanics run from this follow-up. No vertical bias, no micro-align, no lateral gate. Initial lateral was sub-millimeter, contact occurred, backoff reached `15 mm`, force release passed at `1.5 N`, return-to-preinsert gates passed, and retries reached `retry_count=2`. It still did not insert; each retry eventually contacted before insertion. GPT-5 central-camera analysis recommended an explicit near-preinsert S-curve target and ensuring recovery resets and re-runs a real guarded descent after backoff/realign.

Compacted policy-data version:

```text
/home/ubuntu/ws_aic/src/aic/outputs/expert_debug/nominalrecovery_z45_ft15_release15_control_v1_20260502T210206Z/replay_attempts/attempt_000001_candidate_00/dataset_compacted_stalls_cadence15_v1
```

Compaction reduced `2430 -> 468` frames and removed `64` cadence windows. Videos are copied unchanged; state/action data is recomputed and retimed.

Recommended next steps:

1. Implement a real near-preinsert aligned target: compute a pose from plug/port TFs, place it about `6-8 mm` from the mouth along the port axis, fully match port orientation, and move there with one smooth S-curve before guarded descent.
2. Keep the lateral gate, but use it as a reject/recover safety filter after the near-preinsert target is improved. The controller is too variable for a strict gate to work with the current coarse handoff.
3. Make recovery explicitly reset descent state after backoff and realign, then consume a retry only after a real guarded descent attempt. The best run shows backoff/release/return are working, but insertion retries still need better re-entry geometry.
4. Prefer `ft-threshold` around `2.0-2.5 N` with a debounce for nominalrecovery exploration. The official scorer still reports no force penalty in these light-contact failures, while the teacher threshold at `1.0-1.5 N` often stops before useful insertion progress.
5. Continue post-replay stall compaction at `1.5 s` cadence, but only after generating runs that contain meaningful insertion or recovery motion.

## Tests Last Run

After the latest code changes:

```text
.pixi/envs/default/bin/python -m pytest aic_teacher_official/test/test_official_teacher_pipeline.py aic_teacher_official/test/test_expert_generator.py -q
65 passed
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
