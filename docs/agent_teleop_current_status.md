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
     --backup-distance-m 0.006 \
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

