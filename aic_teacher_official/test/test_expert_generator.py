import json
from pathlib import Path
import sys
from types import SimpleNamespace
import importlib.util

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "generate_expert_trajectories.py"
spec = importlib.util.spec_from_file_location("generate_expert_trajectories", SCRIPT)
generate_expert_trajectories = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(generate_expert_trajectories)

from aic_teacher_official.expert_generator.candidate_generation import generate_approach_candidates
from aic_teacher_official.expert_generator.dataset_writer import DatasetMetadataWriter, ExpertEpisodeMetadata
from aic_teacher_official.expert_generator.debug_artifacts import (
    aggregate_ft_windows,
    build_gpt5_failure_payload,
    compute_phase_speed_metrics,
    compact_payload_with_retry,
    compute_transition_metrics,
    write_debug_artifacts,
)
from aic_teacher_official.expert_generator.ft_guard import FTGuard, FTGuardConfig, RecoveryPhase
from aic_teacher_official.expert_generator.generation_loop import ExpertGenerationLoop, GenerationConfig
from aic_teacher_official.expert_generator.collision_scene import object_geometries_from_engine_config
from aic_teacher_official.expert_generator.moveit_planner import (
    MoveItPlanner,
    MoveItUnavailableError,
    PlannedApproach,
    waypoints_from_candidate,
)
from aic_teacher_official.expert_generator.moveit_py_backend import MoveItPyBackendConfig, MoveItPyPlanningBackend
import aic_teacher_official.expert_generator.moveit_py_backend as moveit_py_backend
from aic_teacher_official.expert_generator.planner_runner import (
    ExpertPlannerRecordingRunner,
    ExpertPlannerRunConfig,
)
from aic_teacher_official.expert_generator.replay_runner import (
    OfficialRecordingReplayRunner,
    OfficialReplayConfig,
    metrics_from_scoring_yaml,
)
from aic_teacher_official.expert_generator.nominal_expert import NominalExpert
from aic_teacher_official.expert_generator.recovery_expert import RecoveryExpert
from aic_teacher_official.expert_generator.scene_snapshot import SceneSnapshot, SerializablePose
from aic_teacher_official.expert_generator.trajectory_repair import repair_precontact_approach
from aic_teacher_official.expert_generator.trajectory_validator import TrajectoryValidator, ValidationCriteria
from aic_teacher_official.expert_generator.vlm_strategy import ExpertMode, parse_vlm_strategy
from aic_teacher_official.expert_generator.vlm_strategy_client import OpenAIVLMStrategyProvider
from aic_teacher_official.replay import SmoothTrajectoryReplayPolicy
from aic_teacher_official.trajectory import SmoothTrajectory, PhaseLabel, SourceLabel, TCPPose, TrajectoryWaypoint


def _snapshot(mode="nominal"):
    return SceneSnapshot(
        run_id="run",
        seed=7,
        scene_id="scene_1",
        mode=mode,
        tcp_pose=SerializablePose([0.0, 0.0, 0.30], [0.0, 0.0, 0.0, 1.0]),
        target_port_pose=SerializablePose([0.10, 0.20, 0.12], [0.0, 0.0, 0.0, 1.0]),
        camera_images=["/tmp/live_left.png"],
    )


def _strategy(mode="nominal", **overrides):
    payload = {
        "mode": mode,
        "approach_side": "above_left",
        "cable_risk": "medium",
        "reason": "Cable trails near the front edge.",
        "mitigation": "Use high clearance and avoid lateral sweep.",
        "preferred_clearance_m": 0.12,
        "avoid_regions": ["front_of_nic"],
        "insertion_strategy": "straight_slow_descent" if mode == "nominal" else "guarded_descent_with_backoff",
        "recovery_allowed": mode == "recovery",
        "probe_pattern": "small_cross",
        "backup_distance_m": 0.002,
        "retry_count": 3,
    }
    payload.update(overrides)
    return parse_vlm_strategy(payload, expected_mode=mode)


class FakeMoveItBackend:
    def plan_free_space_approach(self, snapshot, strategy, candidate):
        objects = []
        return PlannedApproach(
            candidate_name=candidate.name,
            waypoints=waypoints_from_candidate(candidate),
            planning_scene_objects=objects,
            metadata={"backend": "fake_moveit", "moveit_used": True},
        )


class FakePlanningComponent:
    def __init__(self):
        self.goal_count = 0
        self.start_state_count = 0

    def set_start_state_to_current_state(self):
        pass

    def set_start_state(self, *_args, **_kwargs):
        self.start_state_count += 1

    def set_goal_state(self, **_kwargs):
        self.goal_count += 1

    def plan(self, _params=None):
        class Result:
            success = True
            trajectory = SimpleNamespace(
                robot_model=object(),
                get_robot_trajectory_msg=lambda: SimpleNamespace(
                    joint_trajectory=SimpleNamespace(
                        joint_names=[
                            "shoulder_pan_joint",
                            "shoulder_lift_joint",
                            "elbow_joint",
                            "wrist_1_joint",
                            "wrist_2_joint",
                            "wrist_3_joint",
                        ],
                        points=[
                            SimpleNamespace(
                                positions=[0.0, -1.0, 1.0, -1.5, 0.0, 0.0],
                                velocities=[0.0] * 6,
                                time_from_start=SimpleNamespace(sec=0, nanosec=0),
                            ),
                            SimpleNamespace(
                                positions=[0.1, -1.1, 1.1, -1.4, 0.1, 0.0],
                                velocities=[0.1] * 6,
                                time_from_start=SimpleNamespace(sec=1, nanosec=0),
                            ),
                        ],
                    )
                ),
            )

        return Result()


class FakeMoveItPy:
    def __init__(self):
        self.component = FakePlanningComponent()

    def get_planning_component(self, _name):
        return self.component


def test_vlm_strategy_schema_parsing_and_validation():
    strategy = _strategy()

    assert strategy.mode == ExpertMode.NOMINAL
    assert strategy.cable_risk == "medium"
    assert strategy.recovery_allowed is False
    assert strategy.retry_count == 0


def test_cli_mode_flags_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_expert_trajectories.py",
            "--nominal",
            "--recovery",
            "--config",
            "config.yaml",
            "--output-dir",
            "out",
        ],
    )

    with pytest.raises(SystemExit):
        generate_expert_trajectories.parse_args()


def test_cli_selects_new_nominal_mode_flag(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_expert_trajectories.py",
            "--nominal",
            "--config",
            "config.yaml",
            "--output-dir",
            "out",
        ],
    )

    args = generate_expert_trajectories.parse_args()

    assert generate_expert_trajectories._selected_mode(args) == ExpertMode.NOMINAL


def test_vlm_strategy_rejects_malformed_json():
    with pytest.raises(ValueError, match="Malformed"):
        parse_vlm_strategy("{not json", expected_mode="nominal")


def test_vlm_strategy_numeric_clamping():
    strategy = _strategy(preferred_clearance_m=99.0)

    assert strategy.preferred_clearance_m == pytest.approx(0.25)


def test_vlm_strategy_normalizes_simple_approach_side_aliases():
    strategy = _strategy(approach_side="right")

    assert strategy.approach_side == "above_right"


def test_vlm_strategy_normalizes_robot_frame_approach_side_aliases():
    strategy = _strategy(approach_side="robot_front")

    assert strategy.approach_side == "front"


def test_vlm_strategy_normalizes_descriptive_approach_side_aliases():
    strategy = _strategy(approach_side="port-normal (front-facing)")

    assert strategy.approach_side == "front"


def test_vlm_strategy_normalizes_cable_risk_aliases():
    strategy = _strategy(cable_risk="moderate")

    assert strategy.cable_risk == "medium"


def test_vlm_strategy_normalizes_avoid_region_variants():
    strategy = _strategy(avoid_regions={"front_of_nic": True, "rear": False, "note": "left sweep"})

    assert strategy.avoid_regions == ["front_of_nic", "left sweep"]


def test_recovery_probe_pattern_aliases_are_normalized():
    strategy = _strategy(
        mode="nominalrecovery",
        insertion_strategy="guarded_descent_with_backoff",
        recovery_allowed=True,
        probe_pattern={"type": "small_z_probe_then_spiral_xy"},
    )

    assert strategy.probe_pattern == "small_spiral"


def test_candidate_pose_generation_uses_strategy_and_port_orientation():
    candidates = generate_approach_candidates(_snapshot(), _strategy(), count=5)

    assert [c.name for c in candidates][:2] == ["above_left", "above"]
    assert len(candidates) == 5
    assert candidates[0].pre_insert_pose.position[:2] == pytest.approx([0.10, 0.20])
    assert candidates[0].metadata["orientation_source"] == "cheatcode_style_target_port_geometry"


def test_moveit_wrapper_fails_when_unavailable_and_has_no_fallback():
    with pytest.raises(MoveItUnavailableError, match="No geometric fallback"):
        MoveItPlanner(required=True, import_names=("definitely_missing_moveit_pkg",))


def test_moveit_wrapper_returns_structured_failure_without_required_import():
    planner = MoveItPlanner(required=False, import_names=("definitely_missing_moveit_pkg",))
    result = planner.plan_free_space_approach(_snapshot(), _strategy(), generate_approach_candidates(_snapshot(), _strategy(), count=1)[0])

    assert result.reason == "moveit_unavailable"
    assert result.recoverable is False


def test_moveit_py_backend_requires_successful_moveit_plan():
    snapshot = _snapshot()
    strategy = _strategy()
    candidate = generate_approach_candidates(snapshot, strategy, count=1)[0]
    fake_moveit = FakeMoveItPy()
    backend = MoveItPyPlanningBackend(moveit_py=fake_moveit)
    original_fk = moveit_py_backend._tcp_pose_from_joint_point
    moveit_py_backend._tcp_pose_from_joint_point = lambda _robot_model, **_kwargs: (
        TCPPose([0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0]),
        object(),
    )

    try:
        result = backend.plan_free_space_approach(snapshot, strategy, candidate)
    finally:
        moveit_py_backend._tcp_pose_from_joint_point = original_fk

    assert result.metadata["backend"] == "moveit_py"
    assert result.metadata["replay_space"] == "joint_position"
    assert fake_moveit.component.goal_count == 1
    assert fake_moveit.component.start_state_count == 0
    assert result.waypoints[-1].phase == "pre_insertion"
    assert result.waypoints[-1].joint_positions is not None


def test_moveit_py_backend_can_plan_legacy_three_stage_sequence():
    snapshot = _snapshot()
    strategy = _strategy()
    candidate = generate_approach_candidates(snapshot, strategy, count=1)[0]
    fake_moveit = FakeMoveItPy()
    backend = MoveItPyPlanningBackend(
        config=MoveItPyBackendConfig(approach_segment_mode="three_stage"),
        moveit_py=fake_moveit,
    )
    original_fk = moveit_py_backend._tcp_pose_from_joint_point
    moveit_py_backend._tcp_pose_from_joint_point = lambda _robot_model, **_kwargs: (
        TCPPose([0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0]),
        object(),
    )

    try:
        result = backend.plan_free_space_approach(snapshot, strategy, candidate)
    finally:
        moveit_py_backend._tcp_pose_from_joint_point = original_fk

    assert fake_moveit.component.goal_count == 3
    assert fake_moveit.component.start_state_count == 2
    assert [segment["segment"] for segment in result.metadata["segments"]] == [
        "safe_lift",
        "approach_standoff",
        "pre_insert",
    ]


def test_smooth_replay_interpolates_joint_targets():
    trajectory = SmoothTrajectory(
        waypoints=[
            TrajectoryWaypoint(
                timestamp=0.0,
                tcp_pose=TCPPose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                phase=PhaseLabel.APPROACH,
                source=SourceLabel.OPTIMIZER,
                joint_names=["j0", "j1"],
                joint_positions=[0.0, 1.0],
                joint_velocities=[0.0, 0.0],
            ),
            TrajectoryWaypoint(
                timestamp=2.0,
                tcp_pose=TCPPose([1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
                phase=PhaseLabel.PRE_INSERTION,
                source=SourceLabel.OPTIMIZER,
                joint_names=["j0", "j1"],
                joint_positions=[2.0, 3.0],
                joint_velocities=[0.2, 0.4],
            ),
        ]
    )

    target = SmoothTrajectoryReplayPolicy(trajectory).sample(1.0)

    assert target.joint_names == ["j0", "j1"]
    assert target.joint_positions == pytest.approx([1.0, 2.0])
    assert target.joint_velocities == pytest.approx([0.1, 0.2])


def test_nominal_mode_does_not_use_ft_correction():
    snapshot = _snapshot()
    strategy = _strategy()
    candidate = generate_approach_candidates(snapshot, strategy, count=1)[0]
    expert = NominalExpert(moveit_planner=MoveItPlanner(backend=FakeMoveItBackend()))

    result = expert.generate_candidate(snapshot, strategy, candidate=candidate)

    assert result.trajectory is not None
    assert result.trajectory.metadata.planning["vlm_waypoints_used"] is False
    assert result.trajectory.metadata.planning["ft_correction_used"] is False
    assert all(
        waypoint.diagnostics.get("ft_correction_used") is not True
        for waypoint in result.trajectory.waypoints
    )


def test_final_insertion_defaults_are_slow_with_settle_and_handoff():
    snapshot = _snapshot()
    strategy = _strategy()
    candidate = generate_approach_candidates(snapshot, strategy, count=1)[0]
    expert = NominalExpert(moveit_planner=MoveItPlanner(backend=FakeMoveItBackend()))

    result = expert.generate_candidate(snapshot, strategy, candidate=candidate)
    waypoints = result.trajectory.waypoints
    final = waypoints[-1]

    assert any(w.diagnostics.get("command_source") == "blended_handoff" for w in waypoints)
    assert any(w.phase == PhaseLabel.LOCAL_PREINSERT_ALIGN for w in waypoints)
    assert any(w.phase == PhaseLabel.HOLD for w in waypoints)
    assert final.diagnostics["insertion_speed_mps"] == pytest.approx(0.0013)
    assert final.timestamp - waypoints[-2].timestamp == pytest.approx(0.070 / 0.0013)


def test_precontact_repair_recomputes_actions_and_pins_preinsert_pose():
    trajectory = _debug_trajectory()

    repaired, metrics = repair_precontact_approach(trajectory, sample_dt=0.1)

    original_preinsert = trajectory.waypoints[-2]
    repaired_preinsert = repaired.waypoints[-2]
    assert metrics["action_recomputed"] is True
    assert metrics["preinsert_pose_pinned"] is True
    assert repaired_preinsert.tcp_pose.position == pytest.approx(original_preinsert.tcp_pose.position)
    assert repaired_preinsert.tcp_pose.orientation_xyzw == pytest.approx(original_preinsert.tcp_pose.orientation_xyzw)
    assert repaired.waypoints[-1].phase == PhaseLabel.FINAL_INSERTION
    assert repaired.waypoints[-1].tcp_pose.position == pytest.approx(trajectory.waypoints[-1].tcp_pose.position)


def test_phase_speed_metrics_reports_guarded_insert_speed():
    metrics = compute_phase_speed_metrics(
        [
            {
                "observation": {
                    "phase": "final_insertion",
                    "command_source": "guarded_insert",
                    "elapsed": 1.0,
                    "actual_tcp_velocity": {"linear": [0.0, 0.0, -0.01]},
                }
            },
            {
                "observation": {
                    "phase": "final_insertion",
                    "command_source": "guarded_insert",
                    "elapsed": 1.5,
                    "actual_tcp_velocity": {"linear": [0.0, 0.0, -0.03]},
                }
            },
        ]
    )

    assert metrics["max_guarded_insert_speed_mps"] == pytest.approx(0.03)


def test_recovery_state_machine_transitions():
    guard = FTGuard(FTGuardConfig(soft_threshold_n=1.0, hard_threshold_n=3.0, max_retries=1))

    assert guard.update({"force": [0.1, 0.0, 0.0]}) == RecoveryPhase.GUARDED_DESCENT
    assert guard.update({"force": [1.2, 0.0, 0.0]}) == RecoveryPhase.SOFT_CONTACT
    assert guard.update({"force": [0.2, 0.0, 0.0]}) == RecoveryPhase.BACKOFF
    assert guard.update({"force": [0.2, 0.0, 0.0]}) == RecoveryPhase.REALIGN
    assert guard.update({"force": [0.2, 0.0, 0.0]}) == RecoveryPhase.RETRY
    assert guard.update({"force": [0.2, 0.0, 0.0]}) == RecoveryPhase.GUARDED_DESCENT


def test_recovery_expert_online_api_returns_recovery_action():
    expert = RecoveryExpert(moveit_planner=MoveItPlanner(backend=FakeMoveItBackend()))

    action = expert.recover_from_state(
        {"wrench_force_torque": {"force": [2.0, 0.0, 0.0]}},
        _snapshot(mode="recovery"),
    )

    assert action.phase == RecoveryPhase.SOFT_CONTACT
    assert action.command == "stop_descent"


def test_nominalrecovery_strategy_and_recovery_expert_are_supported():
    snapshot = _snapshot(mode="nominalrecovery")
    strategy = _strategy(
        mode="nominalrecovery",
        insertion_strategy="guarded_descent_with_backoff",
        recovery_allowed=True,
    )
    candidate = generate_approach_candidates(snapshot, strategy, count=1)[0]
    expert = RecoveryExpert(moveit_planner=MoveItPlanner(backend=FakeMoveItBackend()))

    result = expert.generate_candidate(snapshot, strategy, candidate=candidate)

    assert result.trajectory is not None
    assert result.metadata["mode"] == "nominalrecovery"
    assert result.trajectory.metadata.planning["ft_correction_used"] is True


def test_dataset_metadata_writer_outputs_sidecars(tmp_path):
    writer = DatasetMetadataWriter(tmp_path)
    writer.append_episode(
        ExpertEpisodeMetadata(
            episode_index=0,
            mode="nominal",
            scene_id="scene",
            candidate_index=0,
            trajectory_path=None,
            validation={"accepted": True},
            vlm_strategy={"cable_risk": "low"},
            moveit={"success": True},
            phase_labels=[{"phase": "approach"}],
            extra={},
        )
    )

    assert (tmp_path / "meta" / "expert_trajectory_metadata.jsonl").exists()
    assert (tmp_path / "meta" / "validation_results.jsonl").exists()
    assert json.loads((tmp_path / "meta" / "vlm_strategy.jsonl").read_text().splitlines()[0])["cable_risk"] == "low"


def test_ft_window_aggregation_min_max_median():
    windows = aggregate_ft_windows(
        [
            {"timestamp": 0.0, "force": [1.0, 2.0, 3.0], "torque": [0.1, 0.2, 0.3]},
            {"timestamp": 0.2, "force": [3.0, 4.0, 5.0], "torque": [0.3, 0.4, 0.5]},
            {"timestamp": 0.7, "fx": 10.0, "fy": 0.0, "fz": 0.0, "tx": 1.0, "ty": 0.0, "tz": 0.0},
        ],
        window_sec=0.5,
    )

    assert len(windows) == 2
    assert windows[0]["fx"]["min"] == pytest.approx(1.0)
    assert windows[0]["fx"]["max"] == pytest.approx(3.0)
    assert windows[0]["fx"]["median"] == pytest.approx(2.0)
    assert windows[1]["force_norm"]["median"] == pytest.approx(10.0)


def _debug_trajectory():
    return SmoothTrajectory(
        waypoints=[
            TrajectoryWaypoint(
                timestamp=0.0,
                tcp_pose=TCPPose([0.0, 0.0, 0.10], [0.0, 0.0, 0.0, 1.0]),
                phase=PhaseLabel.APPROACH,
                source=SourceLabel.OPTIMIZER,
                joint_names=["j0"],
                joint_positions=[0.0],
                joint_velocities=[0.0],
            ),
            TrajectoryWaypoint(
                timestamp=1.0,
                tcp_pose=TCPPose([0.0, 0.0, 0.08], [0.0, 0.0, 0.0, 1.0]),
                phase=PhaseLabel.PRE_INSERTION,
                source=SourceLabel.OPTIMIZER,
                joint_names=["j0"],
                joint_positions=[0.1],
                joint_velocities=[0.0],
            ),
            TrajectoryWaypoint(
                timestamp=1.75,
                tcp_pose=TCPPose([0.0, 0.0, 0.08], [0.0, 0.0, 0.0, 1.0]),
                phase=PhaseLabel.HOLD,
                source=SourceLabel.CHEATCODE,
                joint_names=["j0"],
                joint_positions=[0.1],
                joint_velocities=[0.0],
            ),
            TrajectoryWaypoint(
                timestamp=16.75,
                tcp_pose=TCPPose([0.0, 0.0, 0.035], [0.0, 0.0, 0.0, 1.0]),
                tcp_velocity=[0.0, 0.0, -0.002],
                phase=PhaseLabel.FINAL_INSERTION,
                source=SourceLabel.CHEATCODE,
            ),
        ]
    )


def test_debug_artifact_writer_outputs_expected_files(tmp_path):
    paths = write_debug_artifacts(tmp_path, trajectory=_debug_trajectory())

    assert paths.observations.exists()
    assert paths.actions.exists()
    assert paths.transition_metrics.exists()
    assert (tmp_path / "debug" / "sampled_images" / "center").is_dir()
    assert json.loads(paths.transition_metrics.read_text())["boundary_count"] >= 2


def test_transition_metrics_marks_large_phase_boundary_suspicious():
    metrics = compute_transition_metrics(_debug_trajectory())

    assert metrics["boundary_count"] >= 2
    assert any(boundary["phase_after"] == "pre_insertion" for boundary in metrics["boundaries"])


def test_gpt5_payload_fails_fast_when_required_artifacts_missing(tmp_path):
    (tmp_path / "debug").mkdir()

    with pytest.raises(FileNotFoundError):
        build_gpt5_failure_payload(tmp_path / "debug")


def test_gpt5_payload_downsamples_to_one_second(tmp_path, monkeypatch):
    paths = write_debug_artifacts(tmp_path, trajectory=_debug_trajectory())
    with paths.ft_windows.open("w", encoding="utf-8") as f:
        for i in range(8):
            f.write(json.dumps({"window_start": i * 0.5, "force_norm": {"max": i}}) + "\n")
    monkeypatch.setattr(
        "aic_teacher_official.expert_generator.debug_artifacts.MAX_PROMPT_BYTES",
        1,
    )

    with pytest.raises(ValueError, match="1.0 second sampling"):
        compact_payload_with_retry(paths.debug_dir)


def test_collision_scene_extracts_rigid_objects_from_engine_config(tmp_path):
    config = {
        "trials": {
            "trial_000001": {
                "scene": {
                    "task_board": {
                        "pose": {"x": 0.1, "y": -0.2, "z": 1.14, "yaw": 3.14},
                        "nic_rail_0": {
                            "entity_present": True,
                            "entity_name": "nic_card_0",
                            "entity_pose": {"translation": 0.01, "yaw": 0.0},
                        },
                        "nic_rail_1": {"entity_present": False},
                    }
                }
            }
        }
    }

    objects = object_geometries_from_engine_config(config)

    assert [obj.role for obj in objects] == ["task_board", "nic_card"]
    assert objects[1].metadata["rail"] == "nic_rail_0"


def test_replay_runner_builds_official_recording_command(tmp_path):
    runner = OfficialRecordingReplayRunner(
        OfficialReplayConfig(
            repo_root=Path("/repo"),
            engine_config=Path("/repo/config.yaml"),
            output_dir=tmp_path,
            require_recorder_save_log=True,
        )
    )

    cmd = runner.build_command(
        trajectory_path=tmp_path / "smooth.json",
        attempt_dir=tmp_path / "attempt",
        attempt_index=1,
        candidate_index=2,
    )

    assert "./aic_utils/lerobot_robot_aic/scripts/launch_policy_recording_per_trial.sh" in cmd
    assert "aic_teacher_official.OfficialTeacherReplay" in cmd
    assert "--teacher-trajectory" in cmd
    assert "--teacher-action-mode" in cmd
    assert cmd[cmd.index("--teacher-action-mode") + 1] == "joint_position_then_cheatcode"


def test_replay_runner_passes_recovery_env(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, cwd, env, text, stdout, stderr, check):
        captured["env"] = env
        return SimpleNamespace(returncode=0)

    class FakeTrajectory:
        def save_json(self, path):
            path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr("subprocess.run", fake_run)
    runner = OfficialRecordingReplayRunner(
        OfficialReplayConfig(
            repo_root=Path("/repo"),
            engine_config=Path("/repo/config.yaml"),
            output_dir=tmp_path,
            ft_threshold_n=1.0,
            recovery_backoff_distance_m=0.005,
            recovery_backoff_increment_m=0.015,
            recovery_backoff_sec=0.5,
            recovery_min_backoff_distance_m=0.015,
            recovery_max_retries=2,
            recovery_release_force_threshold_n=2.0,
            force_confirm_sec=0.0,
            cartesian_stiffness="70,70,70,40,40,40",
            cartesian_damping="55",
            recovery_cartesian_stiffness="45,45,55,35,35,35",
            recovery_cartesian_damping="70,70,80,30,30,30",
            joint_stiffness="80",
            joint_damping="30",
        )
    )

    runner.replay_and_score(FakeTrajectory(), attempt_index=1, candidate_index=1)

    env = captured["env"]
    assert env["AIC_OFFICIAL_TEACHER_FT_THRESHOLD_N"] == "1.0"
    assert env["AIC_OFFICIAL_TEACHER_RECOVERY_MAX_BACKOFF_DISTANCE_M"] == "0.005"
    assert env["AIC_OFFICIAL_TEACHER_RECOVERY_BACKOFF_DISTANCE_M"] == "0.015"
    assert env["AIC_OFFICIAL_TEACHER_RECOVERY_BACKOFF_SEC"] == "0.5"
    assert env["AIC_OFFICIAL_TEACHER_RECOVERY_MIN_BACKOFF_DISTANCE_M"] == "0.015"
    assert env["AIC_OFFICIAL_TEACHER_RECOVERY_MAX_RETRIES"] == "2"
    assert env["AIC_OFFICIAL_TEACHER_RECOVERY_RELEASE_FORCE_THRESHOLD_N"] == "2.0"
    assert env["AIC_OFFICIAL_TEACHER_FORCE_CONFIRM_SEC"] == "0.0"
    assert env["AIC_OFFICIAL_TEACHER_CARTESIAN_STIFFNESS"] == "70,70,70,40,40,40"
    assert env["AIC_OFFICIAL_TEACHER_CARTESIAN_DAMPING"] == "55"
    assert env["AIC_OFFICIAL_TEACHER_RECOVERY_CARTESIAN_STIFFNESS"] == "45,45,55,35,35,35"
    assert env["AIC_OFFICIAL_TEACHER_RECOVERY_CARTESIAN_DAMPING"] == "70,70,80,30,30,30"
    assert env["AIC_OFFICIAL_TEACHER_JOINT_STIFFNESS"] == "80"
    assert env["AIC_OFFICIAL_TEACHER_JOINT_DAMPING"] == "30"


def test_expert_planner_runner_builds_live_policy_command(tmp_path):
    runner = ExpertPlannerRecordingRunner(
        ExpertPlannerRunConfig(
            repo_root=Path("/repo"),
            engine_config=Path("/repo/config.yaml"),
            output_dir=tmp_path,
            expert_mode="nominal",
        )
    )

    cmd = runner.build_command(attempt_dir=tmp_path / "attempt")

    assert "aic_teacher_official.OfficialExpertGeneratorPlanner" in cmd
    assert "--engine-config" in cmd
    assert "--require-recorder-save-log" in cmd
    assert "--launch-moveit" in cmd
    assert "aic_moveit_config moveit.launch.py" in cmd


def test_moveit_config_assets_exist():
    root = Path(__file__).resolve().parents[2] / "aic_moveit_config"

    assert (root / "config" / "aic.srdf").exists()
    assert (root / "config" / "ompl_planning.yaml").exists()
    assert (root / "config" / "moveit_cpp.yaml").exists()
    assert (root / "launch" / "moveit.launch.py").exists()
    assert "ur_manipulator" in (root / "config" / "aic.srdf").read_text(encoding="utf-8")


def test_vlm_strategy_client_requires_live_images_before_api_key():
    snapshot = SceneSnapshot(
        run_id="run",
        seed=1,
        scene_id="scene",
        mode="nominal",
        camera_images=[],
    )

    with pytest.raises(RuntimeError, match="live validated camera images"):
        OpenAIVLMStrategyProvider().strategy_for_scene(snapshot, mode=ExpertMode.NOMINAL)


def test_scoring_yaml_metrics_parser(tmp_path):
    scoring = tmp_path / "scoring.yaml"
    scoring.write_text(
        """
total: 96.5
trial_000001:
  tier_2:
    categories:
      contacts:
        message: No contact detected.
      duration:
        message: "Task duration: 12.34 seconds."
      insertion force:
        message: "Max detected force: 2.50N."
  tier_3:
    message: Cable insertion successful.
""",
        encoding="utf-8",
    )

    metrics = metrics_from_scoring_yaml(scoring)

    assert metrics["score"] == pytest.approx(96.5)
    assert metrics["official_total_score"] == pytest.approx(96.5)
    assert metrics["insertion_event_reached"] is True
    assert metrics["max_force_n"] == pytest.approx(2.5)
    assert metrics["official_max_force_n"] == pytest.approx(2.5)
    assert metrics["insertion_force_penalty_applied"] is True
    assert metrics["offlimit_contact_count"] == 0
    assert metrics["trajectory_duration_s"] == pytest.approx(12.34)


def test_scoring_yaml_metrics_parser_preserves_official_total_and_task_score(tmp_path):
    scoring = tmp_path / "scoring.yaml"
    scoring.write_text(
        """
total: 1
trial_000001:
  tier_1:
    score: 1
    message: Model validation succeeded.
  tier_2:
    score: 0
    message: Scoring succeeded.
    categories:
      contacts:
        score: 0
        message: No contact detected.
      duration:
        score: 0
        message: Task not completed.
      insertion force:
        score: 0
        message: No excessive force detected
      trajectory efficiency:
        score: 0
        message: Task not completed.
      trajectory smoothness:
        score: 0
        message: Task not completed.
  tier_3:
    score: 0
    message: Task not completed.
""",
        encoding="utf-8",
    )

    metrics = metrics_from_scoring_yaml(scoring)

    assert metrics["official_total_score"] == pytest.approx(1.0)
    assert metrics["score"] == pytest.approx(1.0)
    assert metrics["task_score_excluding_tier_1"] == pytest.approx(0.0)
    assert metrics["score_source"] == "official_total"
    assert metrics["tier_1_score"] == pytest.approx(1.0)
    assert metrics["tier_2_score"] == pytest.approx(0.0)
    assert metrics["tier_3_score"] == pytest.approx(0.0)
    assert metrics["tier_3_message"] == "Task not completed."
    assert metrics["has_partial_or_full_task_progress"] is False
    assert metrics["insertion_event_reached"] is False
    assert metrics["max_force_n"] == pytest.approx(0.0)
    assert metrics["official_max_force_n"] == pytest.approx(0.0)
    assert metrics["insertion_force_penalty_applied"] is False


def test_scoring_yaml_metrics_parser_ignores_unpenalized_force_spike(tmp_path):
    scoring = tmp_path / "scoring.yaml"
    scoring.write_text(
        """
total: 89.9
trial_000001:
  tier_2:
    categories:
      contacts:
        message: No contact detected.
      duration:
        message: "Task duration: 45.29 seconds."
      insertion force:
        message: "Insertion force above 20.00 N, detected for a time of 0.02 seconds. Max detected force: 27.46N. This is below the threshold of 1.00 seconds. Penalty not applied."
  tier_3:
    message: Cable insertion successful.
""",
        encoding="utf-8",
    )

    metrics = metrics_from_scoring_yaml(scoring)

    assert metrics["max_force_n"] == pytest.approx(0.0)
    assert metrics["official_max_force_n"] == pytest.approx(27.46)
    assert metrics["insertion_force_penalty_applied"] is False


def test_scoring_yaml_metrics_parser_preserves_partial_success_points(tmp_path):
    scoring = tmp_path / "scoring.yaml"
    scoring.write_text(
        """
total: 59
trial_000001:
  tier_1:
    score: 1
    message: Model validation succeeded.
  tier_2:
    score: 18
    message: Scoring succeeded.
    categories:
      contacts:
        score: 0
        message: No contact detected.
      duration:
        score: 8
        message: "Task duration: 20.0 seconds."
      insertion force:
        score: 0
        message: No excessive force detected
      trajectory efficiency:
        score: 5
        message: "Total end-effector path length: 0.40 m"
      trajectory smoothness:
        score: 5
        message: "Average linear jerk magnitude of the end effector: 10.0 m/s^3"
  tier_3:
    score: 40
    message: Partial insertion detected with distance of 0.01m.
""",
        encoding="utf-8",
    )

    metrics = metrics_from_scoring_yaml(scoring)

    assert metrics["score"] == pytest.approx(59.0)
    assert metrics["official_total_score"] == pytest.approx(59.0)
    assert metrics["task_score_excluding_tier_1"] == pytest.approx(58.0)
    assert metrics["tier_1_score"] == pytest.approx(1.0)
    assert metrics["tier_2_score"] == pytest.approx(18.0)
    assert metrics["tier_3_score"] == pytest.approx(40.0)
    assert metrics["tier_3_message"] == "Partial insertion detected with distance of 0.01m."
    assert metrics["has_partial_or_full_task_progress"] is True
    assert metrics["insertion_event_reached"] is False


class OneSceneProvider:
    def next_scene(self, *, attempt_index, rerandomize_scene, respawn_assets):
        return _snapshot()


class StaticStrategyProvider:
    def strategy_for_scene(self, snapshot, *, mode, output_dir=None):
        return _strategy(mode.value)


class AcceptingReplayRunner:
    def replay_and_score(self, trajectory, *, attempt_index, candidate_index):
        return {
            "score": 99.0,
            "insertion_event_reached": True,
            "max_force_n": 0.2,
            "offlimit_contact_count": 0,
            "trajectory_duration_s": 5.0,
        }


def test_generation_loop_targets_accepted_and_stops(tmp_path):
    config = GenerationConfig(
        expert_mode=ExpertMode.NOMINAL,
        target_accepted_trajectories=2,
        max_total_attempts=5,
        candidates_per_scene=3,
    )
    loop = ExpertGenerationLoop(
        config=config,
        scene_provider=OneSceneProvider(),
        strategy_provider=StaticStrategyProvider(),
        expert=NominalExpert(moveit_planner=MoveItPlanner(backend=FakeMoveItBackend())),
        replay_runner=AcceptingReplayRunner(),
        validator=TrajectoryValidator(ValidationCriteria(score_threshold=95.0)),
        metadata_writer=DatasetMetadataWriter(tmp_path),
    )

    summary = loop.run()

    assert summary.accepted == 2
    assert summary.attempts == 2
    assert summary.stopped_reason == "target_reached"
