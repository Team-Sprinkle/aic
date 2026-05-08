import json
from pathlib import Path
import sys
from types import SimpleNamespace
import importlib.util

import pytest
import yaml

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
from aic_teacher_official.expert_generator.scene_snapshot import ObjectGeometry, SceneSnapshot, SerializablePose
from aic_teacher_official.expert_generator.trajectory_repair import repair_precontact_approach
from aic_teacher_official.expert_generator.trajectory_validator import TrajectoryValidator, ValidationCriteria
from aic_teacher_official.expert_generator.vlm_strategy import ExpertMode, build_strategy_prompt, parse_vlm_strategy
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


def _sc_snapshot_with_nics(mode="nominal"):
    snapshot = _snapshot(mode)
    return SceneSnapshot(
        **{
            **snapshot.__dict__,
            "task_config": {
                "plug_type": "sc",
                "port_type": "sc",
                "target_module_name": "sc_port_0",
            },
            "collision_objects": [
                ObjectGeometry(
                    name="trial_nic_card_0",
                    pose=SerializablePose([0.0, 0.04, 0.02], [0.0, 0.0, 0.0, 1.0]),
                    shape="box",
                    dimensions=[0.12, 0.06, 0.03],
                    role="nic_card",
                    metadata={"rail": "nic_rail_0"},
                ),
                ObjectGeometry(
                    name="trial_nic_card_1",
                    pose=SerializablePose([0.0, 0.09, 0.02], [0.0, 0.0, 0.0, 1.0]),
                    shape="box",
                    dimensions=[0.12, 0.06, 0.03],
                    role="nic_card",
                    metadata={"rail": "nic_rail_1"},
                ),
            ],
        }
    )


def test_planner_result_quota_failure_is_detected():
    result = {
        "success": False,
        "reason": "exception",
        "type": "RateLimitError",
        "error": "Error code: 429 - {'error': {'code': 'insufficient_quota'}}",
    }

    assert generate_expert_trajectories._planner_result_is_quota_failure(result)


def test_planner_result_non_quota_failure_is_not_detected():
    result = {
        "success": False,
        "reason": "planner_did_not_write_piecewise",
        "type": "MoveItError",
        "error": "No valid plan found",
    }

    assert not generate_expert_trajectories._planner_result_is_quota_failure(result)


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


def test_sc_to_sc_candidates_route_around_present_nic_cards():
    snapshot = _sc_snapshot_with_nics()
    candidates = generate_approach_candidates(snapshot, _strategy(approach_side="front"), count=3)

    assert [c.name for c in candidates] == ["above_left", "back", "high_clearance_vertical"]
    assert candidates[0].metadata["sc_nic_obstacle_route"] is True
    assert snapshot.tcp_pose is not None
    assert snapshot.target_port_pose is not None
    route = candidates[0].route_subgoals
    assert [subgoal.name for subgoal in route] == [
        "camera_left_clearance",
        "left_lane_descent",
        "outside_lane_forward_past_cards",
        "right_sweep_toward_port",
        "port_standoff",
        "pre_insert",
    ]
    assert route[0].pose.position[0] < snapshot.tcp_pose.position[0]
    assert route[0].pose.position[1] == pytest.approx(snapshot.tcp_pose.position[1])
    assert route[1].pose.position[0] == pytest.approx(route[0].pose.position[0])
    assert route[1].pose.position[1] == pytest.approx(route[0].pose.position[1])
    assert route[1].pose.position[2] < route[0].pose.position[2]
    assert route[2].pose.position[0] == pytest.approx(route[1].pose.position[0])
    assert route[2].pose.position[1] > 0.12
    assert route[3].pose.position[1] == pytest.approx(route[2].pose.position[1])
    assert route[3].pose.position[0] > route[2].pose.position[0]
    assert route[4].pose.position[0] == pytest.approx(route[3].pose.position[0])
    assert route[4].pose.position[1] < route[3].pose.position[1]
    assert route[5].pose.position[0] > route[4].pose.position[0]
    assert route[5].pose.position[1] == pytest.approx(route[4].pose.position[1])
    assert route[5].pose.position[2] < route[4].pose.position[2]
    assert candidates[0].approach_standoff_pose.position[0] < candidates[0].pre_insert_pose.position[0]
    assert candidates[0].approach_standoff_pose.position[1] > 0.12
    assert candidates[0].metadata["diagonal_bypass_progress_fraction"] == pytest.approx(0.0)
    assert candidates[0].metadata["route_policy"] == "camera_left_then_outside_lane_moveit_around_present_nic_cards"


def test_sc_to_sc_near_start_uses_short_approach_instead_of_high_bypass():
    snapshot = _sc_snapshot_with_nics()
    snapshot = SceneSnapshot(
        **{
            **snapshot.__dict__,
            "tcp_pose": SerializablePose([-0.37, 0.19, 0.326], [0.0, 0.0, 0.0, 1.0]),
            "target_port_pose": SerializablePose([-0.41, 0.204, 0.319], [0.0, 0.0, 0.0, 1.0]),
            "plug_pose": SerializablePose([-0.37, 0.205, 0.310], [0.0, 0.0, 0.0, 1.0]),
        }
    )
    candidates = generate_approach_candidates(snapshot, _strategy(approach_side="front"), count=1)

    candidate = candidates[0]
    assert candidate.metadata["near_start_short_approach"] is True
    assert candidate.metadata["route_policy"] == "near_start_short_approach"
    assert candidate.safe_lift_pose.position[1] == pytest.approx(snapshot.tcp_pose.position[1])
    assert candidate.approach_standoff_pose.position[1] == pytest.approx(candidate.pre_insert_pose.position[1])
    assert candidate.safe_lift_pose.position[2] < 0.40
    assert candidate.pre_insert_pose.position[2] < 0.38


def test_strategy_prompt_lists_sc_to_sc_nic_obstacles():
    prompt = build_strategy_prompt(_sc_snapshot_with_nics().to_dict(), mode=ExpertMode.NOMINAL)

    assert "SC-to-SC obstacle guidance" in prompt
    assert "trial_nic_card_0" in prompt
    assert "avoid_regions" in prompt
    assert "wide outside lane" in prompt


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


def test_moveit_py_backend_uses_explicit_sc_to_sc_bypass_subgoals():
    snapshot = _sc_snapshot_with_nics()
    strategy = _strategy(approach_side="front")
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

    expected_segments = [subgoal.name for subgoal in candidate.route_subgoals]
    assert fake_moveit.component.goal_count == len(expected_segments)
    assert fake_moveit.component.start_state_count == len(expected_segments) - 1
    assert [segment["segment"] for segment in result.metadata["segments"]] == expected_segments


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


def test_live_generation_resume_counts_existing_accepted(tmp_path):
    writer = DatasetMetadataWriter(tmp_path / "accepted_metadata")
    for episode_index in range(3):
        writer.append_episode(
            ExpertEpisodeMetadata(
                episode_index=episode_index,
                mode="nominal",
                scene_id=f"attempt_{episode_index + 1:06d}",
                candidate_index=0,
                trajectory_path=None,
                validation={"accepted": True},
                vlm_strategy={},
                moveit={},
                phase_labels=[],
                extra={},
            )
        )

    assert generate_expert_trajectories._count_existing_accepted(tmp_path) == 3


def test_live_generation_resume_uses_next_unused_attempt_index(tmp_path):
    for relative in [
        "planner_attempts/attempt_000007_candidate_00",
        "replay_attempts/attempt_000009_candidate_00",
        "rejected_attempts/attempt_000006",
        "rejected_attempts/attempt_000008_repair_00",
    ]:
        (tmp_path / relative).mkdir(parents=True)

    assert generate_expert_trajectories._max_existing_attempt_index(tmp_path) == 9


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


def test_replay_runner_uses_single_trial_config_and_matching_score(monkeypatch, tmp_path):
    root_config = tmp_path / "engine_config.yaml"
    root_config.write_text("trials: {}\n", encoding="utf-8")
    trial_config = tmp_path / "trials" / "trial_000002.yaml"
    trial_config.parent.mkdir()
    trial_config.write_text("trials: {trial_000002: {}}\n", encoding="utf-8")
    captured = {}

    def fake_run(cmd, cwd, env, text, stdout, stderr, check):
        captured["cmd"] = cmd
        score = tmp_path / "attempt_000002_candidate_00" / "results" / "trial_1_trial_000002" / "scoring.yaml"
        score.parent.mkdir(parents=True)
        score.write_text("total: 92\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    class FakeTrajectory:
        def save_json(self, path):
            path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr("subprocess.run", fake_run)
    runner = OfficialRecordingReplayRunner(
        OfficialReplayConfig(
            repo_root=Path("/repo"),
            engine_config=root_config,
            output_dir=tmp_path,
        )
    )

    metrics = runner.replay_and_score(FakeTrajectory(), attempt_index=2, candidate_index=0)

    assert captured["cmd"][captured["cmd"].index("--engine-config") + 1] == str(trial_config)
    assert metrics["engine_config"] == str(trial_config)
    assert metrics["scoring_yaml"].endswith("trial_1_trial_000002/scoring.yaml")


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


def test_expert_planner_runner_can_skip_intermediate_dataset_recording(tmp_path):
    runner = ExpertPlannerRecordingRunner(
        ExpertPlannerRunConfig(
            repo_root=Path("/repo"),
            engine_config=Path("/repo/config.yaml"),
            output_dir=tmp_path,
            expert_mode="nominal",
            record_dataset=False,
        )
    )

    cmd = runner.build_command(attempt_dir=tmp_path / "attempt")

    assert "--record-episode" in cmd
    assert cmd[cmd.index("--record-episode") + 1] == "false"


def test_expert_planner_runner_uses_single_trial_config(monkeypatch, tmp_path):
    root_config = tmp_path / "engine_config.yaml"
    root_config.write_text("trials: {}\n", encoding="utf-8")
    trial_config = tmp_path / "trials" / "trial_000003.yaml"
    trial_config.parent.mkdir()
    trial_config.write_text("trials: {trial_000003: {}}\n", encoding="utf-8")
    captured = {}

    def fake_run(cmd, cwd, text, stdout, stderr, env, check):
        captured["cmd"] = cmd
        captured["env"] = env
        Path(env["AIC_EXPERT_PIECEWISE_OUTPUT"]).write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    runner = ExpertPlannerRecordingRunner(
        ExpertPlannerRunConfig(
            repo_root=Path("/repo"),
            engine_config=root_config,
            output_dir=tmp_path / "planner_attempts",
            expert_mode="nominal",
        )
    )

    result = runner.run_planner(attempt_index=3, candidate_index=0, seed=11)

    assert captured["cmd"][captured["cmd"].index("--engine-config") + 1] == str(trial_config)
    assert captured["env"]["AIC_EXPERT_ENGINE_CONFIG"] == str(trial_config)
    assert result["engine_config"] == str(trial_config)
    assert result["piecewise_exists"] is True


def test_expert_planner_runner_applies_per_trial_registry_env(monkeypatch, tmp_path):
    root_config = tmp_path / "engine_config.yaml"
    root_config.write_text("trials: {}\n", encoding="utf-8")
    trial_config = tmp_path / "trials" / "trial_000001.yaml"
    trial_config.parent.mkdir()
    trial_config.write_text(
        yaml.safe_dump(
            {
                "trials": {
                    "trial_000001": {
                        "scene": {
                            "task_board": {
                                "sc_rail_0": {"entity_present": True},
                                "nic_rail_1": {"entity_present": True},
                            }
                        },
                        "tasks": {
                            "task_1": {
                                "target_module_name": "sc_port_0",
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_run(cmd, cwd, text, stdout, stderr, env, check):
        captured["env"] = env
        Path(env["AIC_EXPERT_PIECEWISE_OUTPUT"]).write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setenv("AIC_EXPERT_TASK_FAMILY", "sc_to_sc")
    monkeypatch.setenv(
        "AIC_EXPERT_REGISTRY_MODE_ENV_BY_SUFFIX",
        json.dumps(
            {
                "matrix_sc2sc_sc1_present0_target0_nic1": {
                    "AIC_OFFICIAL_TEACHER_SC_ENABLE_LIVE_Z_REPAIR": "true",
                }
            }
        ),
    )
    runner = ExpertPlannerRecordingRunner(
        ExpertPlannerRunConfig(
            repo_root=Path("/repo"),
            engine_config=root_config,
            output_dir=tmp_path / "planner_attempts",
            expert_mode="nominal",
        )
    )

    runner.run_planner(attempt_index=1, candidate_index=0, seed=11)

    assert captured["env"]["AIC_OFFICIAL_TEACHER_SC_ENABLE_LIVE_Z_REPAIR"] == "true"


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


def test_near_gate_acceptance_overrides_standard_score_acceptance():
    validation = TrajectoryValidator(
        ValidationCriteria(score_threshold=10.0, require_insertion_event=False)
    ).evaluate(
        {
            "score": 60.0,
            "offlimit_contact_count": 0,
            "moveit_success": True,
            "mode": "nominal",
        }
    )
    args = SimpleNamespace(
        allow_near_gate_acceptance=True,
        near_gate_max_lateral_error_m=0.004,
        near_gate_max_axial_error_m=None,
        near_gate_max_tcp_speed_mps=None,
        near_gate_max_force_delta_n=None,
        near_gate_max_force_n=None,
        score_threshold=10.0,
    )
    assessment = {
        "metrics": {
            "preinsert_tracking_gate_checked": True,
            "preinsert_tracking_gate_passed": False,
            "preinsert_tracking_gate_final_lateral_error_m": 0.005,
            "preinsert_tracking_gate_final_tcp_speed_mps": 0.001,
            "preinsert_tracking_gate_force_delta_n": 1.0,
        }
    }

    updated, metadata = generate_expert_trajectories._maybe_accept_near_gate(
        validation,
        assessment,
        args=args,
    )

    assert validation.accepted is True
    assert updated.accepted is False
    assert "near_gate_acceptance_failed" in updated.reasons
    assert metadata["near_gate_acceptance"]["accepted"] is False
    assert "standard_validation_would_accept" in metadata["near_gate_acceptance"]["original_reasons"]


def test_near_gate_acceptance_requires_axial_error_when_configured():
    validation = TrajectoryValidator(
        ValidationCriteria(score_threshold=90.0, require_insertion_event=False)
    ).evaluate(
        {
            "score": 10.0,
            "offlimit_contact_count": 0,
            "moveit_success": True,
            "mode": "nominal",
        }
    )
    args = SimpleNamespace(
        allow_near_gate_acceptance=True,
        near_gate_max_lateral_error_m=0.004,
        near_gate_max_axial_error_m=0.004,
        near_gate_max_tcp_speed_mps=None,
        near_gate_max_force_delta_n=None,
        near_gate_max_force_n=None,
        score_threshold=90.0,
    )
    assessment = {
        "metrics": {
            "preinsert_tracking_gate_checked": True,
            "preinsert_tracking_gate_passed": True,
            "preinsert_tracking_gate_final_lateral_error_m": 0.003,
            "preinsert_tracking_gate_final_axial_error_m": 0.006,
            "preinsert_tracking_gate_final_tcp_speed_mps": 0.001,
            "preinsert_tracking_gate_force_delta_n": 1.0,
            "offlimit_contact_count": 0,
            "moveit_success": True,
        }
    }

    updated, metadata = generate_expert_trajectories._maybe_accept_near_gate(
        validation,
        assessment,
        args=args,
    )

    assert updated.accepted is False
    assert metadata["near_gate_acceptance"]["checks"]["lateral_error_within_threshold"] is True
    assert metadata["near_gate_acceptance"]["checks"]["axial_error_within_threshold"] is False
    assert "near_gate_acceptance_failed" in updated.reasons


def test_near_gate_acceptance_accepts_when_lateral_and_axial_error_pass():
    validation = TrajectoryValidator(
        ValidationCriteria(score_threshold=90.0, require_insertion_event=False)
    ).evaluate(
        {
            "score": 10.0,
            "offlimit_contact_count": 0,
            "moveit_success": True,
            "mode": "nominal",
        }
    )
    args = SimpleNamespace(
        allow_near_gate_acceptance=True,
        near_gate_max_lateral_error_m=0.004,
        near_gate_max_axial_error_m=0.004,
        near_gate_max_tcp_speed_mps=None,
        near_gate_max_force_delta_n=None,
        near_gate_max_force_n=None,
        score_threshold=90.0,
    )
    assessment = {
        "metrics": {
            "preinsert_tracking_gate_checked": True,
            "preinsert_tracking_gate_passed": True,
            "preinsert_tracking_gate_final_lateral_error_m": 0.003,
            "preinsert_tracking_gate_final_axial_error_m": 0.002,
            "preinsert_tracking_gate_final_tcp_speed_mps": 0.001,
            "preinsert_tracking_gate_force_delta_n": 1.0,
            "offlimit_contact_count": 0,
            "moveit_success": True,
        }
    }

    updated, metadata = generate_expert_trajectories._maybe_accept_near_gate(
        validation,
        assessment,
        args=args,
    )

    assert updated.accepted is True
    assert metadata["score"] == pytest.approx(90.0)
    assert metadata["acceptance_type"] == "near_gate"
    assert metadata["near_gate_acceptance"]["accepted"] is True


def test_near_gate_acceptance_uses_configured_errors_over_legacy_gate_bool():
    validation = TrajectoryValidator(
        ValidationCriteria(score_threshold=90.0, require_insertion_event=False)
    ).evaluate(
        {
            "score": 1.0,
            "max_force_n": 0.0,
            "offlimit_contact_count": 0,
            "moveit_success": True,
            "mode": "nominal",
        }
    )
    args = SimpleNamespace(
        allow_near_gate_acceptance=True,
        near_gate_max_lateral_error_m=0.005,
        near_gate_max_axial_error_m=0.005,
        near_gate_max_tcp_speed_mps=None,
        near_gate_max_force_delta_n=5.0,
        near_gate_max_force_n=60.0,
        score_threshold=10.0,
    )
    assessment = {
        "metrics": {
            "preinsert_tracking_gate_checked": True,
            "preinsert_tracking_gate_passed": False,
            "preinsert_tracking_gate_final_lateral_error_m": 0.0016,
            "preinsert_tracking_gate_final_axial_error_m": 0.0007,
            "preinsert_tracking_gate_final_tcp_speed_mps": 0.008,
            "preinsert_tracking_gate_force_delta_n": 1.7,
            "offlimit_contact_count": 0,
            "moveit_success": True,
        }
    }

    updated, metadata = generate_expert_trajectories._maybe_accept_near_gate(
        validation,
        assessment,
        args=args,
    )

    assert updated.accepted is True
    assert metadata["acceptance_type"] == "near_gate"
    assert metadata["near_gate_acceptance"]["checks"]["preinsert_tracking_gate_passed"] is False
    assert metadata["near_gate_acceptance"]["accepted"] is True


def test_nominal_repair_is_skipped_after_near_gate_acceptance():
    validation = TrajectoryValidator(
        ValidationCriteria(score_threshold=90.0, require_insertion_event=False)
    ).evaluate(
        {
            "score": 10.0,
            "offlimit_contact_count": 0,
            "moveit_success": True,
            "mode": "nominal",
        }
    )
    accepted = validation.__class__(
        **{
            **validation.__dict__,
            "accepted": True,
            "reasons": [],
        }
    )
    assessment = {
        "phase_speed_metrics": {
            "phases": {
                "local_preinsert_align": {"max_speed_mps": 0.1},
                "moveit_approach": {"max_speed_mps": 0.2},
            }
        }
    }

    assert generate_expert_trajectories._should_attempt_nominal_repair(accepted, assessment) is False
