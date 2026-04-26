import json
import importlib
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic_teacher_official.generate_piecewise import (
    PiecewiseGeneratorConfig,
    generate_piecewise_file,
    generate_piecewise_trajectory,
)
from aic_teacher_official.context import OfficialTeacherContext
from aic_teacher_official.postprocess import (
    minimum_jerk_fraction,
    postprocess_piecewise_trajectory,
)
from aic_teacher_official.review import (
    _compact_manifest_for_gpt,
    build_comparison_review_bundle,
    build_review_bundle,
)
from aic_teacher_official.replay import SmoothTrajectoryReplayPolicy
from aic_teacher_official.iteration import (
    build_recording_command,
    loop_roots,
    parse_total_score,
    run_name_for_loop,
)
from aic_teacher_official.trajectory import (
    PhaseLabel,
    PiecewiseTrajectory,
    SourceLabel,
    TCPPose,
    TrajectoryWaypoint,
    assert_monotonic_timestamps,
)


def _piecewise() -> PiecewiseTrajectory:
    return PiecewiseTrajectory(
        waypoints=[
            TrajectoryWaypoint(
                timestamp=0.0,
                tcp_pose=TCPPose([0.0, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0]),
                phase=PhaseLabel.APPROACH,
                source=SourceLabel.VLM,
            ),
            TrajectoryWaypoint(
                timestamp=1.0,
                tcp_pose=TCPPose([0.1, 0.0, 0.2], [0.0, 0.0, 0.0, 1.0]),
                phase=PhaseLabel.ALIGNMENT,
                source=SourceLabel.OPTIMIZER,
            ),
            TrajectoryWaypoint(
                timestamp=2.0,
                tcp_pose=TCPPose([0.1, 0.0, 0.1], [0.0, 0.0, 0.0, 1.0]),
                phase=PhaseLabel.FINAL_INSERTION,
                source=SourceLabel.CHEATCODE,
            ),
        ]
    )


def test_piecewise_trajectory_loading(tmp_path):
    path = tmp_path / "piecewise.json"
    path.write_text(
        """
{
  "metadata": {"schema_version": "official_teacher_trajectory/v0"},
  "waypoints": [
    {
      "timestamp": 0.0,
      "tcp_pose": {"position": [0, 0, 0.3], "orientation_xyzw": [0, 0, 0, 1]},
      "phase": "approach",
      "source": "vlm"
    },
    {
      "timestamp": 1.0,
      "tcp_pose": {"position": [0.1, 0, 0.2], "orientation_xyzw": [0, 0, 0, 1]},
      "phase": "final_insertion",
      "source": "cheatcode"
    }
  ]
}
""",
        encoding="utf-8",
    )

    trajectory = PiecewiseTrajectory.load_json(path)

    assert len(trajectory.waypoints) == 2
    assert trajectory.waypoints[0].phase == PhaseLabel.APPROACH
    assert trajectory.waypoints[1].source == SourceLabel.CHEATCODE


def test_minimum_jerk_interpolation_endpoints_and_midpoint():
    assert minimum_jerk_fraction(0.0) == pytest.approx(0.0)
    assert minimum_jerk_fraction(1.0) == pytest.approx(1.0)
    assert minimum_jerk_fraction(0.5) == pytest.approx(0.5)


def test_smooth_trajectory_has_no_timestamp_regressions():
    smooth = postprocess_piecewise_trajectory(_piecewise(), sample_dt=0.2)

    assert_monotonic_timestamps(smooth.waypoints)
    assert smooth.waypoints[0].timestamp == pytest.approx(0.0)
    assert smooth.waypoints[-1].timestamp == pytest.approx(2.0)


def test_smooth_trajectory_preserves_phase_labels():
    piecewise = _piecewise()
    smooth = postprocess_piecewise_trajectory(piecewise, sample_dt=0.2)
    phases = {waypoint.phase for waypoint in smooth.waypoints}

    assert {PhaseLabel.APPROACH, PhaseLabel.ALIGNMENT, PhaseLabel.FINAL_INSERTION} <= phases


def test_final_insertion_phase_is_marked_cheatcode_derived():
    smooth = postprocess_piecewise_trajectory(_piecewise(), sample_dt=0.2)
    insertion_waypoints = [
        w for w in smooth.waypoints if w.phase == PhaseLabel.FINAL_INSERTION
    ]

    assert insertion_waypoints
    assert all(w.source == SourceLabel.CHEATCODE for w in insertion_waypoints)
    assert all(w.diagnostics["cheatcode_derived"] for w in insertion_waypoints)


def test_replay_policy_imports_no_vlm_backend():
    before = set(sys.modules)
    module = importlib.import_module("aic_teacher_official.replay")
    loaded = set(sys.modules) - before

    assert hasattr(module, "SmoothTrajectoryReplayPolicy")
    assert not any("openai" in name.lower() or "vlm" in name.lower() for name in loaded)


def test_replay_policy_samples_without_vlm_calls():
    smooth = postprocess_piecewise_trajectory(_piecewise(), sample_dt=0.2)
    replay = SmoothTrajectoryReplayPolicy(smooth)

    target = replay.sample(0.5)

    assert 0.0 < target.tcp_pose.position[0] < 0.1
    assert target.waypoint.phase in {PhaseLabel.ALIGNMENT, PhaseLabel.APPROACH}


def test_piecewise_generator_emits_valid_cheatcode_insertion(tmp_path):
    output = tmp_path / "piecewise.json"
    trajectory = generate_piecewise_file(
        PiecewiseGeneratorConfig(
            start_position=[-0.35, 0.35, 0.32],
            port_position=[-0.10, 0.45, 0.12],
            orientation_xyzw=[1.0, 0.0, 0.0, 0.0],
            approach_offset=[-0.08, -0.08, 0.22],
        ),
        output,
    )
    loaded = PiecewiseTrajectory.load_json(output)

    assert len(trajectory.waypoints) == len(loaded.waypoints)
    assert loaded.waypoints[0].source == SourceLabel.PLACEHOLDER_VLM
    assert loaded.waypoints[-2].phase == PhaseLabel.PRE_INSERTION
    assert loaded.waypoints[-1].source == SourceLabel.CHEATCODE
    assert loaded.waypoints[-1].diagnostics["vlm_used"] is False
    assert loaded.waypoints[-1].timestamp - loaded.waypoints[-2].timestamp == pytest.approx(12.0)


def test_postprocessor_accepts_generated_piecewise_output():
    piecewise = generate_piecewise_trajectory(
        PiecewiseGeneratorConfig(
            start_position=[-0.35, 0.35, 0.32],
            port_position=[-0.10, 0.45, 0.12],
            orientation_xyzw=[1.0, 0.0, 0.0, 0.0],
            approach_offset=[-0.08, -0.08, 0.22],
        )
    )
    smooth = postprocess_piecewise_trajectory(piecewise, sample_dt=0.2)

    assert_monotonic_timestamps(smooth.waypoints)
    assert any(w.phase == PhaseLabel.PRE_INSERTION for w in smooth.waypoints)
    assert all(
        w.source == SourceLabel.CHEATCODE
        for w in smooth.waypoints
        if w.phase == PhaseLabel.FINAL_INSERTION
    )


def test_replay_policy_loads_trajectory_from_env(monkeypatch, tmp_path):
    smooth_path = tmp_path / "smooth.json"
    smooth = postprocess_piecewise_trajectory(_piecewise(), sample_dt=0.2)
    smooth.save_json(smooth_path)
    monkeypatch.setenv("AIC_OFFICIAL_TEACHER_TRAJECTORY", str(smooth_path))

    class Logger:
        def info(self, _msg):
            pass

    class Parent:
        def get_logger(self):
            return Logger()

    from aic_teacher_official.OfficialTeacherReplay import OfficialTeacherReplay

    policy = OfficialTeacherReplay(Parent())

    assert policy._trajectory_path == str(smooth_path)


def test_replay_action_shape_is_geometry_pose():
    from geometry_msgs.msg import Pose

    from aic_teacher_official.OfficialTeacherReplay import OfficialTeacherReplay

    smooth = postprocess_piecewise_trajectory(_piecewise(), sample_dt=0.2)
    target = SmoothTrajectoryReplayPolicy(smooth).sample(0.1)
    pose = OfficialTeacherReplay.target_to_pose(target)

    assert isinstance(pose, Pose)
    assert pose.position.x == pytest.approx(target.tcp_pose.position[0])
    assert pose.orientation.w == pytest.approx(target.tcp_pose.orientation_xyzw[3])


def test_replay_delta_action_shape_is_relative_pose():
    from geometry_msgs.msg import Point, Pose, Quaternion

    from aic_teacher_official.OfficialTeacherReplay import OfficialTeacherReplay

    smooth = postprocess_piecewise_trajectory(_piecewise(), sample_dt=0.2)
    target = SmoothTrajectoryReplayPolicy(smooth).sample(0.1)
    current = Pose(
        position=Point(x=0.0, y=0.0, z=0.3),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    delta = OfficialTeacherReplay.target_to_delta_pose(target, current)

    assert isinstance(delta, Pose)
    assert delta.position.z < 0.0


def test_vlm_delta_plan_generates_vlm_waypoints_and_cheatcode_insertion():
    context = OfficialTeacherContext(
        start_position=[0.0, 0.0, 0.3],
        port_position=[0.1, 0.0, 0.1],
        orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
    )
    piecewise = generate_piecewise_trajectory(
        PiecewiseGeneratorConfig(
            start_position=[-0.35, 0.35, 0.32],
            port_position=[-0.10, 0.45, 0.12],
            orientation_xyzw=[1.0, 0.0, 0.0, 0.0],
            approach_offset=[-0.08, -0.08, 0.22],
            context=context,
            vlm_delta_plan={
                "waypoints": [
                    {
                        "phase": "approach",
                        "delta_xyz": [0.02, 0.0, 0.05],
                        "duration": 1.0,
                        "rationale": "clearance",
                    },
                    {
                        "phase": "alignment",
                        "delta_xyz": [0.04, 0.0, -0.03],
                        "duration": 1.0,
                        "rationale": "align",
                    },
                ]
            },
        )
    )

    assert any(w.source == SourceLabel.VLM for w in piecewise.waypoints)
    assert piecewise.waypoints[-1].source == SourceLabel.CHEATCODE
    assert piecewise.metadata.planning["vlm_delta_plan"] is not None


def test_orchestration_cli_dry_run(tmp_path):
    script = Path("scripts/official_teacher_build_and_replay.py")
    piecewise = tmp_path / "piecewise.json"
    smooth = tmp_path / "smooth.json"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--piecewise-output",
            str(piecewise),
            "--smooth-output",
            str(smooth),
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert piecewise.exists()
    assert smooth.exists()
    assert "--teacher-trajectory" in result.stdout
    assert "--teacher-action-mode" in result.stdout
    assert "aic_teacher_official.OfficialTeacherReplay" in result.stdout


def test_orchestration_cli_dataset_layout(tmp_path):
    script = Path("scripts/official_teacher_build_and_replay.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--use-dataset-layout",
            "--root-dir",
            str(tmp_path),
            "--timestamp",
            "2026_0425_205620",
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "vlm_planner/nic_cards_2/n1/trial9_2026_0425_205620" in result.stdout
    assert "vlm_planner_postprocessed/nic_cards_2/n1/trial9_2026_0425_205620" in result.stdout


def test_review_bundle_generation_missing_images_gracefully(tmp_path):
    smooth_path = tmp_path / "smooth.json"
    review_path = tmp_path / "review.json"
    smooth = postprocess_piecewise_trajectory(_piecewise(), sample_dt=0.2)
    smooth.save_json(smooth_path)

    manifest = build_review_bundle(
        smooth_path,
        review_path,
        wrist_image_dir=tmp_path / "missing_wrist",
        gazebo_image_dir=tmp_path / "missing_gazebo",
        samples=4,
    )
    loaded = json.loads(review_path.read_text(encoding="utf-8"))

    assert manifest["images"]["missing_wrist_images"] is True
    assert manifest["images"]["missing_gazebo_images"] is True
    assert loaded["critique"]["api_called"] is False


def test_comparison_review_bundle_includes_multiple_loop_contexts(tmp_path):
    smooth = postprocess_piecewise_trajectory(_piecewise(), sample_dt=0.2)
    loop1_smooth = tmp_path / "loop1_smooth.json"
    loop2_smooth = tmp_path / "loop2_smooth.json"
    smooth.save_json(loop1_smooth)
    smooth.save_json(loop2_smooth)
    loop1_score = tmp_path / "loop1_score.yaml"
    loop2_score = tmp_path / "loop2_score.yaml"
    loop1_score.write_text("total: 97.0\n", encoding="utf-8")
    loop2_score.write_text("total: 44.0\n", encoding="utf-8")

    manifest = build_comparison_review_bundle(
        [
            {
                "label": "loop_1",
                "trajectory_path": loop1_smooth,
                "scoring_path": loop1_score,
            },
            {
                "label": "loop_2",
                "trajectory_path": loop2_smooth,
                "scoring_path": loop2_score,
            },
        ],
        tmp_path / "comparison_review.json",
        samples=10,
    )

    assert [run["label"] for run in manifest["runs"]] == ["loop_1", "loop_2"]
    assert manifest["runs"][0]["score_total"] == pytest.approx(97.0)
    assert manifest["runs"][1]["score_total"] == pytest.approx(44.0)
    assert manifest["samples_per_run"] == 10
    assert manifest["runs"][0]["samples"][0]["planned"]["tcp_pose"]


def test_review_bundle_includes_container_scoring_messages(tmp_path):
    smooth_path = tmp_path / "smooth.json"
    smooth = postprocess_piecewise_trajectory(_piecewise(), sample_dt=0.2)
    smooth.save_json(smooth_path)
    scoring_path = tmp_path / "scores" / "trial_1_trial_000001" / "scoring.yaml"
    scoring_path.parent.mkdir(parents=True)
    scoring_path.write_text(
        "total: 58.0\ntrial_000001:\n  tier_3:\n    message: Partial insertion detected with distance of 0.05m.\n",
        encoding="utf-8",
    )
    log_dir = tmp_path / "logs" / "per_trial_tmp"
    log_dir.mkdir(parents=True)
    (log_dir / "trial_1_trial_000001_simulation.log").write_text(
        "scoring total: 58.0\nresult: OK\n",
        encoding="utf-8",
    )

    manifest = build_review_bundle(
        smooth_path,
        tmp_path / "review.json",
        scoring_path=scoring_path,
        samples=4,
    )

    context = manifest["container_scoring_context"]
    assert context["available"] is True
    assert "Partial insertion detected" in context["scoring_yaml_text"]
    assert context["log_excerpts"][0]["matching_lines"] == ["scoring total: 58.0", "result: OK"]


def test_gpt_review_payload_is_compacted(tmp_path):
    smooth_path = tmp_path / "smooth.json"
    smooth = postprocess_piecewise_trajectory(_piecewise(), sample_dt=0.2)
    smooth.save_json(smooth_path)
    scoring_path = tmp_path / "scores" / "trial_1_trial_000001" / "scoring.yaml"
    scoring_path.parent.mkdir(parents=True)
    scoring_path.write_text("total: 97.0\n", encoding="utf-8")

    manifest = build_review_bundle(
        smooth_path,
        tmp_path / "review.json",
        scoring_path=scoring_path,
        samples=4,
    )
    compact = _compact_manifest_for_gpt(manifest)

    assert "recorded_dataset" not in compact
    assert "trajectory_metadata" not in compact
    assert compact["score"]["total"] == pytest.approx(97.0)


def test_iteration_loop_names_and_score_parse(tmp_path):
    roots = loop_roots(
        root_dir=tmp_path,
        task_family="sfp_to_nic",
        scene_count_label="nic_cards_2",
        attempt_label="n1",
        base_run_name="trial9_2026_0425_205620",
        loop_index=2,
    )
    scoring = roots.scoring_path
    scoring.parent.mkdir(parents=True)
    scoring.write_text("total: 97.5\n", encoding="utf-8")

    assert run_name_for_loop("trial9_2026_0425_205620", 1) == "trial9_2026_0425_205620_loop_1"
    assert run_name_for_loop("trial9_2026_0425_205620", 2) == "trial9_2026_0425_205620_loop_2"
    assert "vlm_planner_postprocessed" in str(roots.postprocessed_root)
    assert parse_total_score(scoring) == pytest.approx(97.5)


def test_iteration_recording_command_uses_stable_waits(tmp_path):
    cmd = build_recording_command(
        engine_config="engine.yaml",
        sim_distrobox="aic_eval",
        smooth_path=tmp_path / "smooth.json",
        dataset_repo_id="local/test",
        dataset_root=tmp_path / "raw_dataset",
        scores_root=tmp_path / "scores",
        tmp_dir=tmp_path / "tmp",
    )

    assert "--startup-delay-sec" in cmd
    assert cmd[cmd.index("--startup-delay-sec") + 1] == "8"
    assert "--per-trial-timeout-sec" in cmd
    assert cmd[cmd.index("--per-trial-timeout-sec") + 1] == "0"


def test_iteration_cli_dry_run_writes_loop_dirs(tmp_path):
    engine_config = tmp_path / "engine.yaml"
    engine_config.write_text("trials: {trial_000001: {}}\n", encoding="utf-8")
    script = Path("scripts/official_teacher_iterate.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root-dir",
            str(tmp_path),
            "--base-run-name",
            "trial9_2026_0425_205620",
            "--engine-config",
            str(engine_config),
            "--max-loops",
            "2",
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "trial9_2026_0425_205620_loop_1" in result.stdout
    assert "trial9_2026_0425_205620_loop_2" in result.stdout
    assert (
        tmp_path
        / "sfp_to_nic"
        / "vlm_planner_postprocessed"
        / "nic_cards_2"
        / "n1"
        / "trial9_2026_0425_205620_loop_2"
        / "smooth_trajectory.json"
    ).exists()
