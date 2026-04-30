import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from aic_teacher_official.debug_recorder import (
    DebugRecorder,
    build_failure_analysis_payload,
    build_failure_analysis_prompt,
    validate_image,
    write_bundle,
    write_image_manifest,
)
from aic_teacher_official.context import OfficialTeacherContext
from aic_teacher_official.postprocess import postprocess_piecewise_trajectory
from aic_teacher_official.trajectory import (
    PhaseLabel,
    PiecewiseTrajectory,
    SourceLabel,
    TCPPose,
    TrajectoryWaypoint,
)
from aic_teacher_official.vlm_planner import call_gpt5_mini_delta_planner
from analyze_agent_teleop_failure import run_analysis
from run_agent_teleop_failure_debug import _recording_command, parse_args


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
                phase=PhaseLabel.PRE_INSERTION,
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


def _write_attempt(run_dir: Path, index: int) -> None:
    attempt = run_dir / f"attempt_{index}"
    attempt.mkdir(parents=True)
    piecewise = _piecewise()
    smooth = postprocess_piecewise_trajectory(piecewise, sample_dt=0.1)
    piecewise.save_json(attempt / "piecewise_trajectory.json")
    smooth.save_json(attempt / "smooth_trajectory.json")
    recorder = DebugRecorder(attempt, sample_period=0.5)
    recorder.sample_smooth_trajectory(smooth, piecewise=piecewise)
    recorder.write_trace(metadata={"attempt_index": index})
    write_image_manifest(attempt, validate=True, describe=True, dry_run_descriptions=True)
    (attempt / "planner_prompt.json").write_text(
        json.dumps({"prompt": "planner/smoother/policy/CheatCode context"}) + "\n",
        encoding="utf-8",
    )
    (attempt / "planner_response.json").write_text(
        json.dumps({"waypoints": []}) + "\n",
        encoding="utf-8",
    )
    (attempt / "command_result.json").write_text(
        json.dumps({"exit_code": 0}) + "\n",
        encoding="utf-8",
    )


def test_bundle_represents_three_attempts(tmp_path):
    for index in range(1, 4):
        _write_attempt(tmp_path, index)
    write_bundle(tmp_path)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "bundle_manifest.json").read_text(encoding="utf-8"))
    payload = build_failure_analysis_payload(tmp_path)

    assert summary["attempt_count"] == 3
    assert payload["attempt_count"] == 3
    assert any("attempt_3/trace.json" == item["relative_path"] for item in manifest["files"])


def test_recorder_samples_every_half_second(tmp_path):
    piecewise = _piecewise()
    smooth = postprocess_piecewise_trajectory(piecewise, sample_dt=0.1)
    recorder = DebugRecorder(tmp_path, sample_period=0.5)
    recorder.sample_smooth_trajectory(smooth, piecewise=piecewise)
    trace_path = recorder.write_trace()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))

    timestamps = [sample["timestamp"] for sample in trace["samples"]]
    assert trace["schema_version"] == "agent_teleop_failure_trace/v1"
    assert trace["sample_period"] == pytest.approx(0.5)
    assert timestamps[:5] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])


def test_trace_schema_contains_stable_required_fields(tmp_path):
    piecewise = _piecewise()
    smooth = postprocess_piecewise_trajectory(piecewise, sample_dt=0.1)
    recorder = DebugRecorder(tmp_path, sample_period=0.5)
    recorder.sample_smooth_trajectory(smooth, piecewise=piecewise)
    trace = json.loads(recorder.write_trace().read_text(encoding="utf-8"))
    sample = trace["samples"][0]

    for key in [
        "timestamp",
        "sim_time",
        "pipeline_phase",
        "robot_tcp_pose",
        "commanded_target_pose",
        "commanded_actual_delta",
        "observation",
        "camera_metadata",
        "policy_command",
        "cheatcode_command",
        "errors_warnings_exceptions",
    ]:
        assert key in sample


def test_image_manifest_blank_detection_and_description_storage(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    image_path = tmp_path / "blank.png"
    Image.new("RGB", (8, 8), (0, 0, 0)).save(image_path)

    validation = validate_image(image_path)
    manifest = write_image_manifest(
        tmp_path,
        validate=True,
        describe=True,
        dry_run_descriptions=True,
    )

    assert validation["valid"] is True
    assert validation["near_constant"] is True
    assert validation["blank_reason"] == "all_black_or_zero"
    assert manifest["image_count"] == 1
    assert manifest["images"][0]["description"]["reason"] == "invalid_or_blank_image_not_sent_to_model"


def test_gpt5_prompt_includes_required_failure_questions(tmp_path):
    for index in range(1, 4):
        _write_attempt(tmp_path, index)
    payload = build_failure_analysis_payload(tmp_path)
    prompt = build_failure_analysis_prompt(payload)

    assert "Compare all 3 attempts" in prompt
    assert "GPT-5-mini is capable" in prompt
    assert "MoveIt, cuRobo, FCL" in prompt
    assert "planner" in prompt
    assert "smoother" in prompt
    assert "policy compatible with aic_model" in prompt
    assert "CheatCode.py" in prompt


def test_dry_run_analysis_works_without_openai_api_key(monkeypatch, tmp_path):
    for index in range(1, 4):
        _write_attempt(tmp_path, index)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    paths = run_analysis(run_dir=tmp_path, bundle=None, dry_run=True)

    assert paths["analysis"].exists()
    assert paths["analysis_json"].exists()
    result = json.loads(paths["analysis_json"].read_text(encoding="utf-8"))
    assert result["dry_run"] is True
    assert result["attempt_count"] == 3


def test_runner_command_uses_live_official_recording_policy(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_agent_teleop_failure_debug.py",
            "--output-dir",
            str(tmp_path),
            "--dry-run-image-descriptions",
        ],
    )
    args = parse_args()

    cmd = _recording_command(
        args,
        attempt_dir=tmp_path / "attempt_1",
        policy_class="aic_teacher_official.OfficialTeacherOraclePlanner",
        dataset_root=tmp_path / "dataset",
        results_root=tmp_path / "results",
        tmp_dir=tmp_path / "tmp",
    )

    assert "aic_teacher_official.OfficialTeacherOraclePlanner" in cmd
    assert "--policy-class" in cmd
    assert "--dry-run" not in cmd
    assert "--run" not in cmd
    assert "--image" not in cmd


def test_vlm_planner_requires_images_before_model_call(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    context = OfficialTeacherContext(
        start_position=[0.0, 0.0, 0.3],
        port_position=[0.1, 0.0, 0.1],
        orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
    )

    with pytest.raises(RuntimeError, match="requires at least one validated scene image"):
        call_gpt5_mini_delta_planner(context, image_paths=[])
