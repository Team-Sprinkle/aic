"""Debug trace utilities for agentic official-teacher failure analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import base64
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any
import zipfile

import numpy as np

from aic_teacher_official.replay import SmoothTrajectoryReplayPolicy
from aic_teacher_official.trajectory import PiecewiseTrajectory, SmoothTrajectory, TrajectoryWaypoint
from aic_teacher_official.vlm_planner import _extract_output_text, load_openai_api_key


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TRACE_SCHEMA_VERSION = "agent_teleop_failure_trace/v1"
BUNDLE_SCHEMA_VERSION = "agent_teleop_failure_bundle/v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def unavailable(reason: str) -> dict[str, Any]:
    return {"value": None, "available": False, "reason": reason}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def read_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def git_metadata(repo_root: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                text=True,
                capture_output=True,
            )
        except Exception:
            return None
        return result.stdout.strip()

    status = run_git(["status", "--short"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "dirty": bool(status),
        "status_short": status,
    }


def environment_metadata(repo_root: Path) -> dict[str, Any]:
    return {
        "created_at": utc_now_iso(),
        "cwd": str(repo_root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": git_metadata(repo_root),
    }


def observation_summary(observation: Any) -> dict[str, Any]:
    if observation is None:
        return {"available": False, "reason": "observation_not_provided", "keys": [], "shapes": {}}
    if isinstance(observation, dict):
        keys = sorted(str(key) for key in observation.keys())
        shapes: dict[str, Any] = {}
        for key, value in observation.items():
            shape = getattr(value, "shape", None)
            if shape is not None:
                shapes[str(key)] = list(shape)
            elif isinstance(value, (list, tuple)):
                shapes[str(key)] = [len(value)]
            else:
                shapes[str(key)] = None
        return {"available": True, "keys": keys, "shapes": shapes}
    fields = [name for name in dir(observation) if not name.startswith("_")]
    return {"available": True, "keys": fields, "shapes": {}}


def _pose_delta(commanded: dict[str, Any] | None, actual: dict[str, Any] | None) -> dict[str, Any]:
    if not commanded or not actual:
        return unavailable("commanded_or_actual_pose_unavailable")
    try:
        c = np.asarray(commanded["position"], dtype=np.float64)
        a = np.asarray(actual["position"], dtype=np.float64)
    except Exception:
        return unavailable("pose_position_unavailable")
    delta = c - a
    return {
        "value": {
            "position_delta": delta.tolist(),
            "position_delta_norm": float(np.linalg.norm(delta)),
        },
        "available": True,
    }


@dataclass
class DebugSample:
    timestamp: float
    sim_time: float | None
    wall_time: float
    pipeline_phase: str
    robot_tcp_pose: dict[str, Any] | None = None
    robot_tcp_velocity: list[float] | None = None
    joint_positions: list[float] | None = None
    joint_velocities: list[float] | None = None
    commanded_target_pose: dict[str, Any] | None = None
    commanded_action: dict[str, Any] | None = None
    actual_reached_pose: dict[str, Any] | None = None
    gripper_state: dict[str, Any] | None = None
    cable_state: dict[str, Any] | None = None
    wrench_force_torque: dict[str, Any] | None = None
    target_port_pose: dict[str, Any] | None = None
    scene_object_poses: dict[str, Any] | None = None
    transform_frames: dict[str, Any] = field(default_factory=dict)
    observation: dict[str, Any] = field(default_factory=dict)
    camera_metadata: dict[str, Any] = field(default_factory=dict)
    local_planner_input: dict[str, Any] | None = None
    local_planner_output: dict[str, Any] | None = None
    parsed_planner_decision: dict[str, Any] | None = None
    smoothing_input_waypoints: list[dict[str, Any]] | None = None
    smoothing_output_waypoints: list[dict[str, Any]] | None = None
    global_smoothed_trajectory_point: dict[str, Any] | None = None
    policy_command: dict[str, Any] | None = None
    cheatcode_command: dict[str, Any] | None = None
    final_score_reward: dict[str, Any] | None = None
    errors_warnings_exceptions: list[str] = field(default_factory=list)
    unavailable_fields: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        commanded_pose = self.commanded_target_pose
        actual_pose = self.actual_reached_pose or self.robot_tcp_pose
        return {
            "timestamp": self.timestamp,
            "sim_time": self.sim_time,
            "wall_time": self.wall_time,
            "pipeline_phase": self.pipeline_phase,
            "robot_tcp_pose": self.robot_tcp_pose,
            "robot_tcp_velocity": self.robot_tcp_velocity,
            "joint_positions": self.joint_positions,
            "joint_velocities": self.joint_velocities,
            "commanded_target_pose": commanded_pose,
            "commanded_action": self.commanded_action,
            "actual_reached_pose": self.actual_reached_pose,
            "commanded_actual_delta": _pose_delta(commanded_pose, actual_pose),
            "gripper_state": self.gripper_state,
            "cable_state": self.cable_state,
            "wrench_force_torque": self.wrench_force_torque,
            "target_port_pose": self.target_port_pose,
            "scene_object_poses": self.scene_object_poses,
            "transform_frames": self.transform_frames,
            "observation": self.observation,
            "camera_metadata": self.camera_metadata,
            "local_planner_input": self.local_planner_input,
            "local_planner_output": self.local_planner_output,
            "parsed_planner_decision": self.parsed_planner_decision,
            "smoothing_input_waypoints": self.smoothing_input_waypoints,
            "smoothing_output_waypoints": self.smoothing_output_waypoints,
            "global_smoothed_trajectory_point": self.global_smoothed_trajectory_point,
            "policy_command": self.policy_command,
            "cheatcode_command": self.cheatcode_command,
            "final_score_reward": self.final_score_reward,
            "errors_warnings_exceptions": self.errors_warnings_exceptions,
            "unavailable_fields": self.unavailable_fields,
        }


class DebugRecorder:
    """Collects periodic trace samples without requiring ROS/Gazebo."""

    def __init__(self, output_dir: str | Path, *, sample_period: float = 0.5):
        if sample_period <= 0.0:
            raise ValueError("sample_period must be positive")
        self.output_dir = Path(output_dir)
        self.sample_period = float(sample_period)
        self.samples: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.started_wall_time = time.time()

    def record_event(self, phase: str, payload: dict[str, Any]) -> None:
        self.events.append({"wall_time": time.time(), "phase": phase, "payload": payload})

    def record_sample(self, sample: DebugSample) -> None:
        self.samples.append(sample.to_dict())

    def sample_smooth_trajectory(
        self,
        smooth: SmoothTrajectory,
        *,
        piecewise: PiecewiseTrajectory | None = None,
        planner_prompt: dict[str, Any] | None = None,
        planner_response: dict[str, Any] | None = None,
        parsed_decision: dict[str, Any] | None = None,
        final_score_reward: dict[str, Any] | None = None,
    ) -> None:
        replay = SmoothTrajectoryReplayPolicy(smooth)
        duration = max(0.0, replay.end_time - replay.start_time)
        query_times = np.arange(0.0, duration + self.sample_period * 0.5, self.sample_period)
        smoothing_input = [w.to_dict() for w in piecewise.waypoints] if piecewise is not None else None
        smoothing_output = [w.to_dict() for w in smooth.waypoints]
        for elapsed in query_times:
            target = replay.sample(float(elapsed))
            pose = target.tcp_pose.to_dict()
            phase = str(target.waypoint.phase.value)
            self.record_sample(
                DebugSample(
                    timestamp=float(elapsed),
                    sim_time=float(elapsed),
                    wall_time=self.started_wall_time + float(elapsed),
                    pipeline_phase=phase,
                    robot_tcp_pose=None,
                    robot_tcp_velocity=None,
                    joint_positions=None,
                    joint_velocities=None,
                    commanded_target_pose=pose,
                    commanded_action={
                        "action_mode": smooth.metadata.diagnostics.get(
                            "action_mode_assumption",
                            "relative_delta_gripper_tcp",
                        ),
                        "source": "smooth_trajectory_replay_target",
                    },
                    actual_reached_pose=None,
                    gripper_state=target.waypoint.gripper_state,
                    cable_state=target.waypoint.cable_state,
                    target_port_pose=None,
                    transform_frames={
                        "commanded_target_pose": "base_link",
                        "policy_delta_command": "gripper/tcp",
                        "actual_reached_pose": None,
                    },
                    observation=observation_summary(None),
                    camera_metadata={"available": False, "reason": "dry_run_no_camera_stream"},
                    local_planner_input=planner_prompt,
                    local_planner_output=planner_response,
                    parsed_planner_decision=parsed_decision,
                    smoothing_input_waypoints=smoothing_input,
                    smoothing_output_waypoints=smoothing_output,
                    global_smoothed_trajectory_point=target.waypoint.to_dict(),
                    policy_command={
                        "target_pose": pose,
                        "target_velocity": target.tcp_velocity,
                        "frame_id": "base_link",
                    },
                    cheatcode_command=(
                        {
                            "handoff": True,
                            "phase": phase,
                            "command_source": "online_cheatcode_or_cheatcode_derived_segment",
                            "target_pose": pose,
                        }
                        if phase == "final_insertion"
                        else None
                    ),
                    final_score_reward=final_score_reward,
                    unavailable_fields={
                        "robot_tcp_pose": "dry_run_no_ros_execution",
                        "robot_tcp_velocity": "dry_run_no_ros_execution",
                        "joint_positions": "dry_run_no_ros_execution",
                        "joint_velocities": "dry_run_no_ros_execution",
                        "actual_reached_pose": "dry_run_no_controller_feedback",
                        "wrench_force_torque": "dry_run_no_force_sensor_stream",
                        "target_port_pose": "privileged_pose_not_available_without_debug_context",
                        "scene_object_poses": "dry_run_no_tf_stream",
                    },
                )
            )

    def trace_payload(self, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "sample_period": self.sample_period,
            "metadata": metadata or {},
            "events": self.events,
            "samples": self.samples,
        }

    def write_trace(self, *, metadata: dict[str, Any] | None = None) -> Path:
        path = self.output_dir / "trace.json"
        write_json(path, self.trace_payload(metadata=metadata))
        return path


def _image_stats_with_pillow(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float64)
    return {
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "channels": int(arr.shape[2]),
        "mean": float(arr.mean()),
        "stddev": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "unique_sample_count": int(len(np.unique(arr.reshape(-1, arr.shape[2]), axis=0))),
    }


def validate_image(path: str | Path) -> dict[str, Any]:
    image_path = Path(path)
    result: dict[str, Any] = {
        "path": str(image_path),
        "exists": image_path.exists(),
        "valid": False,
        "non_empty": False,
        "near_constant": None,
        "blank_reason": None,
        "error": None,
    }
    if not image_path.exists():
        result["error"] = "image_missing"
        return result
    size = image_path.stat().st_size
    result["bytes"] = int(size)
    result["non_empty"] = size > 0
    if size <= 0:
        result["error"] = "image_empty_file"
        result["blank_reason"] = "empty_file"
        return result
    try:
        stats = _image_stats_with_pillow(image_path)
    except Exception as ex:
        result["error"] = f"image_decode_failed: {ex}"
        return result
    result.update(stats)
    result["valid"] = True
    near_constant = stats["stddev"] < 1.0 or stats["max"] - stats["min"] < 3.0
    result["near_constant"] = bool(near_constant)
    if near_constant:
        if stats["max"] <= 3.0:
            result["blank_reason"] = "all_black_or_zero"
        elif stats["min"] >= 252.0:
            result["blank_reason"] = "all_white"
        else:
            result["blank_reason"] = "near_constant"
    return result


def _encode_image_data_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix or "png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


def describe_image_with_model(
    path: str | Path,
    *,
    model: str = "gpt-5-mini",
    dry_run: bool = False,
    max_size_bytes: int = 5_000_000,
) -> dict[str, Any]:
    image_path = Path(path)
    if dry_run:
        return {"available": False, "reason": "dry_run_description_requested", "model": model}
    api_key = load_openai_api_key()
    if not api_key:
        return {"available": False, "reason": "OPENAI_API_KEY_not_available", "model": model}
    if image_path.stat().st_size > max_size_bytes:
        return {"available": False, "reason": "image_too_large_for_description", "model": model}
    try:
        from openai import OpenAI
    except Exception as ex:
        return {"available": False, "reason": f"openai_package_unavailable: {ex}", "model": model}
    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            instructions=(
                "Briefly describe what is visible in this robotics observation image. "
                "Mention whether the robot, cable/plug, task board, and target port are visible."
            ),
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this AIC teleop debug image."},
                        {
                            "type": "input_image",
                            "image_url": _encode_image_data_url(image_path),
                            "detail": "low",
                        },
                    ],
                }
            ],
        )
        return {"available": True, "model": model, "description": _extract_output_text(response)}
    except Exception as ex:
        return {"available": False, "reason": f"image_description_failed: {ex}", "model": model}


def find_images(root: str | Path, *, max_images: int | None = None) -> list[Path]:
    root_path = Path(root)
    images = sorted(path for path in root_path.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if max_images is not None:
        return images[: max(0, max_images)]
    return images


def write_image_manifest(
    root: str | Path,
    *,
    validate: bool = True,
    describe: bool = False,
    dry_run_descriptions: bool = False,
    max_images: int | None = None,
    model: str = "gpt-5-mini",
) -> dict[str, Any]:
    root_path = Path(root)
    entries = []
    for path in find_images(root_path, max_images=max_images):
        validation = validate_image(path) if validate else {"path": str(path), "valid": None}
        description = None
        if describe:
            if validation.get("valid") and not validation.get("near_constant"):
                description = describe_image_with_model(
                    path,
                    model=model,
                    dry_run=dry_run_descriptions,
                )
            else:
                description = {
                    "available": False,
                    "reason": "invalid_or_blank_image_not_sent_to_model",
                    "model": model,
                }
        entries.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(root_path)),
                "validation": validation,
                "description": description,
            }
        )
    manifest = {
        "schema_version": "agent_teleop_image_manifest/v1",
        "root": str(root_path),
        "image_count": len(entries),
        "images": entries,
    }
    write_json(root_path / "image_manifest.json", manifest)
    return manifest


def load_attempt(attempt_dir: str | Path) -> dict[str, Any]:
    attempt = Path(attempt_dir)
    piecewise = read_json_if_exists(attempt / "piecewise_trajectory.json")
    smooth = read_json_if_exists(attempt / "smooth_trajectory.json")
    trace = read_json_if_exists(attempt / "trace.json")
    image_manifest = read_json_if_exists(attempt / "image_manifest.json")
    planner_prompt = read_json_if_exists(attempt / "planner_prompt.json")
    planner_response = read_json_if_exists(attempt / "planner_response.json")
    command_result = read_json_if_exists(attempt / "command_result.json")
    segment_manifest = read_json_if_exists(attempt / "segment_vlm_trajectory.json")
    replay_score_summary = _read_optional_text(attempt / "replay_results" / "score_summary.csv", 4000)
    planner_score_summary = _read_optional_text(attempt / "planner_results" / "score_summary.csv", 4000)
    return {
        "attempt_dir": str(attempt),
        "piecewise_trajectory": piecewise,
        "smooth_trajectory": smooth,
        "trace": trace,
        "image_manifest": image_manifest,
        "planner_prompt": planner_prompt,
        "planner_response": planner_response,
        "command_result": command_result,
        "segment_manifest": segment_manifest,
        "replay_score_summary_csv": replay_score_summary,
        "planner_score_summary_csv": planner_score_summary,
        "stdout_excerpt": _read_optional_text(attempt / "command_stdout.txt", 8000),
        "stderr_excerpt": _read_optional_text(attempt / "command_stderr.txt", 8000),
    }


def _read_optional_text(path: Path, max_chars: int) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[-max_chars:]


def compact_attempt_for_analysis(attempt: dict[str, Any], *, max_samples: int = 12) -> dict[str, Any]:
    trace = attempt.get("trace") or {}
    samples = trace.get("samples") or []
    if len(samples) > max_samples:
        indices = sorted({int(round(v)) for v in np.linspace(0, len(samples) - 1, max_samples)})
        samples = [samples[i] for i in indices]
    samples = [_compact_trace_sample(sample) for sample in samples]
    piecewise = attempt.get("piecewise_trajectory") or {}
    smooth = attempt.get("smooth_trajectory") or {}
    return {
        "attempt_dir": attempt.get("attempt_dir"),
        "command_result": attempt.get("command_result"),
        "planner_prompt": attempt.get("planner_prompt"),
        "planner_response": attempt.get("planner_response"),
        "vlm_delta_plan": (
            piecewise.get("metadata", {}).get("planning", {}).get("vlm_delta_plan")
            if isinstance(piecewise, dict)
            else None
        ),
        "piecewise_waypoints": piecewise.get("waypoints") if isinstance(piecewise, dict) else None,
        "smooth_metadata": smooth.get("metadata") if isinstance(smooth, dict) else None,
        "smooth_waypoint_count": len(smooth.get("waypoints", [])) if isinstance(smooth, dict) else 0,
        "trace_sample_period": trace.get("sample_period"),
        "trace_samples": samples,
        "image_manifest": attempt.get("image_manifest"),
        "segment_manifest": attempt.get("segment_manifest"),
        "replay_score_summary_csv": attempt.get("replay_score_summary_csv"),
        "planner_score_summary_csv": attempt.get("planner_score_summary_csv"),
        "stdout_excerpt": attempt.get("stdout_excerpt"),
        "stderr_excerpt": attempt.get("stderr_excerpt"),
    }


def _compact_trace_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Keep per-sample signal, drop repeated full artifacts already stored elsewhere."""
    keep_keys = {
        "timestamp",
        "sim_time",
        "pipeline_phase",
        "robot_tcp_pose",
        "robot_tcp_velocity",
        "joint_positions",
        "joint_velocities",
        "commanded_target_pose",
        "commanded_action",
        "actual_reached_pose",
        "commanded_actual_delta",
        "gripper_state",
        "cable_state",
        "wrench_force_torque",
        "target_port_pose",
        "scene_object_poses",
        "transform_frames",
        "observation",
        "camera_metadata",
        "parsed_planner_decision",
        "global_smoothed_trajectory_point",
        "policy_command",
        "cheatcode_command",
        "final_score_reward",
        "errors_warnings_exceptions",
        "unavailable_fields",
    }
    compact = {key: value for key, value in sample.items() if key in keep_keys}
    point = compact.get("global_smoothed_trajectory_point")
    if isinstance(point, dict):
        compact["global_smoothed_trajectory_point"] = {
            key: point.get(key)
            for key in ("timestamp", "tcp_pose", "tcp_velocity", "phase", "source", "diagnostics")
        }
    planner_decision = compact.get("parsed_planner_decision")
    if isinstance(planner_decision, dict) and "waypoints" in planner_decision:
        compact["parsed_planner_decision"] = {
            "waypoint_count": len(planner_decision.get("waypoints") or []),
            "diagnostics": planner_decision.get("diagnostics"),
        }
    return compact


FAILURE_ANALYSIS_CONTEXT = (
    "This system solves the AIC cable insertion task using an agentic teleop teacher. "
    "GPT-5-mini is used as a local planner to propose movements from visual observations "
    "and robot state. Its output is smoothed locally, then globally smoothed into a "
    "continuous trajectory. That trajectory is converted into a policy compatible with "
    "aic_model and replayed in the official/sim environment. The final insertion phase "
    "is intentionally delegated to CheatCode.py, which uses privileged target-pose "
    "information and is known to score about 96 when run directly. Therefore, if the "
    "combined pipeline fails, the likely issue is before final insertion, during "
    "smoothing/replay, or at the transition into CheatCode.\n\n"
    "A key question is whether GPT-5-mini should be expected to produce detailed metric "
    "waypoints at all. It may be better used as a supplementary component for scene "
    "interpretation, high-level strategy, waypoint review, or choosing subgoals, while "
    "deterministic robotics tools such as IK, MoveIt, cuRobo, collision checking, visual "
    "servoing, and trajectory optimization generate executable robot motions."
)


FAILURE_ANALYSIS_QUESTIONS = [
    "Compare all 3 attempts.",
    "Identify common failure patterns.",
    "Rank likely root causes with evidence.",
    "Decide whether GPT-5-mini is capable of producing full detailed robot waypoints for this task.",
    "If GPT-5-mini is not enough, explain whether it should be replaced by a robotics planner, used only as a high-level semantic planner, used only for scene understanding, used only for choosing intermediate goals, used as a critic/reviewer, or removed from the control loop.",
    "Evaluate whether tools like MoveIt, cuRobo, FCL, IK solvers, collision checkers, trajectory optimizers, or visual servoing should be added.",
    "Explain exactly how such tools should fit into the pipeline.",
    "Identify missing context/tools for the local planner.",
    "Check frame/coordinate inconsistencies.",
    "Check timing/action-rate/replay issues.",
    "Check smoothing artifacts.",
    "Check whether the policy reaches a valid pre-insertion pose before CheatCode handoff.",
    "Check whether images are useful, valid, and correctly framed.",
    "Check whether observations are sufficient.",
    "Recommend concrete code-level fixes.",
    "Recommend new assertions/tests.",
    "Recommend a minimal experiment matrix to isolate the root cause.",
]


def build_failure_analysis_prompt(payload: dict[str, Any]) -> str:
    questions = "\n".join(f"{index + 1}. {question}" for index, question in enumerate(FAILURE_ANALYSIS_QUESTIONS))
    return (
        "# AIC Agent Teleop Failure Analysis\n\n"
        f"{FAILURE_ANALYSIS_CONTEXT}\n\n"
        "Analyze the structured payload below. Use concrete evidence from the three attempts.\n\n"
        f"## Required Questions\n{questions}\n\n"
        "## Payload\n"
        "```json\n"
        f"{json.dumps(payload, indent=2, default=_json_default)}\n"
        "```\n"
    )


def build_failure_analysis_payload(run_dir: str | Path, *, max_samples_per_attempt: int = 12) -> dict[str, Any]:
    run_path = Path(run_dir)
    attempts = [load_attempt(path) for path in sorted(run_path.glob("attempt_*")) if path.is_dir()]
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "run_dir": str(run_path),
        "attempt_count": len(attempts),
        "attempts": [
            compact_attempt_for_analysis(attempt, max_samples=max_samples_per_attempt)
            for attempt in attempts
        ],
        "run_summary": read_json_if_exists(run_path / "summary.json"),
        "bundle_manifest": read_json_if_exists(run_path / "bundle_manifest.json"),
    }


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    attempts = [load_attempt(path) for path in sorted(run_path.glob("attempt_*")) if path.is_dir()]
    scores = []
    image_counts = []
    sample_counts = []
    for attempt in attempts:
        trace = attempt.get("trace") or {}
        image_manifest = attempt.get("image_manifest") or {}
        sample_counts.append(len(trace.get("samples") or []))
        image_counts.append(int(image_manifest.get("image_count") or 0))
        recording = (
            (attempt.get("segment_manifest") or {}).get("recording")
            or (attempt.get("smooth_trajectory") or {}).get("metadata", {}).get("recording")
            or {}
        )
        if recording.get("score_total") is not None:
            scores.append(float(recording["score_total"]))
        else:
            parsed_score = _score_from_score_summary_text(attempt.get("replay_score_summary_csv"))
            if parsed_score is not None:
                scores.append(parsed_score)
    return {
        "schema_version": "agent_teleop_failure_summary/v1",
        "run_dir": str(run_path),
        "attempt_count": len(attempts),
        "attempts": [
            {
                "attempt_dir": attempt.get("attempt_dir"),
                "exit_code": (attempt.get("command_result") or {}).get("exit_code"),
                "image_count": (attempt.get("image_manifest") or {}).get("image_count", 0),
                "trace_sample_count": len((attempt.get("trace") or {}).get("samples") or []),
                "score_total": (
                    ((attempt.get("segment_manifest") or {}).get("recording") or {}).get("score_total")
                    or _score_from_score_summary_text(attempt.get("replay_score_summary_csv"))
                ),
            }
            for attempt in attempts
        ],
        "image_count_total": sum(image_counts),
        "trace_sample_count_total": sum(sample_counts),
        "score_total_mean": statistics.mean(scores) if scores else None,
        "score_total_values": scores,
    }


def _score_from_score_summary_text(text: str | None) -> float | None:
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    parts = lines[-1].split(",")
    if len(parts) < 4:
        return None
    try:
        return float(parts[3])
    except ValueError:
        return None


def write_bundle(run_dir: str | Path, *, include_zip: bool = False) -> dict[str, Any]:
    run_path = Path(run_dir)
    summary = summarize_run(run_path)
    write_json(run_path / "summary.json", summary)
    payload = build_failure_analysis_payload(run_path)
    prompt = build_failure_analysis_prompt(payload)
    (run_path / "prompt.md").write_text(prompt, encoding="utf-8")
    files = sorted(path for path in run_path.rglob("*") if path.is_file())
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "run_dir": str(run_path),
        "created_at": utc_now_iso(),
        "files": [
            {
                "path": str(path),
                "relative_path": str(path.relative_to(run_path)),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    if include_zip:
        zip_path = run_path / "bundle.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                if path.name == "bundle.zip":
                    continue
                archive.write(path, path.relative_to(run_path))
        manifest["bundle_zip"] = str(zip_path)
    write_json(run_path / "bundle_manifest.json", manifest)
    return manifest
