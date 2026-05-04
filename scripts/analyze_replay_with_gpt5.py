#!/usr/bin/env python3
"""Build a central-camera replay failure bundle and optionally ask GPT-5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "aic_teacher_official"))

from aic_teacher_official.vlm_planner import _extract_output_text, _image_items, load_openai_api_key  # noqa: E402


STATE_KEYS = [
    "tcp_pose.position.x",
    "tcp_pose.position.y",
    "tcp_pose.position.z",
    "tcp_pose.orientation.x",
    "tcp_pose.orientation.y",
    "tcp_pose.orientation.z",
    "tcp_pose.orientation.w",
    "tcp_velocity.linear.x",
    "tcp_velocity.linear.y",
    "tcp_velocity.linear.z",
    "tcp_velocity.angular.x",
    "tcp_velocity.angular.y",
    "tcp_velocity.angular.z",
    "tcp_error.x",
    "tcp_error.y",
    "tcp_error.z",
    "tcp_error.rx",
    "tcp_error.ry",
    "tcp_error.rz",
    "joint_positions.0",
    "joint_positions.1",
    "joint_positions.2",
    "joint_positions.3",
    "joint_positions.4",
    "joint_positions.5",
    "joint_positions.6",
    "wrist_wrench.force.x",
    "wrist_wrench.force.y",
    "wrist_wrench.force.z",
    "wrist_wrench.torque.x",
    "wrist_wrench.torque.y",
    "wrist_wrench.torque.z",
]


def quaternion_xyzw_to_rotation_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        quat = quat / norm
    x, y, z, w = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", required=True, help="Replay attempt directory.")
    parser.add_argument("--output-dir", help="Output analysis directory. Defaults to attempt_dir/gpt5_replay_analysis.")
    parser.add_argument("--sample-period-sec", type=float, default=0.25)
    parser.add_argument("--camera", default="all", choices=["all", "center_camera", "left_camera", "right_camera"])
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--use-gpt5", action="store_true")
    parser.add_argument("--max-images", type=int, default=80)
    parser.add_argument("--timeout-sec", type=float, default=240.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    attempt_dir = Path(args.attempt_dir)
    output_dir = Path(args.output_dir) if args.output_dir else attempt_dir / "gpt5_replay_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = attempt_dir / "dataset"
    data_path = dataset_root / "data" / "chunk-000" / "file-000.parquet"
    cameras = ["center_camera", "left_camera", "right_camera"] if args.camera == "all" else [args.camera]
    runtime_trace_path = attempt_dir / "runtime_trace.jsonl"
    scoring_path = attempt_dir / "results" / "trial_1_trial_000001" / "scoring.yaml"
    summary_path = attempt_dir.parents[1] / "generation_summary.json"

    if not data_path.exists():
        raise SystemExit(f"missing LeRobot data parquet: {data_path}")
    video_paths = {
        camera: dataset_root / "videos" / f"observation.images.{camera}" / "chunk-000" / "file-000.mp4"
        for camera in cameras
    }
    missing_videos = [str(path) for path in video_paths.values() if not path.exists()]
    if missing_videos:
        raise SystemExit(f"missing replay camera video(s): {missing_videos}")

    per_camera_max = max(1, args.max_images // max(1, len(video_paths)))
    frame_rows = []
    for camera, video_path in video_paths.items():
        frame_rows.extend(
            extract_camera_frames(
                video_path,
                output_dir / f"{camera}_frames",
                camera=camera,
                sample_period_sec=args.sample_period_sec,
                max_images=per_camera_max,
            )
        )
    frame_rows.sort(key=lambda row: (float(row["timestamp"]), str(row["camera"])))
    observations = sample_observations(data_path, [float(row["timestamp"]) for row in frame_rows])
    payload = {
        "schema_version": "aic_replay_gpt5_failure_bundle/v2",
        "attempt_dir": str(attempt_dir),
        "dataset_root": str(dataset_root),
        "analysis_sampling": {
            "camera": args.camera,
            "cameras": cameras,
            "sample_period_sec": args.sample_period_sec,
            "image_count": len(frame_rows),
            "note": "Frames are aligned to the nearest LeRobot observation row. Time history is sampled at 0.25 seconds by default and may be truncated by --max-images.",
        },
        "coordinate_frame_contract": {
            "tcp_pose_frame": "base_link",
            "absolute_cartesian_action_frame": "base_link",
            "relative_delta_action_frame": "gripper/tcp",
            "force_frame_note": (
                "Dataset wrist force is raw wrist wrench. The sampled observation includes the raw "
                "TCP-assumed force and a base_link estimate obtained by rotating through actual TCP orientation."
            ),
        },
        "pipeline_strategy": pipeline_strategy_description(),
        "ranked_directions_context": ranked_directions_context(),
        "score_summary": load_scoring(scoring_path),
        "generation_summary_excerpt": load_generation_summary(summary_path),
        "runtime_trace_events": select_runtime_events(runtime_trace_path),
        "samples": [
            {
                **row,
                "observation": observations.get(round(float(row["timestamp"]), 3)),
            }
            for row in frame_rows
        ],
    }
    prompt = build_prompt(payload)
    (output_dir / "payload.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "prompt.md").write_text(prompt, encoding="utf-8")

    if args.use_gpt5:
        analysis = call_gpt5(prompt, [Path(row["path"]) for row in frame_rows], model=args.model, timeout_sec=args.timeout_sec)
    else:
        analysis = "# Dry Run\n\nGPT-5 was not called. Re-run with `--use-gpt5`.\n"
    (output_dir / "analysis.md").write_text(analysis, encoding="utf-8")
    print(f"payload: {output_dir / 'payload.json'}")
    print(f"prompt: {output_dir / 'prompt.md'}")
    print(f"analysis: {output_dir / 'analysis.md'}")
    print(f"images: {len(frame_rows)}")
    return 0


def extract_camera_frames(
    video_path: Path,
    output_dir: Path,
    *,
    camera: str,
    sample_period_sec: float,
    max_images: int,
) -> list[dict[str, Any]]:
    try:
        import cv2
    except Exception as ex:
        raise RuntimeError("opencv-python is required to extract central camera frames") from ex
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"unable to open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0.0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(f"invalid video metadata for {video_path}")
    duration = frame_count / fps
    rows: list[dict[str, Any]] = []
    timestamp = 0.0
    while timestamp <= duration + 1e-9 and len(rows) < max_images:
        frame_index = int(round(timestamp * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break
        out = output_dir / f"{camera}_t{timestamp:07.3f}_f{frame_index:06d}.jpg"
        cv2.imwrite(str(out), frame)
        rows.append(
            {
                "timestamp": round(timestamp, 3),
                "camera": camera,
                "frame_index": frame_index,
                "path": str(out),
            }
        )
        timestamp += sample_period_sec
    cap.release()
    return rows


def sample_observations(data_path: Path, timestamps: list[float]) -> dict[float, dict[str, Any]]:
    df = pd.read_parquet(data_path)
    if df.empty:
        return {}
    timestamp_values = df["timestamp"].to_numpy(dtype=np.float64)
    out: dict[float, dict[str, Any]] = {}
    for timestamp in timestamps:
        idx = int(np.argmin(np.abs(timestamp_values - timestamp)))
        row = df.iloc[idx]
        state = [float(v) for v in list(row["observation.state"])]
        action = [float(v) for v in list(row["action"])]
        state_dict = {key: state[i] for i, key in enumerate(STATE_KEYS[: len(state)])}
        force = np.asarray(
            [
                state_dict.get("wrist_wrench.force.x", 0.0),
                state_dict.get("wrist_wrench.force.y", 0.0),
                state_dict.get("wrist_wrench.force.z", 0.0),
            ],
            dtype=np.float64,
        )
        velocity = np.asarray(
            [
                state_dict.get("tcp_velocity.linear.x", 0.0),
                state_dict.get("tcp_velocity.linear.y", 0.0),
                state_dict.get("tcp_velocity.linear.z", 0.0),
            ],
            dtype=np.float64,
        )
        tracking = np.asarray(
            [
                state_dict.get("tcp_error.x", 0.0),
                state_dict.get("tcp_error.y", 0.0),
                state_dict.get("tcp_error.z", 0.0),
            ],
            dtype=np.float64,
        )
        tcp_quat = np.asarray(
            [
                state_dict.get("tcp_pose.orientation.x", 0.0),
                state_dict.get("tcp_pose.orientation.y", 0.0),
                state_dict.get("tcp_pose.orientation.z", 0.0),
                state_dict.get("tcp_pose.orientation.w", 1.0),
            ],
            dtype=np.float64,
        )
        force_base_estimated = quaternion_xyzw_to_rotation_matrix(tcp_quat) @ force
        out[round(float(timestamp), 3)] = {
            "nearest_dataset_timestamp": float(row["timestamp"]),
            "row_index": int(row["index"]),
            "frame_index": int(row["frame_index"]),
            "tcp_position": [
                state_dict.get("tcp_pose.position.x"),
                state_dict.get("tcp_pose.position.y"),
                state_dict.get("tcp_pose.position.z"),
            ],
            "tcp_orientation_xyzw": [
                state_dict.get("tcp_pose.orientation.x"),
                state_dict.get("tcp_pose.orientation.y"),
                state_dict.get("tcp_pose.orientation.z"),
                state_dict.get("tcp_pose.orientation.w"),
            ],
            "tcp_speed_mps": float(np.linalg.norm(velocity)),
            "tracking_error_m": float(np.linalg.norm(tracking)),
            "wrist_force_tcp_assumed": force.tolist(),
            "wrist_force_base_link_estimated": force_base_estimated.tolist(),
            "wrist_force_norm": float(np.linalg.norm(force)),
            "action": action,
        }
    return out


def pipeline_strategy_description() -> dict[str, Any]:
    return {
        "two_step_process": [
            "First trial: plan and execute segment-wise. VLM/GPT-5-mini gives only symbolic strategy; MoveIt transports near the port; deterministic geometry handles final insertion/recovery. Stalls during this phase are acceptable.",
            "Postprocessing: delete stall intervals at image-retrieval cadence units, globally smooth the retained trajectory except near the last insertion, convert it into a policy/action dataset, and rerun that policy to record the final trajectory.",
        ],
        "important_implication": (
            "Do not over-optimize the first trial for perfect timing. Stalls can be deleted later. "
            "The first trial should prioritize safe geometry, interpretable recovery, and enough signal for postprocessing."
        ),
        "current_modes": {
            "nominal": "No post-contact recovery; should either insert cleanly or reject contact.",
            "nominalrecovery": "On F/T threshold, stop descent, back off, wait for force release, realign, retry, then postprocess recorded stalls.",
        },
        "additional_candidate_approach": (
            "Use measured plug/port pose to compute a smooth transport target similar to CheatCode alignment, "
            "but place it slightly closer to the port than the standard preinsert point. Then descend. "
            "This could be a three-run loop: find aligned pose, run VLM+MoveIt plus deterministic insertion, "
            "postprocess to smooth/remove stalls and convert to policy, then replay policy to save final trajectory."
        ),
    }


def ranked_directions_context() -> list[dict[str, Any]]:
    return [
        {
            "rank": 1,
            "direction": "Gate or replace live-Z repair when lateral/force context is bad.",
            "why": "Recent runs show recovery/backoff succeeds, but live-Z repaired descent contacts immediately.",
        },
        {
            "rank": 2,
            "direction": "Try a smooth aligned transport target computed from plug/port pose before descent.",
            "why": "This may avoid the artificial straight-line then straight-line CheatCode profile while still aligning laterally.",
        },
        {
            "rank": 3,
            "direction": "Keep post-replay stall compaction and policy conversion as the smoothing path.",
            "why": "Stalls are acceptable in the first trial if they can be removed at image-retrieval cadence before policy replay.",
        },
    ]


def load_scoring(path: Path) -> Any:
    if not path.exists():
        return {"available": False, "path": str(path)}
    try:
        import yaml

        return {"available": True, "path": str(path), "scoring": yaml.safe_load(path.read_text(encoding="utf-8"))}
    except Exception:
        return {"available": True, "path": str(path), "text": path.read_text(encoding="utf-8", errors="ignore")}


def load_generation_summary(path: Path) -> Any:
    if not path.exists():
        return {"available": False, "path": str(path)}
    summary = json.loads(path.read_text(encoding="utf-8"))
    records = summary.get("records") or []
    return {
        "available": True,
        "path": str(path),
        "mode": summary.get("mode"),
        "accepted": summary.get("accepted"),
        "attempts": summary.get("attempts"),
        "stopped_reason": summary.get("stopped_reason"),
        "latest_replay_metrics": (records[-1].get("replay_metrics") if records else None),
    }


def select_runtime_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    selected = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        name = str(event.get("event", ""))
        if (
            "gate" in name
            or name.startswith("recovery")
            or name.startswith("guarded")
            or name == "contact_detected"
        ):
            selected.append(event)
    return selected[-120:]


def build_prompt(payload: dict[str, Any]) -> str:
    return (
        "# GPT-5 Replay Failure Analysis\n\n"
        "You are analyzing an AIC cable-insertion replay. Use the attached camera images and the aligned observation/action rows. "
        "Images are sampled every 0.25 seconds by default, with truncation if the bundle becomes too large.\n\n"
        "The pipeline is explicitly two-step: first, a segment-wise trial is planned/executed, where stalls are acceptable; "
        "second, postprocessing deletes stall intervals in image-retrieval cadence units, globally smooths the retained motion except near the last insertion, "
        "converts it to a policy/action dataset, and reruns that policy to save the final trajectory.\n\n"
        "Focus on ranking next changes for both nominal and nominalrecovery. Explain whether the first-trial failure should be fixed online, left for postprocessing, "
        "or rejected/regenerated. Be concrete about coordinate frames, live-Z repair, measured recovery backoff, aligned transport target ideas, and policy conversion.\n\n"
        "Return concise Markdown with: Findings, Root Cause, Ranked Next Directions, and Next Trial Settings.\n\n"
        "## Payload\n```json\n"
        + json.dumps(payload, indent=2)
        + "\n```\n"
    )


def call_gpt5(prompt: str, images: list[Path], *, model: str, timeout_sec: float) -> str:
    api_key = load_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for GPT-5 analysis")
    from openai import OpenAI

    response = OpenAI(api_key=api_key, timeout=timeout_sec).responses.create(
        model=model,
        instructions="You are a senior robotics trajectory and contact-debugging engineer.",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    *_image_items(images),
                ],
            }
        ],
    )
    text = _extract_output_text(response)
    if not text.strip():
        raise RuntimeError("GPT-5 returned empty output")
    return text


if __name__ == "__main__":
    raise SystemExit(main())
