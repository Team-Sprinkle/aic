"""Prepare review bundles for future GPT-5 VLM trajectory critique."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from aic_teacher_official.trajectory import SmoothTrajectory
from aic_teacher_official.vlm_planner import _image_items, _extract_output_text, load_openai_api_key


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
SCORING_LOG_PATTERNS = (
    "score",
    "scoring",
    "tier",
    "total",
    "result",
    "insertion",
    "distance",
    "contact",
    "force",
    "duration",
    "trajectory",
    "episode saved",
)


def _find_images(image_dir: Path | None) -> list[Path]:
    if image_dir is None or not image_dir.exists():
        return []
    return sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def _equidistant_indices(count: int, samples: int) -> list[int]:
    if count <= 0:
        return []
    if samples <= 1:
        return [0]
    return sorted({int(round(v)) for v in np.linspace(0, count - 1, samples)})


def _load_lerobot_rows(dataset_root: Path | None, samples: int) -> dict[str, Any]:
    if dataset_root is None:
        return {"available": False, "reason": "dataset_root_not_provided", "samples": []}
    data_path = dataset_root / "data" / "chunk-000" / "file-000.parquet"
    info_path = dataset_root / "meta" / "info.json"
    if not data_path.exists() or not info_path.exists():
        return {
            "available": False,
            "reason": "missing_lerobot_parquet_or_info",
            "samples": [],
            "dataset_root": str(dataset_root),
        }
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    state_names = features.get("observation.state", {}).get("names") or []
    action_names = features.get("action", {}).get("names") or []
    df = pq.read_table(data_path).to_pandas()
    return _load_lerobot_rows_for_indices(dataset_root, _equidistant_indices(len(df), samples))


def _load_lerobot_rows_for_indices(
    dataset_root: Path | None,
    indices: list[int],
) -> dict[str, Any]:
    if dataset_root is None:
        return {"available": False, "reason": "dataset_root_not_provided", "samples": []}
    data_path = dataset_root / "data" / "chunk-000" / "file-000.parquet"
    info_path = dataset_root / "meta" / "info.json"
    if not data_path.exists() or not info_path.exists():
        return {
            "available": False,
            "reason": "missing_lerobot_parquet_or_info",
            "samples": [],
            "dataset_root": str(dataset_root),
        }
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    state_names = features.get("observation.state", {}).get("names") or []
    action_names = features.get("action", {}).get("names") or []
    df = pq.read_table(data_path).to_pandas()
    selected = []
    for index in sorted({i for i in indices if 0 <= i < len(df)}):
        row = df.iloc[index]
        state = [float(v) for v in row["observation.state"]]
        action = [float(v) for v in row["action"]]
        selected.append(
            {
                "row_index": int(index),
                "timestamp": float(row["timestamp"]),
                "frame_index": int(row["frame_index"]),
                "observation_state": {
                    name: state[i] for i, name in enumerate(state_names[: len(state)])
                },
                "action": {
                    name: action[i] for i, name in enumerate(action_names[: len(action)])
                },
            }
        )
    return {
        "available": True,
        "dataset_root": str(dataset_root),
        "data_path": str(data_path),
        "row_count": int(len(df)),
        "state_names": state_names,
        "action_names": action_names,
        "samples": selected,
    }


def _extract_video_frames(
    dataset_root: Path | None,
    output_dir: Path,
    *,
    samples: int,
) -> list[dict[str, Any]]:
    if dataset_root is None or not dataset_root.exists():
        return []
    try:
        import cv2
    except Exception:
        return []

    videos = sorted((dataset_root / "videos").glob("observation.images.*/*/*.mp4"))
    if not videos:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, Any]] = []
    for video in videos:
        view = video.parts[-3].replace("observation.images.", "")
        cap = cv2.VideoCapture(str(video))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_index in _equidistant_indices(frame_count, samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            out = output_dir / f"{view}_frame_{frame_index:06d}.png"
            if cv2.imwrite(str(out), frame):
                extracted.append(
                    {
                        "view": view,
                        "frame_index": int(frame_index),
                        "path": str(out),
                    }
                )
        cap.release()
    return extracted


def _extract_video_frames_for_indices(
    dataset_root: Path | None,
    output_dir: Path,
    *,
    frame_indices: list[int],
) -> list[dict[str, Any]]:
    if dataset_root is None or not dataset_root.exists():
        return []
    try:
        import cv2
    except Exception:
        return []

    videos = sorted((dataset_root / "videos").glob("observation.images.*/*/*.mp4"))
    if not videos:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, Any]] = []
    for video in videos:
        view = video.parts[-3].replace("observation.images.", "")
        cap = cv2.VideoCapture(str(video))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_index in sorted({i for i in frame_indices if 0 <= i < frame_count}):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            out = output_dir / f"{view}_frame_{frame_index:06d}.png"
            if cv2.imwrite(str(out), frame):
                extracted.append(
                    {
                        "view": view,
                        "frame_index": int(frame_index),
                        "path": str(out),
                    }
                )
        cap.release()
    return extracted


def _load_score(scoring_path: Path | None) -> dict[str, Any]:
    if scoring_path is None or not scoring_path.exists():
        return {"available": False}
    try:
        import yaml
    except Exception:
        return {"available": True, "path": str(scoring_path), "raw_text": scoring_path.read_text()}
    return {
        "available": True,
        "path": str(scoring_path),
        "scoring": yaml.safe_load(scoring_path.read_text(encoding="utf-8")),
    }


def _read_text_excerpt(path: Path, *, max_chars: int = 12000) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _matching_log_lines(path: Path, *, max_lines: int = 120) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    matches = [
        line
        for line in lines
        if any(pattern in line.lower() for pattern in SCORING_LOG_PATTERNS)
    ]
    return matches[-max_lines:]


def _load_container_scoring_context(scoring_path: Path | None) -> dict[str, Any]:
    if scoring_path is None:
        return {"available": False, "reason": "scoring_path_not_provided"}
    scoring_path = Path(scoring_path)
    if not scoring_path.exists():
        return {
            "available": False,
            "reason": "scoring_path_missing",
            "scoring_path": str(scoring_path),
        }

    scores_root = scoring_path.parents[1]
    postprocessed_root = scoring_path.parents[2]
    score_summary = scores_root / "score_summary.csv"
    log_dir = postprocessed_root / "logs" / "per_trial_tmp"
    log_paths = sorted(log_dir.glob("trial_*_*.log")) if log_dir.exists() else []
    return {
        "available": True,
        "description": (
            "Container/eval scoring context preserved for GPT-5 failure analysis. "
            "Includes official scorer YAML messages, score summary CSV, and "
            "scoring-related excerpts from per-trial logs."
        ),
        "scoring_yaml_path": str(scoring_path),
        "scoring_yaml_text": _read_text_excerpt(scoring_path),
        "score_summary_csv_path": str(score_summary) if score_summary.exists() else None,
        "score_summary_csv_text": (
            _read_text_excerpt(score_summary, max_chars=4000)
            if score_summary.exists()
            else None
        ),
        "log_excerpts": [
            {
                "path": str(path),
                "matching_lines": _matching_log_lines(path),
            }
            for path in log_paths
        ],
    }


def _score_total(score: dict[str, Any]) -> float | None:
    scoring = score.get("scoring")
    if isinstance(scoring, dict) and scoring.get("total") is not None:
        try:
            return float(scoring["total"])
        except (TypeError, ValueError):
            return None
    return None


def _norm(values: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64)))


def _port_position_from_metadata(metadata: dict[str, Any]) -> list[float] | None:
    context = metadata.get("planning", {}).get("context")
    if not isinstance(context, dict):
        return None
    port = context.get("port_position")
    if not isinstance(port, list) or len(port) != 3:
        return None
    return [float(v) for v in port]


def _sample_analysis(
    *,
    planned: dict[str, Any] | None,
    recorded: dict[str, Any] | None,
    port_position: list[float] | None,
) -> dict[str, Any]:
    analysis: dict[str, Any] = {}
    planned_position = (
        planned.get("tcp_pose", {}).get("position")
        if planned is not None
        else None
    )
    recorded_state = recorded.get("observation_state", {}) if recorded is not None else {}
    action = recorded.get("action", {}) if recorded is not None else {}
    recorded_position = [
        recorded_state.get("tcp_pose.position.x"),
        recorded_state.get("tcp_pose.position.y"),
        recorded_state.get("tcp_pose.position.z"),
    ]
    if all(value is not None for value in recorded_position):
        recorded_position = [float(value) for value in recorded_position]
        analysis["recorded_tcp_position"] = recorded_position
    else:
        recorded_position = None
    if planned_position is not None and recorded_position is not None:
        delta = (
            np.asarray(recorded_position, dtype=np.float64)
            - np.asarray(planned_position, dtype=np.float64)
        )
        analysis["recorded_minus_planned_tcp_position"] = delta.tolist()
        analysis["recorded_planned_position_error_norm_m"] = float(np.linalg.norm(delta))
    if port_position is not None:
        for prefix, position in [
            ("planned", planned_position),
            ("recorded", recorded_position),
        ]:
            if position is None:
                continue
            vector = np.asarray(port_position, dtype=np.float64) - np.asarray(
                position,
                dtype=np.float64,
            )
            analysis[f"{prefix}_tcp_to_port_vector_m"] = vector.tolist()
            analysis[f"{prefix}_tcp_to_port_distance_m"] = float(np.linalg.norm(vector))
    tcp_error = [
        recorded_state.get("tcp_error.x"),
        recorded_state.get("tcp_error.y"),
        recorded_state.get("tcp_error.z"),
    ]
    if all(value is not None for value in tcp_error):
        analysis["recorded_tcp_error_position_norm_m"] = _norm([float(v) for v in tcp_error])
    force = [
        recorded_state.get("wrist_wrench.force.x"),
        recorded_state.get("wrist_wrench.force.y"),
        recorded_state.get("wrist_wrench.force.z"),
    ]
    if all(value is not None for value in force):
        analysis["wrist_force_norm_n"] = _norm([float(v) for v in force])
    delta_action = [
        action.get("delta_position.x"),
        action.get("delta_position.y"),
        action.get("delta_position.z"),
    ]
    if all(value is not None for value in delta_action):
        analysis["delta_position_action_norm_m"] = _norm([float(v) for v in delta_action])
    return analysis


def _build_geometry_plots(
    paired_samples: list[dict[str, Any]],
    port_position: list[float] | None,
    output_dir: Path,
) -> list[dict[str, Any]]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    planned = []
    recorded = []
    for sample in paired_samples:
        planned_position = (
            sample.get("planned", {}).get("tcp_pose", {}).get("position")
            if sample.get("planned") is not None
            else None
        )
        recorded_position = sample.get("analysis", {}).get("recorded_tcp_position")
        if planned_position is not None:
            planned.append((sample["sample_index"], planned_position))
        if recorded_position is not None:
            recorded.append((sample["sample_index"], recorded_position))

    def _plot(
        filename: str,
        axes: tuple[int, int],
        labels: tuple[str, str],
        title: str,
    ) -> dict[str, Any] | None:
        if not planned and not recorded:
            return None
        fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
        for values, color, name, marker in [
            (planned, "tab:blue", "planned TCP", "o"),
            (recorded, "tab:green", "recorded TCP", "x"),
        ]:
            if not values:
                continue
            xs = [position[axes[0]] for _, position in values]
            ys = [position[axes[1]] for _, position in values]
            ax.plot(xs, ys, color=color, marker=marker, label=name)
            for sample_index, position in values:
                ax.annotate(
                    str(sample_index),
                    (position[axes[0]], position[axes[1]]),
                    fontsize=7,
                    color=color,
                )
        if port_position is not None:
            ax.scatter(
                [port_position[axes[0]]],
                [port_position[axes[1]]],
                color="tab:red",
                marker="*",
                s=180,
                label="port",
            )
        ax.set_title(title)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.grid(True, alpha=0.35)
        ax.axis("equal")
        ax.legend(loc="best")
        path = output_dir / filename
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return {
            "kind": "geometry_plot",
            "path": str(path),
            "description": title,
        }

    plots = [
        _plot("tcp_path_xy.png", (0, 1), ("base_link x (m)", "base_link y (m)"), "TCP path top-down XY with port marker"),
        _plot("tcp_path_xz.png", (0, 2), ("base_link x (m)", "base_link z (m)"), "TCP path XZ height profile with port marker"),
    ]
    return [plot for plot in plots if plot is not None]


def _trajectory_samples(trajectory: SmoothTrajectory, samples: int) -> list[dict[str, Any]]:
    indices = _equidistant_indices(len(trajectory.waypoints), samples)
    if indices and indices[-1] != len(trajectory.waypoints) - 1:
        indices.append(len(trajectory.waypoints) - 1)
    selected = []
    for waypoint_index in indices:
        waypoint = trajectory.waypoints[waypoint_index]
        selected.append(
            {
                "waypoint_index": int(waypoint_index),
                "timestamp": waypoint.timestamp,
                "phase": waypoint.phase.value,
                "source": waypoint.source.value,
                "tcp_pose": waypoint.tcp_pose.to_dict(),
                "tcp_velocity": waypoint.tcp_velocity,
                "diagnostics": waypoint.diagnostics,
            }
        )
    return selected


def build_run_review_context(
    *,
    label: str,
    trajectory_path: str | Path,
    output_dir: str | Path,
    dataset_root: str | Path | None = None,
    scoring_path: str | Path | None = None,
    samples: int = 10,
) -> dict[str, Any]:
    """Build per-loop context with observations/actions/images at sampled times."""
    trajectory_path = Path(trajectory_path)
    output_dir = Path(output_dir)
    trajectory = SmoothTrajectory.load_json(trajectory_path)
    if samples <= 0:
        raise ValueError("samples must be positive")

    dataset_root_path = Path(dataset_root) if dataset_root else None
    data_path = (
        dataset_root_path / "data" / "chunk-000" / "file-000.parquet"
        if dataset_root_path is not None
        else None
    )
    if data_path is not None and data_path.exists():
        row_count = pq.read_table(data_path, columns=[]).num_rows
        row_indices = _equidistant_indices(row_count, samples)
    else:
        row_indices = []
    rows = _load_lerobot_rows_for_indices(dataset_root_path, row_indices)
    frame_indices = [
        int(sample["frame_index"])
        for sample in rows.get("samples", [])
        if sample.get("frame_index") is not None
    ]
    extracted_images = _extract_video_frames_for_indices(
        dataset_root_path,
        output_dir / "sampled_video_frames",
        frame_indices=frame_indices,
    )
    images_by_frame: dict[int, list[dict[str, Any]]] = {}
    for item in extracted_images:
        images_by_frame.setdefault(int(item["frame_index"]), []).append(item)

    planned_samples = _trajectory_samples(trajectory, samples)
    row_samples = rows.get("samples", [])
    paired_samples = []
    port_position = _port_position_from_metadata(trajectory.metadata.to_dict())
    for index in range(max(len(planned_samples), len(row_samples))):
        row = row_samples[index] if index < len(row_samples) else None
        frame_index = int(row["frame_index"]) if row is not None else None
        planned = planned_samples[index] if index < len(planned_samples) else None
        paired_samples.append(
            {
                "sample_index": index,
                "planned": planned,
                "recorded": row,
                "images": images_by_frame.get(frame_index, []) if frame_index is not None else [],
                "analysis": _sample_analysis(
                    planned=planned,
                    recorded=row,
                    port_position=port_position,
                ),
            }
        )
    visual_aids = _build_geometry_plots(
        paired_samples,
        port_position,
        output_dir / "visual_aids",
    )

    score = _load_score(Path(scoring_path) if scoring_path else None)
    container_scoring_context = _load_container_scoring_context(
        Path(scoring_path) if scoring_path else None
    )
    return {
        "label": label,
        "trajectory": str(trajectory_path),
        "trajectory_metadata": trajectory.metadata.to_dict(),
        "dataset_root": str(dataset_root_path) if dataset_root_path is not None else None,
        "score": score,
        "score_total": _score_total(score),
        "container_scoring_context": container_scoring_context,
        "samples": paired_samples,
        "recorded_dataset": rows,
        "images": {
            "wrist": [item["path"] for item in extracted_images],
            "gazebo": [],
            "missing_wrist_images": len(extracted_images) == 0,
            "missing_gazebo_images": True,
        },
        "visual_aids": visual_aids,
        "derived_geometry": {
            "port_position": port_position,
            "description": (
                "Per-sample analysis includes TCP-to-port vectors/distances, "
                "recorded-vs-planned TCP position error, TCP controller error norm, "
                "delta action norm, and wrist force norm when the source fields exist."
            ),
        },
    }


def build_comparison_review_bundle(
    runs: list[dict[str, Any]],
    output_path: str | Path,
    *,
    samples: int = 10,
) -> dict[str, Any]:
    """Build GPT-5 critique context for multiple prior loops.

    Each run dict must contain label, trajectory_path, and may include
    dataset_root and scoring_path. The resulting manifest carries per-run
    equidistant planned trajectory samples, recorded observations/actions, and
    frame images, so GPT-5 can compare a high-score loop against a regressed
    loop directly.
    """
    output_path = Path(output_path)
    output_root = output_path.with_suffix("")
    run_contexts = [
        build_run_review_context(
            label=str(run["label"]),
            trajectory_path=run["trajectory_path"],
            output_dir=output_root / str(run["label"]),
            dataset_root=run.get("dataset_root"),
            scoring_path=run.get("scoring_path"),
            samples=samples,
        )
        for run in runs
    ]
    manifest: dict[str, Any] = {
        "schema_version": "official_teacher_comparison_review_bundle/v0",
        "comparison_goal": (
            "Compare prior loops, especially a high-score run and any regressed "
            "run, using score breakdowns plus equidistant planned/executed samples."
        ),
        "samples_per_run": samples,
        "runs": run_contexts,
        "score_summary": [
            {
                "label": run["label"],
                "score_total": run.get("score_total"),
                "score": run.get("score"),
            }
            for run in run_contexts
        ],
        "critique": {
            "model": "gpt-5-vlm",
            "api_called": False,
            "required_context": (
                "For each sampled timestep inspect planned TCP pose, recorded "
                "observation values, action values, wrist images, phase/source "
                "labels, and official score breakdown."
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_review_bundle(
    trajectory_path: str | Path,
    output_path: str | Path,
    *,
    wrist_image_dir: str | Path | None = None,
    gazebo_image_dir: str | Path | None = None,
    dataset_root: str | Path | None = None,
    scoring_path: str | Path | None = None,
    samples: int = 8,
) -> dict[str, Any]:
    trajectory_path = Path(trajectory_path)
    output_path = Path(output_path)
    trajectory = SmoothTrajectory.load_json(trajectory_path)
    if samples <= 0:
        raise ValueError("samples must be positive")

    stride = max(1, (len(trajectory.waypoints) - 1) // max(1, samples - 1))
    selected = trajectory.waypoints[::stride][:samples]
    if selected[-1] is not trajectory.waypoints[-1]:
        selected.append(trajectory.waypoints[-1])

    dataset_root_path = Path(dataset_root) if dataset_root else None
    extracted_dir = output_path.with_suffix("") / "sampled_video_frames"
    extracted_video_images = _extract_video_frames(
        dataset_root_path,
        extracted_dir,
        samples=samples,
    )

    wrist_images = _find_images(Path(wrist_image_dir) if wrist_image_dir else None)
    gazebo_images = _find_images(Path(gazebo_image_dir) if gazebo_image_dir else None)
    if extracted_video_images:
        wrist_images = [*wrist_images, *[Path(item["path"]) for item in extracted_video_images]]

    manifest: dict[str, Any] = {
        "schema_version": "official_teacher_review_bundle/v0",
        "trajectory": str(trajectory_path),
        "trajectory_metadata": trajectory.metadata.to_dict(),
        "samples": [
            {
                "timestamp": waypoint.timestamp,
                "phase": waypoint.phase.value,
                "source": waypoint.source.value,
                "tcp_pose": waypoint.tcp_pose.to_dict(),
                "tcp_velocity": waypoint.tcp_velocity,
                "diagnostics": waypoint.diagnostics,
            }
            for waypoint in selected
        ],
        "images": {
            "wrist": [str(path) for path in wrist_images],
            "gazebo": [str(path) for path in gazebo_images],
            "missing_wrist_images": len(wrist_images) == 0,
            "missing_gazebo_images": len(gazebo_images) == 0,
        },
        "recorded_dataset": _load_lerobot_rows(dataset_root_path, samples),
        "score": _load_score(Path(scoring_path) if scoring_path else None),
        "container_scoring_context": _load_container_scoring_context(
            Path(scoring_path) if scoring_path else None
        ),
        "sampled_execution_context": (
            build_run_review_context(
                label="current_run",
                trajectory_path=trajectory_path,
                output_dir=output_path.with_suffix("") / "paired_context",
                dataset_root=dataset_root_path,
                scoring_path=scoring_path,
                samples=samples,
            )
            if dataset_root_path is not None
            else None
        ),
        "force_summary": {
            "available": False,
            "todo": "Attach wrist force/torque summary from LeRobot/ROS bag when available.",
        },
        "actions": {
            "available": False,
            "todo": "Attach executed action rows from LeRobot dataset after recording.",
        },
        "critique": {
            "model": "gpt-5-vlm",
            "api_called": False,
            "todo": "Send this manifest plus equidistant images to GPT-5 VLM in a later job.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _compact_score(score: dict[str, Any]) -> dict[str, Any]:
    scoring = score.get("scoring")
    if not isinstance(scoring, dict):
        return score
    return scoring


def _compact_container_scoring(context: dict[str, Any]) -> dict[str, Any]:
    if not context or not context.get("available"):
        return context
    return {
        "available": True,
        "scoring_yaml_text": context.get("scoring_yaml_text"),
        "score_summary_csv_text": context.get("score_summary_csv_text"),
        "log_excerpts": [
            {
                "path": item.get("path"),
                "matching_lines": item.get("matching_lines", [])[-30:],
            }
            for item in context.get("log_excerpts", [])
        ],
    }


def _compact_recorded_sample(recorded: dict[str, Any] | None) -> dict[str, Any] | None:
    if recorded is None:
        return None
    state = recorded.get("observation_state", {})
    keys = [
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
        "tcp_error.x",
        "tcp_error.y",
        "tcp_error.z",
        "wrist_wrench.force.x",
        "wrist_wrench.force.y",
        "wrist_wrench.force.z",
    ]
    return {
        "row_index": recorded.get("row_index"),
        "timestamp": recorded.get("timestamp"),
        "frame_index": recorded.get("frame_index"),
        "observation_state": {key: state.get(key) for key in keys if key in state},
        "action": recorded.get("action"),
    }


def _compact_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_index": sample.get("sample_index"),
        "planned": sample.get("planned"),
        "recorded": _compact_recorded_sample(sample.get("recorded")),
        "analysis": sample.get("analysis"),
        "images": sample.get("images"),
    }


def _compact_run_context(run: dict[str, Any]) -> dict[str, Any]:
    metadata = run.get("trajectory_metadata", {})
    planning = metadata.get("planning", {}) if isinstance(metadata, dict) else {}
    return {
        "label": run.get("label"),
        "trajectory": run.get("trajectory"),
        "score_total": run.get("score_total"),
        "score": _compact_score(run.get("score", {})),
        "container_scoring_context": _compact_container_scoring(
            run.get("container_scoring_context", {})
        ),
        "planning_context": planning.get("context"),
        "vlm_delta_plan": planning.get("vlm_delta_plan"),
        "derived_geometry": run.get("derived_geometry"),
        "visual_aids": run.get("visual_aids"),
        "samples": [_compact_sample(sample) for sample in run.get("samples", [])],
    }


def _compact_manifest_for_gpt(manifest: dict[str, Any]) -> dict[str, Any]:
    sampled_execution_context = manifest.get("sampled_execution_context")
    metadata = manifest.get("trajectory_metadata", {})
    planning = metadata.get("planning", {}) if isinstance(metadata, dict) else {}
    compact: dict[str, Any] = {
        "schema_version": manifest.get("schema_version"),
        "comparison_goal": manifest.get("comparison_goal"),
        "score_summary": manifest.get("score_summary"),
        "score": _compact_score(manifest.get("score", {})),
        "container_scoring_context": _compact_container_scoring(
            manifest.get("container_scoring_context", {})
        ),
        "planning_context": planning.get("context"),
        "vlm_delta_plan": planning.get("vlm_delta_plan"),
        "critique": manifest.get("critique"),
    }
    if manifest.get("runs"):
        compact["runs"] = [
            _compact_run_context(run)
            for run in manifest.get("runs", [])
        ]
    elif sampled_execution_context:
        compact["run"] = _compact_run_context(sampled_execution_context)
    else:
        compact["samples"] = manifest.get("samples")
        compact["images"] = manifest.get("images")
    return compact


def call_gpt5_failure_review(
    manifest: dict[str, Any],
    *,
    model: str = "gpt-5",
    request_timeout_sec: float = 120.0,
) -> dict[str, Any]:
    """Run one GPT-5 VLM failure-analysis call for a prepared review manifest."""
    api_key = load_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set and could not be read from ~/.zshrc."
        )
    try:
        from openai import OpenAI
    except Exception as ex:
        raise RuntimeError("The openai Python package is required for GPT-5 review.") from ex

    image_paths = [
        Path(path)
        for path in [
            *manifest.get("images", {}).get("wrist", []),
            *manifest.get("images", {}).get("gazebo", []),
        ]
    ]
    sampled_execution_context = manifest.get("sampled_execution_context") or {}
    for visual in sampled_execution_context.get("visual_aids", []):
        path = visual.get("path")
        if path:
            image_paths.append(Path(path))
    for sample in sampled_execution_context.get("samples", []):
        for image in sample.get("images", []):
            path = image.get("path")
            if path:
                image_paths.append(Path(path))
    for run in manifest.get("runs", []):
        for visual in run.get("visual_aids", []):
            path = visual.get("path")
            if path:
                image_paths.append(Path(path))
        for sample in run.get("samples", []):
            for image in sample.get("images", []):
                path = image.get("path")
                if path:
                    image_paths.append(Path(path))
    compact_manifest = _compact_manifest_for_gpt(manifest)
    prompt = {
        "task": "Analyze trajectory failure or quality risks for AIC cable insertion.",
        "success_preferences": {
            "score_target": ">80",
            "final_distance_to_port_m": "<0.05",
        },
        "required_analysis": [
            "Use the score breakdown from manifest.score.scoring to explain which score components should improve next.",
            "Use manifest.container_scoring_context and each run.container_scoring_context for engine/container-returned scoring messages and log excerpts.",
            "Use derived per-sample geometry fields instead of estimating coordinates only from images.",
            "If manifest.runs is present, compare all provided runs. Explain why a lower-scoring loop regressed relative to the higher-scoring loop.",
            "For each compared run, inspect the equidistant sampled timestamps, observation values, action values, and per-view images.",
            "Check whether VLM delta commands were well-formed for optimizer/replay.",
            "Compare phase labels, timestamps, TCP poses, actions if available, and images.",
            "Identify likely bugs in relative pose or coordinate calculations.",
            "Suggest concrete next trajectory/planner changes.",
        ],
        "manifest": compact_manifest,
    }
    unique_image_paths = list(dict.fromkeys(path for path in image_paths if path.exists()))
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        timeout=request_timeout_sec,
        instructions="You are a robotics failure-analysis reviewer. Output JSON only.",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": json.dumps(prompt, indent=2)},
                    *_image_items(unique_image_paths),
                ],
            }
        ],
    )
    text = _extract_output_text(response)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {"raw_text": text}
    parsed.setdefault("diagnostics", {})
    parsed["diagnostics"].update(
        {
            "model": model,
            "api_calls_used": 1,
            "image_count": len(unique_image_paths),
        }
    )
    return parsed
