"""Parity tooling for comparing rollout traces."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from aic_gym_gz.reward import AicScoreCalculator
else:
    from .reward import AicScoreCalculator


@dataclass(frozen=True)
class ParityMetric:
    name: str
    max_abs_error: float
    mean_abs_error: float


def _metric(name: str, reference: np.ndarray, candidate: np.ndarray) -> ParityMetric:
    diff = np.abs(reference - candidate)
    return ParityMetric(
        name=name,
        max_abs_error=float(diff.max(initial=0.0)),
        mean_abs_error=float(diff.mean() if diff.size else 0.0),
    )


class AicParityHarness:
    """Compares key trajectories from rollout logs."""

    def compare_rollouts(
        self,
        *,
        reference_steps: list[dict[str, Any]],
        candidate_steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if len(reference_steps) != len(candidate_steps):
            raise ValueError("Reference and candidate rollouts must have the same length.")
        metrics = []
        for key in ("tcp_x", "tcp_y", "tcp_z", "plug_x", "plug_y", "plug_z"):
            ref = np.array([row[key] for row in reference_steps], dtype=np.float64)
            cand = np.array([row[key] for row in candidate_steps], dtype=np.float64)
            metrics.append(_metric(key, ref, cand).__dict__)
        return {
            "num_steps": len(reference_steps),
            "metrics": metrics,
            "final_task_classification_match": (
                reference_steps[-1].get("classification") == candidate_steps[-1].get("classification")
                if reference_steps
                else True
            ),
        }

    def compare_csv_files(self, reference_csv: Path | str, candidate_csv: Path | str) -> dict[str, Any]:
        reference = self._read_csv(reference_csv)
        candidate = self._read_csv(candidate_csv)
        return self.compare_rollouts(reference_steps=reference, candidate_steps=candidate)

    def compare_trace_json(
        self,
        *,
        reference_report: dict[str, Any],
        candidate_report: dict[str, Any],
    ) -> dict[str, Any]:
        reference_steps = self._trace_steps(reference_report)
        candidate_steps = self._trace_steps(candidate_report)
        if len(reference_steps) != len(candidate_steps):
            raise ValueError("Reference and candidate trace reports must have the same step count.")
        metrics: list[dict[str, Any]] = []
        for key in (
            "tcp_x",
            "tcp_y",
            "tcp_z",
            "plug_x",
            "plug_y",
            "plug_z",
            "target_x",
            "target_y",
            "target_z",
            "distance_to_target",
            "orientation_error",
        ):
            ref = np.array([row[key] for row in reference_steps], dtype=np.float64)
            cand = np.array([row[key] for row in candidate_steps], dtype=np.float64)
            metrics.append(_metric(key, ref, cand).__dict__)
        joint_metrics = self._joint_metrics(reference_steps, candidate_steps)
        metrics.extend(joint_metrics)
        return {
            "num_steps": len(reference_steps),
            "metrics": metrics,
            "final_success_like_match": (
                reference_steps[-1]["success_like"] == candidate_steps[-1]["success_like"]
                if reference_steps
                else True
            ),
        }

    def compare_score_json(
        self,
        *,
        reference_report: dict[str, Any],
        candidate_report: dict[str, Any],
    ) -> dict[str, Any]:
        calculator = AicScoreCalculator()
        reference_summary = calculator.evaluate(self._episode_from_trace_report(reference_report))
        candidate_summary = calculator.evaluate(self._episode_from_trace_report(candidate_report))
        tier2_keys = sorted(set(reference_summary.tier2) | set(candidate_summary.tier2))
        tier2_deltas = {
            key: {
                "reference": float(reference_summary.tier2.get(key, 0.0)),
                "candidate": float(candidate_summary.tier2.get(key, 0.0)),
                "abs_error": abs(
                    float(reference_summary.tier2.get(key, 0.0))
                    - float(candidate_summary.tier2.get(key, 0.0))
                ),
            }
            for key in tier2_keys
        }
        reference_tier3 = float(reference_summary.tier3.get("score", 0.0))
        candidate_tier3 = float(candidate_summary.tier3.get("score", 0.0))
        return {
            "reference": {
                "score_label": "gym_final_score",
                "gym_reward": reference_summary.total_score,
                "gym_final_score": reference_summary.total_score,
                "official_eval_score": None,
                "tier2": reference_summary.tier2,
                "tier3": reference_summary.tier3,
                "total_score": reference_summary.total_score,
                "message": reference_summary.message,
            },
            "candidate": {
                "score_label": "gym_final_score",
                "gym_reward": candidate_summary.total_score,
                "gym_final_score": candidate_summary.total_score,
                "official_eval_score": None,
                "tier2": candidate_summary.tier2,
                "tier3": candidate_summary.tier3,
                "total_score": candidate_summary.total_score,
                "message": candidate_summary.message,
            },
            "deltas": {
                "tier2": tier2_deltas,
                "tier3_score_abs_error": abs(reference_tier3 - candidate_tier3),
                "total_score_abs_error": abs(
                    float(reference_summary.total_score) - float(candidate_summary.total_score)
                ),
                "message_match": reference_summary.message == candidate_summary.message,
            },
            "approximation_notes": [
                "This report uses the local `gym_final_score` / `gym_reward` path on both traces rather than invoking the official toolkit.",
                "If you need `official_eval_score`, run the official `aic_scoring` / toolkit path directly against the ROS/Gazebo rollout.",
                "Force and off-limit contact terms are limited to what the current live trace schema records.",
            ],
        }

    def compare_image_trace_json(
        self,
        *,
        reference_report: dict[str, Any],
        candidate_report: dict[str, Any],
    ) -> dict[str, Any]:
        reference_records = reference_report.get("records", [])
        candidate_records = candidate_report.get("records", [])
        if len(reference_records) != len(candidate_records):
            raise ValueError("Reference and candidate image traces must have the same step count.")
        cameras = ("left", "center", "right")
        per_camera: dict[str, dict[str, Any]] = {}
        for camera in cameras:
            ref_pixels: list[float] = []
            cand_pixels: list[float] = []
            ref_timestamps: list[float] = []
            cand_timestamps: list[float] = []
            ref_present = True
            cand_present = True
            ref_shape = None
            cand_shape = None
            for reference, candidate in zip(reference_records, candidate_records):
                ref_image = (reference.get("images") or {}).get(camera)
                cand_image = (candidate.get("images") or {}).get(camera)
                if not isinstance(ref_image, dict):
                    ref_present = False
                    continue
                if not isinstance(cand_image, dict):
                    cand_present = False
                    continue
                ref_pixels.append(float(ref_image.get("pixel_sum", 0.0)))
                cand_pixels.append(float(cand_image.get("pixel_sum", 0.0)))
                ref_timestamps.append(float(ref_image.get("timestamp", 0.0)))
                cand_timestamps.append(float(cand_image.get("timestamp", 0.0)))
                ref_shape = ref_image.get("shape")
                cand_shape = cand_image.get("shape")
                ref_present = ref_present and bool(ref_image.get("present", False))
                cand_present = cand_present and bool(cand_image.get("present", False))
            metric = (
                _metric(
                    f"{camera}_pixel_sum",
                    np.asarray(ref_pixels, dtype=np.float64),
                    np.asarray(cand_pixels, dtype=np.float64),
                ).__dict__
                if ref_pixels and cand_pixels
                else None
            )
            per_camera[camera] = {
                "reference_present_all_steps": ref_present,
                "candidate_present_all_steps": cand_present,
                "reference_shape": ref_shape,
                "candidate_shape": cand_shape,
                "reference_timestamp_monotonic": self._is_non_decreasing(ref_timestamps),
                "candidate_timestamp_monotonic": self._is_non_decreasing(cand_timestamps),
                "pixel_sum_metric": metric,
            }
        return {
            "num_steps": len(reference_records),
            "cameras": per_camera,
            "notes": [
                "Absolute timestamps are not compared across runs because official and candidate traces come from different fresh launches.",
                "Image parity checks presence, declared shape, monotonic timestamps, and coarse pixel statistics.",
            ],
        }

    def analyze_official_trace(self, report: dict[str, Any]) -> dict[str, Any]:
        steps = self._trace_steps(report)
        if not steps:
            return {"num_steps": 0, "metrics": [], "notes": ["Trace contains no rollout steps."]}

        native_tcp = np.array([[step["tcp_x"], step["tcp_y"], step["tcp_z"]] for step in steps], dtype=np.float64)
        native_joint = np.array([step["joint_positions"] for step in steps], dtype=np.float64)
        native_delta = native_tcp - native_tcp[:1]
        native_joint_delta = native_joint - native_joint[:1]

        controller_positions = []
        for step in steps:
            controller = step.get("controller_tcp_position")
            if controller is None:
                controller_positions = []
                break
            controller_positions.append(controller)
        controller_delta_metrics: list[dict[str, Any]] = []
        if controller_positions:
            controller_tcp = np.asarray(controller_positions, dtype=np.float64)
            controller_delta = controller_tcp - controller_tcp[:1]
            for axis, name in enumerate(("tcp_delta_x", "tcp_delta_y", "tcp_delta_z")):
                controller_delta_metrics.append(
                    _metric(name, native_delta[:, axis], controller_delta[:, axis]).__dict__
                )

        total_tcp_delta = native_tcp[-1] - native_tcp[0]
        joint_max_displacement = np.max(np.abs(native_joint_delta), axis=0)
        return {
            "num_steps": len(steps),
            "metrics": controller_delta_metrics,
            "summary": {
                "native_tcp_total_delta": total_tcp_delta.tolist(),
                "native_joint_max_displacement": joint_max_displacement.tolist(),
                "final_distance_to_target": float(steps[-1]["distance_to_target"]),
                "final_success_like": bool(steps[-1]["success_like"]),
            },
            "notes": [
                "Controller TCP positions and native Gazebo TCP positions are reported in different frames, so this analysis compares only per-rollout deltas.",
                "Joint displacement is reported from the native Gazebo trace because controller-state joint references are not currently recorded.",
            ],
        }

    def load_trace_report(self, path: Path | str) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def write_report(
        self,
        report: dict[str, Any],
        *,
        output_json: Path | str,
    ) -> None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    def _read_csv(self, path: Path | str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                parsed: dict[str, Any] = {}
                for key, value in row.items():
                    try:
                        parsed[key] = float(value)
                    except (TypeError, ValueError):
                        parsed[key] = value
                rows.append(parsed)
        return rows

    def _trace_steps(self, report: dict[str, Any]) -> list[dict[str, Any]]:
        parsed: list[dict[str, Any]] = []
        for item in report.get("records", []):
            native = item.get("native") or {}
            tcp = native.get("tcp_position") or [math.nan, math.nan, math.nan]
            plug = native.get("plug_position") or [math.nan, math.nan, math.nan]
            target = native.get("target_position") or [math.nan, math.nan, math.nan]
            parsed.append(
                {
                    "step_idx": int(item.get("step_idx", len(parsed))),
                    "tcp_x": float(tcp[0]),
                    "tcp_y": float(tcp[1]),
                    "tcp_z": float(tcp[2]),
                    "plug_x": float(plug[0]),
                    "plug_y": float(plug[1]),
                    "plug_z": float(plug[2]),
                    "target_x": float(target[0]),
                    "target_y": float(target[1]),
                    "target_z": float(target[2]),
                    "distance_to_target": float(native.get("distance_to_target", math.nan)),
                    "orientation_error": float(native.get("orientation_error", math.nan)),
                    "success_like": bool(native.get("success_like", False)),
                    "sim_time": (
                        float(native["sim_time"])
                        if isinstance(native.get("sim_time"), (int, float))
                        else math.nan
                    ),
                    "force_magnitude": float(native.get("force_magnitude", 0.0)),
                    "off_limit_contact": bool(native.get("off_limit_contact", False)),
                    "joint_positions": list(native.get("joint_positions") or []),
                    "controller_tcp_position": native.get("controller_tcp_position"),
                }
            )
        return parsed

    def _episode_from_trace_report(self, report: dict[str, Any]) -> dict[str, Any]:
        steps = self._trace_steps(report)
        if not steps:
            raise ValueError("Trace report does not contain rollout records.")

        initial_native = report.get("initial_native") or {}
        initial_distance_value = initial_native.get("distance_to_target")
        initial_distance = (
            float(initial_distance_value)
            if isinstance(initial_distance_value, (int, float))
            else float(steps[0]["distance_to_target"])
        )
        sim_time = [float(step["sim_time"]) for step in steps]
        if any(math.isnan(value) for value in sim_time) or not self._is_strictly_increasing(sim_time):
            sim_time = self._fallback_sim_time(report, steps)
        tcp_positions = [
            np.array([step["tcp_x"], step["tcp_y"], step["tcp_z"]], dtype=np.float64)
            for step in steps
        ]
        return {
            "initial_distance": initial_distance,
            "sim_time": sim_time,
            "tcp_positions": tcp_positions,
            "tcp_linear_velocity": self._finite_difference_velocities(tcp_positions, sim_time),
            "distances": [float(step["distance_to_target"]) for step in steps],
            "force_magnitudes": [float(step["force_magnitude"]) for step in steps],
            "off_limit_contacts": [bool(step["off_limit_contact"]) for step in steps],
            "success": bool(steps[-1]["success_like"]),
            "wrong_port": False,
        }

    def _fallback_sim_time(
        self,
        report: dict[str, Any],
        steps: list[dict[str, Any]],
    ) -> list[float]:
        del steps
        fallback: list[float] = []
        elapsed = 0.0
        sim_dt = 0.001
        for record in report.get("records", []):
            action = record.get("action") or {}
            ticks = action.get("sim_steps")
            if isinstance(ticks, int):
                elapsed += float(ticks) * sim_dt
            else:
                elapsed += 1.0
            fallback.append(elapsed)
        return fallback

    def _is_strictly_increasing(self, values: list[float]) -> bool:
        return all(current > previous for previous, current in zip(values, values[1:]))

    def _is_non_decreasing(self, values: list[float]) -> bool:
        return all(current >= previous for previous, current in zip(values, values[1:]))

    def _finite_difference_velocities(
        self,
        tcp_positions: list[np.ndarray],
        sim_time: list[float],
    ) -> list[np.ndarray]:
        velocities: list[np.ndarray] = [np.zeros(3, dtype=np.float64)]
        for index in range(1, len(tcp_positions)):
            dt = sim_time[index] - sim_time[index - 1]
            if dt <= 0.0:
                velocities.append(np.zeros(3, dtype=np.float64))
            else:
                velocities.append((tcp_positions[index] - tcp_positions[index - 1]) / dt)
        return velocities

    def _joint_metrics(
        self,
        reference_steps: list[dict[str, Any]],
        candidate_steps: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        reference_joint_count = len(reference_steps[0]["joint_positions"]) if reference_steps else 0
        candidate_joint_count = len(candidate_steps[0]["joint_positions"]) if candidate_steps else 0
        joint_count = min(reference_joint_count, candidate_joint_count)
        metrics: list[dict[str, Any]] = []
        for index in range(joint_count):
            ref = np.array([row["joint_positions"][index] for row in reference_steps], dtype=np.float64)
            cand = np.array([row["joint_positions"][index] for row in candidate_steps], dtype=np.float64)
            metrics.append(_metric(f"joint_{index}", ref, cand).__dict__)
        return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-trace", type=str, default=None)
    parser.add_argument("--candidate-trace", type=str, default=None)
    parser.add_argument("--analyze-trace", type=str, default=None)
    parser.add_argument("--compare-scores", action="store_true")
    parser.add_argument("--compare-images", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    harness = AicParityHarness()
    if args.analyze_trace:
        report = harness.analyze_official_trace(harness.load_trace_report(args.analyze_trace))
    elif args.compare_scores and args.reference_trace and args.candidate_trace:
        report = harness.compare_score_json(
            reference_report=harness.load_trace_report(args.reference_trace),
            candidate_report=harness.load_trace_report(args.candidate_trace),
        )
    elif args.compare_images and args.reference_trace and args.candidate_trace:
        report = harness.compare_image_trace_json(
            reference_report=harness.load_trace_report(args.reference_trace),
            candidate_report=harness.load_trace_report(args.candidate_trace),
        )
    elif args.reference_trace and args.candidate_trace:
        report = harness.compare_trace_json(
            reference_report=harness.load_trace_report(args.reference_trace),
            candidate_report=harness.load_trace_report(args.candidate_trace),
        )
    else:
        raise SystemExit(
            "Usage: parity.py --analyze-trace TRACE.json [--output REPORT.json] "
            "or --reference-trace REF.json --candidate-trace CAND.json [--compare-scores] [--output REPORT.json]"
        )

    if args.output:
        harness.write_report(report, output_json=args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
