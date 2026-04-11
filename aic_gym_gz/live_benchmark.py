"""Benchmark the live official-control path against the live aic_gym_gz path.

This runs inside the ROS/Gazebo container and reuses the same fixed-rollout
trace and replay paths that were used for parity. The reset metric for the
official path is intentionally a readiness surrogate rather than a simulator
reset latency because the current official bringup is unstable under reset on
this machine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from aic_gym_gz.official_trace import capture_official_and_native_trace
    from aic_gym_gz.parity import AicParityHarness
    from aic_gym_gz.replay_trace import replay_trace_against_attached_runtime
else:
    from .official_trace import capture_official_and_native_trace
    from .parity import AicParityHarness
    from .replay_trace import replay_trace_against_attached_runtime


def _summary_from_timing(trace: dict[str, Any]) -> dict[str, Any]:
    timing = trace.get("timing") or {}
    return {
        "ready_to_first_sane_state_latency_s": timing.get("ready_to_first_sane_state_latency_s"),
        "mean_step_latency_s": timing.get("mean_step_latency_s"),
        "simulated_seconds_per_wall_second": timing.get("simulated_seconds_per_wall_second"),
        "samples_per_second": timing.get("samples_per_second"),
        "total_wall_s": timing.get("total_wall_s"),
        "simulated_seconds": timing.get("simulated_seconds"),
        "num_steps": trace.get("num_steps", len(trace.get("records", []))),
    }


def _speedup(numerator: Any, denominator: Any) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if denominator == 0.0:
        return None
    return float(numerator) / float(denominator)


def run_live_benchmark(*, include_images: bool = False) -> dict[str, Any]:
    official = capture_official_and_native_trace(include_images=include_images)
    candidate = replay_trace_against_attached_runtime(
        trace_report=official,
        include_images=include_images,
    )
    harness = AicParityHarness()
    state_parity = harness.compare_trace_json(reference_report=official, candidate_report=candidate)
    image_parity = (
        harness.compare_image_trace_json(reference_report=official, candidate_report=candidate)
        if include_images
        else None
    )
    score_parity = harness.compare_score_json(reference_report=official, candidate_report=candidate)

    official_summary = _summary_from_timing(official)
    candidate_summary = _summary_from_timing(candidate)
    comparison = {
        "reset_latency_speedup": _speedup(
            official_summary.get("ready_to_first_sane_state_latency_s"),
            candidate_summary.get("ready_to_first_sane_state_latency_s"),
        ),
        "step_latency_speedup": _speedup(
            official_summary.get("mean_step_latency_s"),
            candidate_summary.get("mean_step_latency_s"),
        ),
        "simulated_seconds_per_wall_second_speedup": _speedup(
            candidate_summary.get("simulated_seconds_per_wall_second"),
            official_summary.get("simulated_seconds_per_wall_second"),
        ),
        "samples_per_second_speedup": _speedup(
            candidate_summary.get("samples_per_second"),
            official_summary.get("samples_per_second"),
        ),
    }
    notes = [
        "The official reset metric is a ready-to-first-sane-state surrogate, not /gz_server/reset_simulation latency, because reset is unstable in the official bringup on this machine.",
        "The aic_gym_gz benchmark path is the attached live replay path against the same fixed rollout, which isolates hot-loop overhead on the same machine and same world.",
        "Scaling beyond one environment is not measured here because the current exact-step parity harness is single-world and the official control path is not multi-env capable.",
    ]
    return {
        "mode": "image" if include_images else "state_only",
        "official_path": official_summary,
        "aic_gym_gz_path": candidate_summary,
        "comparison": comparison,
        "parity": {
            "state": state_parity,
            "images": image_parity,
            "scores": score_parity,
        },
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    payload = run_live_benchmark(include_images=args.include_images)
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
