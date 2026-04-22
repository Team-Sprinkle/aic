"""Run the fixed deterministic parity gate against official and live paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from aic_gym_gz.official_trace import FixedVelocityAction, capture_official_and_native_trace
    from aic_gym_gz.parity import AicParityHarness
    from aic_gym_gz.policies import deterministic_policy_actions
    from aic_gym_gz.replay_trace import replay_trace_against_attached_runtime
    from aic_gym_gz.utils import to_jsonable
else:
    from .official_trace import FixedVelocityAction, capture_official_and_native_trace
    from .parity import AicParityHarness
    from .policies import deterministic_policy_actions
    from .replay_trace import replay_trace_against_attached_runtime
    from .utils import to_jsonable


def _official_actions() -> tuple[FixedVelocityAction, ...]:
    return tuple(
        FixedVelocityAction(
            linear_xyz=action.linear_xyz,
            angular_xyz=action.angular_xyz,
            frame_id=action.frame_id,
            sim_steps=action.sim_steps,
        )
        for action in deterministic_policy_actions()
    )


def _within_tolerance(metrics: list[dict[str, Any]], *, default_tol: float, joint_tol: float) -> bool:
    for metric in metrics:
        name = str(metric.get("name"))
        value = float(metric.get("max_abs_error", 0.0))
        limit = joint_tol if name.startswith("joint_") else default_tol
        if value > limit:
            return False
    return True


def run_deterministic_policy_parity(*, include_images: bool = False) -> dict[str, Any]:
    harness = AicParityHarness()
    official = capture_official_and_native_trace(
        actions=_official_actions(),
        include_images=include_images,
    )
    candidate = replay_trace_against_attached_runtime(
        trace_report=official,
        include_images=include_images,
    )
    state = harness.compare_trace_json(reference_report=official, candidate_report=candidate)
    scores = harness.compare_score_json(reference_report=official, candidate_report=candidate)
    images = (
        harness.compare_image_trace_json(reference_report=official, candidate_report=candidate)
        if include_images
        else None
    )
    state_pass = bool(state.get("final_success_like_match")) and _within_tolerance(
        state.get("metrics", []),
        default_tol=1e-3 if not include_images else 2e-4,
        joint_tol=1e-3,
    )
    score_pass = (
        float(scores["deltas"]["total_score_abs_error"]) == 0.0
        and bool(scores["deltas"]["message_match"])
    )
    image_pass = True
    if images is not None:
        image_pass = all(
            bool(camera_report.get("reference_present_all_steps"))
            and bool(camera_report.get("candidate_present_all_steps"))
            and bool(camera_report.get("reference_timestamp_monotonic"))
            and bool(camera_report.get("candidate_timestamp_monotonic"))
            for camera_report in images["cameras"].values()
        )
    return {
        "mode": "image" if include_images else "state_only",
        "passed": state_pass and score_pass and image_pass,
        "state_pass": state_pass,
        "score_pass": score_pass,
        "image_pass": image_pass,
        "official": official,
        "candidate": candidate,
        "parity": {
            "state": state,
            "scores": scores,
            "images": images,
        },
        "tolerances": {
            "state_default_max_abs_error": 1e-3 if not include_images else 2e-4,
            "joint_max_abs_error": 1e-3,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    report = run_deterministic_policy_parity(include_images=args.include_images)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "official_trace.json").write_text(
            json.dumps(to_jsonable(report["official"]), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "candidate_trace.json").write_text(
            json.dumps(to_jsonable(report["candidate"]), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "parity_report.json").write_text(
            json.dumps(to_jsonable(report["parity"]), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "summary.json").write_text(
            json.dumps(
                to_jsonable(
                    {
                        key: value
                        for key, value in report.items()
                        if key not in ("official", "candidate", "parity")
                    }
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
