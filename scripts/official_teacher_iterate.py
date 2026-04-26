#!/usr/bin/env python3
"""Iterate official teacher planning, replay, recording, and GPT-5 feedback."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aic_teacher_official"))

from aic_teacher_official.context import OfficialTeacherContext
from aic_teacher_official.generate_piecewise import (
    PiecewiseGeneratorConfig,
    generate_piecewise_file,
)
from aic_teacher_official.iteration import (
    build_recording_command,
    loop_roots,
    parse_total_score,
    shell_join,
    write_loop_manifest,
)
from aic_teacher_official.postprocess import postprocess_file
from aic_teacher_official.review import (
    build_comparison_review_bundle,
    build_review_bundle,
    call_gpt5_failure_review,
)
from aic_teacher_official.vlm_planner import call_gpt5_mini_delta_planner

from official_teacher_generate_piecewise import _vector


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        import yaml
    except Exception:
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _trial_scoring(scoring: dict[str, Any] | None) -> dict[str, Any] | None:
    if not scoring:
        return None
    for key, value in scoring.items():
        if key.startswith("trial_") and isinstance(value, dict):
            return value
    return None


def _has_tier3_success(scoring_path: Path) -> bool:
    trial = _trial_scoring(_load_yaml(scoring_path))
    tier3 = trial.get("tier_3", {}) if trial else {}
    message = str(tier3.get("message", "")).lower()
    return "successful" in message and float(tier3.get("score", 0.0)) >= 75.0


def _tier2_categories(scoring_path: Path) -> dict[str, Any]:
    trial = _trial_scoring(_load_yaml(scoring_path))
    tier2 = trial.get("tier_2", {}) if trial else {}
    categories = tier2.get("categories", {})
    return categories if isinstance(categories, dict) else {}


def _best_successful_prior(args: argparse.Namespace, loop_index: int):
    best = None
    for prior_loop in range(1, loop_index):
        roots = loop_roots(
            root_dir=args.root_dir,
            task_family=args.task_family,
            scene_count_label=args.scene_count_label,
            attempt_label=args.attempt_label,
            base_run_name=args.base_run_name,
            loop_index=prior_loop,
        )
        score = parse_total_score(roots.scoring_path)
        if score is None or not roots.piecewise_path.exists() or not _has_tier3_success(roots.scoring_path):
            continue
        if best is None or score > best["score"]:
            best = {"loop_index": prior_loop, "roots": roots, "score": score}
    return best


def _retime_piecewise(
    piecewise: dict[str, Any],
    *,
    final_scale: float = 1.0,
    nonfinal_scale: float = 1.0,
) -> dict[str, Any]:
    waypoints = piecewise["waypoints"]
    old_times = [float(w["timestamp"]) for w in waypoints]
    new_times = [0.0]
    for index, (prev, curr) in enumerate(zip(waypoints, waypoints[1:]), start=1):
        duration = old_times[index] - old_times[index - 1]
        is_final_segment = curr.get("phase") == "final_insertion"
        scale = final_scale if is_final_segment else nonfinal_scale
        new_times.append(new_times[-1] + max(0.25, duration * scale))
    for waypoint, timestamp in zip(waypoints, new_times):
        waypoint["timestamp"] = float(timestamp)
    return piecewise


def _write_conservative_success_piecewise(
    *,
    reference_piecewise_path: Path,
    reference_scoring_path: Path,
    output_path: Path,
    planner_feedback: dict[str, Any] | None,
    vlm_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    piecewise = json.loads(reference_piecewise_path.read_text(encoding="utf-8"))
    categories = _tier2_categories(reference_scoring_path)
    force_message = str(categories.get("insertion force", {}).get("message", "")).lower()
    duration_score = float(categories.get("duration", {}).get("score", 0.0) or 0.0)
    smoothness_score = float(categories.get("trajectory smoothness", {}).get("score", 0.0) or 0.0)

    final_scale = 1.0
    nonfinal_scale = 1.0
    changes: list[str] = []
    if "above 20" in force_message or "excessive force" in force_message:
        final_scale = 1.03
        changes.append("final insertion timing increased by 3% due to insertion-force scorer message")
    elif duration_score < 10.0:
        nonfinal_scale = 0.97
        changes.append("non-final timing reduced by 3% due to duration score below 10")
    elif smoothness_score < 4.5:
        final_scale = 1.02
        changes.append("final insertion timing increased by 2% due to smoothness score below 4.5")
    else:
        changes.append("no timing change; prior successful geometry and timing preserved")

    piecewise = _retime_piecewise(
        piecewise,
        final_scale=final_scale,
        nonfinal_scale=nonfinal_scale,
    )
    metadata = piecewise.setdefault("metadata", {})
    planning = metadata.setdefault("planning", {})
    planning["method"] = "conservative_success_retime_v0"
    planning["conservative_success_guard"] = {
        "reference_piecewise": str(reference_piecewise_path),
        "reference_scoring": str(reference_scoring_path),
        "policy": (
            "prior official tier_3 insertion success locks all TCP poses and "
            "final insertion geometry; only small scorer-targeted timing changes are allowed"
        ),
        "final_scale": final_scale,
        "nonfinal_scale": nonfinal_scale,
        "changes": changes,
        "vlm_plan_recorded_but_geometry_ignored": vlm_plan,
        "planner_feedback": planner_feedback,
    }
    for waypoint in piecewise["waypoints"]:
        diagnostics = waypoint.setdefault("diagnostics", {})
        diagnostics["conservative_success_guard"] = True
        diagnostics["geometry_locked_from_successful_prior"] = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(piecewise, indent=2) + "\n", encoding="utf-8")
    return planning["conservative_success_guard"]


def _planner_images_from_review(manifest: dict[str, Any] | None, limit: int = 8) -> list[Path]:
    if not manifest:
        return []
    image_paths = [
        Path(path)
        for path in [
            *manifest.get("images", {}).get("wrist", []),
            *manifest.get("images", {}).get("gazebo", []),
        ]
    ]
    return [path for path in image_paths if path.exists()][:limit]


def _score_meets_threshold(score: float | None, threshold: float) -> bool:
    return score is not None and score >= threshold


def _call_with_timeout(fn, *, timeout_sec: float, label: str):
    def _target(queue):
        try:
            queue.put(("ok", fn()))
        except BaseException as ex:
            queue.put(("error", type(ex).__name__, str(ex)))

    queue = mp.Queue(maxsize=1)
    process = mp.Process(target=_target, args=(queue,), daemon=True)
    process.start()
    process.join(timeout_sec)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        raise TimeoutError(f"{label} exceeded {timeout_sec:.1f}s")
    if queue.empty():
        if process.exitcode == 0:
            return None
        raise RuntimeError(f"{label} exited with code {process.exitcode}")
    status, *payload = queue.get()
    if status == "ok":
        return payload[0]
    error_type, message = payload
    raise RuntimeError(f"{label} failed with {error_type}: {message}")


def _review_runs_for_loop(args: argparse.Namespace, loop_index: int) -> list[dict[str, Any]]:
    previous = []
    for prior_loop in range(1, loop_index):
        roots = loop_roots(
            root_dir=args.root_dir,
            task_family=args.task_family,
            scene_count_label=args.scene_count_label,
            attempt_label=args.attempt_label,
            base_run_name=args.base_run_name,
            loop_index=prior_loop,
        )
        if not roots.smooth_path.exists():
            continue
        previous.append(
            {
                "loop_index": prior_loop,
                "label": f"loop_{prior_loop}",
                "trajectory_path": str(roots.smooth_path),
                "dataset_root": str(roots.dataset_root),
                "scoring_path": str(roots.scoring_path),
                "score": parse_total_score(roots.scoring_path),
            }
        )
    if not previous:
        return []
    selected = {previous[-1]["loop_index"]: previous[-1]}
    scored = [run for run in previous if run["score"] is not None]
    if scored:
        best = max(scored, key=lambda run: run["score"])
        selected[best["loop_index"]] = best
    return [selected[index] for index in sorted(selected)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default="outputs/trajectory_datasets")
    parser.add_argument("--task-family", default="sfp_to_nic")
    parser.add_argument("--scene-count-label", default="nic_cards_2")
    parser.add_argument("--attempt-label", default="n1")
    parser.add_argument(
        "--base-run-name",
        required=True,
        help="Stable trial timestamp label, e.g. trial9_2026_0425_205620.",
    )
    parser.add_argument("--engine-config", required=True)
    parser.add_argument("--context-json", help="Official oracle context JSON for same scene.")
    parser.add_argument("--seed-piecewise", help="Optional piecewise JSON to use for loop 1.")
    parser.add_argument("--start-loop", type=int, default=1)
    parser.add_argument("--max-loops", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=80.0)
    parser.add_argument("--force-all-loops", action="store_true")
    parser.add_argument(
        "--disable-conservative-success-edits",
        action="store_true",
        help=(
            "Allow VLM to rewrite geometry even after a prior official tier_3 "
            "insertion success. By default successful geometry is locked."
        ),
    )
    parser.add_argument(
        "--stop-if-not-improved",
        action="store_true",
        help="Stop after a recorded loop if its score does not exceed the best prior loop score.",
    )
    parser.add_argument("--use-vlm", action="store_true")
    parser.add_argument("--use-gpt5-review", action="store_true")
    parser.add_argument("--vlm-model", default="gpt-5-mini")
    parser.add_argument("--max-vlm-calls", type=int, default=20)
    parser.add_argument("--openai-timeout-sec", type=float, default=180.0)
    parser.add_argument("--review-samples", type=int, default=10)
    parser.add_argument("--sample-dt", type=float, default=0.05)
    parser.add_argument("--start-position", default="-0.35,0.35,0.32")
    parser.add_argument("--port-position", default="-0.10,0.45,0.12")
    parser.add_argument("--orientation-xyzw", default="1,0,0,0")
    parser.add_argument("--approach-offset", default="-0.08,-0.08,0.22")
    parser.add_argument("--sim-distrobox", default="aic_eval")
    parser.add_argument("--dataset-repo-prefix", default="local/official_teacher")
    parser.add_argument(
        "--action-mode",
        choices=["relative_delta_gripper_tcp", "absolute_cartesian_pose_base_link"],
        default="relative_delta_gripper_tcp",
    )
    parser.add_argument("--gazebo-gui", action="store_true")
    parser.add_argument("--launch-rviz", action="store_true")
    parser.add_argument("--startup-delay-sec", type=int, default=8)
    parser.add_argument("--per-trial-timeout-sec", type=int, default=0)
    parser.add_argument("--recorder-drain-sec", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    if args.max_loops < 1 or args.max_loops > 4:
        raise SystemExit("--max-loops must be in [1, 4]")
    if args.start_loop < 1 or args.start_loop > args.max_loops:
        raise SystemExit("--start-loop must be in [1, --max-loops]")
    if not args.dry_run and not args.record:
        raise SystemExit("Use --dry-run to print/build locally or --record to execute official replay.")

    context = OfficialTeacherContext.load_json(args.context_json) if args.context_json else None
    planner_feedback: dict[str, Any] | None = None
    previous_manifest: dict[str, Any] | None = None
    loop_results: list[dict[str, Any]] = []
    best_prior_score: float | None = None

    for loop_index in range(args.start_loop, args.max_loops + 1):
        roots = loop_roots(
            root_dir=args.root_dir,
            task_family=args.task_family,
            scene_count_label=args.scene_count_label,
            attempt_label=args.attempt_label,
            base_run_name=args.base_run_name,
            loop_index=loop_index,
        )
        roots.planner_root.mkdir(parents=True, exist_ok=True)
        roots.postprocessed_root.mkdir(parents=True, exist_ok=True)

        if loop_index > 1:
            prev = loop_roots(
                root_dir=args.root_dir,
                task_family=args.task_family,
                scene_count_label=args.scene_count_label,
                attempt_label=args.attempt_label,
                base_run_name=args.base_run_name,
                loop_index=loop_index - 1,
            )
            review_manifest_path = roots.planner_root / "previous_loop_review_bundle.json"
            if prev.smooth_path.exists():
                review_runs = _review_runs_for_loop(args, loop_index)
                if len(review_runs) > 1:
                    previous_manifest = build_comparison_review_bundle(
                        review_runs,
                        review_manifest_path,
                        samples=args.review_samples,
                    )
                else:
                    previous_manifest = build_review_bundle(
                        prev.smooth_path,
                        review_manifest_path,
                        dataset_root=prev.dataset_root,
                        scoring_path=prev.scoring_path,
                        samples=args.review_samples,
                    )
                if args.use_gpt5_review and not args.dry_run:
                    try:
                        planner_feedback = _call_with_timeout(
                            lambda: call_gpt5_failure_review(
                                previous_manifest,
                                request_timeout_sec=args.openai_timeout_sec,
                            ),
                            timeout_sec=args.openai_timeout_sec + 10.0,
                            label="GPT-5 failure review",
                        )
                        (roots.planner_root / "previous_loop_gpt5_review.json").write_text(
                            json.dumps(planner_feedback, indent=2) + "\n",
                            encoding="utf-8",
                        )
                    except Exception as ex:
                        error = {
                            "review_bundle": previous_manifest,
                            "api_error": {
                                "stage": "gpt5_failure_review",
                                "type": type(ex).__name__,
                                "message": str(ex),
                            },
                            "note": (
                                "GPT-5 review failed. Exiting because iterative improvement "
                                "requires VLM feedback, including the score breakdown."
                            ),
                        }
                        (roots.planner_root / "previous_loop_gpt5_review_error.json").write_text(
                            json.dumps(error, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        raise SystemExit(
                            "GPT-5 failure review failed. The loop requires VLM feedback; "
                            f"see {roots.planner_root / 'previous_loop_gpt5_review_error.json'}"
                        ) from ex
                else:
                    planner_feedback = {
                        "review_bundle": previous_manifest,
                        "note": (
                            "GPT-5 review not called. Use --use-gpt5-review without --dry-run "
                            "to request one failure-analysis call."
                        ),
                    }
            else:
                previous_manifest = None
                planner_feedback = None

        vlm_plan = None
        if args.use_vlm and not args.dry_run:
            vlm_context = context or OfficialTeacherContext(
                start_position=_vector(args.start_position, length=3, name="--start-position"),
                port_position=_vector(args.port_position, length=3, name="--port-position"),
                orientation_xyzw=_vector(args.orientation_xyzw, length=4, name="--orientation-xyzw"),
                diagnostics={"source": "explicit_cli"},
            )
            try:
                vlm_plan = _call_with_timeout(
                    lambda: call_gpt5_mini_delta_planner(
                        vlm_context,
                        image_paths=_planner_images_from_review(previous_manifest),
                        planner_feedback=planner_feedback,
                        max_calls=args.max_vlm_calls,
                        model=args.vlm_model,
                        request_timeout_sec=args.openai_timeout_sec,
                    ),
                    timeout_sec=args.openai_timeout_sec + 10.0,
                    label="GPT-5 mini planner",
                )
            except Exception as ex:
                error = {
                    "planner_feedback": planner_feedback,
                    "api_error": {
                        "stage": "gpt5_mini_delta_planner",
                        "type": type(ex).__name__,
                        "message": str(ex),
                    },
                    "note": (
                        "GPT-5 mini planning failed. Exiting because iterative improvement "
                        "requires a fresh VLM plan."
                    ),
                }
                (roots.planner_root / "vlm_planner_error.json").write_text(
                    json.dumps(error, indent=2) + "\n",
                    encoding="utf-8",
                )
                raise SystemExit(
                    "GPT-5 mini planner failed. The loop requires a VLM plan; "
                    f"see {roots.planner_root / 'vlm_planner_error.json'}"
                ) from ex

        if loop_index == 1 and args.seed_piecewise:
            shutil.copyfile(args.seed_piecewise, roots.piecewise_path)
        else:
            best_success = (
                None
                if args.disable_conservative_success_edits
                else _best_successful_prior(args, loop_index)
            )
            if best_success is not None:
                guard = _write_conservative_success_piecewise(
                    reference_piecewise_path=best_success["roots"].piecewise_path,
                    reference_scoring_path=best_success["roots"].scoring_path,
                    output_path=roots.piecewise_path,
                    planner_feedback=planner_feedback,
                    vlm_plan=vlm_plan,
                )
                (roots.planner_root / "conservative_success_guard.json").write_text(
                    json.dumps(
                        {
                            "reference_loop": best_success["loop_index"],
                            "reference_score": best_success["score"],
                            **guard,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                config = PiecewiseGeneratorConfig(
                    start_position=_vector(args.start_position, length=3, name="--start-position"),
                    port_position=_vector(args.port_position, length=3, name="--port-position"),
                    orientation_xyzw=_vector(args.orientation_xyzw, length=4, name="--orientation-xyzw"),
                    approach_offset=_vector(args.approach_offset, length=3, name="--approach-offset"),
                    context=context,
                    vlm_delta_plan=vlm_plan,
                    planner_feedback=planner_feedback,
                )
                generate_piecewise_file(config, roots.piecewise_path)

        postprocess_file(roots.piecewise_path, roots.smooth_path, args.sample_dt)
        if Path(args.engine_config).exists():
            shutil.copyfile(args.engine_config, roots.planner_root / "engine_config.yaml")

        record_cmd = build_recording_command(
            engine_config=args.engine_config,
            sim_distrobox=args.sim_distrobox,
            smooth_path=roots.smooth_path,
            dataset_repo_id=f"{args.dataset_repo_prefix}_{args.base_run_name}_loop_{loop_index}",
            dataset_root=roots.dataset_root,
            scores_root=roots.scores_root,
            tmp_dir=roots.postprocessed_root / "logs" / "per_trial_tmp",
            action_mode=args.action_mode,
            gazebo_gui=args.gazebo_gui,
            launch_rviz=args.launch_rviz,
            startup_delay_sec=args.startup_delay_sec,
            per_trial_timeout_sec=args.per_trial_timeout_sec,
            recorder_drain_sec=args.recorder_drain_sec,
        )
        score = parse_total_score(roots.scoring_path)
        if args.record:
            exit_code = subprocess.call(record_cmd)
            if exit_code != 0:
                raise SystemExit(exit_code)
            score = parse_total_score(roots.scoring_path)

        loop_record = {
            "loop_index": loop_index,
            "planner_root": str(roots.planner_root),
            "postprocessed_root": str(roots.postprocessed_root),
            "piecewise": str(roots.piecewise_path),
            "smooth": str(roots.smooth_path),
            "dataset_root": str(roots.dataset_root),
            "scoring_path": str(roots.scoring_path),
            "score": score,
            "record_command": shell_join(record_cmd),
            "threshold_met": _score_meets_threshold(score, args.score_threshold),
            "best_prior_score": best_prior_score,
            "improved_over_prior_best": (
                None if score is None or best_prior_score is None else score > best_prior_score
            ),
        }
        write_loop_manifest(roots.postprocessed_root / "loop_manifest.json", loop_record)
        loop_results.append(loop_record)
        print(json.dumps(loop_record, indent=2))

        if (
            args.stop_if_not_improved
            and score is not None
            and best_prior_score is not None
            and score <= best_prior_score
        ):
            print(
                f"Stopping after loop {loop_index}: score {score:.3f} did not exceed "
                f"best prior score {best_prior_score:.3f}."
            )
            break

        if score is not None and (best_prior_score is None or score > best_prior_score):
            best_prior_score = score

        if _score_meets_threshold(score, args.score_threshold) and not args.force_all_loops:
            print(
                f"Stopping after loop {loop_index}: score {score:.3f} >= "
                f"threshold {args.score_threshold:.3f}."
            )
            break

    summary_path = (
        loop_roots(
            root_dir=args.root_dir,
            task_family=args.task_family,
            scene_count_label=args.scene_count_label,
            attempt_label=args.attempt_label,
            base_run_name=args.base_run_name,
            loop_index=args.start_loop,
        ).postprocessed_root
        / "iteration_summary.json"
    )
    write_loop_manifest(summary_path, {"loops": loop_results})
    print(f"Wrote iteration summary: {summary_path}")


if __name__ == "__main__":
    main()
