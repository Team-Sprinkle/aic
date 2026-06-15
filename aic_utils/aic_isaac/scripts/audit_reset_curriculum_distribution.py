#!/usr/bin/env python3
"""Summarize reset curriculum episode metadata and validation metrics.

This is a lightweight pre-training audit.  It does not run Isaac; it reads the
episode YAMLs plus any existing reset-validation summaries and writes a compact
JSON/Markdown report so candidate curricula can be accepted or rejected before
long SERL runs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


def _numbers(values: list[float]) -> dict[str, float] | None:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return None
    return {
        "min": min(finite),
        "mean": sum(finite) / len(finite),
        "max": max(finite),
        "count": float(len(finite)),
    }


def _vec(raw: Any, *, length: int) -> list[float] | None:
    if not isinstance(raw, list | tuple) or len(raw) != length:
        return None
    try:
        return [float(v) for v in raw]
    except Exception:
        return None


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(a[i] * b[i] for i in range(3))


def _episode_summary(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    start = (((data or {}).get("scene") or {}).get("start_near_gate") or {})
    reference = _vec(start.get("reference_tip_center_position_world") or start.get("reference_body_position"), length=3)
    gate = _vec(start.get("target_gate_position"), length=3)
    axis = _vec(start.get("target_gate_axis_world"), length=3)
    body = _vec(start.get("body_start_position_world") or start.get("tcp_start_position_world"), length=3)
    offset = _vec(start.get("reset_body_offset_from_reference_world"), length=3)
    reward_body = start.get("reference_reward_body_name")
    reward_offset = _vec(start.get("reference_reward_body_position_offset"), length=3)
    variant = start.get("randomized_curriculum_variant") or {}
    rotvec = variant.get("rotvec_rad")
    rotvec_norm = None
    if isinstance(rotvec, list | tuple) and len(rotvec) == 3:
        try:
            rotvec_norm = _norm([float(v) for v in rotvec])
        except Exception:
            rotvec_norm = None
    out: dict[str, Any] = {
        "episode": path.name,
        "episode_id": data.get("episode_id") if isinstance(data, dict) else None,
        "reset_body_name": start.get("reset_body_name"),
        "reference_reward_body_name": reward_body,
        "has_body_start_orientation": isinstance(start.get("body_start_orientation_wxyz"), list),
        "has_reference_tip_center": reference is not None,
        "has_reset_body_offset": offset is not None,
        "has_reference_reward_body_offset": reward_offset is not None,
        "requested_axial_distance_m": start.get("axial_distance_m"),
        "requested_lateral_distance_m": start.get("lateral_distance_m"),
        "variant_bucket": variant.get("bucket"),
        "variant_target_signed_depth_m": variant.get("target_signed_depth_m"),
        "variant_requested_lateral_m": variant.get("requested_lateral_m"),
        "variant_tip_preserving_rotation": variant.get("tip_preserving_rotation"),
        "variant_rotvec_norm_rad": rotvec_norm,
    }
    if reference is not None and gate is not None and axis is not None:
        axis_norm = _norm(axis)
        if axis_norm > 1.0e-9:
            unit_axis = [v / axis_norm for v in axis]
            delta = _sub(reference, gate)
            axial = _dot(delta, unit_axis)
            lateral = math.sqrt(max(0.0, _norm(delta) ** 2 - axial * axial))
            out["metadata_reference_axial_m"] = axial
            out["metadata_reference_lateral_m"] = lateral
    if body is not None and reference is not None:
        out["body_reference_distance_m"] = _norm(_sub(body, reference))
    if offset is not None:
        out["reset_body_offset_norm_m"] = _norm(offset)
    if reward_offset is not None:
        out["reference_reward_body_offset_norm_m"] = _norm(reward_offset)
    if reward_body is not None and out["reset_body_name"] is not None:
        out["reset_body_matches_reward_body"] = bool(str(out["reset_body_name"]) == str(reward_body))
    return out


def _load_validation(run_dir: Path) -> dict[str, Any] | None:
    post_step = run_dir / "post_step_reset_metrics_summary.json"
    if post_step.exists():
        raw = json.loads(post_step.read_text(encoding="utf-8"))
        steps = raw.get("steps")
        if isinstance(steps, dict):
            return steps
        return raw
    summary = run_dir / "reset_validation_summary_codex.json"
    if summary.exists():
        return json.loads(summary.read_text(encoding="utf-8"))
    metrics = run_dir / "metrics_summary_codex.json"
    if metrics.exists():
        return json.loads(metrics.read_text(encoding="utf-8"))
    return None


def _markdown(report: dict[str, Any]) -> str:
    lines = [f"# Reset Curriculum Audit - {report['name']}", ""]
    lines.append(f"Config dir: `{report['config_dir']}`")
    if report.get("validation_run"):
        lines.append(f"Validation run: `{report['validation_run']}`")
    lines.append("")
    lines.append("## Episode Metadata")
    lines.append("")
    if report.get("categorical_counts"):
        lines.append("### Categorical Counts")
        lines.append("")
        for key, counts in report["categorical_counts"].items():
            lines.append(f"- `{key}`: `{counts}`")
        lines.append("")
    lines.append("| metric | min | mean | max | count |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, stats in report["episode_stats"].items():
        if stats is None:
            continue
        lines.append(
            f"| {key} | {stats['min']:.6g} | {stats['mean']:.6g} | {stats['max']:.6g} | {int(stats['count'])} |"
        )
    if report.get("validation"):
        lines.extend(["", "## Validation Summary", ""])
        validation = report["validation"]
        step_keys = [
            key
            for key in sorted(validation)
            if key.startswith("step") and isinstance(validation.get(key), dict)
        ] if isinstance(validation, dict) else []
        if step_keys:
            lines.append("| step | metric | min | mean | max | count |")
            lines.append("|---|---|---:|---:|---:|---:|")
            for step_key in step_keys:
                step = validation.get(step_key)
                if not isinstance(step, dict):
                    continue
                for key in ("s", "r", "theta", "module_s", "module_r"):
                    value = step.get(key)
                    if isinstance(value, dict):
                        lines.append(
                            f"| `{step_key}` | `{key}` | {value.get('min'):.6g} | {value.get('mean'):.6g} | {value.get('max'):.6g} | {int(value.get('count', 0))} |"
                        )
                if step.get("terminated_by_env") is not None:
                    lines.append(f"- `{step_key}.terminated_by_env`: `{step['terminated_by_env']}`")
                if step.get("force_norm_mean") is not None:
                    lines.append(f"- `{step_key}.force_norm_mean`: `{step['force_norm_mean']:.6g}`")
        else:
            for key in ("s_m", "r_m", "theta_rad", "module_s_m", "module_r_m"):
                value = validation.get(key)
                if isinstance(value, dict):
                    lines.append(
                        f"- `{key}` min/mean/max: {value.get('min'):.6g} / {value.get('mean'):.6g} / {value.get('max'):.6g}"
                    )
            if validation.get("terminated_by_env") is not None:
                lines.append(f"- `terminated_by_env`: `{validation['terminated_by_env']}`")
    lines.append("")
    if report.get("warnings"):
        lines.append("## Warnings")
        lines.append("")
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    lines.append(f"Decision hint: {report['decision_hint']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--name", default="")
    parser.add_argument("--validation-run", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    config_dir = args.config_dir.resolve()
    episodes = sorted((config_dir / "episodes").glob("episode_*.yaml"))
    if not episodes:
        raise FileNotFoundError(f"no episode_*.yaml under {config_dir / 'episodes'}")
    summaries = [_episode_summary(path) for path in episodes]
    numeric_keys = sorted({key for row in summaries for key, value in row.items() if isinstance(value, int | float)})
    episode_stats = {key: _numbers([row[key] for row in summaries if isinstance(row.get(key), int | float)]) for key in numeric_keys}
    categorical_keys = ("reset_body_name", "reference_reward_body_name", "variant_bucket", "reset_body_matches_reward_body")
    categorical_counts: dict[str, dict[str, int]] = {}
    for key in categorical_keys:
        counts: dict[str, int] = {}
        for row in summaries:
            value = row.get(key)
            label = "<missing>" if value is None else str(value)
            counts[label] = counts.get(label, 0) + 1
        categorical_counts[key] = counts
    validation = _load_validation(args.validation_run.resolve()) if args.validation_run else None

    warnings: list[str] = []
    reset_body_counts = categorical_counts.get("reset_body_name", {})
    if len([name for name, count in reset_body_counts.items() if name != "<missing>" and count > 0]) > 1:
        warnings.append("Episode dir mixes reset bodies; Isaac reset batches cannot mix reset_body_name values.")
    mismatch_count = categorical_counts.get("reset_body_matches_reward_body", {}).get("False", 0)
    if mismatch_count:
        warnings.append(
            f"{mismatch_count} episodes reset a body different from the semantic reward body; this is valid only if reset_body_offset_from_reference_world is calibrated."
        )
    missing_reference = sum(1 for row in summaries if not row.get("has_reference_tip_center"))
    if missing_reference:
        warnings.append(f"{missing_reference} episodes lack reference_tip_center/reference_body metadata.")
    target_depth_mismatches = [
        row
        for row in summaries
        if isinstance(row.get("variant_target_signed_depth_m"), int | float)
        and isinstance(row.get("metadata_reference_axial_m"), int | float)
        and abs(float(row["variant_target_signed_depth_m"]) - float(row["metadata_reference_axial_m"])) > 0.001
    ]
    if target_depth_mismatches:
        sample = target_depth_mismatches[0]
        warnings.append(
            f"{len(target_depth_mismatches)} episodes have semantic-tip signed depth inconsistent with "
            "randomized_curriculum_variant.target_signed_depth_m; "
            f"sample {sample.get('episode')} target={float(sample['variant_target_signed_depth_m']):+.4f}m "
            f"metadata={float(sample['metadata_reference_axial_m']):+.4f}m."
        )
    large_offsets = [
        row for row in summaries if isinstance(row.get("reset_body_offset_norm_m"), float) and row["reset_body_offset_norm_m"] > 0.20
    ]
    if large_offsets:
        warnings.append(
            f"{len(large_offsets)} episodes have reset body/reference offsets above 20 cm; audit gripper-vs-tip calibration before training."
        )
    non_tip_preserving_rotations = [
        row
        for row in summaries
        if row.get("variant_tip_preserving_rotation") is False
        and isinstance(row.get("variant_rotvec_norm_rad"), float)
        and row["variant_rotvec_norm_rad"] > 1.0e-9
    ]
    if non_tip_preserving_rotations:
        warnings.append(
            f"{len(non_tip_preserving_rotations)} episodes apply orientation perturbations with tip_preserving_rotation=false; "
            "post-step reset validation is required because the semantic tip may leave the requested s/r/theta bucket."
        )

    decision = "needs Isaac validation before training"
    if warnings:
        decision = "audit warnings before training"
    if validation:
        if isinstance(validation.get("step1"), dict):
            validation_for_decision = validation["step1"]
            theta = validation_for_decision.get("theta")
            r = validation_for_decision.get("r")
            terminated = validation_for_decision.get("terminated_by_env")
        else:
            theta = validation.get("theta_rad") if isinstance(validation, dict) else None
            r = validation.get("r_m") if isinstance(validation, dict) else None
            terminated = validation.get("terminated_by_env") if isinstance(validation, dict) else None
        if isinstance(theta, dict) and float(theta.get("min", 1.0)) < 0.03 and isinstance(r, dict) and float(r.get("max", 1.0)) < 0.003 and not any(bool(v) for v in (terminated or [])) and not warnings:
            decision = "candidate for short training smoke"
        elif isinstance(theta, dict) and float(theta.get("min", 1.0)) >= 0.03:
            decision = "reject for strict-orientation curriculum until low-theta starts are added"
        elif isinstance(r, dict) and float(r.get("max", 0.0)) >= 0.003:
            decision = "reject or tighten lateral distribution before training"
        elif warnings and all("reset a body different from the semantic reward body" in warning for warning in warnings):
            decision = "candidate for short training smoke after validated gripper-to-tip reset calibration"

    report = {
        "name": args.name or config_dir.name,
        "config_dir": str(config_dir),
        "episode_count": len(episodes),
        "episode_stats": episode_stats,
        "categorical_counts": categorical_counts,
        "sample_episodes": summaries[:8],
        "warnings": warnings,
        "validation_run": str(args.validation_run.resolve()) if args.validation_run else None,
        "validation": validation,
        "decision_hint": decision,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or config_dir.name
    (output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / f"{stem}.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(output_dir / f"{stem}.json"), "markdown": str(output_dir / f"{stem}.md"), "decision_hint": decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
