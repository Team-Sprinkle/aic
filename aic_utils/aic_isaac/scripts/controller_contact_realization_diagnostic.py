#!/usr/bin/env python3
"""Run and summarize controller/contact realization probes near final SFP insertion.

This script intentionally reuses the SERL trainer's existing debug-audit paths
instead of adding a second control stack.  The default base command is loaded
from a known near-seated run's ``train_config.json`` and each case only changes
the run name/output directory plus the debug-audit action.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DATE = "20260524"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / f"agentic_reward_curriculum_{DATE}_controller_contact"
DEFAULT_BASE_CONFIG = (
    REPO_ROOT
    / "outputs/agentic_reward_curriculum_20260524_direct_axis_audit/runs/"
    / "2026-05-24_01-25-34_guarded_approach_direct_axis_backout_start105/train_config.json"
)


@dataclass(frozen=True)
class StrictThresholds:
    min_depth_m: float = 0.008
    max_lateral_m: float = 0.0005
    max_theta_rad: float = 0.030
    min_module_consistency: float = 0.80


@dataclass(frozen=True)
class ProbeCase:
    name: str
    hypothesis: str
    replacements: dict[str, list[str] | None]
    add_flags: tuple[str, ...] = ()
    remove_flags: tuple[str, ...] = ()


DEFAULT_CASES: tuple[ProbeCase, ...] = (
    ProbeCase(
        name="semantic_axis_backout_300um_start105",
        hypothesis="A direct semantic-axis backout should reduce tip and module signed depth under contact; failure indicates contact/IK realization mismatch.",
        replacements={
            "--debug_audit_steps": ["130"],
            "--debug_audit_start_step": ["105"],
            "--debug_audit_insertion_axis_action": ["backout"],
            "--debug_audit_insertion_axis_magnitude": ["0.00030"],
            "--debug_audit_axis_magnitude": ["0.0"],
            "--max_logged_image_steps": ["130"],
            "--image_log_every": ["10"],
        },
        remove_flags=("--debug_audit_rotation_axes",),
    ),
    ProbeCase(
        name="semantic_axis_forward_100um_start105",
        hypothesis="A tiny direct semantic-axis inward command should increase depth without lateral or module-consistency collapse if the near-seat controller path is usable.",
        replacements={
            "--debug_audit_steps": ["125"],
            "--debug_audit_start_step": ["105"],
            "--debug_audit_insertion_axis_action": ["forward"],
            "--debug_audit_insertion_axis_magnitude": ["0.00010"],
            "--debug_audit_axis_magnitude": ["0.0"],
            "--max_logged_image_steps": ["125"],
            "--image_log_every": ["10"],
        },
        remove_flags=("--debug_audit_rotation_axes",),
    ),
    ProbeCase(
        name="pure_rotation_axes_4mrad_start103",
        hypothesis="Pure wrist rotation probes identify whether any bounded axis can close the remaining theta gap without increasing r or losing module consistency.",
        replacements={
            "--debug_audit_steps": ["126"],
            "--debug_audit_start_step": ["103"],
            "--debug_audit_insertion_axis_action": ["none"],
            "--debug_audit_insertion_axis_magnitude": ["0.0"],
            "--debug_audit_axis_magnitude": ["0.004"],
            "--max_logged_image_steps": ["126"],
            "--image_log_every": ["10"],
        },
        add_flags=("--debug_audit_rotation_axes",),
    ),
    ProbeCase(
        name="shallow_rotation_axes_4mrad_start90",
        hypothesis="At shallow positive insertion, bounded wrist rotations may still reduce semantic theta before final seated contact makes the error unrecoverable.",
        replacements={
            "--debug_audit_steps": ["116"],
            "--debug_audit_start_step": ["90"],
            "--debug_audit_insertion_axis_action": ["none"],
            "--debug_audit_insertion_axis_magnitude": ["0.0"],
            "--debug_audit_axis_magnitude": ["0.004"],
            "--max_logged_image_steps": ["116"],
            "--image_log_every": ["10"],
        },
        add_flags=("--debug_audit_rotation_axes",),
    ),
    ProbeCase(
        name="shallow_semantic_axis_backout_200um_start90",
        hypothesis="At shallow positive insertion, semantic-axis backout should reduce depth if contact is not yet locking the module against the cage.",
        replacements={
            "--debug_audit_steps": ["116"],
            "--debug_audit_start_step": ["90"],
            "--debug_audit_insertion_axis_action": ["backout"],
            "--debug_audit_insertion_axis_magnitude": ["0.00020"],
            "--debug_audit_axis_magnitude": ["0.0"],
            "--max_logged_image_steps": ["116"],
            "--image_log_every": ["10"],
        },
        remove_flags=("--debug_audit_rotation_axes",),
    ),
)


def _run_git(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        ).stdout
    except Exception as exc:  # pragma: no cover - diagnostic best effort
        return f"<git failed: {exc}>"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_base_argv(path: Path) -> list[str]:
    cfg = _load_json(path)
    argv = cfg.get("argv")
    if isinstance(argv, list) and argv:
        return [str(x) for x in argv]
    command = cfg.get("command")
    if isinstance(command, str) and command.strip():
        return shlex.split(command)
    raise ValueError(f"{path} does not contain argv or command")


def _remove_flag(argv: list[str], flag: str) -> list[str]:
    out: list[str] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token == flag:
            idx += 1
            while idx < len(argv) and not argv[idx].startswith("--"):
                idx += 1
            continue
        out.append(token)
        idx += 1
    return out


def _replace_flag(argv: list[str], flag: str, values: list[str] | None) -> list[str]:
    out = _remove_flag(argv, flag)
    out.append(flag)
    if values:
        out.extend(values)
    return out


def _ensure_flag(argv: list[str], flag: str) -> list[str]:
    if flag in argv:
        return argv
    return [*argv, flag]


def _case_argv(base_argv: list[str], case: ProbeCase, output_dir_in_container: str, run_name: str) -> list[str]:
    argv = list(base_argv)
    argv = _replace_flag(argv, "--output_dir", [output_dir_in_container])
    argv = _replace_flag(argv, "--run_name", [run_name])
    argv = _replace_flag(argv, "--updates", [case.replacements.get("--debug_audit_steps", ["130"])[0]])
    argv = _replace_flag(argv, "--diagnostics_every", ["1"])
    for flag in case.remove_flags:
        argv = _remove_flag(argv, flag)
    for flag, values in case.replacements.items():
        argv = _replace_flag(argv, flag, values)
    for flag in case.add_flags:
        argv = _ensure_flag(argv, flag)
    argv = _ensure_flag(argv, "--insertion_action_guard_disable_after_debug_audit_start")
    argv = _ensure_flag(argv, "--debug_diagnostics")
    argv = _ensure_flag(argv, "--save_step_images")
    return argv


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            pass
    return out


def _by_env(metric: dict[str, Any], key: str) -> list[float]:
    vals = _float_list(metric.get(f"{key}_by_env"))
    if vals:
        return vals
    val = metric.get(f"{key}_env0")
    if val is None:
        val = metric.get(f"{key}_mean")
    try:
        return [] if val is None else [float(val)]
    except (TypeError, ValueError):
        return []


def _geom(row: dict[str, Any]) -> dict[str, Any]:
    geom = row.get("post_step_insertion_geometry")
    return geom if isinstance(geom, dict) else {}


def _module_consistency(row: dict[str, Any], env_count: int) -> list[float]:
    geom = _geom(row)
    vals = _float_list(geom.get("consistency_gate_by_env"))
    if vals:
        return vals
    phase = geom.get("cheatcode_phase_reward")
    if isinstance(phase, dict):
        vals = _float_list(phase.get("g_semantic_by_env"))
        if vals:
            return vals
    val = geom.get("consistency_gate_env0") or geom.get("consistency_gate_mean")
    try:
        return [float(val)] if val is not None else [0.0 for _ in range(env_count)]
    except (TypeError, ValueError):
        return [0.0 for _ in range(env_count)]


def _audit_start_step(run_dir: Path, rows: list[dict[str, Any]]) -> int | None:
    cfg_path = run_dir / "train_config.json"
    if cfg_path.is_file():
        cfg = _load_json(cfg_path)
        args = cfg.get("args") if isinstance(cfg.get("args"), dict) else {}
        try:
            return int(args.get("debug_audit_start_step"))
        except (TypeError, ValueError):
            pass
    for row in rows:
        reps = row.get("action_representations")
        if isinstance(reps, dict):
            try:
                return int(reps.get("debug_audit_start_step"))
            except (TypeError, ValueError):
                continue
    return None


def _row_env_metrics(row: dict[str, Any], thresholds: StrictThresholds) -> list[dict[str, Any]]:
    geom = _geom(row)
    s_vals = _by_env(geom, "signed_depth_m")
    r_vals = _by_env(geom, "lateral_error_m")
    theta_vals = _by_env(geom, "orientation_error_rad")
    target_depth_vals = _by_env(geom, "target_depth_m")
    n_env = max(len(s_vals), len(r_vals), len(theta_vals), len(target_depth_vals), 1)
    module_vals = _module_consistency(row, n_env)
    out: list[dict[str, Any]] = []
    for env_id in range(n_env):
        s = s_vals[min(env_id, len(s_vals) - 1)] if s_vals else None
        r = r_vals[min(env_id, len(r_vals) - 1)] if r_vals else None
        theta = theta_vals[min(env_id, len(theta_vals) - 1)] if theta_vals else None
        target_depth = (
            target_depth_vals[min(env_id, len(target_depth_vals) - 1)] if target_depth_vals else thresholds.min_depth_m
        )
        module = module_vals[min(env_id, len(module_vals) - 1)] if module_vals else None
        strict = (
            s is not None
            and r is not None
            and theta is not None
            and module is not None
            and s >= thresholds.min_depth_m
            and r <= thresholds.max_lateral_m
            and theta <= thresholds.max_theta_rad
            and module >= thresholds.min_module_consistency
        )
        score = (
            (0.0 if s is None else 10.0 * min(max(s, 0.0) / max(target_depth, 1.0e-9), 1.0))
            - (1.0 if r is None else 1000.0 * max(r - thresholds.max_lateral_m, 0.0))
            - (1.0 if theta is None else 20.0 * max(theta - thresholds.max_theta_rad, 0.0))
            + (0.0 if module is None else module)
        )
        out.append(
            {
                "env": env_id,
                "s_m": s,
                "r_m": r,
                "theta_rad": theta,
                "target_depth_m": target_depth,
                "module_consistency": module,
                "strict_success": strict,
                "score": score,
            }
        )
    return out


def _best_metric_row(rows: list[dict[str, Any]], thresholds: StrictThresholds) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for row in rows:
        for metrics in _row_env_metrics(row, thresholds):
            candidate = {"step": row.get("step"), **metrics}
            if best is None or bool(candidate["strict_success"]) or float(candidate["score"]) > float(best["score"]):
                best = candidate
                if bool(candidate["strict_success"]):
                    return best
    return best


def _best_theta_row(rows: list[dict[str, Any]], thresholds: StrictThresholds) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for row in rows:
        for metrics in _row_env_metrics(row, thresholds):
            theta = metrics.get("theta_rad")
            if theta is None:
                continue
            candidate = {"step": row.get("step"), **metrics}
            if best is None or float(theta) < float(best["theta_rad"]):
                best = candidate
    return best


def _audit_action_label(row: dict[str, Any], audit_start: int | None) -> str | None:
    if audit_start is None:
        return None
    try:
        step = int(row.get("step"))
    except (TypeError, ValueError):
        return None
    if step < audit_start:
        return None
    action = row.get("debug_audit_insertion_axis_action")
    if action in {"forward", "backout"}:
        return str(action)
    reps = row.get("action_representations")
    if isinstance(reps, dict) and reps.get("debug_audit_insertion_axis_action") in {"forward", "backout"}:
        return str(reps["debug_audit_insertion_axis_action"])
    axis_index = (step - audit_start) % 6
    sign = "+" if axis_index < 3 else "-"
    axis = ("rx", "ry", "rz")[axis_index % 3]
    return f"{sign}{axis}"


def summarize_run(run_dir: Path, thresholds: StrictThresholds = StrictThresholds()) -> dict[str, Any]:
    rows = _jsonl_rows(run_dir / "metrics.jsonl")
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "has_metrics": bool(rows),
        "strict_thresholds": thresholds.__dict__,
        "strict_success": False,
        "failure_label": "insufficient_logs",
    }
    if not rows:
        return summary
    audit_start = _audit_start_step(run_dir, rows)
    audit_rows = [row for row in rows if audit_start is not None and int(row.get("step", -1)) >= audit_start]
    baseline_candidates = [row for row in rows if audit_start is not None and int(row.get("step", -1)) < audit_start]
    baseline = baseline_candidates[-1] if baseline_candidates else rows[0]
    final = rows[-1]
    best = _best_metric_row(rows, thresholds)
    best_audit = _best_metric_row(audit_rows or rows, thresholds)
    best_theta = _best_theta_row(audit_rows or rows, thresholds)
    baseline_best = _best_metric_row([baseline], thresholds)
    final_best = _best_metric_row([final], thresholds)
    by_action: dict[str, dict[str, Any]] = {}
    for row in audit_rows:
        label = _audit_action_label(row, audit_start)
        if label is None:
            continue
        row_best = _best_metric_row([row], thresholds)
        if row_best is None:
            continue
        previous = by_action.get(label)
        if previous is None or float(row_best["score"]) > float(previous["score"]):
            by_action[label] = row_best
    delta = _delta(final_best, baseline_best)
    summary.update(
        {
            "audit_start_step": audit_start,
            "num_rows": len(rows),
            "num_audit_rows": len(audit_rows),
            "baseline": baseline_best,
            "final": final_best,
            "best_overall": best,
            "best_after_audit_start": best_audit,
            "best_theta_after_audit_start": best_theta,
            "best_by_audit_action": by_action,
            "delta_final_minus_baseline": delta,
            "strict_success": bool(best and best.get("strict_success")),
            "failure_label": _classify(best or {}, best_audit or {}, final_best or {}, delta),
            "images": sorted(str(p.relative_to(run_dir)) for p in run_dir.glob("**/*.png"))[:30],
        }
    )
    return summary


def _delta(final: dict[str, Any] | None, baseline: dict[str, Any] | None) -> dict[str, float | None]:
    if final is None or baseline is None:
        return {"s_m": None, "r_m": None, "theta_rad": None, "module_consistency": None}
    out: dict[str, float | None] = {}
    for key in ("s_m", "r_m", "theta_rad", "module_consistency"):
        try:
            out[key] = float(final[key]) - float(baseline[key])
        except (TypeError, ValueError, KeyError):
            out[key] = None
    return out


def _classify(
    best: dict[str, Any],
    best_audit: dict[str, Any],
    final: dict[str, Any],
    delta: dict[str, float | None],
) -> str:
    if best.get("strict_success"):
        return "strict_success"
    s = _num(best.get("s_m"), -1.0)
    r = _num(best.get("r_m"), 1.0)
    theta = _num(best.get("theta_rad"), 1.0)
    module = _num(best.get("module_consistency"), 0.0)
    ds = _num(delta.get("s_m"), 0.0)
    dtheta = _num(delta.get("theta_rad"), 0.0)
    dmodule = _num(delta.get("module_consistency"), 0.0)
    if s >= 0.008 and r <= 0.0005 and module >= 0.80 and theta > 0.030:
        return "near_success_orientation_blocked"
    if s >= 0.008 and r <= 0.0005 and theta <= 0.030 and module < 0.80:
        return "near_success_module_consistency_blocked"
    if s > 0.0 and module < 0.40:
        return "tip_depth_false_positive"
    if s > 0.0 and r > 0.0015:
        return "lateral_bypass"
    if abs(ds) < 0.00005 and abs(dtheta) < 0.001 and abs(dmodule) < 0.02:
        return "controller_realization_mismatch"
    if s < 0.008 and r <= 0.0005 and theta <= 0.040:
        return "no_axial_progress"
    if best_audit and _num(best_audit.get("theta_rad"), 1.0) < 0.035 and _num(best_audit.get("r_m"), 1.0) > 0.0005:
        return "rotation_induced_lateral_sweep"
    if dtheta < -0.001 and dmodule < -0.05:
        return "orientation_plateau_env_or_card_dependent"
    return "insufficient_logs"


def _num(value: Any, default: float) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_summary_md(output_root: Path, run_summaries: list[dict[str, Any]], dry_run_commands: list[str]) -> None:
    lines = [
        "# Controller/contact realization diagnostic",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Strict success requires post-step depth >= 8 mm, r <= 0.5 mm, theta <= 0.030 rad, and module consistency >= 0.80.",
        "",
    ]
    if dry_run_commands:
        lines.extend(["## Dry-run commands", ""])
        lines.extend(f"- `{cmd}`" for cmd in dry_run_commands)
        lines.append("")
    if run_summaries:
        lines.extend(
            [
                "## Results",
                "",
                "| run | strict | label | best s mm | best r mm | best theta rad | best consistency | final ds mm | final dtheta rad |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for summary in run_summaries:
            best = summary.get("best_overall") or {}
            delta = summary.get("delta_final_minus_baseline") or {}
            lines.append(
                "| {run} | {strict} | {label} | {s:.3f} | {r:.3f} | {theta:.5f} | {cons:.3f} | {ds:.3f} | {dtheta:.5f} |".format(
                    run=Path(str(summary.get("run_dir", ""))).name,
                    strict=str(bool(summary.get("strict_success"))).lower(),
                    label=summary.get("failure_label", "unknown"),
                    s=1000.0 * _num(best.get("s_m"), float("nan")),
                    r=1000.0 * _num(best.get("r_m"), float("nan")),
                    theta=_num(best.get("theta_rad"), float("nan")),
                    cons=_num(best.get("module_consistency"), float("nan")),
                    ds=1000.0 * _num(delta.get("s_m"), float("nan")),
                    dtheta=_num(delta.get("theta_rad"), float("nan")),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "Promote a probe only if post-step semantic metrics improve strict insertion criteria together; reward return and tip-depth-only rows are ignored.",
        ]
    )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _docker_command(container: str, train_argv: list[str], *, gpu: str, num_envs: int) -> list[str]:
    inner = "cd /workspace/isaaclab && ./isaaclab.sh -p " + " ".join(shlex.quote(x) for x in train_argv)
    return [
        "docker",
        "exec",
        "-e",
        f"LC_USER_ID={os.environ.get('LC_USER_ID', 'yoonjung')}",
        "-e",
        "AIC_ISAAC_ALLOW_UNSUPPORTED_RTX_DRIVER=1",
        "-e",
        "AIC_ISAAC_RANDOMIZATION_PROFILE=none",
        "-e",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "-e",
        f"DEVICE=cuda:{gpu}",
        "-e",
        f"NUM_ENVS={num_envs}",
        "-e",
        "RENDERING_MODE=performance",
        container,
        "bash",
        "-lc",
        inner,
    ]


def _selected_cases(names: list[str] | None) -> list[ProbeCase]:
    cases = list(DEFAULT_CASES)
    if not names:
        return cases
    wanted = set(names)
    selected = [case for case in cases if case.name in wanted]
    missing = sorted(wanted - {case.name for case in selected})
    if missing:
        raise ValueError(f"unknown case(s): {', '.join(missing)}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--container", default="isaac-lab-base")
    parser.add_argument("--case", action="append", choices=[case.name for case in DEFAULT_CASES])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--timeout-minutes", type=float, default=55.0)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    runs_dir = output_root / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "git_status.txt").write_text(_run_git(["status", "--short", "--branch"]), encoding="utf-8")
    (output_root / "git_diff.patch").write_text(_run_git(["diff", "--", "."]), encoding="utf-8")

    cases = _selected_cases(args.case)
    commands: list[str] = []
    summaries: list[dict[str, Any]] = []

    if not args.analyze_only:
        base_argv = _load_base_argv(args.base_config)
        for case in cases:
            run_name = case.name
            train_argv = _case_argv(base_argv, case, f"aic/{runs_dir.relative_to(REPO_ROOT)}", run_name)
            docker_cmd = _docker_command(args.container, train_argv, gpu=str(args.gpu), num_envs=args.num_envs)
            commands.append(" ".join(shlex.quote(x) for x in docker_cmd))
            decision = {
                "hypothesis": case.hypothesis,
                "config_changes": case.replacements,
                "command": commands[-1],
                "decision": "dry_run" if args.dry_run else "run",
            }
            if args.dry_run:
                run_dir = runs_dir / f"dry_run_{run_name}"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "agent_decision.json").write_text(
                    json.dumps(decision, indent=2) + "\n",
                    encoding="utf-8",
                )
                continue
            before_dirs = {path for path in runs_dir.iterdir() if path.is_dir()}
            proc = subprocess.run(
                docker_cmd,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=max(60.0, args.timeout_minutes * 60.0),
                check=False,
            )
            after_dirs = {path for path in runs_dir.iterdir() if path.is_dir()}
            created = sorted(
                after_dirs - before_dirs,
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            matching = [path for path in created if path.name.endswith(run_name)]
            run_dir = matching[0] if matching else (created[0] if created else runs_dir / run_name)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "agent_decision.json").write_text(
                json.dumps(decision, indent=2) + "\n",
                encoding="utf-8",
            )
            (run_dir / "launcher_output.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
            if proc.returncode != 0:
                (run_dir / "launcher_error.json").write_text(
                    json.dumps({"returncode": proc.returncode}, indent=2) + "\n",
                    encoding="utf-8",
                )

    for run_dir in sorted(runs_dir.iterdir() if runs_dir.is_dir() else []):
        if not run_dir.is_dir() or not (run_dir / "metrics.jsonl").is_file():
            continue
        summary = summarize_run(run_dir)
        summaries.append(summary)
        (run_dir / "controller_contact_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    (output_root / "controller_contact_summary.json").write_text(
        json.dumps({"runs": summaries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary_md(output_root, summaries, commands if args.dry_run else [])
    print(json.dumps({"output_root": str(output_root), "runs": len(summaries), "dry_run": args.dry_run}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
