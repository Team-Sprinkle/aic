#!/usr/bin/env python3
"""Build and query the expert setting pass/skip registry."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml


MODES = ("nominal", "nominalrecovery", "recovery")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _matrix_run_config(root: Path) -> dict[str, Any]:
    for name in ("matrix_run_config.json", "matrix_run_config.posthoc.json"):
        value = _load_json(root / name)
        if value:
            return value
    return {}


def _with_fallback_run_config(row: dict[str, Any], run_config: dict[str, Any]) -> dict[str, Any]:
    if not run_config:
        return row
    mode = str(row.get("mode") or "")
    if mode not in MODES:
        return row
    result = dict(row.get("result") or {})
    changed = False
    mode_env = (run_config.get("mode_env") or {}).get(mode)
    if isinstance(mode_env, dict) and not isinstance(result.get("mode_env"), dict):
        result["mode_env"] = {str(key): str(value) for key, value in mode_env.items()}
        changed = True
    mode_args = (run_config.get("mode_args") or {}).get(mode)
    if isinstance(mode_args, list) and not isinstance(result.get("mode_args"), list):
        result["mode_args"] = [str(item) for item in mode_args]
        changed = True
    if not changed:
        return row
    patched = dict(row)
    patched["result"] = result
    return patched


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = data.get("settings") if isinstance(data, dict) else None
    if not isinstance(settings, list):
        raise ValueError(f"Manifest missing settings list: {path}")
    return settings


def _empty_mode() -> dict[str, Any]:
    return {
        "status": "unknown_not_logged",
        "attempts": 0,
        "passed_runs": 0,
        "best_score": None,
        "best_summary": None,
        "best_mode_env": None,
        "best_mode_args": None,
        "best_cmd": None,
        "last_summary": None,
        "last_reason": None,
        "last_mode_env": None,
        "last_mode_args": None,
        "last_cmd": None,
        "history": [],
    }


def _setting_entry(setting: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": int(setting["index"]),
        "suffix": str(setting["suffix"]),
        "task_family": setting.get("task_family"),
        "request_yaml": setting.get("request_yaml"),
        "engine_config": setting.get("engine_config"),
        "modes": {mode: _empty_mode() for mode in MODES},
    }


def _score_values(row: dict[str, Any]) -> list[float]:
    result = row.get("result") or {}
    scores = result.get("scores") or []
    values: list[float] = []
    for score in scores:
        if score is None:
            continue
        try:
            values.append(float(score))
        except (TypeError, ValueError):
            continue
    return values


def _failure_reason(row: dict[str, Any]) -> str | None:
    result = row.get("result") or {}
    if row.get("passed"):
        return None
    summary = result.get("summary")
    stderr = str(result.get("stderr_tail") or "").strip().splitlines()
    stdout = str(result.get("stdout_tail") or "").strip().splitlines()
    if stderr:
        return stderr[-1][-240:]
    if stdout:
        return stdout[-1][-240:]
    return f"not accepted; summary={summary}"


def _mode_env(row: dict[str, Any]) -> dict[str, str] | None:
    result = row.get("result") or {}
    value = result.get("mode_env")
    if not isinstance(value, dict):
        return None
    return {str(key): str(val) for key, val in value.items()}


def _mode_args(row: dict[str, Any]) -> list[str] | None:
    result = row.get("result") or {}
    value = result.get("mode_args")
    if not isinstance(value, list):
        return None
    return [str(item) for item in value]


def _cmd(row: dict[str, Any]) -> list[str] | None:
    result = row.get("result") or {}
    value = result.get("cmd")
    if not isinstance(value, list):
        return None
    return [str(item) for item in value]


def _apply_row(registry: dict[str, Any], row: dict[str, Any], max_attempts: int) -> None:
    setting = row.get("setting") or {}
    suffix = str(setting.get("suffix") or "")
    mode = str(row.get("mode") or "")
    if not suffix or mode not in MODES or suffix not in registry["settings"]:
        return
    mode_entry = registry["settings"][suffix]["modes"][mode]
    scores = _score_values(row)
    best_score = max(scores) if scores else None
    result = row.get("result") or {}
    summary = result.get("summary")
    mode_env = _mode_env(row)
    mode_args = _mode_args(row)
    cmd = _cmd(row)
    mode_entry["attempts"] = int(mode_entry.get("attempts", 0)) + 1
    mode_entry["last_summary"] = summary
    mode_entry["last_reason"] = _failure_reason(row)
    mode_entry["last_mode_env"] = mode_env
    mode_entry["last_mode_args"] = mode_args
    mode_entry["last_cmd"] = cmd
    if best_score is not None and (
        mode_entry.get("best_score") is None or best_score > float(mode_entry["best_score"])
    ):
        mode_entry["best_score"] = best_score
        mode_entry["best_summary"] = summary
        if mode_env is not None:
            mode_entry["best_mode_env"] = mode_env
        if mode_args is not None:
            mode_entry["best_mode_args"] = mode_args
        if cmd is not None:
            mode_entry["best_cmd"] = cmd
    if row.get("passed"):
        mode_entry["status"] = "passed"
        mode_entry["passed_runs"] = int(mode_entry.get("passed_runs", 0)) + 1
        mode_entry["best_summary"] = summary
        if mode_env is not None:
            mode_entry["best_mode_env"] = mode_env
        if mode_args is not None:
            mode_entry["best_mode_args"] = mode_args
        if cmd is not None:
            mode_entry["best_cmd"] = cmd
    elif mode_entry.get("status") != "passed" and int(mode_entry["attempts"]) >= max_attempts:
        mode_entry["status"] = "skipped_exhausted"
    elif mode_entry.get("status") != "passed":
        mode_entry["status"] = "attempted_not_passing"
    mode_entry["history"].append(
        {
            "output_root": str(Path(str(summary)).parents[3]) if summary and len(Path(str(summary)).parents) > 3 else None,
            "summary": summary,
            "passed": bool(row.get("passed")),
            "scores": scores,
            "reason": mode_entry["last_reason"],
            "mode_env": mode_env,
            "mode_args": mode_args,
            "cmd": cmd,
        }
    )


def _setting_mode_from_summary_path(path: Path) -> tuple[str, str] | None:
    parts = path.parts
    for idx, part in enumerate(parts):
        if part.startswith("setting_") and idx > 0:
            mode = parts[idx - 1]
            if mode not in MODES:
                return None
            suffix = part.split("_", 2)
            if len(suffix) < 3:
                return None
            return suffix[2], mode
    return None


def _apply_generation_summary(
    registry: dict[str, Any],
    summary_path: Path,
    max_attempts: int,
    score_threshold: float,
    run_config: dict[str, Any] | None = None,
) -> None:
    parsed = _setting_mode_from_summary_path(summary_path)
    if parsed is None:
        return
    suffix, mode = parsed
    if suffix not in registry["settings"]:
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    records = summary.get("records") or []
    scores: list[float] = []
    for record in records:
        validation = record.get("validation") or {}
        score = validation.get("score")
        if score is None:
            replay_metrics = record.get("replay_metrics") or {}
            score = replay_metrics.get("score")
        if score is None:
            continue
        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            continue
    passed = int(summary.get("accepted", 0) or 0) > 0 or any(score >= score_threshold for score in scores)
    row = {
        "setting": registry["settings"][suffix],
        "mode": mode,
        "passed": passed,
        "score_threshold": score_threshold,
        "result": {
            "summary": str(summary_path),
            "accepted": int(summary.get("accepted", 0) or 0),
            "scores": scores,
        },
    }
    row = _with_fallback_run_config(row, run_config or {})
    _apply_row(registry, row, max_attempts)


def build_registry(
    manifest: Path,
    matrix_roots: list[Path],
    summary_roots: list[Path],
    max_attempts: int,
    score_threshold: float,
) -> dict[str, Any]:
    settings = _load_manifest(manifest)
    registry = {
        "schema_version": "aic_expert_setting_registry/v1",
        "manifest": str(manifest),
        "max_attempts_before_skip": max_attempts,
        "score_threshold": score_threshold,
        "missing_evidence_status": "unknown_not_logged",
        "settings": {str(setting["suffix"]): _setting_entry(setting) for setting in settings},
    }
    for root in matrix_roots:
        run_config = _matrix_run_config(root)
        for row in _read_jsonl(root / "matrix_results.jsonl"):
            row = _with_fallback_run_config(row, run_config)
            _apply_row(registry, row, max_attempts)
    for root in summary_roots:
        paths = [root] if root.name == "generation_summary.json" else root.glob("**/generation_summary.json")
        for summary_path in paths:
            parsed = _setting_mode_from_summary_path(summary_path)
            run_config = {}
            if parsed is not None and len(summary_path.parents) > 3:
                run_config = _matrix_run_config(summary_path.parents[3])
            _apply_generation_summary(registry, summary_path, max_attempts, score_threshold, run_config)
    return registry


def _status_counts(registry: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for setting in registry.get("settings", {}).values():
        for mode_entry in setting.get("modes", {}).values():
            status = str(mode_entry.get("status", "unresolved"))
            counts[status] = counts.get(status, 0) + 1
    return counts


def _next_unresolved(registry: dict[str, Any], seed: int) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for suffix, setting in registry.get("settings", {}).items():
        for mode, mode_entry in setting.get("modes", {}).items():
            if mode_entry.get("status") in {"unknown_not_logged", "attempted_not_passing"}:
                candidates.append(
                    {
                        "index": setting["index"],
                        "suffix": suffix,
                        "mode": mode,
                        "attempts": mode_entry.get("attempts", 0),
                        "engine_config": setting.get("engine_config"),
                        "request_yaml": setting.get("request_yaml"),
                    }
                )
    if not candidates:
        return None
    rng = random.Random(seed)
    return rng.choice(candidates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/expert_matrix_configs_v2/matrix_manifest.yaml"))
    parser.add_argument("--registry", type=Path, default=Path("outputs/expert_setting_registry.json"))
    parser.add_argument("--matrix-root", action="append", type=Path, default=[])
    parser.add_argument("--summary-root", action="append", type=Path, default=[])
    parser.add_argument("--discover-roots", action="store_true")
    parser.add_argument("--max-attempts-before-skip", type=int, default=7)
    parser.add_argument("--score-threshold", type=float, default=92.0)
    parser.add_argument("--next", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    roots = list(args.matrix_root)
    summary_roots = list(args.summary_root)
    if args.discover_roots:
        for pattern in ("expert_matrix*/matrix_results.jsonl", "expert_registry*/matrix_results.jsonl"):
            roots.extend(path.parent for path in Path("outputs").glob(pattern))
        summary_roots.append(Path("outputs"))
    registry = build_registry(
        args.manifest,
        sorted(set(roots)),
        sorted(set(summary_roots)),
        args.max_attempts_before_skip,
        args.score_threshold,
    )
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    args.registry.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote registry: {args.registry}")
    print(json.dumps({"status_counts": _status_counts(registry)}, sort_keys=True))
    if args.next:
        print(json.dumps({"next": _next_unresolved(registry, args.seed)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
