#!/usr/bin/env python3
"""Rank insertion run folders by post-step geometry.

This is intentionally lightweight: it scans metrics.jsonl files, extracts the
best signed depth per run, and reports the corresponding lateral, orientation,
module-consistency, and checkpoint evidence. It is a triage helper, not a
success checker.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Candidate:
    signed_depth_m: float
    step: int | None
    lateral_error_m: float | None
    theta_rad: float | None
    consistency: float | None
    strict_success: bool | None
    run_dir: Path
    checkpoint: Path | None


def _first(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _float(value: Any) -> float | None:
    value = _first(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _bool(value: Any) -> bool | None:
    value = _first(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _geometry(row: dict[str, Any]) -> dict[str, Any]:
    geom = row.get("post_step_insertion_geometry")
    if isinstance(geom, dict) and geom:
        return geom
    geom = row.get("insertion_geometry")
    if isinstance(geom, dict) and geom:
        return geom
    all_body = row.get("post_step_all_body_insertion_geometry")
    if isinstance(all_body, dict):
        tip_geom = all_body.get("sfp_tip_link")
        if isinstance(tip_geom, dict):
            return tip_geom
    return {}


def _checkpoint_for(run_dir: Path, step: int | None) -> Path | None:
    if step is not None:
        exact = run_dir / "checkpoints" / f"checkpoint_{int(step):06d}.pt"
        if exact.is_file():
            return exact
    candidates = sorted((run_dir / "checkpoints").glob("checkpoint_*.pt"))
    if candidates:
        return candidates[-1]
    latest = run_dir / "checkpoint_latest.pt"
    return latest if latest.is_file() else None


def scan_metrics(path: Path, *, max_rows: int) -> Candidate | None:
    best: Candidate | None = None
    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return None
    with handle:
        for idx, line in enumerate(handle):
            if idx >= max_rows:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            geom = _geometry(row)
            signed_depth = _float(geom.get("signed_depth_m_by_env") or geom.get("signed_depth_m"))
            if signed_depth is None:
                continue
            lateral = _float(geom.get("lateral_error_m_by_env") or geom.get("lateral_error_m"))
            theta = _float(geom.get("orientation_error_rad_by_env") or geom.get("orientation_error_rad"))
            consistency = _float(geom.get("consistency_gate_by_env") or geom.get("consistency_gate"))
            strict_success = _bool(
                geom.get("strict_success_by_env")
                or geom.get("success_by_env")
                or row.get("target_success_by_env")
                or row.get("target_success_fraction")
            )
            step = row.get("step")
            step = int(step) if isinstance(step, int) else None
            candidate = Candidate(
                signed_depth_m=signed_depth,
                step=step,
                lateral_error_m=lateral,
                theta_rad=theta,
                consistency=consistency,
                strict_success=strict_success,
                run_dir=path.parent,
                checkpoint=_checkpoint_for(path.parent, step),
            )
            if best is None or candidate.signed_depth_m > best.signed_depth_m:
                best = candidate
    return best


def _fmt_mm(value: float | None) -> str:
    return "NA" if value is None else f"{value * 1000.0:.2f}"


def _fmt_float(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("outputs")])
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max-rows-per-file", type=int, default=5000)
    parser.add_argument("--max-files", type=int, default=1000)
    parser.add_argument("--min-depth-mm", type=float, default=-1000.0)
    args = parser.parse_args()

    candidates: list[Candidate] = []
    files_seen = 0
    for root in args.roots:
        for path in root.rglob("metrics.jsonl"):
            files_seen += 1
            if files_seen > max(1, int(args.max_files)):
                break
            candidate = scan_metrics(path, max_rows=max(1, int(args.max_rows_per_file)))
            if candidate is None:
                continue
            if candidate.signed_depth_m * 1000.0 < float(args.min_depth_mm):
                continue
            candidates.append(candidate)
        if files_seen > max(1, int(args.max_files)):
            break

    candidates.sort(key=lambda item: item.signed_depth_m, reverse=True)
    print("depth_mm,step,r_mm,theta_rad,consistency,strict_success,checkpoint,run_dir")
    for item in candidates[: max(1, int(args.limit))]:
        checkpoint = "" if item.checkpoint is None else str(item.checkpoint)
        print(
            ",".join(
                [
                    f"{item.signed_depth_m * 1000.0:.2f}",
                    "" if item.step is None else str(item.step),
                    _fmt_mm(item.lateral_error_m),
                    _fmt_float(item.theta_rad),
                    _fmt_float(item.consistency),
                    "" if item.strict_success is None else str(item.strict_success).lower(),
                    checkpoint,
                    str(item.run_dir),
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
