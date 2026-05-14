#!/usr/bin/env python3
"""Summarize offline/online SERL metrics JSONL with trend windows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_KEYS = [
    "reward_mean",
    "target_distance_tanh",
    "target_distance_exp",
    "target_distance_progress",
    "target_orientation_gated_exp",
    "target_success_once_bonus",
    "force_delta_penalty",
    "force_delta_norm",
    "target_distance_mean",
    "target_distance_min",
    "target_distance_max",
    "signed_axial_error",
    "signed_lateral_error",
    "orientation_error",
    "base_action_norm",
    "act_base_action_norm",
    "adapter_delta_norm",
    "raw_adapter_delta_norm",
    "final_action_norm",
    "adapter_clipped_fraction",
    "critic_loss",
    "td_loss",
    "q_mean",
    "actor_loss",
    "bc_loss",
    "updates_done",
    "replay_size",
]


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    else:
        out[prefix] = value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        flat: dict[str, Any] = {}
        _flatten("", raw, flat)
        rows.append(flat)
    return rows


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _window(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0 or n >= len(rows):
        return rows
    return rows[-n:]


def _series(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [v for row in rows if (v := _float(row, key)) is not None]


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return {
        "count": float(len(values)),
        "mean": mean,
        "min": min(values),
        "max": max(values),
        "first": values[0],
        "last": values[-1],
        "last_minus_first": values[-1] - values[0],
    }


def _slope(rows: list[dict[str, Any]], key: str) -> float | None:
    pairs = []
    for idx, row in enumerate(rows):
        y = _float(row, key)
        if y is None:
            continue
        x = _float(row, "step")
        pairs.append((float(idx if x is None else x), y))
    if len(pairs) < 2:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denom


def analyze(path: Path, keys: list[str], windows: list[int]) -> dict[str, Any]:
    rows = _read_rows(path)
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
    )
    selected = []
    for key in keys:
        if key in numeric_keys and key not in selected:
            selected.append(key)
    for key in numeric_keys:
        if (
            key.endswith("_mean")
            or "reward" in key
            or "distance" in key
            or "force" in key
            or "action" in key
            or "loss" in key
            or key in {"q_mean", "updates_done", "replay_size"}
        ) and key not in selected:
            selected.append(key)
    summary: dict[str, Any] = {
        "path": str(path),
        "rows": len(rows),
        "first_step": rows[0].get("step") if rows else None,
        "last_step": rows[-1].get("step") if rows else None,
        "numeric_keys": numeric_keys,
        "keys": {},
    }
    for key in selected:
        entry: dict[str, Any] = {
            "overall": _stats(_series(rows, key)),
            "slope_per_step": _slope(rows, key),
            "windows": {},
        }
        for n in windows:
            entry["windows"][str(n)] = _stats(_series(_window(rows, n), key))
        summary["keys"][key] = entry
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--keys", default="")
    parser.add_argument("--windows", default="100,500")
    args = parser.parse_args()
    keys = [item.strip() for item in args.keys.replace(",", " ").split() if item.strip()] or DEFAULT_KEYS
    windows = [int(item) for item in args.windows.replace(",", " ").split() if item.strip()]
    result = analyze(args.metrics, keys, windows)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
