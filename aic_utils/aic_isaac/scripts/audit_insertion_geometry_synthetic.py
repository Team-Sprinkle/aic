#!/usr/bin/env python3
"""Synthetic insertion-geometry audit cases.

This is intentionally Isaac-free. It exercises the same pure geometry helper
used by Isaac rewards/diagnostics against hand-placed port-frame points.
"""

from __future__ import annotations

import json
import importlib.util
import math
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
INSERTION_GEOMETRY_PATH = (
    REPO_ROOT
    / "aic_utils"
    / "aic_isaac"
    / "aic_isaaclab"
    / "source"
    / "aic_task"
    / "aic_task"
    / "tasks"
    / "manager_based"
    / "aic_task"
    / "mdp"
    / "insertion_geometry.py"
)
spec = importlib.util.spec_from_file_location("aic_insertion_geometry", INSERTION_GEOMETRY_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to load insertion geometry helper from {INSERTION_GEOMETRY_PATH}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
compute_insertion_geometry = module.compute_insertion_geometry


def classify(depth_fraction: float, centered_depth_fraction: float, lateral_error_m: float) -> dict[str, bool]:
    return {
        "strict_partial": bool(depth_fraction > 0.25 and lateral_error_m < 0.0025),
        "strict_full": bool(depth_fraction > 0.95 and lateral_error_m < 0.0005 and centered_depth_fraction > 0.90),
    }


def main() -> int:
    entrance = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    axis = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    target_depth = 0.024
    target = entrance + target_depth * axis
    sigma = 0.0025
    cases = {
        "centered_before_entrance": entrance - 0.004 * axis,
        "centered_at_entrance": entrance.clone(),
        "centered_half_depth": entrance + 0.5 * target_depth * axis,
        "off_center_5mm_half_depth": entrance + 0.5 * target_depth * axis + torch.tensor([[0.005, 0.0, 0.0]]),
        "centered_seated": target.clone(),
    }
    rows = []
    for name, body in cases.items():
        geom = compute_insertion_geometry(
            body_pos_w=body,
            entrance_pos_w=entrance,
            target_pos_w=target,
            axis_w=axis,
            lateral_gate_sigma=sigma,
        )
        depth_fraction = float(geom.depth_fraction[0])
        lateral_error = float(geom.lateral_error[0])
        centered_depth_fraction = float((geom.depth_fraction * geom.lateral_gate)[0])
        row = {
            "case": name,
            "signed_depth_m": float(geom.axial_depth[0]),
            "target_depth_m": float(geom.target_depth[0]),
            "lateral_error_m": lateral_error,
            "lateral_gate": float(geom.lateral_gate[0]),
            "depth_fraction": depth_fraction,
            "centered_depth_fraction": centered_depth_fraction,
            "target_lateral_residual_m": float(geom.target_lateral_residual[0]),
            "classification": classify(depth_fraction, centered_depth_fraction, lateral_error),
        }
        rows.append(row)
    print(json.dumps({"sigma_m": sigma, "orientation_error_rad_assumed": 0.0, "cases": rows}, indent=2))
    expected = {
        "centered_at_entrance": (False, False),
        "centered_half_depth": (True, False),
        "off_center_5mm_half_depth": (False, False),
        "centered_seated": (True, True),
    }
    by_name = {row["case"]: row["classification"] for row in rows}
    for name, (partial, full) in expected.items():
        got = by_name[name]
        if got["strict_partial"] != partial or got["strict_full"] != full:
            raise SystemExit(f"unexpected classification for {name}: {got}, expected partial={partial} full={full}")
    if not math.isclose(rows[1]["depth_fraction"], 0.0, abs_tol=1.0e-6):
        raise SystemExit("entrance case should have depth_fraction ~= 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
