#!/usr/bin/env python3
"""Plot Isaac online SERL insertion trajectory diagnostics from metrics.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_path = args.run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    output_dir = args.output_dir or (args.run_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = []
    for row in rows:
        geom = ((row.get("diagnostics") or {}).get("insertion_geometry") or {})
        if not geom:
            continue
        samples.append(
            {
                "step": int(row["step"]),
                "reward_mean": float(row.get("reward_mean", 0.0)),
                "axial_depth_m": float(geom.get("signed_depth_m_mean", 0.0)),
                "target_depth_m": float(geom.get("target_depth_m_mean", 0.0)),
                "depth_fraction": float(geom.get("depth_fraction_mean", 0.0)),
                "lateral_error_m": float(geom.get("lateral_error_m_mean", 0.0)),
                "lateral_gate": float(geom.get("lateral_gate_mean", 0.0)),
                "bypass_risk_fraction": float(geom.get("bypass_risk_fraction", 0.0)),
            }
        )
    if not samples:
        raise ValueError(f"No insertion_geometry diagnostics found in {metrics_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [s["step"] for s in samples]
    axial_mm = [s["axial_depth_m"] * 1000.0 for s in samples]
    lateral_mm = [s["lateral_error_m"] * 1000.0 for s in samples]
    target_mm = samples[0]["target_depth_m"] * 1000.0
    max_axial_depth_m = max(s["axial_depth_m"] for s in samples)
    final_axial_depth_m = samples[-1]["axial_depth_m"]
    retreat_after_entry_m = max(0.0, max_axial_depth_m - final_axial_depth_m) if max_axial_depth_m > 0.0 else 0.0

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lateral_mm, axial_mm, marker="o")
    ax.axhline(0.0, color="black", linewidth=0.8, label="entrance")
    ax.axhline(target_mm, color="black", linewidth=0.8, linestyle="--", label="seated target")
    ax.axvline(2.0, color="tab:green", linewidth=0.8, linestyle=":", label="2mm lateral")
    ax.set_xlabel("lateral error r (mm)")
    ax.set_ylabel("axial depth s (mm)")
    ax.set_title("Insertion Trajectory: Lateral Error vs Axial Depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "lateral_vs_axial.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, axial_mm, marker="o")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(target_mm, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("step")
    ax.set_ylabel("axial depth s (mm)")
    ax.set_title("Axial Depth Over Time")
    fig.tight_layout()
    fig.savefig(output_dir / "axial_depth_over_time.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, lateral_mm, marker="o")
    ax.axhline(2.0, color="tab:green", linewidth=0.8, linestyle=":")
    ax.set_xlabel("step")
    ax.set_ylabel("lateral error r (mm)")
    ax.set_title("Lateral Error Over Time")
    fig.tight_layout()
    fig.savefig(output_dir / "lateral_error_over_time.png", dpi=160)
    plt.close(fig)

    summary = {
        "run_dir": str(args.run_dir),
        "sample_count": len(samples),
        "final_lateral_error_mm": lateral_mm[-1],
        "best_lateral_error_mm": min(lateral_mm),
        "final_axial_depth_mm": axial_mm[-1],
        "max_axial_depth_mm": max(axial_mm),
        "entered_positive_depth": max_axial_depth_m > 0.0,
        "retreat_after_entry_mm": retreat_after_entry_m * 1000.0,
        "retreated_after_entry": retreat_after_entry_m > 0.00025,
        "max_centered_depth_fraction": max(
            s["depth_fraction"] for s in samples if s["lateral_error_m"] <= 0.002
        )
        if any(s["lateral_error_m"] <= 0.002 for s in samples)
        else 0.0,
        "bypass_seen": any(s["bypass_risk_fraction"] > 0.0 for s in samples),
        "plots": {
            "lateral_vs_axial": str(output_dir / "lateral_vs_axial.png"),
            "axial_depth_over_time": str(output_dir / "axial_depth_over_time.png"),
            "lateral_error_over_time": str(output_dir / "lateral_error_over_time.png"),
        },
    }
    (output_dir / "trajectory_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
