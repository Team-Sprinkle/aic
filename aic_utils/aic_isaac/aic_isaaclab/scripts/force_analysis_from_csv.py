#!/usr/bin/env python3
"""Generate force/torque plots from a cheatcode_modified_eval CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Plot Fx/Fy/Fz and Tx/Ty/Tz vs time from CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV.")
    parser.add_argument("--out", type=str, default="", help="Path to output PNG. Defaults to <csv>.force_analysis.png")
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        raise RuntimeError(f"CSV has no data rows: {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for PNG generation.") from exc

    t = [float(r["time_s"]) for r in rows]
    fx = [float(r["force_x_n"]) for r in rows]
    fy = [float(r["force_y_n"]) for r in rows]
    fz = [float(r["force_z_n"]) for r in rows]
    tx = [float(r["torque_x_nm"]) for r in rows]
    ty = [float(r["torque_y_nm"]) for r in rows]
    tz = [float(r["torque_z_nm"]) for r in rows]

    out_path = Path(args.out).expanduser() if args.out else csv_path.with_suffix(".force_analysis.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(t, fx, label="Fx [N]", linewidth=1.0)
    axes[0].plot(t, fy, label="Fy [N]", linewidth=1.0)
    axes[0].plot(t, fz, label="Fz [N]", linewidth=1.0)
    axes[0].set_ylabel("Force [N]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, tx, label="Tx [Nm]", linewidth=1.0)
    axes[1].plot(t, ty, label="Ty [Nm]", linewidth=1.0)
    axes[1].plot(t, tz, label="Tz [Nm]", linewidth=1.0)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Torque [Nm]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.suptitle("Force / Torque vs Time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

