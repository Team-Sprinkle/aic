#!/usr/bin/env python3
"""Audit insertion reward geometry without launching Isaac Lab."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MDP_DIR = (
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
)
if str(MDP_DIR) not in sys.path:
    sys.path.insert(0, str(MDP_DIR))

from insertion_geometry import compute_insertion_geometry, insertion_corridor_reward, signed_axial_progress_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/reward_audits"))
    parser.add_argument(
        "--target-depth-m",
        type=float,
        default=0.008,
        help=(
            "Seated depth from entrance along the insertion axis. The reward helper "
            "intentionally rejects collapsed geometry below 3 mm."
        ),
    )
    parser.add_argument("--sigma-r", type=float, default=0.0025)
    parser.add_argument("--bypass-penalty-scale", type=float, default=2.0)
    parser.add_argument("--axial-progress-scale", type=float, default=0.001)
    parser.add_argument("--axial-min-m", type=float, default=-0.004)
    parser.add_argument("--axial-max-m", type=float, default=0.012)
    parser.add_argument("--lateral-max-m", type=float, default=0.012)
    parser.add_argument("--grid", type=int, default=161)
    return parser.parse_args()


def _geometry_for_grid(axial: torch.Tensor, lateral: torch.Tensor, *, target_depth: float, sigma: float):
    body = torch.stack([axial, lateral, torch.zeros_like(axial)], dim=1)
    entrance = torch.zeros_like(body)
    target = torch.zeros_like(body)
    target[:, 0] = float(target_depth)
    axis = torch.zeros_like(body)
    axis[:, 0] = 1.0
    return compute_insertion_geometry(
        body_pos_w=body,
        entrance_pos_w=entrance,
        target_pos_w=target,
        axis_w=axis,
        lateral_gate_sigma=sigma,
    )


def main() -> int:
    args = parse_args()
    run_name = args.run_name or datetime.utcnow().strftime("%Y%m%d_%H%M%S_corridor_audit")
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    axial_values = torch.linspace(float(args.axial_min_m), float(args.axial_max_m), int(args.grid))
    lateral_values = torch.linspace(0.0, float(args.lateral_max_m), int(args.grid))
    axial_grid, lateral_grid = torch.meshgrid(axial_values, lateral_values, indexing="ij")
    axial_flat = axial_grid.reshape(-1)
    lateral_flat = lateral_grid.reshape(-1)
    geometry = _geometry_for_grid(
        axial_flat,
        lateral_flat,
        target_depth=float(args.target_depth_m),
        sigma=float(args.sigma_r),
    )
    corridor = insertion_corridor_reward(geometry, bypass_penalty_scale=float(args.bypass_penalty_scale))
    forward_progress = signed_axial_progress_reward(
        previous_depth=geometry.axial_depth - 0.0005,
        current_depth=geometry.axial_depth,
        lateral_gate=geometry.lateral_gate,
        scale=float(args.axial_progress_scale),
    )
    total = corridor + 0.25 * forward_progress
    corridor_grid = corridor.reshape(axial_grid.shape).numpy()
    progress_grid = forward_progress.reshape(axial_grid.shape).numpy()
    total_grid = total.reshape(axial_grid.shape).numpy()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = [
        0.0,
        float(args.lateral_max_m) * 1000.0,
        float(args.axial_min_m) * 1000.0,
        float(args.axial_max_m) * 1000.0,
    ]
    for name, values, title in [
        ("corridor_reward_slice.png", corridor_grid, "Corridor Reward"),
        ("signed_axial_progress_slice.png", progress_grid, "Signed Axial Progress Reward"),
        ("total_reward_slice.png", total_grid, "Corridor + 0.25 Axial Progress"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(values, origin="lower", aspect="auto", extent=extent, cmap="coolwarm")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axhline(float(args.target_depth_m) * 1000.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("lateral error r (mm)")
        ax.set_ylabel("axial depth s (mm)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / name, dpi=160)
        plt.close(fig)

    sample = slice(None, None, max(1, int(axial_flat.numel() // 6000)))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    points = ax.scatter(
        axial_flat[sample].numpy() * 1000.0,
        lateral_flat[sample].numpy() * 1000.0,
        torch.zeros_like(axial_flat[sample]).numpy(),
        c=total[sample].numpy(),
        cmap="coolwarm",
        s=4,
    )
    ax.set_xlabel("axial depth s (mm)")
    ax.set_ylabel("lateral error r (mm)")
    ax.set_zlabel("orthogonal lateral dim (mm)")
    ax.set_title("3D Port-Frame Reward Sheet")
    fig.colorbar(points, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "reward_3d_port_frame.png", dpi=160)
    plt.close(fig)

    def reward_at(s: float, r: float) -> dict[str, float]:
        g = _geometry_for_grid(
            torch.tensor([s], dtype=torch.float32),
            torch.tensor([r], dtype=torch.float32),
            target_depth=float(args.target_depth_m),
            sigma=float(args.sigma_r),
        )
        corr = insertion_corridor_reward(g, bypass_penalty_scale=float(args.bypass_penalty_scale))
        prog = signed_axial_progress_reward(
            previous_depth=g.axial_depth - 0.0005,
            current_depth=g.axial_depth,
            lateral_gate=g.lateral_gate,
            scale=float(args.axial_progress_scale),
        )
        return {
            "depth_fraction": float(g.depth_fraction[0]),
            "lateral_gate": float(g.lateral_gate[0]),
            "corridor_reward": float(corr[0]),
            "forward_progress_reward": float(prog[0]),
            "total_probe_reward": float(corr[0] + 0.25 * prog[0]),
        }

    probes = {
        "outside_center_4mm": reward_at(-0.004, 0.0),
        "entrance_center": reward_at(0.0, 0.0),
        "half_insert_center": reward_at(float(args.target_depth_m) * 0.5, 0.0),
        "seated_center": reward_at(float(args.target_depth_m), 0.0),
        "deep_side_5mm": reward_at(float(args.target_depth_m) * 0.5, 0.005),
        "deep_side_12mm": reward_at(float(args.target_depth_m) * 0.5, 0.012),
    }
    summary = {
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "outputs": {
            "corridor_slice": str(output_dir / "corridor_reward_slice.png"),
            "signed_axial_progress_slice": str(output_dir / "signed_axial_progress_slice.png"),
            "total_slice": str(output_dir / "total_reward_slice.png"),
            "reward_3d": str(output_dir / "reward_3d_port_frame.png"),
        },
        "min_total_reward": float(np.min(total_grid)),
        "max_total_reward": float(np.max(total_grid)),
        "max_total_reward_location": {
            "axial_depth_m": float(axial_flat[int(torch.argmax(total))]),
            "lateral_error_m": float(lateral_flat[int(torch.argmax(total))]),
        },
        "off_axis_forward_motion_penalized": probes["deep_side_5mm"]["forward_progress_reward"] < 0.0,
        "seated_center_highest_probe": probes["seated_center"]["total_probe_reward"]
        >= max(v["total_probe_reward"] for k, v in probes.items() if k != "seated_center"),
        "probes": probes,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
