#!/usr/bin/env python3
"""Audit phase-conditioned insertion reward funnels over `s`, `r`, `theta`, and action direction."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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

from insertion_geometry import cheatcode_insertion_phase_reward, compute_insertion_geometry


@dataclass
class FunnelConfig:
    target_depth_m: float = 0.0458
    lateral_gate_width_m: float = 0.0006
    lateral_gate_width_far_m: float = 0.0040
    orientation_gate_width_rad: float = 0.030
    orientation_gate_width_far_rad: float = 0.100
    axial_progress_gate_m: float = 0.001
    action_axis_gate: bool = True
    action_lateral_sigma_m: float = 0.00005
    module_consistency_penalty: float = 0.80
    bypass_penalty: float = 8.0
    contact_recovery_penalty: float = 0.0
    smoothness_penalty: float = 0.0


def _load_auto_config(path: Path) -> FunnelConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    summaries = data.get("summaries", [])
    near = [
        item
        for item in summaries
        if item.get("strict_success")
        or (
            item.get("final_s_m", -1.0) > 0.004
            and item.get("final_r_m", 99.0) < 0.001
            and item.get("final_theta_rad", 99.0) < 0.06
        )
    ]
    if not near:
        return FunnelConfig()
    lateral = float(np.percentile([item["final_r_m"] for item in near], 75)) * 1.5
    theta = float(np.percentile([item["final_theta_rad"] for item in near], 75)) * 1.2
    module = float(np.percentile([item["final_module_consistency"] for item in near], 25))
    return FunnelConfig(
        lateral_gate_width_m=max(0.0004, min(0.0015, lateral)),
        orientation_gate_width_rad=max(0.025, min(0.030, theta)),
        module_consistency_penalty=max(0.60, min(0.95, module)),
        bypass_penalty=8.0,
    )


def _geometry(s: torch.Tensor, r: torch.Tensor, cfg: FunnelConfig):
    body = torch.stack([s, r, torch.zeros_like(s)], dim=1)
    entrance = torch.zeros_like(body)
    target = torch.zeros_like(body)
    target[:, 0] = cfg.target_depth_m
    axis = torch.zeros_like(body)
    axis[:, 0] = 1.0
    return compute_insertion_geometry(
        body_pos_w=body,
        entrance_pos_w=entrance,
        target_pos_w=target,
        axis_w=axis,
        lateral_gate_sigma=cfg.lateral_gate_width_m,
    )


def _reward_grid(cfg: FunnelConfig, theta: float, action_lateral_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    s_values = torch.linspace(-0.006, max(0.012, float(cfg.target_depth_m) * 1.10), 181)
    r_values = torch.linspace(0.0, 0.010, 141)
    s_grid, r_grid = torch.meshgrid(s_values, r_values, indexing="ij")
    s_flat = s_grid.reshape(-1)
    r_flat = r_grid.reshape(-1)
    theta_t = torch.full_like(s_flat, theta)
    action = torch.stack(
        [
            torch.full_like(s_flat, 0.0001),
            torch.full_like(s_flat, action_lateral_m),
            torch.zeros_like(s_flat),
        ],
        dim=1,
    )
    semantic_gate = torch.ones_like(s_flat)
    comp = cheatcode_insertion_phase_reward(
        geometry=_geometry(s_flat, r_flat, cfg),
        previous_depth=s_flat - cfg.axial_progress_gate_m,
        previous_lateral_error=r_flat,
        orientation_error=theta_t,
        previous_orientation_error=theta_t,
        sigma_lat_pre=cfg.lateral_gate_width_m,
        sigma_lat_insert=cfg.lateral_gate_width_m,
        schedule_lateral_radius=True,
        sigma_lat_pre_far=cfg.lateral_gate_width_far_m,
        sigma_lat_insert_far=cfg.lateral_gate_width_far_m,
        sigma_theta_pre=cfg.orientation_gate_width_rad,
        sigma_theta_insert=cfg.orientation_gate_width_rad,
        schedule_orientation_tolerance=True,
        sigma_theta_pre_far=cfg.orientation_gate_width_far_rad,
        sigma_theta_insert_far=cfg.orientation_gate_width_far_rad,
        axial_progress_scale=cfg.axial_progress_gate_m,
        bypass_penalty_scale=cfg.bypass_penalty,
        action_delta_w=action,
        action_axis_gate=cfg.action_axis_gate,
        action_lateral_sigma=cfg.action_lateral_sigma_m,
        semantic_gate=semantic_gate,
        semantic_loss_weight=cfg.module_consistency_penalty,
    )
    total = comp.total.reshape(s_grid.shape).numpy()
    near_insert_region = s_flat >= 0.0
    bad_alignment_region = near_insert_region & (
        (r_flat > cfg.lateral_gate_width_m * 2.0) | (theta_t > cfg.orientation_gate_width_rad * 2.0)
    )
    bad_forward = bool((comp.axial_progress[bad_alignment_region] > 0.0).any())
    summary = {
        "theta_rad": theta,
        "action_lateral_m": action_lateral_m,
        "max_total": float(np.max(total)),
        "min_total": float(np.min(total)),
        "bad_forward_positive_reward": bad_forward,
        "forward_bad_region_max_axial_reward": float(comp.axial_progress[bad_alignment_region].max()),
    }
    return s_grid.numpy(), r_grid.numpy(), total, summary


def _write_plots(run_dir: Path, cfg: FunnelConfig) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries: list[dict[str, Any]] = []
    for theta in (0.0, cfg.orientation_gate_width_rad, cfg.orientation_gate_width_rad * 2.5):
        for action_lateral in (0.0, cfg.action_lateral_sigma_m * 8.0):
            s_grid, r_grid, total, summary = _reward_grid(cfg, theta, action_lateral)
            summaries.append(summary)
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(
                total,
                origin="lower",
                aspect="auto",
                extent=[0.0, float(r_grid.max()) * 1000.0, float(s_grid.min()) * 1000.0, float(s_grid.max()) * 1000.0],
                cmap="coolwarm",
            )
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.axhline(cfg.target_depth_m * 1000.0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("r lateral error (mm)")
            ax.set_ylabel("s axial depth (mm)")
            ax.set_title(f"Reward funnel theta={theta:.3f}, action_lat={action_lateral * 1000.0:.2f}mm")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(run_dir / f"reward_surface_theta_{theta:.3f}_action_lat_{action_lateral:.5f}.png", dpi=160)
            plt.close(fig)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/agent_reward_funnel/reward_audits"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--version", choices=["hand_tuned", "auto_from_servo"], default="hand_tuned")
    parser.add_argument("--servo-metrics-json", type=Path, default=None)
    parser.add_argument("--target-depth-m", type=float, default=None)
    parser.add_argument("--lateral-gate-width-m", type=float, default=None)
    parser.add_argument("--orientation-gate-width-rad", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = FunnelConfig()
    if args.version == "auto_from_servo":
        if args.servo_metrics_json is None:
            raise SystemExit("--servo-metrics-json is required for --version auto_from_servo")
        cfg = _load_auto_config(args.servo_metrics_json)
    if args.lateral_gate_width_m is not None:
        cfg.lateral_gate_width_m = float(args.lateral_gate_width_m)
    if args.target_depth_m is not None:
        cfg.target_depth_m = float(args.target_depth_m)
    if args.orientation_gate_width_rad is not None:
        cfg.orientation_gate_width_rad = float(args.orientation_gate_width_rad)

    run_name = args.run_name or datetime.utcnow().strftime(f"%Y%m%d_%H%M%S_{args.version}")
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "version": args.version,
        "funnel_config": asdict(cfg),
        "servo_metrics_json": None if args.servo_metrics_json is None else str(args.servo_metrics_json),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    summaries = _write_plots(run_dir, cfg)
    bad = [item for item in summaries if item["bad_forward_positive_reward"]]
    (run_dir / "summary.json").write_text(
        json.dumps({"config": config, "surface_summaries": summaries, "bad_forward_surface_count": len(bad)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Phase Reward Funnel Audit",
                "",
                f"Version: `{args.version}`",
                f"Run folder: `{run_dir}`",
                f"Bad forward-positive surfaces: {len(bad)}/{len(summaries)}",
                "",
                "The audit sweeps `s`, `r`, `theta`, and action-axis lateral error. It flags any positive axial reward when lateral or orientation error is outside the tight gate.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(run_dir), "bad_forward_surface_count": len(bad)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
