#!/usr/bin/env python3
"""Audit Gazebo SDF SFP/NIC collision geometry used by insertion experiments."""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SFP_MODULE = REPO_ROOT / "aic_assets/models/SFP Module/model.sdf"
DEFAULT_SFP_CABLE = REPO_ROOT / "aic_assets/models/sfp_sc_cable/model.sdf"
DEFAULT_NIC_CARD = REPO_ROOT / "aic_assets/models/NIC Card/model.sdf"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/agentic_reward_curriculum_20260529/collision_audits"


@dataclass(frozen=True)
class BoxCollision:
    name: str
    pose_xyz_rpy: tuple[float, float, float, float, float, float]
    size_xyz: tuple[float, float, float]

    @property
    def volume_m3(self) -> float:
        x, y, z = self.size_xyz
        return x * y * z

    @property
    def local_aabb_min(self) -> tuple[float, float, float]:
        x, y, z, *_ = self.pose_xyz_rpy
        sx, sy, sz = self.size_xyz
        return (x - sx / 2.0, y - sy / 2.0, z - sz / 2.0)

    @property
    def local_aabb_max(self) -> tuple[float, float, float]:
        x, y, z, *_ = self.pose_xyz_rpy
        sx, sy, sz = self.size_xyz
        return (x + sx / 2.0, y + sy / 2.0, z + sz / 2.0)


def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(v) for v in text.split())


def _parse_box_collisions(path: Path) -> list[BoxCollision]:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    out: list[BoxCollision] = []
    for collision in root.findall(".//collision"):
        name = str(collision.attrib.get("name", ""))
        box = collision.find("./geometry/box")
        if box is None:
            continue
        pose_elem = collision.find("./pose")
        size_elem = box.find("./size")
        if size_elem is None:
            continue
        pose = _parse_floats(pose_elem.text or "0 0 0 0 0 0") if pose_elem is not None else (0, 0, 0, 0, 0, 0)
        size = _parse_floats(size_elem.text or "")
        if len(pose) != 6 or len(size) != 3:
            raise ValueError(f"unexpected collision pose/size for {name} in {path}")
        out.append(BoxCollision(name=name, pose_xyz_rpy=pose, size_xyz=size))
    return out


def _parse_removed_sfp_collisions(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    removed: set[str] = set()
    for match in re.finditer(r'<collision\s+element_id="sfp_module_link::([^"]+)"\s+action="remove"\s*/>', text):
        removed.add(match.group(1))
    return removed


def _combined_aabb(collisions: list[BoxCollision]) -> dict[str, Any] | None:
    if not collisions:
        return None
    mins = [min(c.local_aabb_min[i] for c in collisions) for i in range(3)]
    maxs = [max(c.local_aabb_max[i] for c in collisions) for i in range(3)]
    return {
        "min_m": mins,
        "max_m": maxs,
        "size_m": [maxs[i] - mins[i] for i in range(3)],
        "min_mm": [1000.0 * v for v in mins],
        "max_mm": [1000.0 * v for v in maxs],
        "size_mm": [1000.0 * (maxs[i] - mins[i]) for i in range(3)],
    }


def _collision_dict(c: BoxCollision) -> dict[str, Any]:
    return {
        "name": c.name,
        "pose_xyz_rpy_m_rad": list(c.pose_xyz_rpy),
        "size_m": list(c.size_xyz),
        "size_mm": [1000.0 * v for v in c.size_xyz],
        "local_aabb_min_m": list(c.local_aabb_min),
        "local_aabb_max_m": list(c.local_aabb_max),
        "volume_m3": c.volume_m3,
    }


def _nic_cage_openings(nic_boxes: list[BoxCollision]) -> list[dict[str, Any]]:
    cage = [c for c in nic_boxes if c.name.startswith("10099100-011lfc001_collider_box")]
    groups: list[list[BoxCollision]] = []
    for x_center in (-0.010237, 0.012963):
        group = [c for c in cage if abs(c.pose_xyz_rpy[0] - x_center) < 0.012 or abs(c.pose_xyz_rpy[0] - (x_center - 0.00755)) < 0.001 or abs(c.pose_xyz_rpy[0] - (x_center + 0.00755)) < 0.001]
        # Keep groups unambiguous by selecting the five boxes nearest each port center.
        group = sorted(group, key=lambda c: abs(c.pose_xyz_rpy[0] - x_center))[:5]
        groups.append(group)
    openings: list[dict[str, Any]] = []
    for idx, group in enumerate(groups):
        if len(group) < 5:
            openings.append({"port_index": idx, "error": "could not identify five cage boxes", "boxes": [c.name for c in group]})
            continue
        left_wall = min(group, key=lambda c: c.pose_xyz_rpy[0])
        right_wall = max(group, key=lambda c: c.pose_xyz_rpy[0])
        bottom = min(group, key=lambda c: c.pose_xyz_rpy[2])
        top = max(group, key=lambda c: c.pose_xyz_rpy[2])
        rear = min(group, key=lambda c: c.size_xyz[1])
        x_min = left_wall.pose_xyz_rpy[0] + left_wall.size_xyz[0] / 2.0
        x_max = right_wall.pose_xyz_rpy[0] - right_wall.size_xyz[0] / 2.0
        z_min = bottom.pose_xyz_rpy[2] + bottom.size_xyz[2] / 2.0
        z_max = top.pose_xyz_rpy[2] - top.size_xyz[2] / 2.0
        depth = max(c.size_xyz[1] for c in group)
        openings.append(
            {
                "port_index": idx,
                "boxes": [c.name for c in group],
                "opening_width_m": x_max - x_min,
                "opening_height_m": z_max - z_min,
                "cage_depth_m": depth,
                "opening_width_mm": 1000.0 * (x_max - x_min),
                "opening_height_mm": 1000.0 * (z_max - z_min),
                "cage_depth_mm": 1000.0 * depth,
                "rear_lip_box": rear.name,
                "x_bounds_m": [x_min, x_max],
                "z_bounds_m": [z_min, z_max],
            }
        )
    return openings


def _load_runtime_reports(paths: list[Path]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - diagnostic path
            reports.append({"path": str(path), "error": str(exc)})
            continue
        reports.append(
            {
                "path": str(path),
                "mode": payload.get("mode"),
                "matched_count": payload.get("matched_count"),
                "created_count": payload.get("created_count"),
                "created_names": [item.get("source_sdf_collision") for item in payload.get("created", [])],
            }
        )
    return reports


def _write_markdown(path: Path, data: dict[str, Any]) -> None:
    body = data["sfp_module"]["active_in_gazebo_body_aabb"]
    openings = data["nic_card"]["cage_openings"]
    lines = [
        "# SFP/NIC Collision Geometry Audit",
        "",
        "This is an offline SDF/report audit. It does not claim insertion success.",
        "",
        "## Key Dimensions",
        "",
        f"- Active Gazebo SFP body AABB size: `{body['size_mm'][0]:.3f} x {body['size_mm'][1]:.3f} x {body['size_mm'][2]:.3f} mm`.",
    ]
    for opening in openings:
        if "error" in opening:
            lines.append(f"- NIC port {opening['port_index']}: {opening['error']}.")
            continue
        lines.append(
            "- NIC port "
            f"{opening['port_index']} opening: width `{opening['opening_width_mm']:.3f} mm`, "
            f"height `{opening['opening_height_mm']:.3f} mm`, cage depth `{opening['cage_depth_mm']:.3f} mm`."
        )
    lines.extend(
        [
            "",
            "## Collider Set Mismatch",
            "",
            f"- SFP Module raw SDF box colliders: `{data['sfp_module']['raw_box_count']}`.",
            f"- Gazebo cable wrapper removes: `{len(data['sfp_module']['removed_by_sfp_sc_cable'])}` SFP module colliders.",
            f"- Gazebo-active SFP module colliders: `{len(data['sfp_module']['active_in_gazebo'])}`.",
            "- The current body-only runtime replacement creates only `body_collider_box*`, which matches the long body shell but not the remaining Gazebo-active front/port detail boxes.",
            "- The all-box replacement would reintroduce colliders that Gazebo explicitly removes in `sfp_sc_cable/model.sdf`.",
            "",
            "## Runtime Reports",
        ]
    )
    for report in data.get("runtime_reports", []):
        lines.append(
            f"- `{report['path']}`: mode `{report.get('mode')}`, matched `{report.get('matched_count')}`, "
            f"created `{report.get('created_count')}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The SFP body shell is almost the same length as the NIC cage depth, leaving little room for contact-model error. "
            "The previous v725/v727 results therefore fit a collision/contact mismatch: disabling the converted mesh helps "
            "partial-depth progress, but the current body-box replacement alone creates early resistance in the full guide rollout.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfp-module-sdf", type=Path, default=DEFAULT_SFP_MODULE)
    parser.add_argument("--sfp-cable-sdf", type=Path, default=DEFAULT_SFP_CABLE)
    parser.add_argument("--nic-card-sdf", type=Path, default=DEFAULT_NIC_CARD)
    parser.add_argument("--runtime-report", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    sfp_boxes = _parse_box_collisions(args.sfp_module_sdf)
    nic_boxes = _parse_box_collisions(args.nic_card_sdf)
    removed = _parse_removed_sfp_collisions(args.sfp_cable_sdf)
    active = [c for c in sfp_boxes if c.name not in removed]
    body_active = [c for c in active if c.name.startswith("body_collider_box")]

    data = {
        "inputs": {
            "sfp_module_sdf": str(args.sfp_module_sdf),
            "sfp_cable_sdf": str(args.sfp_cable_sdf),
            "nic_card_sdf": str(args.nic_card_sdf),
        },
        "sfp_module": {
            "raw_box_count": len(sfp_boxes),
            "removed_by_sfp_sc_cable": sorted(removed),
            "active_in_gazebo": [_collision_dict(c) for c in active],
            "active_in_gazebo_aabb": _combined_aabb(active),
            "active_in_gazebo_body_boxes": [_collision_dict(c) for c in body_active],
            "active_in_gazebo_body_aabb": _combined_aabb(body_active),
        },
        "nic_card": {
            "box_count": len(nic_boxes),
            "cage_openings": _nic_cage_openings(nic_boxes),
        },
        "runtime_reports": _load_runtime_reports(args.runtime_report),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "sfp_nic_collision_geometry_audit.json"
    md_path = args.output_dir / "sfp_nic_collision_geometry_audit.md"
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, data)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
