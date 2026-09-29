#!/usr/bin/env python3
"""Convert recorded SC teacher TCP targets to each episode's selected port frame.

Training-only simulator TF supplies the frame; this script audits labels and
timing. It does not supply geometry to an autonomous policy.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def matrix(edge: dict) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(edge["quat_xyzw"]).as_matrix()
    result[:3, 3] = edge["xyz_m"]
    return result


def pose_matrix(pose: list[float]) -> np.ndarray:
    """AIC recorded pose: XYZ position followed by XYZW quaternion."""
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def pose_list(transform: np.ndarray) -> list[float]:
    xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return [*transform[:3, 3].tolist(), *xyzw.tolist()]


def select(edges: list[dict], parent: str, child: str) -> dict:
    matches = [edge for edge in edges if edge["parent"] == parent and edge["child"] == child]
    if not matches:
        raise ValueError(f"Missing {parent} -> {child} TF")
    return next((edge for edge in matches if edge["topic"] == "/scoring/tf"), matches[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--port-edges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels-output", type=Path, required=True)
    args = parser.parse_args()
    source = {row["episode_index"]: row for row in json.loads(args.manifest.read_text())["episodes"]}
    ports = json.loads(args.port_edges.read_text())["episodes"]
    reports = []
    args.labels_output.parent.mkdir(parents=True, exist_ok=True)
    with args.labels_output.open("w") as labels:
        for port_row in ports:
            episode = source[port_row["episode_index"]]
            edges = port_row["edges"]
            index = port_row["target_port_index"]
            port_parent = f"task_board/sc_port_{index}"
            port_name = f"{port_parent}/sc_port_base_link"
            world_base = matrix(select(edges, "world", "tabletop")) @ matrix(select(edges, "tabletop", "base_link"))
            world_port = (matrix(select(edges, "aic_world", "task_board"))
                          @ matrix(select(edges, "task_board", port_parent))
                          @ matrix(select(edges, port_parent, port_name)))
            base_port = np.linalg.inv(world_base) @ world_port
            port_base = np.linalg.inv(base_port)
            frames = [json.loads(line) for line in (Path(episode["canonical_episode_dir"]) / "frames.jsonl").read_text().splitlines()]
            indices = [int(row["command_index"]) for row in frames]
            times = [float(row["sim_time"]) for row in frames]
            if indices != sorted(set(indices)):
                raise ValueError(f"Duplicate or unordered command indices in episode {episode['episode_index']}")
            if len(frames) != episode["original_recorded_frames"]:
                raise ValueError(f"Frame count mismatch in episode {episode['episode_index']}")
            gap_durations = [b - a for a, b in zip(times, times[1:])]
            valid_chunks = sum(
                all(indices[j + k + 1] == indices[j + k] + 1 and
                    times[j + k + 1] - times[j + k] <= 0.25 for k in range(3))
                for j in range(max(0, len(frames) - 3))
            )
            max_position_roundtrip = 0.0
            max_rotation_roundtrip = 0.0
            terminal_target = None
            for row in frames:
                teacher = pose_matrix(row["teacher_target_pose"])
                relative = port_base @ teacher
                recovered = base_port @ relative
                max_position_roundtrip = max(max_position_roundtrip,
                                             float(np.linalg.norm(recovered[:3, 3] - teacher[:3, 3])))
                angle = Rotation.from_matrix(recovered[:3, :3] @ teacher[:3, :3].T).magnitude()
                max_rotation_roundtrip = max(max_rotation_roundtrip, float(angle))
                terminal_target = pose_list(relative)
                labels.write(json.dumps({
                    "episode_index": episode["episode_index"], "scene_sha256": episode["scene_sha256"],
                    "split": episode["split"], "frame": row["frame"],
                    "command_index": row["command_index"], "sim_time": row["sim_time"],
                    "images": row["images"], "teacher_tcp_target_port_frame": pose_list(relative),
                    "selected_port_frame": port_name,
                }, separators=(",", ":")) + "\n")
            reports.append({
                "episode_index": episode["episode_index"], "split": episode["split"],
                "scene_sha256": episode["scene_sha256"], "nic_count": episode["nic_count"],
                "target_port_index": index, "selected_port_frame": port_name,
                "frames": len(frames), "command_index_span": max(indices) - min(indices) + 1,
                "missing_command_indices": max(indices) - min(indices) + 1 - len(indices),
                "valid_contiguous_four_command_chunks": valid_chunks,
                "possible_four_command_chunks": max(0, len(frames) - 3),
                "max_sim_time_gap_s": max(gap_durations),
                "gaps_over_250ms": sum(gap > 0.25 for gap in gap_durations),
                "terminal_teacher_tcp_target_port_frame": terminal_target,
                "base_selected_port_xyz_m": base_port[:3, 3].tolist(),
                "max_position_roundtrip_m": max_position_roundtrip,
                "max_rotation_roundtrip_rad": max_rotation_roundtrip,
            })
    groups = {row["scene_sha256"] for row in reports}
    output = {
        "schema": "aic_sc_port_tcp_target_audit/v2",
        "pose_quaternion_order": "xyzw",
        "port_frame_source": "Scored training-bag TF; world and aic_world assumed coincident, not autonomous input",
        "warning": "These are TCP teacher targets in the SC receptacle base frame, not connector waypoints or validated fixed-50ms RPDP chunks. Reconstruct opening/tip offsets and observation-only SC pose before policy training.",
        "episodes": reports,
        "counts": {
            "episodes": len(reports), "scene_groups": len(groups),
            "split": dict(Counter(row["split"] for row in reports)),
            "frames": sum(row["frames"] for row in reports),
            "missing_command_indices": sum(row["missing_command_indices"] for row in reports),
            "valid_contiguous_four_command_chunks": sum(row["valid_contiguous_four_command_chunks"] for row in reports),
            "possible_four_command_chunks": sum(row["possible_four_command_chunks"] for row in reports),
            "gaps_over_250ms": sum(row["gaps_over_250ms"] for row in reports),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "counts": output["counts"]}))


if __name__ == "__main__":
    main()
