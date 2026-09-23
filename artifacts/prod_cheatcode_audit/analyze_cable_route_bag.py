#!/usr/bin/env python3
"""Plot observed cable segments around NIC cards from a Gazebo scoring bag.

Run in the official AIC image after sourcing /ws_aic/install/setup.bash.
This is a privileged diagnostic view, never an actor observation.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation


def transform(msg):
    q, p = msg.rotation, msg.translation
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    out[:3, 3] = [p.x, p.y, p.z]
    return out


def point(matrix):
    return matrix[:3, 3].copy()


def main(bag: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {item.name: get_message(item.type) for item in reader.get_all_topics_and_types()}
    tf = {}
    records = []
    forces = []
    start_ns = None
    next_sample_s = 0.0
    board_key = ("aic_world", "task_board")
    cable_key = ("aic_world", "cable_1")
    cards = [("task_board", f"task_board/nic_card_mount_{i}") for i in range(5)]
    card_links = [(f"task_board/nic_card_mount_{i}",
                   f"task_board/nic_card_mount_{i}/nic_card_link") for i in range(5)]
    links = [("cable_1", f"cable_1/link_{i}") for i in range(1, 21)]
    keys = [board_key, cable_key, *cards, *card_links, *links]
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        if start_ns is None:
            start_ns = stamp
        seconds = (stamp - start_ns) / 1e9
        if topic == "/fts_broadcaster/wrench":
            f = deserialize_message(raw, types[topic]).wrench.force
            forces.append((seconds, float(np.linalg.norm([f.x, f.y, f.z]))))
        if topic != "/scoring/tf":
            continue
        for item in deserialize_message(raw, types[topic]).transforms:
            tf[(item.header.frame_id, item.child_frame_id)] = transform(item.transform)
        if seconds < next_sample_s or any(key not in tf for key in keys):
            continue
        board_inv = np.linalg.inv(tf[board_key])
        cable_world = tf[cable_key]
        cable_board = np.stack([point(board_inv @ cable_world @ tf[key]) for key in links])
        card_board = np.stack([point(tf[a] @ tf[b]) for a, b in zip(cards, card_links)])
        # Distance of each cable segment center to the main PCB collider.
        # Other card protrusions and cable radius are omitted deliberately.
        minimum = (float("inf"), None, None)
        for i, (a, b) in enumerate(zip(cards, card_links)):
            card_inv = np.linalg.inv(tf[board_key] @ tf[a] @ tf[b])
            for j, link_key in enumerate(links):
                local = point(card_inv @ cable_world @ tf[link_key])
                offset = np.maximum(np.abs(local - [0, 0, -0.0008]) - [0.028, 0.0725, 0.0008], 0)
                gap = float(np.linalg.norm(offset))
                if gap < minimum[0]:
                    minimum = (gap, i, j + 1)
        records.append({"time_s": round(seconds, 3),
                        "cable_board_m": cable_board.tolist(),
                        "card_centers_board_m": card_board.tolist(),
                        "nearest_main_pcb_center_gap_m": minimum[0],
                        "nearest_card": minimum[1], "nearest_cable_link": minimum[2]})
        next_sample_s = seconds + 0.5
    if not records:
        raise RuntimeError("No complete cable/card poses in scoring bag")
    (destination / "cable_samples.json").write_text(json.dumps(records, separators=(",", ":")) + "\n")
    force = np.asarray(forces, dtype=float)
    peak_index = int(np.argmax(force[:, 1]))
    peak_t, peak_n = force[peak_index].tolist()
    minimum_record = min(records, key=lambda item: item["nearest_main_pcb_center_gap_m"])
    force_at_minimum_n = float(force[np.argmin(np.abs(force[:, 0] - minimum_record["time_s"])), 1])
    times = [records[0]["time_s"], minimum_record["time_s"], 30, 45, 60,
             records[-1]["time_s"]]
    selected = []
    for target in times:
        nearest = min(records, key=lambda item: abs(item["time_s"] - target))
        if nearest not in selected:
            selected.append(nearest)
    fig, axes = plt.subplots(3, 1, figsize=(11, 13), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(selected)))
    card_xyz = np.asarray(records[0]["card_centers_board_m"])
    for center in card_xyz:
        axes[0].plot([center[0] - .032, center[0] + .032], [center[1]] * 2,
                     color="black", lw=6, alpha=.45)
        axes[1].plot([center[1]] * 2, [center[2] - .074, center[2] + .074],
                     color="black", lw=6, alpha=.45)
    for record, color in zip(selected, colors):
        cable = np.asarray(record["cable_board_m"])
        label = f'{record["time_s"]:.1f}s'
        axes[0].plot(cable[:, 0], cable[:, 1], '-o', ms=2.5, color=color, label=label)
        axes[1].plot(cable[:, 1], cable[:, 2], '-o', ms=2.5, color=color, label=label)
    axes[0].set(xlabel="Board x (m)", ylabel="Board y (m)", title="Overhead cable centerline; black bars are NIC card planes")
    axes[1].set(xlabel="Board y (m)", ylabel="Board z (m)", title="Side cable centerline; black bars are NIC card planes")
    axes[0].axis("equal")
    axes[1].axis("equal")
    axes[0].legend(ncol=3)
    axes[2].plot(force[:, 0], force[:, 1], color="tab:red", label="Measured wrist force")
    axes[2].axvline(peak_t, color="tab:red", ls="--", alpha=.5)
    axes[2].set(xlabel="Time from bag start (s)", ylabel="Wrist force (N)")
    twin = axes[2].twinx()
    twin.plot([r["time_s"] for r in records],
              [1000 * r["nearest_main_pcb_center_gap_m"] for r in records],
              color="tab:blue", label="Nearest cable center to main PCB")
    twin.set_ylabel("Approximate center-to-PCB gap (mm)")
    axes[2].set_title("Force and approximate cable/PCB proximity; gap is not a contact label")
    fig.savefig(destination / "cable_overhead_side_force.png", dpi=155)
    plt.close(fig)
    peak_record = min(records, key=lambda item: abs(item["time_s"] - peak_t))
    at_60 = min(records, key=lambda item: abs(item["time_s"] - 60))
    at_80 = min(records, key=lambda item: abs(item["time_s"] - 80))
    link5_motion_mm = (1000 * float(np.linalg.norm(
        np.asarray(at_80["cable_board_m"][4]) - np.asarray(at_60["cable_board_m"][4])))
        if records[-1]["time_s"] >= 80 else None)
    summary = {"schema": "aic_cable_route_geometry/v1", "bag": str(bag),
               "sample_count": len(records), "peak_force_n": peak_n,
               "peak_force_time_s": peak_t,
               "near_peak_main_pcb_center_gap_mm": 1000 * peak_record["nearest_main_pcb_center_gap_m"],
               "near_peak_nearest_card": peak_record["nearest_card"],
               "near_peak_nearest_cable_link": peak_record["nearest_cable_link"],
               "minimum_main_pcb_center_gap_mm": 1000 * minimum_record["nearest_main_pcb_center_gap_m"],
               "minimum_gap_time_s": minimum_record["time_s"],
               "minimum_gap_card": minimum_record["nearest_card"],
               "minimum_gap_cable_link": minimum_record["nearest_cable_link"],
               "wrist_force_at_minimum_gap_n": force_at_minimum_n,
               "cable_link5_motion_60_to_80_s_mm": link5_motion_mm,
               "link5_motion_sample_times_s": [at_60["time_s"], at_80["time_s"]],
               "selected_snapshot_times_s": [r["time_s"] for r in selected],
               "caveat": "Segment-center proximity to main PCB boxes omits cable radius, card protrusions and named contacts; it cannot establish a snag."}
    (destination / "cable_geometry_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(destination / "cable_overhead_side_force.png", summary["peak_force_n"])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
