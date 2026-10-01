#!/usr/bin/env python3
"""Prepare deterministic SC Low Lateral Bypass collection scenes.

The planner is privileged teacher tooling. These scenes never enter a final split.
Every requested scene remains in the collection denominator, including planning
failures. Geometry is transformed from a scored LLB reference, then MoveIt checks
the generated scene's rigid collision objects before any Gazebo replay.
"""

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "outputs/experiments/2026-09-30_transport_continuation"


def pose_matrix(pose):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler(
        "xyz", [pose["roll"], pose["pitch"], pose["yaw"]]
    ).as_matrix()
    result[:3, 3] = [pose["x"], pose["y"], pose["z"]]
    return result


def edge_matrix(edges, key):
    item = edges[key][0][1]
    position, rotation = item["translation"], item["rotation"]
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat([rotation[k] for k in "xyzw"]).as_matrix()
    result[:3, 3] = [position[k] for k in "xyz"]
    return result


def transform_point(matrix, point):
    return (matrix @ np.r_[point, 1.0])[:3]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()
    if args.count <= 0 or args.output.exists():
        parser.error("count must be positive and output must not exist")
    args.output.mkdir(parents=True)
    scenes = args.output / "scenes"
    scenes.mkdir()
    rng = random.Random(args.seed)

    reference_scene = REFERENCE / "scenes/port1_five_shiftminus"
    source_config = yaml.safe_load((reference_scene / "eval_config.yaml").read_text())
    source_board = next(iter(source_config["trials"].values()))["scene"]["task_board"]
    old_board = pose_matrix(source_board["pose"])
    stream = json.loads((REFERENCE / "streams/cartesian_port1_five.json").read_text())
    edges = stream["edges"]
    world_base = edge_matrix(edges, "world|tabletop") @ edge_matrix(edges, "tabletop|base_link")
    observed_board = edge_matrix(edges, "aic_world|task_board")
    assert np.linalg.norm(observed_board[:3, 3] - old_board[:3, 3]) < 1e-4
    old_base_board = np.linalg.inv(world_base) @ old_board
    aligned = json.loads((REFERENCE / "plans/port1_five_shiftminus/plan_summary.json").read_text())[
        "candidate"
    ]["metadata"]["aligned_orientation_xyzw"]
    start_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]
    rows = []
    for index in range(args.start_index, args.start_index + args.count):
        name = f"llb_{index:04d}"
        destination = scenes / name
        destination.mkdir()
        port = index % 2
        card_count = index % 5 + 1
        config = json.loads(json.dumps(source_config))
        trial = next(iter(config["trials"].values()))
        board = trial["scene"]["task_board"]
        board["pose"].update({
            "x": round(rng.uniform(0.125, 0.175), 5),
            "y": round(rng.uniform(-0.235, -0.135), 5),
            "yaw": round(rng.uniform(2.96, 3.16), 5),
        })
        present = set(rng.sample(range(5), card_count))
        card_variations = {}
        for rail in range(5):
            item = board[f"nic_rail_{rail}"]
            item["entity_present"] = rail in present
            if rail in present:
                old = source_board[f"nic_rail_{rail}"]["entity_pose"]
                item["entity_name"] = f"nic_card_{rail}"
                item["entity_pose"] = dict(old)
                item["entity_pose"]["translation"] = round(
                    max(-0.0215, min(0.0234, old["translation"] + rng.uniform(-0.010, 0.010))), 5
                )
                item["entity_pose"]["yaw"] = round(rng.uniform(-0.09, 0.09), 5)
                card_variations[str(rail)] = item["entity_pose"]
            else:
                item.pop("entity_pose", None)
        port_variations = {}
        for rail in range(2):
            item = board[f"sc_rail_{rail}"]
            old = source_board[f"sc_rail_{rail}"]["entity_pose"]["translation"]
            item["entity_present"] = True
            item["entity_name"] = f"sc_mount_{rail}"
            item["entity_pose"] = dict(source_board[f"sc_rail_{rail}"]["entity_pose"])
            item["entity_pose"]["translation"] = round(
                max(-0.06, min(0.055, old + rng.uniform(-0.010, 0.010))), 5
            )
            port_variations[str(rail)] = item["entity_pose"]
        trial["tasks"]["task_1"]["target_module_name"] = f"sc_port_{port}"
        cable = trial["scene"]["cables"]["cable_1"]["pose"]
        for axis in "xyz":
            cable["gripper_offset"][axis] = round(
                cable["gripper_offset"][axis] + rng.uniform(-0.001, 0.001), 5
            )
        for axis in ("roll", "pitch", "yaw"):
            cable[axis] = round(cable[axis] + rng.uniform(-0.015, 0.015), 5)
        start = config["robot"]["home_joint_positions"]
        for joint in start_names:
            start[joint] = round(start[joint] + rng.uniform(-0.012, 0.012), 5)
        joint_state = {
            "name": start_names,
            "position": [start[name] for name in start_names],
            "source": "exact requested home joint positions in this collection config",
        }
        config["trials"] = {f"trial_{1700000 + index}": trial}
        (destination / "eval_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (destination / "measured_start_joint_state.json").write_text(json.dumps(joint_state, indent=2) + "\n")
        for filename in ("world_audit.sdf", "bridge_audit.yaml"):
            shutil.copy2(reference_scene / filename, destination / filename)

        reference_route = REFERENCE / "scenes" / (
            "port0_five_paired" if port == 0 else "port1_five_shiftminus"
        ) / "route_plan.json"
        route = json.loads(reference_route.read_text())
        new_base_board = np.linalg.inv(world_base) @ pose_matrix(board["pose"])
        base_transform = new_base_board @ np.linalg.inv(old_base_board)
        old_port_translation = source_board[f"sc_rail_{port}"]["entity_pose"]["translation"]
        new_port_translation = board[f"sc_rail_{port}"]["entity_pose"]["translation"]
        port_delta_base = new_base_board[:3, :3] @ np.array(
            [new_port_translation - old_port_translation, 0.0, 0.0]
        )
        route["start_tcp_base_m"] = None  # Actual planner FK from exact requested start is authoritative.
        route["port_link_base_m"] = (
            transform_point(base_transform, route["port_link_base_m"]) + port_delta_base
        ).tolist()
        route["nic_card_centers_base_m"] = [
            transform_point(base_transform, point).tolist()
            for point in route["nic_card_centers_base_m"]
        ]
        for waypoint_index, waypoint in enumerate(route["waypoints"]):
            point = transform_point(base_transform, waypoint["tcp_base_m"])
            if waypoint_index >= 4:
                point += port_delta_base
            waypoint["tcp_base_m"] = point.tolist()
        route["lane_x_m"] = route["waypoints"][2]["tcp_base_m"][0]
        route["derived_from"] = str(reference_route)
        (destination / "route_plan.json").write_text(json.dumps(route, indent=2) + "\n")
        orientation_delta = new_base_board[:3, :3] @ old_base_board[:3, :3].T
        new_aligned = (
            Rotation.from_matrix(orientation_delta) * Rotation.from_quat(aligned)
        ).as_quat().tolist()
        case = {
            "name": name,
            "index": index,
            "trial_id": f"trial_{1700000 + index}",
            "selected_port": port,
            "card_count": card_count,
            "present_nic_rails": sorted(present),
            "board_pose": board["pose"],
            "card_poses": card_variations,
            "port_poses": port_variations,
            "robot_home_joint_positions": start,
            "cable_pose": cable,
            "aligned_orientation_xyzw": new_aligned,
            "scene": str(destination),
            "scene_sha256": digest(destination / "eval_config.yaml"),
            "route_sha256": digest(destination / "route_plan.json"),
            "reference_route": str(reference_route),
            "split": "teacher_collection_only",
            "reserved_final": False,
        }
        (destination / "case_manifest.json").write_text(json.dumps(case, indent=2) + "\n")
        rows.append(case)
    manifest = {
        "schema": "aic_sc_llb_collection/v1",
        "route_name": "Low Lateral Bypass (LLB)",
        "requested_attempts": args.count,
        "first_attempt_index": args.start_index,
        "pilot_gate": "At least 10 official full insertions in the first 30 attempted settings",
        "failure_denominator": "all 30 requested settings, including planning and simulator failures",
        "source_scene": str(reference_scene),
        "source_stream": str(REFERENCE / "streams/cartesian_port1_five.json"),
        "cases": rows,
        "reserved_final_scenes_opened": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"count": len(rows), "ports": {str(p): sum(r["selected_port"] == p for r in rows) for p in (0, 1)},
                      "card_counts": {str(n): sum(r["card_count"] == n for r in rows) for n in range(1, 6)},
                      "manifest": str(args.output / "manifest.json")}))


if __name__ == "__main__":
    main()
