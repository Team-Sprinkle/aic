#!/usr/bin/env python3
"""Record a bounded Gazebo control probe, with measured motion and camera evidence."""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from gazebo_rl.gym_env import GazeboRLEnv
from gazebo_rl.score_parser import score_from_scoring_yaml


def motion_plan():
    """Twenty-command phases: four small commands, then sixteen zero commands.

    Requested command spacing is 50 ms; measured simulation spacing can be
    longer because inference, image capture, and IPC run while physics advances.
    """
    yield "settle", np.zeros(6)
    for _ in range(19):
        yield "settle", np.zeros(6)
    for axis in range(6):
        for sign in (1, -1):
            name = f"{'xyzXYZ'[axis]}_{'plus' if sign > 0 else 'minus'}"
            for tick in range(20):
                action = np.zeros(6)
                if tick < 4:
                    action[axis] = sign * (0.00025 if axis < 3 else 0.0025)
                yield name, action


class CameraEvidence:
    def __init__(self, root):
        self.root = root
        self.writers = {}
        self.last_snapshots = {}
        self.counts = {}
        self.timestamps = (root / "camera_timestamps.jsonl").open("w")

    def add(self, obs, index):
        for key, payload in (obs.get("images") or {}).items():
            if not payload or not payload.get("data_b64"):
                continue
            camera = key.rsplit(".", 1)[-1]
            stamp = payload.get("stamp") or obs.get("sim_time")
            bgr = cv2.imdecode(np.frombuffer(base64.b64decode(payload["data_b64"]), np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Cannot decode {camera}")
            if camera not in self.writers:
                writer = cv2.VideoWriter(str(self.root / f"{camera}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20,
                                         (bgr.shape[1], bgr.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot write video for {camera}")
                self.writers[camera] = writer
            self.writers[camera].write(bgr)
            self.counts[camera] = self.counts.get(camera, 0) + 1
            self.timestamps.write(json.dumps({"index": index, "camera": camera, "sim_time": stamp}) + "\n")
            previous = self.last_snapshots.get(camera)
            if stamp is not None and (previous is None or stamp - previous >= 0.99):
                directory = self.root / "frames_1s" / camera
                directory.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(directory / f"step_{index:05d}_sim_{stamp:.3f}.jpg"), bgr)
                self.last_snapshots[camera] = stamp

    def close(self):
        for writer in self.writers.values():
            writer.release()
        self.timestamps.close()


def measured_delta(before, after):
    a = (before.get("oracle") or {}).get("tcp_pose_base_link")
    b = (after.get("oracle") or {}).get("tcp_pose_base_link")
    if not a or not b:
        return None
    rotation = Rotation.from_quat(a["orientation_xyzw"])
    translation = rotation.inv().apply(np.asarray(b["position"]) - np.asarray(a["position"]))
    angular = (rotation.inv() * Rotation.from_quat(b["orientation_xyzw"])).as_rotvec()
    return [*translation.tolist(), *angular.tolist()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True, help="Dedicated rootless evaluation container")
    parser.add_argument("--engine-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-config", type=Path, help="Optional physical pre-position configuration")
    parser.add_argument("--timeout-sec", type=float, default=240)
    parser.add_argument("--zero-action-mode", choices=["retarget_current_tcp", "hold_previous_target"],
                        default="retarget_current_tcp")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    plan = list(motion_plan())
    episode = yaml.safe_load(args.episode_config.read_text()) if args.episode_config else None
    env = GazeboRLEnv(workspace_dir=repo, workspace_container=repo, engine_config=str(args.engine_config.resolve()),
                      sim_docker_container=args.container, docker_host=os.environ.get("DOCKER_HOST"),
                      ground_truth=True, include_images=True, max_steps=len(plan), command_dt_sec=0.05,
                      per_trial_timeout_sec=args.timeout_sec, results_dir=root / "results",
                      episode_config=episode, episode_config_path=args.episode_config)
    # This diagnostic owns only its dedicated container; never kill host routers.
    env.runner.config.clean_stale_zenoh = False
    env.runner.config.zero_action_mode = args.zero_action_mode
    evidence = CameraEvidence(root)
    summary = {"evaluation_kind": "privileged_control_diagnostic", "command_frame": "gripper/tcp",
               "camera_video_fps": 20, "video_timing": "one frame per observation; consult camera_timestamps.jsonl",
               "planned_steps": len(plan), "phases": [], "completed_steps": 0,
               "zero_action_mode": args.zero_action_mode}
    started = time.monotonic()
    try:
        obs, info = env.reset()
        summary["reset_info"] = info
        evidence.add(obs, 0)
        phase, origin, total_action = None, obs, np.zeros(6)
        with (root / "observations.jsonl").open("w") as log:
            log.write(json.dumps({"step": 0, "observation": {k: v for k, v in obs.items() if k != "images"}}) + "\n")
            for index, (name, action) in enumerate(plan, 1):
                if name != phase:
                    if phase is not None:
                        summary["phases"].append({"name": phase, "command_sum": total_action.tolist(),
                                                  "measured_tcp_delta_in_start_frame": measured_delta(origin, obs)})
                    phase, origin, total_action = name, obs, np.zeros(6)
                obs, reward, terminated, truncated, info = env.step(action.tolist())
                total_action += action
                summary["completed_steps"] = index
                log.write(json.dumps({"step": index, "phase": name, "action": action.tolist(), "reward": reward,
                                       "observation": {k: v for k, v in obs.items() if k != "images"}, "info": info}) + "\n")
                log.flush()
                evidence.add(obs, index)
                if terminated or truncated:
                    break
            summary["phases"].append({"name": phase, "command_sum": total_action.tolist(),
                                      "measured_tcp_delta_in_start_frame": measured_delta(origin, obs)})
        # Scoring follows the policy return; do not kill the engine at bridge completion.
        engine = next(p.process for p in env.runner.processes if p.name == "engine")
        summary["engine_returncode"] = engine.wait(timeout=90)
        summary["score"] = score_from_scoring_yaml(env.results_dir)
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        evidence.close()
        env.close()
        summary["camera_frame_counts"] = evidence.counts
        summary["elapsed_wall_seconds"] = time.monotonic() - started
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
