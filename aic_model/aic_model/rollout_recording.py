"""Optional observation snapshots at the policy boundary; no control changes."""
from __future__ import annotations

import json
import time
from pathlib import Path


class RolloutSnapshots:
    def __init__(self, directory, *, interval_sec=1.0):
        self.directory = Path(directory) / f"task_{time.time_ns()}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.interval = float(interval_sec)
        self.last_time = None
        self.frames = 0
        self.log = (self.directory / "frames.jsonl").open("w")

    def capture(self, observation):
        if observation is None:
            return observation
        image = observation.center_image
        stamp = image.header.stamp
        sim_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if self.last_time is not None and 0 <= sim_time - self.last_time < self.interval:
            return observation
        import cv2
        import numpy as np

        paths = {}
        for camera in ("center", "left", "right"):
            msg = getattr(observation, camera + "_image")
            if not msg.height or not msg.width or not msg.data:
                continue
            encoding = msg.encoding.lower()
            channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(encoding)
            if channels is None:
                raise ValueError(f"Unsupported camera encoding: {encoding}")
            # Respect ROS row padding.
            raw = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
            raw = raw[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
            if channels == 1:
                bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            else:
                bgr = raw[..., :3]
                if encoding.startswith("rgb"):
                    bgr = bgr[..., ::-1]
            path = self.directory / f"{camera}_{self.frames:05d}_{sim_time:.3f}.jpg"
            if not cv2.imwrite(str(path), bgr):
                raise RuntimeError(f"Could not write {path}")
            paths[camera] = path.name
        pose = observation.controller_state.tcp_pose
        row = {"frame": self.frames, "sim_time": sim_time, "wall_time": time.time(), "images": paths,
               "tcp_position": [pose.position.x, pose.position.y, pose.position.z],
               "camera_encoding": {camera: getattr(observation, camera + "_image").encoding
                                   for camera in ("center", "left", "right")}}
        def vector(value, fields=("x", "y", "z")):
            return [float(getattr(value, key)) for key in fields]
        if hasattr(pose, "orientation"):
            row["tcp_orientation_xyzw"] = vector(pose.orientation, ("x", "y", "z", "w"))
        controller = observation.controller_state
        if hasattr(controller, "target_mode"):
            row["controller_target_mode"] = int(controller.target_mode.mode)
        if hasattr(controller, "tcp_velocity"):
            row["tcp_velocity_linear"] = vector(controller.tcp_velocity.linear)
            row["tcp_velocity_angular"] = vector(controller.tcp_velocity.angular)
            row["tcp_error"] = [float(v) for v in controller.tcp_error]
        if hasattr(observation, "joint_states"):
            row["joint_names"] = [str(name) for name in observation.joint_states.name]
            row["joint_positions"] = [float(v) for v in observation.joint_states.position]
        if hasattr(observation, "wrist_wrench"):
            row["wrist_force"] = vector(observation.wrist_wrench.wrench.force)
            row["wrist_torque"] = vector(observation.wrist_wrench.wrench.torque)
        self.log.write(json.dumps(row) + "\n")
        self.log.flush()
        self.last_time = sim_time
        self.frames += 1
        return observation

    def close(self):
        self.log.close()
