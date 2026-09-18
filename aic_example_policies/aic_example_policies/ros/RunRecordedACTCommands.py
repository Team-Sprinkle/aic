"""Diagnostic replay of a recorded trajectory; this is not learned inference.

Use a matching original scene. A TorchScript metadata sidecar must explicitly
contain replay_diagnostic={cache, episode_index, mode}. The inherited ACT model
is loaded solely to reuse its tested runtime/action interface.
"""
from pathlib import Path
import json

import numpy as np
from scipy.spatial.transform import Rotation

from .RunACTTorchScript import RunACTTorchScript


class RunRecordedACTCommands(RunACTTorchScript):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        diagnostic = self.metadata["replay_diagnostic"]
        cache = Path(diagnostic["cache"])
        info = json.loads((cache / "cache.json").read_text())
        episode = next(e for e in info["episodes"] if e["episode_index"] == diagnostic["episode_index"])
        start, stop = episode["cache_from_index"], episode["cache_to_index"]
        self.recorded_actions = np.load(cache / "actions.npy", mmap_mode="r")[start:stop].copy()
        if diagnostic["mode"] == "absolute_pose":
            states = np.load(cache / "states.npy", mmap_mode="r")[start:stop]
            rotation = Rotation.from_quat(states[:, 3:7])
            position = states[:, :3] + rotation.apply(self.recorded_actions[:, :3])
            quaternion = (rotation * Rotation.from_rotvec(self.recorded_actions[:, 3:6])).as_quat()
            quaternion[quaternion[:, 0] < 0] *= -1
            norm = np.linalg.norm(quaternion[:, :3], axis=1)
            rotvec = quaternion[:, :3] * (2 * np.arctan2(norm, quaternion[:, 3]) / np.maximum(norm, 1e-12))[:, None]
            self.recorded_actions = np.concatenate([position, rotvec], axis=1)
        elif diagnostic["mode"] != "delta_pose":
            raise ValueError("Recorded replay requires delta_pose or absolute_pose")
        if diagnostic["mode"] != self.command_mode:
            raise ValueError("Replay mode must match runtime command mode")
        self.recorded_fps = info["fps"]
        self.hold_last_seconds = float(diagnostic.get("hold_last_seconds", 0.))
        self.causal_indexing = bool(diagnostic.get("causal_indexing", False))
        if self.hold_last_seconds < 0:
            raise ValueError("Replay endpoint hold must be nonnegative")
        self.replay_max_runtime_sec = self.max_runtime_sec
        self.replay_start = None
        self.get_logger().warn("RECORDED TRAJECTORY DIAGNOSTIC: no learned actions; exclude from model success rates")

    def insert_cable(self, *args, **kwargs):
        self.replay_start = None
        self.max_runtime_sec = self.replay_max_runtime_sec
        return super().insert_cable(*args, **kwargs)

    def select_delta_action(self, observation):
        stamp = observation.center_image.header.stamp
        now = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if self.replay_start is None:
            self.replay_start = now
        frame_time = (now - self.replay_start) * self.recorded_fps
        index = max(0, min(len(self.recorded_actions) - 1,
                          int(np.floor(frame_time + 1e-6)) if self.causal_indexing else int(round(frame_time))))
        if ((self.hold_last_seconds == 0 and index == len(self.recorded_actions) - 1)
                or frame_time >= len(self.recorded_actions) - 1 + self.hold_last_seconds * self.recorded_fps):
            self.max_runtime_sec = 0.0
        return self.recorded_actions[index].copy()
