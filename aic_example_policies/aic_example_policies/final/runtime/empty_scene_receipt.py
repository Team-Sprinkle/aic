"""Inference definitions extracted from empty_scene_receipt.py; no training entry point."""
import numpy as np
from scipy.spatial.transform import Rotation

class EmptySceneConsensus:

    def __init__(self, *, packet_camera_timing=False):
        self.last_stamps = None
        self.count = 0
        self.first_pose = None
        self.packet_camera_timing = packet_camera_timing

    def add(self, *, stamp_s, image_stamps_ns, tcp_pose, linear_speed, angular_speed, clear, task_code):
        stamps = tuple(image_stamps_ns)
        ages = stamp_s - np.array(stamps) * 1e-09
        distinct = self.last_stamps is None or all((x > y for x, y in zip(stamps, self.last_stamps)))
        valid = distinct and ages.min() >= -0.001 and (ages.max() <= 0.1) and (max(stamps) - min(stamps) <= 50000000)
        if self.packet_camera_timing:
            from .public_packet_camera_timing import camera_receipt
            valid = camera_receipt(stamp_s, stamps, self.last_stamps)['accepted']
        valid = valid and linear_speed <= 0.005 and (angular_speed <= 0.03) and clear
        self.last_stamps = stamps
        if not valid:
            self.count = 0
            self.first_pose = None
            return None
        pose = np.asarray(tcp_pose)
        if self.first_pose is None:
            self.first_pose = pose.copy()
        motion = np.linalg.norm(pose[:3] - self.first_pose[:3])
        angle = np.degrees((Rotation.from_quat(self.first_pose[3:]).inv() * Rotation.from_quat(pose[3:])).magnitude())
        if motion > 0.002 or angle > 0.5:
            self.count = 0
            self.first_pose = pose.copy()
            return None
        self.count += 1
        if self.count < 3:
            return None
        return dict(task_code=[float(x) for x in task_code], stamp_s=float(stamp_s), tcp_pose=pose.tolist(), image_stamps_ns=list(stamps), distinct_consistent_observations=self.count, geometry_input=False)

def receipt_matches(receipt, task_code, stamp_s, current_matrix):
    if receipt is None or tuple(receipt['task_code']) != tuple(task_code):
        return False
    if not 0 <= stamp_s - receipt['stamp_s'] <= 1.5:
        return False
    pose = np.asarray(receipt['tcp_pose'])
    if np.linalg.norm(current_matrix[:3, 3] - pose[:3]) > 0.003:
        return False
    angle = np.degrees((Rotation.from_quat(pose[3:]).inv() * Rotation.from_matrix(current_matrix[:3, :3])).magnitude())
    return angle <= 1.0
