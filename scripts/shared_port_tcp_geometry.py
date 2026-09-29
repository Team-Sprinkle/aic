"""Convert complete port-frame TCP targets to executable TCP-body commands.

These functions use only a predicted current TCP-in-port pose and measured TCP
pose. No simulator port transform is read by the autonomous converter.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def matrix_from_pose9(pose):
    pose = np.asarray(pose, dtype=float).reshape(9)
    first = pose[3:6]
    second = pose[6:9]
    first = first / max(np.linalg.norm(first), 1e-9)
    second = second - np.dot(first, second) * first
    second = second / max(np.linalg.norm(second), 1e-9)
    third = np.cross(first, second)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((first, second, third))
    matrix[:3, 3] = pose[:3]
    return matrix


def matrix_from_xyz_xyzw(pose):
    pose = np.asarray(pose, dtype=float).reshape(7)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def inferred_base_port(observed_tcp_base_xyzw, predicted_tcp_port_pose9):
    """Infer base→port from measured TCP and the actor's estimated relative pose."""
    base_tcp = matrix_from_xyz_xyzw(observed_tcp_base_xyzw)
    port_tcp = matrix_from_pose9(predicted_tcp_port_pose9)
    return base_tcp @ np.linalg.inv(port_tcp)


def body_delta_from_target(current_tcp_base_xyzw, estimated_base_port, target_tcp_port_pose9):
    base_tcp = matrix_from_xyz_xyzw(current_tcp_base_xyzw)
    port_target = matrix_from_pose9(target_tcp_port_pose9)
    body_target = np.linalg.inv(base_tcp) @ estimated_base_port @ port_target
    return np.concatenate((body_target[:3, 3], Rotation.from_matrix(body_target[:3, :3]).as_rotvec()))
