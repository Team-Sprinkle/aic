"""SE(3) utilities and the deterministic RPDP connector-to-TCP adapter.

Quaternion arguments use Isaac's scalar-first ``wxyz`` convention.  RPDP poses
are connector poses relative to a fixed port-opening frame.  Translation is in
metres and orientation uses the continuous first-two-columns 6D encoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    qv, v = torch.broadcast_tensors(q[..., 1:], v)
    qw = torch.broadcast_to(q[..., :1], qv.shape[:-1] + (1,))
    return v + 2 * (qw * torch.cross(qv, v, dim=-1)
                    + torch.cross(qv, torch.cross(qv, v, dim=-1), dim=-1))


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ), dim=-1).reshape(q.shape[:-1] + (3, 3))


def matrix_to_quat(m: torch.Tensor) -> torch.Tensor:
    """Stable matrix to wxyz quaternion conversion for small batches."""
    shape = m.shape[:-2]
    flat = m.reshape(-1, 3, 3)
    out = []
    for r in flat:
        tr = float(torch.trace(r))
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2
            q = [0.25*s, float(r[2,1]-r[1,2])/s,
                 float(r[0,2]-r[2,0])/s, float(r[1,0]-r[0,1])/s]
        else:
            i = int(torch.argmax(torch.diag(r)))
            if i == 0:
                s = math.sqrt(1 + float(r[0,0]-r[1,1]-r[2,2])) * 2
                q = [float(r[2,1]-r[1,2])/s, .25*s,
                     float(r[0,1]+r[1,0])/s, float(r[0,2]+r[2,0])/s]
            elif i == 1:
                s = math.sqrt(1 + float(r[1,1]-r[0,0]-r[2,2])) * 2
                q = [float(r[0,2]-r[2,0])/s, float(r[0,1]+r[1,0])/s,
                     .25*s, float(r[1,2]+r[2,1])/s]
            else:
                s = math.sqrt(1 + float(r[2,2]-r[0,0]-r[1,1])) * 2
                q = [float(r[1,0]-r[0,1])/s, float(r[0,2]+r[2,0])/s,
                     float(r[1,2]+r[2,1])/s, .25*s]
        out.append(torch.tensor(q, dtype=m.dtype, device=m.device))
    return quat_normalize(torch.stack(out).reshape(shape + (4,)))


def quat_to_rot6d(q: torch.Tensor) -> torch.Tensor:
    # Column-major: first two rotation-matrix columns.
    m = quat_to_matrix(q)
    return torch.cat((m[..., :, 0], m[..., :, 1]), dim=-1)


def rot6d_to_quat(x: torch.Tensor) -> torch.Tensor:
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1*a2).sum(-1, keepdim=True)*b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return matrix_to_quat(torch.stack((b1, b2, b3), dim=-1))


def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    q = torch.where(q[..., :1] < 0, -q, q)
    sin_half = q[..., 1:].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sin_half, q[..., :1].clamp_min(1e-12))
    return q[..., 1:] * (angle / sin_half.clamp_min(1e-12))


def rotvec_to_quat(v: torch.Tensor) -> torch.Tensor:
    angle = v.norm(dim=-1, keepdim=True)
    half = .5 * angle
    scale = torch.sin(half) / angle.clamp_min(1e-12)
    xyz = v * scale
    xyz = torch.where(angle < 1e-7, .5*v, xyz)
    return quat_normalize(torch.cat((torch.cos(half), xyz), dim=-1))


def compose(pa: torch.Tensor, qa: torch.Tensor, pb: torch.Tensor, qb: torch.Tensor):
    return pa + quat_apply(qa, pb), quat_normalize(quat_mul(qa, qb))


def inverse(p: torch.Tensor, q: torch.Tensor):
    qi = quat_conjugate(quat_normalize(q))
    return quat_apply(qi, -p), qi


def relative_pose(reference_p: torch.Tensor, reference_q: torch.Tensor,
                  body_p: torch.Tensor, body_q: torch.Tensor):
    pi, qi = inverse(reference_p, reference_q)
    return compose(pi, qi, body_p, body_q)


def interpolate_pose(p0: torch.Tensor, q0: torch.Tensor,
                     p1: torch.Tensor, q1: torch.Tensor, fraction: float):
    # Relative axis-angle interpolation is sufficient for the short 50 ms steps.
    dp, dq = relative_pose(p0, q0, p1, q1)
    step_p = dp * fraction
    step_q = rotvec_to_quat(quat_to_rotvec(dq) * fraction)
    return compose(p0, q0, step_p, step_q)


@dataclass
class ConnectorTCPAdapter:
    """Map connector waypoints to directly executable TCP-frame deltas."""

    tcp_to_connector_p: torch.Tensor
    tcp_to_connector_q: torch.Tensor
    microsteps: int = 4

    def connector_to_tcp(self, connector_p: torch.Tensor, connector_q: torch.Tensor):
        ci_p, ci_q = inverse(self.tcp_to_connector_p, self.tcp_to_connector_q)
        return compose(connector_p, connector_q, ci_p, ci_q)

    def action_chunk(self, current_connector_p: torch.Tensor,
                     current_connector_q: torch.Tensor,
                     next_connector_p: torch.Tensor,
                     next_connector_q: torch.Tensor) -> torch.Tensor:
        cur_p, cur_q = self.connector_to_tcp(current_connector_p, current_connector_q)
        goal_p, goal_q = self.connector_to_tcp(next_connector_p, next_connector_q)
        actions = []
        prev_p, prev_q = cur_p, cur_q
        for index in range(1, self.microsteps + 1):
            waypoint_p, waypoint_q = interpolate_pose(cur_p, cur_q, goal_p, goal_q,
                                                       index / self.microsteps)
            dp, dq = relative_pose(prev_p, prev_q, waypoint_p, waypoint_q)
            actions.append(torch.cat((dp, quat_to_rotvec(dq)), dim=-1))
            prev_p, prev_q = waypoint_p, waypoint_q
        return torch.stack(actions, dim=-2)

    def waypoint_chunk(self, current_connector_p: torch.Tensor,
                       current_connector_q: torch.Tensor,
                       connector_waypoint_p: torch.Tensor,
                       connector_waypoint_q: torch.Tensor) -> torch.Tensor:
        """Convert four successive connector waypoints to four TCP deltas.

        Unlike :meth:`action_chunk`, this preserves the path encoded by the
        recorded 50 ms teacher commands instead of interpolating only the final
        200 ms endpoint.
        """
        previous_p, previous_q = self.connector_to_tcp(current_connector_p, current_connector_q)
        actions=[]
        for index in range(connector_waypoint_p.shape[-2]):
            goal_p,goal_q=self.connector_to_tcp(connector_waypoint_p[...,index,:],
                                                connector_waypoint_q[...,index,:])
            dp,dq=relative_pose(previous_p,previous_q,goal_p,goal_q)
            actions.append(torch.cat((dp,quat_to_rotvec(dq)),dim=-1))
            previous_p,previous_q=goal_p,goal_q
        return torch.stack(actions,dim=-2)


def pose9(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.cat((p, quat_to_rot6d(q)), dim=-1)


def unpack_pose9(x: torch.Tensor):
    return x[..., :3], rot6d_to_quat(x[..., 3:9])
