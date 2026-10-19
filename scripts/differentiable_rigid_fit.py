"""Rigid-fit rotation with the deployment SVD forward and stable SO(3) gradient.

The derivative solves the three-dimensional rotation stationarity equation.
It avoids differentiating arbitrary singular vectors, whose derivatives can
be undefined for repeated singular values even when the rotation is unique.
"""
import torch


def skew(vector):
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)


class RigidRotation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, covariance):
        u, _, vh = torch.linalg.svd(covariance)
        v = vh.transpose(-1, -2)
        ut = u.transpose(-1, -2)
        diagonal = torch.eye(3, dtype=covariance.dtype, device=covariance.device).expand(covariance.shape).clone()
        diagonal[..., 2, 2] = torch.linalg.det(v @ ut)
        rotation = v @ diagonal @ ut
        symmetric = rotation @ covariance
        symmetric = .5 * (symmetric + symmetric.transpose(-1, -2))
        identity = torch.eye(3, dtype=covariance.dtype, device=covariance.device)
        normal = symmetric.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None] * identity - symmetric
        ctx.save_for_backward(rotation, normal)
        return rotation

    @staticmethod
    def backward(ctx, gradient):
        rotation, normal = ctx.saved_tensors
        tangent = gradient @ rotation.transpose(-1, -2)
        vector = torch.stack((tangent[..., 2, 1]-tangent[..., 1, 2],
                              tangent[..., 0, 2]-tangent[..., 2, 0],
                              tangent[..., 1, 0]-tangent[..., 0, 1]), -1)
        identity = torch.eye(3, dtype=normal.dtype, device=normal.device)
        solved = torch.linalg.solve(normal.transpose(-1, -2) + 1e-12*identity, vector[..., None]).squeeze(-1)
        return -rotation.transpose(-1, -2) @ skew(solved)


def rigid_pose(cloud, reference):
    centered_reference = reference-reference.mean(-2, keepdim=True)
    centered_cloud = cloud-cloud.mean(-2, keepdim=True)
    covariance = centered_reference.transpose(-1, -2) @ centered_cloud
    rotation = RigidRotation.apply(covariance)
    translation = cloud.mean(-2)-(rotation @ reference.mean(-2)[..., None]).squeeze(-1)
    return translation, rotation
