"""Inference definitions extracted from train_sfp_native_grasp_keypoints.py; no training entry point."""
import numpy as np
from .geometry_sc_native_triangulation import FOCAL
CAMS = ('center', 'left', 'right')

def poses_from_pixels(pixels, optics, reference):
    """Batched camera-ray intersection followed by rigid registration."""
    batch = len(pixels)
    lhs = np.zeros((batch, 5, 3, 3))
    rhs = np.zeros((batch, 5, 3))
    for ci, c in enumerate(CAMS):
        t = np.asarray(optics[c])
        ray = np.concatenate(((pixels[:, ci] - [576, 512]) / FOCAL, np.ones((batch, 5, 1))), axis=-1) @ t[:3, :3].T
        ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
        projection = np.eye(3) - ray[..., None] * ray[..., None, :]
        lhs += projection
        rhs += projection @ t[:3, 3]
    cloud = np.linalg.solve(lhs + 1e-09 * np.eye(3), rhs[..., None])[..., 0]
    centred = cloud - cloud.mean(axis=1, keepdims=True)
    a = np.asarray(reference) - np.mean(reference, axis=0)
    u, _, vh = np.linalg.svd(np.einsum('ki,nkj->nij', a, centred))
    v = vh.transpose(0, 2, 1)
    ut = u.transpose(0, 2, 1)
    diagonal = np.broadcast_to(np.eye(3), (batch, 3, 3)).copy()
    diagonal[:, 2, 2] = np.linalg.det(v @ ut)
    r = v @ diagonal @ ut
    result = np.broadcast_to(np.eye(4), (batch, 4, 4)).copy()
    result[:, :3, :3] = r
    result[:, :3, 3] = cloud.mean(axis=1) - np.einsum('nij,j->ni', r, np.mean(reference, axis=0))
    return result
