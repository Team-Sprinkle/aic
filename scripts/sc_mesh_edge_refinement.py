"""Bounded RGB/CAD edge registration diagnostic; no simulator state input."""
import json
import struct
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from evaluate_sc_native_triangulation import FOCAL


def load_edges(path, mesh_prefixes=('housing', 'ferula', 'parte')):
    blob = Path(path).read_bytes()
    length, _ = struct.unpack_from('<II', blob, 12)
    doc = json.loads(blob[20:20+length])
    count, _ = struct.unpack_from('<II', blob, 20+length)
    data = memoryview(blob)[28+length:28+length+count]
    def accessor(index):
        a = doc['accessors'][index]
        v = doc['bufferViews'][a['bufferView']]
        if 'byteStride' in v or 'sparse' in a:
            raise ValueError('Packed static CAD accessors required')
        width = {'SCALAR': 1, 'VEC3': 3}[a['type']]
        dtype = {5126: '<f4', 5125: '<u4', 5123: '<u2', 5121: 'u1'}[a['componentType']]
        return np.frombuffer(data, dtype=dtype, count=a['count']*width,
            offset=v.get('byteOffset', 0)+a.get('byteOffset', 0)).reshape(a['count'], width)
    edges = []
    # CAD coordinates match the model SDF; every node is identity in this asset.
    assert all(not any(k in n for k in ['matrix', 'translation', 'rotation', 'scale']) for n in doc['nodes'])
    tip_rotation = Rotation.from_euler('xyz', [-1.5708, 0, -1.5708]).as_matrix()
    for mesh in doc['meshes']:
        if not mesh['name'].lower().startswith(mesh_prefixes):
            continue
        for prim in mesh['primitives']:
            assert prim.get('mode', 4) == 4
            original = accessor(prim['attributes']['POSITION']).astype(float)
            vertices, mapping = np.unique(np.round(original, 7), axis=0, return_inverse=True)
            faces = mapping[accessor(prim['indices']).reshape(-1)].reshape(-1, 3)
            normals = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]], vertices[faces[:, 2]]-vertices[faces[:, 0]])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            adjacency = {}
            for face, normal in zip(faces, normals):
                for a, b in zip(face, np.roll(face, -1)):
                    adjacency.setdefault(tuple(sorted((int(a), int(b)))), []).append(normal)
            for (a, b), ns in adjacency.items():
                ns = np.array(ns)
                sharp = len(ns) == 1 or np.min(ns@ns.T) < np.cos(np.deg2rad(30))
                if not sharp:
                    continue
                endpoints = vertices[[a, b]]
                if np.min(endpoints[:, 0]) < -.0044:
                    continue
                tip = (endpoints-[.01165, 0, 0])@tip_rotation
                edges.append((tip, ns@tip_rotation))
    return edges


def refine(images, origins, optics, initial, edges):
    """Five fixed correspondence/pose iterations, with small bounded changes."""
    initial = np.asarray(initial)
    camera_data = []
    for ci, camera in enumerate(['center', 'left', 'right']):
        image = images[camera]
        x, y = origins[ci]
        crop = cv2.warpAffine(image, np.array([[1, 0, -x], [0, 1, -y]], np.float32), (192, 192))
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(gray, 60, 120)
        gy, gx = np.where(canny)
        gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        gradients = np.stack([gradient_x[gy, gx], gradient_y[gy, gx]], axis=1)
        gradients /= np.maximum(np.linalg.norm(gradients, axis=1, keepdims=True), 1e-6)
        pixels = np.stack([gx, gy], axis=1)
        transform = np.linalg.inv(np.asarray(optics[camera]))
        camera_data.append((transform, np.array([576-x, 512-y]), pixels, gradients, cKDTree(pixels)))

    def pose(parameters):
        result = initial.copy()
        result[:3, 3] += parameters[:3]*.001
        result[:3, :3] = initial[:3, :3]@Rotation.from_rotvec(np.deg2rad(parameters[3:])).as_matrix()
        return result

    def project(points, matrix, transform, centre):
        q = (points@matrix[:3, :3].T+matrix[:3, 3])@transform[:3, :3].T+transform[:3, 3]
        return q[:, :2]/q[:, 2:]*FOCAL+centre

    parameters = np.zeros(6)
    history = []
    for iteration in range(5):
        current = pose(parameters)
        pairs = []
        for ci, (transform, centre, pixels, gradients, tree) in enumerate(camera_data):
            camera_in_tip = np.linalg.inv(current)@np.linalg.inv(transform)
            selected_points = []
            selected_pixels = []
            for endpoints, normals in edges:
                if np.max(normals@(camera_in_tip[:3, 3]-endpoints.mean(0))) <= 0:
                    continue
                uv = project(endpoints, current, transform, centre)
                tangent = uv[1]-uv[0]
                length = np.linalg.norm(tangent)
                if length < 2:
                    continue
                normal = np.array([-tangent[1], tangent[0]])/length
                count = min(32, max(3, int(length/2)))
                points = endpoints[0]+np.linspace(.1, .9, count)[:, None]*(endpoints[1]-endpoints[0])
                predicted = project(points, current, transform, centre)
                distances, indices = tree.query(predicted, k=8)
                compatible = np.abs(gradients[indices]@normal) >= np.cos(np.deg2rad(30))
                costs = np.where(compatible & (distances <= 3), distances, np.inf)
                chosen = np.argmin(costs, axis=1)
                valid = np.isfinite(costs[np.arange(count), chosen]) & (predicted.min(1) >= 2) & (predicted.max(1) < 190)
                selected_points.extend(points[valid])
                selected_pixels.extend(pixels[indices[np.arange(count), chosen]][valid])
            if selected_points:
                pairs.append((ci, np.array(selected_points), np.array(selected_pixels)))
        if len(pairs) < 2 or sum(len(p[1]) for p in pairs) < 30:
            return initial.copy(), dict(accepted=False, reason='insufficient_multi_camera_edges', history=history)

        def residual(p):
            matrix = pose(p)
            values = [ (project(points, matrix, camera_data[ci][0], camera_data[ci][1])-observed).reshape(-1)
                       for ci, points, observed in pairs ]
            # Weak prior and fixed trust region avoid large unsupported changes.
            values.append(.1*p/np.array([.3, .3, .3, .5, .5, .5]))
            return np.concatenate(values)

        result = least_squares(residual, parameters, bounds=([-1]*3+[-2]*3, [1]*3+[2]*3),
                               loss='soft_l1', f_scale=1., max_nfev=40)
        parameters = result.x
        history.append(dict(iteration=iteration, correspondences=sum(len(p[1]) for p in pairs),
                            camera_count=len(pairs), residual_rms_px=float(np.sqrt(np.mean(residual(parameters)**2)))))
    return pose(parameters), dict(accepted=True, parameters_mm_deg=parameters.tolist(), history=history,
                                 runtime_true_geometry=False, diagnostic_only=True)
