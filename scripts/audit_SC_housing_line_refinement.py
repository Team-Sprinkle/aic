#!/usr/bin/env python3
"""Single fixed four-episode CAD-edge registration screen; never deploys."""
import cv2
cv2.setNumThreads(2)
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from sc_mesh_edge_refinement import load_edges, FOCAL
from sc_housing_line_refinement import refine
from shared_port_tcp_geometry import matrix_from_pose9

root = Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1')
output = root/'housing_line_registration_screen'
output.mkdir(exist_ok=False)

def predictions(name):
    fit = root/name
    summary = json.loads((fit/'summary.json').read_text())
    source = Path(summary['plan']['source'])
    rows = json.loads((source/'manifest.json').read_text())
    ids = np.load(fit/'heldout_indices.npy')
    poses = np.load(fit/'heldout_pose_predictions.npy')
    truth = np.load(source/'labels.npy')
    return {(rows[int(i)]['episode'], rows[int(i)]['time_s']): (rows[int(i)], poses[j], matrix_from_pose9(truth[int(i)]))
            for j, i in enumerate(ids)}

rotations = predictions('spacefilled_near_fused_orientation_warm')
positions = predictions('expanded_near_spatial_joint')
saved = torch.load(root/'expanded_near_spatial_joint/best.pt', map_location='cpu', weights_only=False)
optics = saved['tcp_to_optical']
origins = np.array([np.rint(np.array(saved['crop_centres'][c])-96) for c in ['center', 'left', 'right']])
edges = load_edges('aic_assets/models/SC Plug/sc_plug_visual.glb', ('housing',))
episodes = sorted({k[0] for k, v in rotations.items() if v[0].get('source_role', '').startswith('Near25mm')})
results = []
for episode in episodes:
    keys = sorted(k for k in rotations if k[0] == episode)
    key = keys[len(keys)//2]
    row, rotation, truth = rotations[key]
    prow, position, ptruth = positions[key]
    assert row['images'] == prow['images'] and np.allclose(truth, ptruth)
    initial = rotation.copy()
    initial[:3, 3] = position[:3, 3]
    images = {c: cv2.cvtColor(cv2.imread(row['images'][c]), cv2.COLOR_BGR2RGB) for c in ['center', 'left', 'right']}
    started = time.perf_counter()
    refined, diagnostic = refine(images, origins, optics, initial, edges)
    def errors(pose):
        relative = truth[:3, :3].T@pose[:3, :3]
        return dict(translation_mm=float(np.linalg.norm(pose[:3, 3]-truth[:3, 3])*1000),
                    orientation_deg=float(np.degrees(Rotation.from_matrix(relative).magnitude())),
                    axial_tilt_deg=float(np.degrees(np.arccos(np.clip(relative[2, 2], -1, 1)))))
    result = dict(episode=episode, initial=errors(initial), refined=errors(refined),
                  runtime_ms=(time.perf_counter()-started)*1000, diagnostic=diagnostic)
    results.append(result)
    panels = []
    for ci, camera in enumerate(['center', 'left', 'right']):
        x, y = origins[ci]
        panel = cv2.warpAffine(cv2.cvtColor(images[camera], cv2.COLOR_RGB2BGR),
                              np.array([[1, 0, -x], [0, 1, -y]], np.float32), (192, 192))
        transform = np.linalg.inv(np.array(optics[camera]))
        for matrix, color in [(initial, (0, 0, 255)), (refined, (0, 255, 0))]:
            for endpoints, _ in edges:
                q = (endpoints@matrix[:3, :3].T+matrix[:3, 3])@transform[:3, :3].T+transform[:3, 3]
                uv = np.rint(q[:, :2]/q[:, 2:]*FOCAL+[576-x, 512-y]).astype(int)
                cv2.line(panel, tuple(uv[0]), tuple(uv[1]), color, 1, cv2.LINE_AA)
        panels.append(panel)
    cv2.imwrite(str(output/(episode+'_predicted_mesh_edges.png')), np.hstack(panels))
    (output/'progress.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(result), flush=True)
(output/'summary.json').write_text(json.dumps(dict(results=results, geometry_for_scoring_only=True,
    fixed_parameters=True, automatic_deployment=False, edge_count=len(edges),
    scope='One midpoint triplet per unchanged near development episode; four episodes, not reliability evidence'), indent=2)+'\n')
