#!/usr/bin/env python3
"""Inspect early complete-target commands for scored SC route families."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train_shared_port_tcp_bc import SharedPortTCPActor


def median(values):
    return np.median(np.asarray(values), axis=0).tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, nargs='+', required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    weights = checkpoint['model']
    stats = {key: weights[key].numpy() for key in ('state_mean', 'state_std',
             'pose_mean', 'pose_std', 'target_mean', 'target_std')}
    model = SharedPortTCPActor(stats, pretrained=False,
                               use_time=bool(checkpoint['include_elapsed']))
    model.load_state_dict(weights, strict=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    episodes = defaultdict(list)
    for path in a.labels:
        for line in path.open():
            row = json.loads(line)
            if row['split'] == 'train':
                episodes[(str(path), int(row['episode_index']))].append(row)
    report_rows = []
    for (source, episode), rows in episodes.items():
        rows.sort(key=lambda row: row['frame'])
        first = rows[0]
        after = min(rows, key=lambda row: abs(row['frame']-100))
        first_xyz = np.asarray(first['observed_tcp_pose_opening_frame'][:3])
        after_xyz = np.asarray(after['observed_tcp_pose_opening_frame'][:3])
        displacement = (after_xyz-first_xyz)*1000
        mode = 'lift' if displacement[0] > -50 else 'direct'
        for frame in (20, 40):
            row = min(rows, key=lambda r: abs(r['frame']-frame))
            views = []
            for camera in ('center', 'left', 'right'):
                with Image.open(row['images'][camera]) as image:
                    rgb = np.asarray(image.convert('RGB').resize(
                        (224, 224), Image.Resampling.BILINEAR)).copy()
                views.append(torch.from_numpy(rgb).permute(2, 0, 1))
            with torch.inference_mode():
                target, pose = model(
                    torch.stack(views)[None].to(device),
                    torch.tensor(row['state'], dtype=torch.float32, device=device)[None],
                    torch.tensor(row['task_vector'], dtype=torch.float32, device=device)[None],
                    torch.tensor([row['sim_time']-first['sim_time']],
                                 dtype=torch.float32, device=device))
            predicted = (target[0, 0, :3]-pose[0, :3]).cpu().numpy()*1000
            teacher = (np.asarray(row['teacher_tcp_target_poses_opening_frame'][0][:3])-
                       np.asarray(row['observed_tcp_pose_opening_frame'][:3]))*1000
            report_rows.append({'source': source, 'episode_index': episode,
                                'route_mode_from_measured_frame100': mode,
                                'frame': frame,
                                'measured_frame100_displacement_mm': displacement.tolist(),
                                'predicted_next_port_translation_mm': predicted.tolist(),
                                'teacher_next_port_translation_mm': teacher.tolist()})
    summary = {}
    for mode in ('direct', 'lift'):
        for frame in (20, 40):
            subset = [r for r in report_rows if r['route_mode_from_measured_frame100']==mode
                      and r['frame']==frame]
            summary[f'{mode}/frame{frame}'] = {
                'episodes': len(subset),
                'predicted_median_xyz_mm': median([r['predicted_next_port_translation_mm'] for r in subset]),
                'teacher_median_xyz_mm': median([r['teacher_next_port_translation_mm'] for r in subset])}
    output = {'schema':'sc_transport_route_mode_audit/v1','checkpoint':str(a.checkpoint),
              'label_sources':[str(p) for p in a.labels],
              'mode_rule':'measured port-frame x travel by frame100 greater than -50 mm means pre-route lift',
              'summary':summary,'rows':report_rows,
              'scope':'training episodes only; route label is posthoc and not an actor input'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':
    main()
