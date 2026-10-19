#!/usr/bin/env python3
"""Project selected SC opening onto scored ordinary RGB; visibility remains unlabeled."""

import argparse
from bisect import bisect_right
import json
from pathlib import Path

import numpy as np

from build_sc_depth_visibility_labels import host, transform


def prior(rows, t):
    i = bisect_right(rows, t, key=lambda row: row[0])-1
    if i < 0:
        raise ValueError('No causal sample')
    return rows[i]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stream', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stride-s', type=float, default=1.)
    p.add_argument('--target-port', type=int, choices=(0, 1),
                   help='Audit a different requested SC port on the same saved images; never a runtime input')
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve old result')
    s = json.loads(a.stream.read_text())
    audit = a.stream.parent if s['audit'] == '/audit' else Path(host(s['audit']))
    selected = (f"sc_port_{a.target_port}" if a.target_port is not None
                else s['task']['target_module_name'])
    edges = s['edges']
    results = []
    counts = {}
    for cam in ('center', 'left', 'right'):
        last = -float('inf')
        for t, image in s['cameras'][cam]:
            if t/1e9-last < a.stride_s:
                continue
            last = t/1e9
            original = Path(host(image, audit))
            if not original.is_file():
                raise FileNotFoundError(original)
            try:
                st, state = prior(s['streams']['/aic_controller/controller_state'], t)
                if t-st > 100_000_000:
                    raise ValueError('Stale controller state')
                def edge(parent, child):
                    return transform(prior(edges[parent+'|'+child], t)[1])
                world_base = edge('world','tabletop') @ edge('tabletop','base_link')
                world_opening = (edge('aic_world','task_board') @
                    edge('task_board','task_board/'+selected) @
                    edge('task_board/'+selected,
                         'task_board/'+selected+'/sc_port_base_link_entrance'))
                base_opening = np.linalg.inv(world_base) @ world_opening
                mount_tcp = (edge('cam_mount/cam_mount_link','ati/base_link') @
                    edge('ati/base_link','ati/tool_link') @
                    edge('ati/tool_link','gripper/hande_base_link') @
                    edge('gripper/hande_base_link','gripper/tcp'))
                mount_optical = (edge('cam_mount/cam_mount_link',cam+'_camera/camera_link') @
                    edge(cam+'_camera/camera_link',cam+'_camera/sensor_link') @
                    edge(cam+'_camera/sensor_link',cam+'_camera/optical'))
                tcp = transform(dict(translation=state['tcp_pose']['position'],
                                     rotation=state['tcp_pose']['orientation']))
                optical = tcp @ np.linalg.inv(mount_tcp) @ mount_optical
                point = np.linalg.inv(optical) @ base_opening[:, 3]
                width, height, hfov = 1152, 1024, .8718
                focal = width/(2*np.tan(hfov/2))
                uv = [width/2+focal*point[0]/point[2],
                      height/2+focal*point[1]/point[2]]
                relative_opening_tcp = np.linalg.inv(tcp) @ base_opening
                results.append(dict(episode_id=audit.name, scene_group_id=s['scene_group_id'],
                                    sim_ns=t, camera=cam, image=str(original.resolve()),
                                    target_port=int(selected.rsplit('_',1)[-1]),
                                    measured_tcp_base_pose=[*[state['tcp_pose']['position'][k] for k in 'xyz'],
                                                            *[state['tcp_pose']['orientation'][k] for k in 'xyzw']],
                                    projected_opening_xy=uv, projected_depth_m=float(point[2]),
                                    projected_inframe=bool(point[2]>.07 and 0<=uv[0]<width and
                                                           0<=uv[1]<height),
                                    relative_opening_tcp_m=relative_opening_tcp[:3,3].tolist(),
                                    relative_opening_rotation_tcp=relative_opening_tcp[:3,:3].tolist(),
                                    identifiable_visibility=None,
                                    simulator_projection_label_only=True))
            except (ValueError, KeyError) as exc:
                counts[str(exc)] = counts.get(str(exc), 0)+1
    a.output.mkdir(parents=True)
    with (a.output/'rows.jsonl').open('w') as f:
        for row in results:
            f.write(json.dumps(row)+'\n')
    report = dict(schema='aic_sc_original_rgb_projection_audit/v1', rows=len(results),
                  by_camera={c:sum(r['camera']==c for r in results)
                             for c in ('center','left','right')},
                  projected_inframe=sum(r['projected_inframe'] for r in results),
                  rejected=counts, visibility_certified=False,
                  final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
