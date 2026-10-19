#!/usr/bin/env python3
"""Build grouped, counterfactual port-projection labels from existing Gazebo data.

These are *projection* labels, not certified visibility. Scene geometry and TF
are used here only. Runtime perception must read RGB, measured state and task.
"""
import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from audit_sc_port_targets import matrix, pose_matrix
from build_sc_native_pose_labels import project
from build_sfp_port_tcp_waypoints import transform


CAMERAS=('center','left','right')


def pose7(state):
    return pose_matrix(np.asarray(state[:7],float))


def edge(edges,parent,child):
    data=edges[parent+'|'+child][0][1]
    out=np.eye(4)
    out[:3,3]=[data['translation'][k] for k in 'xyz']
    out[:3,:3]=Rotation.from_quat([data['rotation'][k] for k in 'xyzw']).as_matrix()
    return out


def from_controller(row):
    p=row['tcp_pose']['position'];q=row['tcp_pose']['orientation']
    return np.array([*[p[k] for k in 'xyz'],*[q[k] for k in 'xyzw']],float)


def sc_cases(spec, stride_s, camera_calibration):
    calibration=json.loads(camera_calibration.read_text())
    optics={c:np.asarray(calibration['tcp_to_optical'][c],float) for c in CAMERAS}
    for episode in spec['episodes']:
        case=episode['case']
        path=Path('outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized/streams')/(case+'.json')
        stream=json.loads(path.read_text());edges=stream['edges']
        world_base=edge(edges,'world','tabletop')@edge(edges,'tabletop','base_link')
        world_board=edge(edges,'aic_world','task_board')
        ports=[]
        for port in (0,1):
            parent=f'task_board/sc_port_{port}'
            world_port=(world_board@edge(edges,'task_board',parent)@
                        edge(edges,parent,parent+'/sc_port_base_link_entrance'))
            ports.append(np.linalg.inv(world_base)@world_port)
        centers=stream['cameras']['center'];controllers=stream['streams']['/aic_controller/controller_state']
        camera_rows={c:stream['cameras'][c] for c in CAMERAS}
        last=-1e12
        for stamp,image in centers:
            if stamp/1e9-last<stride_s:continue
            if not (stream['scored_wall_bounds_ns'][0] <= int(Path(image).stem.split('_')[0]) <= stream['scored_wall_bounds_ns'][1]):continue
            images={};valid=True
            for cam in CAMERAS:
                rows=camera_rows[cam];j=bisect_right(rows,stamp,key=lambda r:r[0])-1
                if j<0 or stamp-rows[j][0]>50_000_000:valid=False;break
                images[cam]=rows[j][1]
            j=bisect_right(controllers,stamp,key=lambda r:r[0])-1
            if not valid or j<0 or stamp-controllers[j][0]>100_000_000:continue
            state=from_controller(controllers[j][1]);base_tcp=pose7(state)
            targets=[]
            for port,m in enumerate(ports):
                target=dict(identity=f'sc_port_{port}',position_base_m=m[:3,3].tolist(),
                            orientation_base_xyzw=Rotation.from_matrix(m[:3,:3]).as_quat().tolist(),
                            cameras={c:project(m,base_tcp@optics[c]) for c in CAMERAS})
                targets.append(target)
            last=stamp/1e9
            yield dict(family='sc',episode=case,scene_group=episode['scene_reset_group'],
                       split=episode['split'],time_s=last,state=state.tolist(),images=images,
                       image_kind='native_rgb_jpeg',source_stream=str(path),targets=targets,
                       selected_identity=f"sc_port_{episode['selected_port']}",card_count=episode['card_count'])


def sfp_ports(episode,calibration):
    config=yaml.safe_load(Path(episode['trial_yaml']).read_text())
    trial=config['trials'][episode['trial_id']];board=trial['scene']['task_board'];bp=board['pose']
    world_board=transform([bp[k] for k in ('x','y','z')],[bp[k] for k in ('roll','pitch','yaw')])
    world_base=matrix(calibration['world_to_base']);mount=calibration['board_to_nic_mount']
    out=[]
    for card in range(5):
        rail=board[f'nic_rail_{card}']
        if not rail['entity_present']:continue
        rp=rail['entity_pose'];board_nic=transform(
            [mount['x_nominal_m']+rp['translation'],mount['y_rail0_m']+mount['y_step_per_rail_m']*card,mount['z_m']],
            [rp[k] for k in ('roll','pitch','yaw')])
        for port in (0,1):
            m=np.linalg.inv(world_base)@world_board@board_nic@matrix(calibration['nic_mount_to_port_opening'][str(port)])
            out.append((card,port,m))
    return out


def sfp_cases(cache_path,stride,calibration_path,camera_calibration):
    info=json.loads((cache_path/'cache.json').read_text())
    calibration=json.loads(calibration_path.read_text())
    camera=json.loads(camera_calibration.read_text())
    optics={c:np.asarray(camera['tcp_to_optical'][c],float) for c in CAMERAS}
    states=np.load(cache_path/'states.npy',mmap_mode='r')
    for episode in info['episodes']:
        if episode['task']['task_family']!='sfp_to_nic':continue
        ports=sfp_ports(episode,calibration)
        selected=f"sfp_card_{episode['task']['target_card_index']}_port_{episode['task']['target_port_index']}"
        if selected not in [f'sfp_card_{card}_port_{port}' for card,port,_ in ports]:raise ValueError(selected)
        start=int(episode['cache_from_index']);end=int(episode['cache_to_index']);length=end-start
        for frame in range(0,length,stride):
            i=start+frame;state=np.asarray(states[i],float);base_tcp=pose7(state)
            targets=[]
            for card,port,m in ports:
                targets.append(dict(identity=f'sfp_card_{card}_port_{port}',
                                    position_base_m=m[:3,3].tolist(),
                                    orientation_base_xyzw=Rotation.from_matrix(m[:3,:3]).as_quat().tolist(),
                                    cameras={c:project(m,base_tcp@optics[c]) for c in CAMERAS}))
            yield dict(family='sfp',episode=f"sfp_{episode['episode_index']}",
                       scene_group=episode['scene_sha256'],split=episode['split'],
                       time_s=frame/20.,state=state[:7].tolist(),
                       images={c:i for c in CAMERAS},image_kind='verified_288x256_rgb_cache',
                       source_cache=str(cache_path),targets=targets,selected_identity=selected,
                       card_count=len(set(card for card,_,_ in ports)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sc-spec',type=Path,required=True)
    p.add_argument('--sfp-cache',type=Path,required=True)
    p.add_argument('--sfp-calibration',type=Path,required=True)
    p.add_argument('--camera-calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sc-stride-seconds',type=float,default=.5)
    p.add_argument('--sfp-stride-frames',type=int,default=10)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing output')
    a.output.mkdir(parents=True)
    spec=json.loads(a.sc_spec.read_text())
    rows=[*sfp_cases(a.sfp_cache,a.sfp_stride_frames,a.sfp_calibration,a.camera_calibration),
          *sc_cases(spec,a.sc_stride_seconds,a.camera_calibration)]
    if not rows:raise ValueError('No examples')
    seen={};counts=Counter();counterfactual=Counter();near=Counter()
    with (a.output/'rows.jsonl').open('w') as out:
        for r in rows:
            prev=seen.setdefault(r['scene_group'],r['split'])
            if prev!=r['split']:raise ValueError('Scene-group split leakage')
            identities={t['identity'] for t in r['targets']}
            if r['selected_identity'] not in identities:raise ValueError('Missing selected target')
            counterfactual[(r['family'],r['split'])]+=len(identities)-1
            counts[(r['family'],r['split'])]+=1
            selected=next(t for t in r['targets'] if t['identity']==r['selected_identity'])
            near[(r['family'],r['split'])]+=int(np.linalg.norm(np.asarray(selected['position_base_m'])-r['state'][:3])<.1)
            out.write(json.dumps(r,separators=(',',':'))+'\n')
    summary=dict(schema='two_family_port_projection_labels/v1',rows=len(rows),
                 examples={str(k):v for k,v in counts.items()},
                 alternate_target_examples={str(k):v for k,v in counterfactual.items()},
                 within_100mm={str(k):v for k,v in near.items()},
                 scene_groups_by_split={s:sum(v==s for v in seen.values()) for s in ('train','validation')},
                 label_kind='projected aperture center; NOT certified visible or identifiable',
                 runtime_geometry_allowed=False,reserved_final_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
