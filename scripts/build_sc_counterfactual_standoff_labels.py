#!/usr/bin/env python3
"""Training-only matched-image labels for selecting either SC port.

The board/rail geometry is read from recorded configurations. It is never a
runtime input. Each row keeps its actual image and measured TCP, while its
public task code can be swapped to teach which physical port is requested.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def pose_matrix(pose):
    x=np.eye(4)
    x[:3,:3]=Rotation.from_euler('xyz',[pose[k] for k in ('roll','pitch','yaw')]).as_matrix()
    x[:3,3]=[pose[k] for k in ('x','y','z')]
    return x


def edge_matrix(edges,key):
    e=edges[key][0][1];x=np.eye(4)
    x[:3,:3]=Rotation.from_quat([e['rotation'][k] for k in 'xyzw']).as_matrix()
    x[:3,3]=[e['translation'][k] for k in 'xyz']
    return x


def board(path):
    config=yaml.safe_load(path.read_text())
    return next(iter(config['trials'].values()))['scene']['task_board']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--rows',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    ref=root/'outputs/experiments/2026-09-30_transport_continuation'
    refboard=board(ref/'scenes/port1_five_shiftminus/eval_config.yaml')
    edges=json.loads((ref/'streams/cartesian_port1_five.json').read_text())['edges']
    world_base=edge_matrix(edges,'world|tabletop')@edge_matrix(edges,'tabletop|base_link')
    ref_base=np.linalg.inv(world_base)@pose_matrix(refboard['pose'])
    def goal(port):
        name='port0_five_paired' if port==0 else 'port1_five_shiftminus'
        plan=json.loads((ref/f'plans/{name}/moveit_joint_trajectory.json').read_text())
        return np.asarray(plan['waypoints'][-1]['tcp_pose']['position'],dtype=float)
    reference_delta=goal(0)-goal(1)
    episodes={e['case']:e for e in json.loads(a.dataset.read_text())['episodes']}
    id_to_case={}
    with a.rows.open() as f:
        for line in f:
            row=json.loads(line)
            key=int(row['episode_index'])
            if key in id_to_case and id_to_case[key]!=row['episode_id']:
                raise ValueError('Episode index maps to two cases')
            id_to_case[key]=row['episode_id']
    result={}
    for episode_index,case in sorted(id_to_case.items()):
        source=episodes[case]
        newboard=board(Path(source['source_episode'])/'eval_config.yaml')
        new_base=np.linalg.inv(world_base)@pose_matrix(newboard['pose'])
        local_delta=ref_base[:3,:3].T@reference_delta
        shift=[]
        for port in (0,1):
            key=f'sc_rail_{port}'
            shift.append(newboard[key]['entity_pose']['translation']-
                         refboard[key]['entity_pose']['translation'])
        local_delta+=np.array([shift[0]-shift[1],0,0])
        delta=new_base[:3,:3]@local_delta
        if not .04<np.linalg.norm(delta)<.11:raise ValueError((case,delta))
        result[str(episode_index)]={'case':case,'split':source['split'],
            'selected_port':source['selected_port'],
            'port0_minus_port1_standoff_base_m':delta.tolist(),
            'scene_reset_group':source['scene_reset_group']}
    report={'schema':'aic_sc_counterfactual_standoff_labels/v1',
        'labels_use':'supervised training only; never autonomous actor or evaluation crop',
        'reference_port0_minus_port1_standoff_base_m':reference_delta.tolist(),
        'source_dataset':str(a.dataset),'source_rows':str(a.rows),
        'source_dataset_sha256':hashlib.sha256(a.dataset.read_bytes()).hexdigest(),
        'episodes':result,'episode_count':len(result),'final_scenes_opened':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'episodes':len(result),'reference_delta_mm':(reference_delta*1000).tolist()}))


if __name__=='__main__':main()
