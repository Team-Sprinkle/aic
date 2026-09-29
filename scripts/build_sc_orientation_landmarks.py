#!/usr/bin/env python3
"""Project fixed SC port-opening axes for perception training and scoring.

These virtual landmarks are defined in the target opening frame. TF is read
only here to create labels; RGB inference never receives a port transform.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from audit_sc_port_targets import matrix, pose_matrix
from build_sc_native_pose_labels import chain, project


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--edges',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--x-offset-m',type=float,default=.010)
    p.add_argument('--y-offset-m',type=float,default=.008)
    args=p.parse_args()
    if not 0<args.x_offset_m<=.015 or not 0<args.y_offset_m<=.013:
        raise ValueError('Landmarks must stay inside the approximate SC port body extent')
    offsets={'opening':(0.,0.,0.),'x_plus':(args.x_offset_m,0.,0.),
             'x_minus':(-args.x_offset_m,0.,0.),'y_plus':(0.,args.y_offset_m,0.),
             'y_minus':(0.,-args.y_offset_m,0.)}
    rows=[json.loads(x) for x in args.labels.open()]
    edges={x['trial']:x for x in json.loads(args.edges.read_text())['rows']}
    first=next(iter(edges.values()))['edges']
    tool_tcp=chain(first,['tool0','cam_mount/cam_mount_link','ati/base_link','ati/tool_link',
                          'gripper/hande_base_link','gripper/tcp'])
    cameras={}
    for cam in ('center','left','right'):
        tool_optical=chain(first,['tool0','cam_mount/cam_mount_link',f'{cam}_camera/camera_link',
                                  f'{cam}_camera/sensor_link',f'{cam}_camera/optical'])
        cameras[cam]=np.linalg.inv(tool_tcp)@tool_optical
    offset=matrix(json.loads(Path('configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json').read_text())['base_to_opening'])
    openings={}
    for trial,item in edges.items():
        port=item['target_module_name'].rsplit('_',1)[-1]
        parent=f'task_board/sc_port_{port}'
        world_base=chain(item['edges'],['world','tabletop','base_link'])
        world_port=chain(item['edges'],['aic_world','task_board',parent,parent+'/sc_port_base_link'])
        openings[trial]=np.linalg.inv(world_base)@world_port@offset
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w') as out:
        for row in rows:
            opening=openings[row['trial']]
            base_tcp=pose_matrix(row['state'][:7])
            projections={}
            for cam,tcp_optical in cameras.items():
                optical=base_tcp@tcp_optical
                projections[cam]={}
                for name,xyz in offsets.items():
                    local=np.eye(4);local[:3,3]=xyz
                    projections[cam][name]=project(opening@local,optical)
                reference=np.array(row['projected_training_pixels'][cam]['opening'][:2])
                if np.linalg.norm(np.array(projections[cam]['opening'][:2])-reference)>1e-3:
                    raise ValueError(f'Opening projection mismatch at {row["trial"]} {row["frame"]}')
            row['projected_orientation_landmarks']=projections
            row['orientation_landmark_source']=(f'scored TF and fixed virtual '
                 f'{args.x_offset_m*1000:g}mm X / {args.y_offset_m*1000:g}mm Y offsets; training/scoring only')
            out.write(json.dumps(row,separators=(',',':'))+'\n')
    print(json.dumps({'rows':len(rows),'output':str(args.output)}))


if __name__=='__main__':main()
