#!/usr/bin/env python3
"""Offline geometric scoring labels for recorded autonomous native SC images.

Fixed scene TF is taken from the scored *source* bag of the same reset group.
Only RGB/task/ordinary measured TCP will enter the frozen locator and refiner.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_sc_port_targets import matrix, pose_list, pose_matrix
from build_sc_native_pose_labels import chain, project


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--edges',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--counterfactual-training',action='store_true')
    p.add_argument('--training-stride',type=int,default=1)
    a=p.parse_args()
    rows=[json.loads(s) for s in (a.cache/'rows.jsonl').open()]
    source={r['trial']:r for r in json.loads(a.edges.read_text())['rows']}
    tcp_tip=matrix(json.loads(Path('configs/hierarchical_recovery/sc_tcp_tip_gazebo_tf_251.json').read_text())['tcp_to_sc_tip'])
    opening_offset=matrix(json.loads(Path('configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json').read_text())['base_to_opening'])
    local={}
    for case,item in source.items():
        edges=item['edges'];port=item['target_module_name'].rsplit('_',1)[-1]
        parent=f'task_board/sc_port_{port}'
        world_base=chain(edges,['world','tabletop','base_link'])
        world_port=chain(edges,['aic_world','task_board',parent,parent+'/sc_port_base_link'])
        base_opening=np.linalg.inv(world_base)@world_port@opening_offset
        alternate_port=str(1-int(port))
        alternate_parent=f'task_board/sc_port_{alternate_port}'
        alternate_world=chain(edges,['aic_world','task_board',alternate_parent,
                                     alternate_parent+'/sc_port_base_link'])
        alternate_opening=np.linalg.inv(world_base)@alternate_world@opening_offset
        tool_tcp=chain(edges,['tool0','cam_mount/cam_mount_link','ati/base_link','ati/tool_link',
                              'gripper/hande_base_link','gripper/tcp'])
        cameras={}
        for cam in ('center','left','right'):
            tool_optical=chain(edges,['tool0','cam_mount/cam_mount_link',f'{cam}_camera/camera_link',
                                      f'{cam}_camera/sensor_link',f'{cam}_camera/optical'])
            cameras[cam]=np.linalg.inv(tool_tcp)@tool_optical
        local[case]=(base_opening,alternate_opening,cameras)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    counts={case:0 for case in local}
    with a.output.open('w') as out:
        for frame,row in enumerate(rows):
            case=row['case']
            if case not in local:continue
            base_opening,alternate_opening,cameras=local[case]
            base_tcp=pose_matrix(row['state'][:7]);base_tip=base_tcp@tcp_tip
            selected_frame=counts[case]
            counts[case]+=1
            if row['split']=='adapt_train' and selected_frame%a.training_stride:continue
            for alternate,opening in ((False,base_opening),(True,alternate_opening)):
                if alternate and (not a.counterfactual_training or row['split']!='adapt_train'):
                    continue
                opening_tip=np.linalg.inv(opening)@base_tip
                pixels={cam:{'tip':project(base_tip,base_tcp@optical),
                             'opening':project(opening,base_tcp@optical)}
                        for cam,optical in cameras.items()}
                task=list(row['task'])
                if alternate:task[2:4]=task[2:4][::-1]
                record={'schema':'sc_actor_native_pose_labels/v1',
                    'trial':case+('_alternate' if alternate else ''),
                    'official_tier3':0,'expert_bc_eligible':False,'pose_label_only':True,
                    'episode_id':case+('_alternate' if alternate else ''),
                    'scene_sha256':row['scene_reset_group'],
                    'split':'train' if row['split']=='adapt_train' else 'validation',
                    'frame':selected_frame,'sim_time':row['sim_time'],
                    'images':dict(zip(('center','left','right'),row['image_paths'])),
                    'state':row['state'],'task_vector':task,
                    'observed_sc_tip_pose_opening_frame':pose_list(opening_tip),
                    'projected_training_pixels':pixels,
                    'pixel_label_source':'scored source-bag TF and fixed camera calibration, offline scoring only'}
                out.write(json.dumps(record,separators=(',',':'))+'\n')
    print(json.dumps(counts))


if __name__=='__main__':main()
