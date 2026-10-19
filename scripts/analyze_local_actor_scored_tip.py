#!/usr/bin/env python3
"""Score recorded SC plug-tip motion against the true opening after a rollout.

All TF here is posthoc scoring only. It is never an actor input.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import matrix


def transform(message):
    p=message['translation'];q=message['rotation'];out=np.eye(4)
    out[:3,3]=[p[k] for k in 'xyz']
    out[:3,:3]=Rotation.from_quat([q[k] for k in 'xyzw']).as_matrix()
    return out


def record_at(rows,t):
    times=[r[0] for r in rows]
    i=max(0,bisect_right(times,t)-1)
    return transform(rows[i][1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stream',type=Path)
    p.add_argument('--opening-calibration',type=Path,default=Path('configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json'))
    p.add_argument('--teacher-chunks',type=Path,help='Successful recorded teacher tip orientation for comparison')
    p.add_argument('--teacher-episode',type=str,default='llb_0001')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=json.loads(a.stream.read_text());edges=r['edges']
    selected=int(r['task']['target_module_name'].rsplit('_',1)[-1])
    prefix=f'task_board/sc_port_{selected}'
    fixed=(record_at(edges['aic_world|task_board'],0)@
           record_at(edges[f'task_board|{prefix}'],0)@
           record_at(edges[f'{prefix}|{prefix}/sc_port_base_link'],0)@
           matrix(json.loads(a.opening_calibration.read_text())['base_to_opening']))
    teacher_orientation=None;teacher_rows=0
    if a.teacher_chunks:
        quats=[]
        for row in map(json.loads,a.teacher_chunks.open()):
            if row.get('episode_id')!=a.teacher_episode:continue
            pose=row['observed_sc_tip_pose_opening_frame']
            if -.10<pose[2]<-.015:
                quats.append(pose[3:7]);teacher_rows+=1
        if not quats:raise ValueError('No teacher orientation reference')
        teacher_orientation=Rotation.from_quat(quats).mean()
    samples=[]
    cable=edges['aic_world|cable_1'];tip=edges['cable_1|cable_1/sc_tip_link']
    for t,_ in cable:
        relative=np.linalg.inv(fixed)@record_at(cable,t)@record_at(tip,t)
        x,y,z=relative[:3,3]
        sample={'sim_time_s':t/1e9,'tip_opening_mm':[float(v*1000) for v in (x,y,z)],
                        'lateral_mm':float(np.hypot(x,y)*1000),
                        # Raw link-frame rotation includes the fixed mating
                        # convention; it is not an insertion alignment error.
                        'raw_link_frame_rotation_deg':float(Rotation.from_matrix(relative[:3,:3]).magnitude()*180/np.pi)}
        if teacher_orientation is not None:
            sample['teacher_tip_orientation_error_deg']=float((
                teacher_orientation.inv()*Rotation.from_matrix(relative[:3,:3])).magnitude()*180/np.pi)
        samples.append(sample)
    closest=min(samples,key=lambda row:abs(row['tip_opening_mm'][2]))
    near=[x for x in samples if abs(x['tip_opening_mm'][2])<30]
    report={'schema':'aic_sc_live_tip_scoring/v1','trial':r['trial_id'],'selected_port':selected,
            'official_tier3':r['official_tier3'],'samples':len(samples),
            'closest_axial':closest,'last':samples[-1],
            'min_lateral_near_30mm':min(near,key=lambda x:x['lateral_mm']) if near else None,
            'teacher_orientation_reference_rows':teacher_rows,
            'true_geometry_used_posthoc_only':True}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
