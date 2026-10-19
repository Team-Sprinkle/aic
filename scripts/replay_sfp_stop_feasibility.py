#!/usr/bin/env python3
"""Score the existing five-estimate SFP lock on frozen native RGB predictions."""
import argparse
from collections import defaultdict,deque
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve earlier replay')
    report=json.loads(a.benchmark.read_text())
    labels={(x['episode'],round(x['time_s'],3)):x for x in map(json.loads,a.labels.open())}
    groups=defaultdict(list)
    for row in report['episode_rows']:groups[row['episode']].append(row)
    episodes=[]
    for episode,rows in groups.items():
        rows.sort(key=lambda x:x['time_s'])
        accepted=deque(maxlen=5);first_lock=None;accepted_rows=[]
        for row in rows:
            if not row['accepted']:continue
            pose=np.array(row['predicted_base_position_m'])
            source=labels[(episode,round(row['time_s'],3))]
            distances={x['identity']:float(np.linalg.norm(pose-np.array(x['position_base_m']))*1000)
                       for x in source['targets']}
            nearest=min(distances,key=distances.get)
            accepted_rows.append({'time_s':row['time_s'],'selected':row['selected_identity'],
                'nearest_physical_port':nearest,'nearest_error_mm':distances[nearest],
                'selected_error_mm':distances[row['selected_identity']],
                'selected_identity_correct':nearest==row['selected_identity'],
                'reprojection_px':row['diagnostic'].get('reprojection_px')})
            accepted.append(row)
            if len(accepted)!=5 or first_lock is not None:continue
            xyz=np.array([x['predicted_base_position_m'] for x in accepted])
            center=np.median(xyz,axis=0)
            quats=np.array([x['predicted_base_quat_xyzw'] for x in accepted])
            rotations=Rotation.from_quat(quats);mean=rotations.mean()
            spread=float(np.linalg.norm(xyz-center,axis=1).max()*1000)
            angle=float(np.rad2deg((mean.inv()*rotations).magnitude().max()))
            if spread<=.5 and angle<=5:
                targets=source['targets']
                nearest_lock=min(targets,key=lambda x:np.linalg.norm(center-np.array(x['position_base_m'])))
                selected=next(x for x in targets if x['identity']==row['selected_identity'])
                first_lock={'time_s':row['time_s'],'position_spread_mm':spread,
                    'orientation_spread_deg':angle,'selected_identity':row['selected_identity'],
                    'nearest_physical_port':nearest_lock['identity'],
                    'wrong_port':nearest_lock['identity']!=row['selected_identity'],
                    'selected_position_error_mm':float(np.linalg.norm(center-np.array(selected['position_base_m']))*1000)}
        episodes.append({'episode':episode,'sampled_frames':len(rows),'accepted_frames':len(accepted_rows),
            'wrong_port_accepted_frames':sum(not x['selected_identity_correct'] for x in accepted_rows),
            'first_lock':first_lock,'stop_feasible_from_identity':first_lock is not None and not first_lock['wrong_port'],
            'accepted_trace':accepted_rows})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'schema':'aic_sfp_identity_stop_feasibility/v1',
        'method':'five accepted RGB poses; median spread <=0.5 mm and mean-rotation spread <=5 deg',
        'episodes':episodes,'sampled_frames_only':True,'target_geometry_scoring_only':True,
        'runtime_geometry_input':False,'reserved_final_opened':False},indent=2)+'\n')
    print(json.dumps([{k:x[k] for k in ('episode','sampled_frames','accepted_frames',
        'wrong_port_accepted_frames','first_lock')} for x in episodes],indent=2))


if __name__=='__main__':main()
