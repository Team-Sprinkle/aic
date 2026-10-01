#!/usr/bin/env python3
"""Offline comparison to a held-out recorded teacher; never an actor input."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def rotation6(v):
    x=np.asarray(v[3:6]); x=x/np.linalg.norm(x)
    y=np.asarray(v[6:9]); y=y-x*np.dot(x,y); y=y/np.linalg.norm(y)
    return np.column_stack((x,y,np.cross(x,y)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit',type=Path);p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--episode',required=True);a=p.parse_args()
    live=[json.loads(s) for s in (a.audit/'runtime_trace.jsonl').open()]
    live=[r for r in live if r['event']=='proposal']
    teacher=[json.loads(s) for s in a.labels.open()]
    teacher=[r for r in teacher if r['episode_id']==a.episode]
    assert teacher and live
    # Align elapsed time only for diagnostics; explicitly report geometric
    # disagreement rather than implying equal physical states after divergence.
    tt=np.array([r['sim_time']-teacher[0]['sim_time'] for r in teacher])
    initial=Rotation.from_quat(live[0]['state'][3:7])
    reports=[]
    for elapsed in (0,1,2,5,10,20,40,60):
        l=min(live,key=lambda r:abs(r['sim_time']-live[0]['sim_time']-elapsed))
        t=teacher[int(np.abs(tt-elapsed).argmin())]
        lr=Rotation.from_quat(l['state'][3:7]);tr=Rotation.from_quat(t['state'][3:7])
        target=t['teacher_tcp_target_poses_opening_frame'][0]
        current=t['observed_tcp_pose_opening_frame']
        expected=(Rotation.from_quat(current[3:]).inv()*Rotation.from_quat(target[3:])).as_rotvec()
        predicted=Rotation.from_matrix(rotation6(l['targets'][0])).as_rotvec()
        reports.append(dict(elapsed_s=elapsed,
            state_translation_difference_mm=float(np.linalg.norm(np.array(l['state'][:3])-t['state'][:3])*1000),
            state_orientation_difference_deg=float((tr.inv()*lr).magnitude()*180/np.pi),
            autonomous_rotation_from_start_deg=float((initial.inv()*lr).magnitude()*180/np.pi),
            teacher_first_body_rotation_deg=(expected*180/np.pi).tolist(),
            teacher_first_body_translation_mm=(Rotation.from_quat(current[3:]).inv().apply(
                np.array(target[:3])-current[:3])*1000).tolist(),
            autonomous_first_body_rotation_deg=(predicted*180/np.pi).tolist(),
            autonomous_first_body_translation_mm=(np.array(l['targets'][0][:3])*1000).tolist()))
    result=dict(schema='aic_transport_first_divergence/v1',episode=a.episode,
        comparison='Elapsed-time alignment, not matched-state counterfactual; teacher geometry is posthoc only.',
        samples=reports,final_scenes_opened=False)
    (a.audit/'first_divergence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
