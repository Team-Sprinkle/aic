#!/usr/bin/env python3
"""Compare live SFP proposals to same-task teacher geometry and history support."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from shared_port_tcp_geometry import matrix_from_pose9


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    teachers=defaultdict(list)
    for r in map(json.loads,a.rows.open()):
        if r['family']=='sfp' and r['split']=='train':teachers[tuple(r['task'])].append(r)
    banks={}
    for key,rows in teachers.items():
        banks[key]=dict(rows=rows,tip=np.asarray([r['predicted_tip_port_pose9'][:3] for r in rows])*1000,
            rotation=np.stack([matrix_from_pose9(r['predicted_port_tcp_pose9'])[:3,:3] for r in rows]),
            delta=np.asarray([r['teacher_tcp_targets_predicted_port_pose9'][0][2]-r['predicted_tip_port_pose9'][2] for r in rows])*1000)
    comparison=json.loads((a.comparison/'summary.json').read_text());results=[]
    for r in comparison['results']:
        if r['family']!='sfp':continue
        folder=Path(r['audit']);case=json.loads((a.comparison/(folder.name+'.case.json')).read_text())
        trace=list(map(json.loads,(folder/'runtime_trace.jsonl').open()))
        planfile=json.loads((folder/'shared150mm_metrics.json').read_text())
        # Exact selected task code is retained in the environment / case; here
        # use the public family+port+NIC request convention from the data bank.
        matching=[k for k in banks if k[0]==1 and k[2+case['port']]==1 and k[4+case['requested_card']]==1]
        if len(matching)!=1:raise ValueError('Ambiguous public task bank')
        bank=banks[matching[0]]
        plans=[t for t in trace if t['event']=='locked_local_perception'];examples=[]
        for t in plans[:40]:
            xyz=np.asarray(t['predicted_tip_port_position_m'])*1000
            rotation=matrix_from_pose9(t['predicted_tcp_pose_in_port'])[:3,:3]
            distance=np.linalg.norm(bank['tip']-xyz,axis=1)
            cosine=(np.einsum('nij,ij->n',bank['rotation'],rotation)-1)/2
            angle=np.degrees(np.arccos(cosine.clip(-1,1)))
            ids=np.flatnonzero((distance<=5)&(angle<=5))
            selected=ids[np.argsort(distance[ids])[:30]]
            delta=(t['actor_control_point_targets_port'][0][2]-xyz[2]/1000)*1000
            examples.append(dict(sim_time=t['sim_time'],tip_z_mm=float(xyz[2]),proposed_delta_mm=delta,
                nearby_teacher_rows=len(ids),nearby_teacher_episodes=len({bank['rows'][i]['episode'] for i in ids}),
                nearest_teacher_distance_mm=float(min(distance)),
                nearest_teacher_orientation_deg=float(angle[np.argmin(distance)]),
                local_teacher_delta_median_mm=float(np.median(bank['delta'][selected])) if len(selected) else None,
                local_teacher_retreat_fraction=float(np.mean(bank['delta'][selected]<-1)) if len(selected) else None,
                nearest_examples=[dict(episode=bank['rows'][i]['episode'],time_s=bank['rows'][i]['time_s'],
                    distance_mm=float(distance[i]),angle_deg=float(angle[i]),delta_mm=float(bank['delta'][i])) for i in selected[:3]]))
        results.append(dict(arm=r['arm'],public_task=list(matching[0]),observations=examples,
            caveat='Geometry-neighbor support does not establish matched images, force, grasp rotation or causal history.'))
    report=dict(results=results,final_scenes_opened=False,source_modified=False)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    for r in results:
        ex=r['observations'];supported=[x for x in ex if x['nearby_teacher_rows']]
        contradict=[x for x in supported if x['proposed_delta_mm'] < -1 and x['local_teacher_delta_median_mm']>1]
        print(r['arm'],'first40',len(ex),'supported',len(supported),'retreat_vs_forward_neighbor',len(contradict))


if __name__=='__main__':main()
