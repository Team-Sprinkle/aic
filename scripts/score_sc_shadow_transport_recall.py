#!/usr/bin/env python3
"""Posthoc true selected-port standoff of a frozen SC shadow rollout."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from analyze_local_actor_scored_tip import record_at
from audit_sc_port_targets import matrix


def score(audit):
    data=json.loads((audit/'scored_stream.json').read_text())
    edges=data['edges']
    selected=int(data['task']['target_module_name'].rsplit('_',1)[-1])
    prefix=f'task_board/sc_port_{selected}'
    world_base=record_at(edges['world|tabletop'],0) @ record_at(edges['tabletop|base_link'],0)
    calib=json.loads(Path('configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json').read_text())
    opening=(np.linalg.inv(world_base) @ record_at(edges['aic_world|task_board'],0) @
             record_at(edges[f'task_board|{prefix}'],0) @
             record_at(edges[f'{prefix}|{prefix}/sc_port_base_link'],0) @
             matrix(calib['base_to_opening']))
    offset=np.asarray(json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
                      ['opening_to_transport_tcp_base_offset_m'])
    standoff=opening[:3,3]+offset
    trace=[json.loads(s) for s in (audit/'runtime_trace.jsonl').open()]
    proposals=[r for r in trace if r.get('event')=='proposal']
    observations=[]
    for r in proposals:
        state=np.asarray(r['state'],float)
        observations.append(dict(sim_time_s=r['sim_time'],
                                 true_standoff_error_mm=float(np.linalg.norm(state[:3]-standoff)*1000),
                                 true_opening_distance_mm=float(np.linalg.norm(state[:3]-opening[:3,3])*1000),
                                 measured_force_norm_n=float(np.linalg.norm(state[26:29]))))
    best=min(observations,key=lambda x:x['true_standoff_error_mm'])
    inspections=[]
    for r in trace:
        if r.get('event')!='shadow_handoff_inspection':continue
        closest=min(observations,key=lambda x:abs(x['sim_time_s']-r['sim_time']))
        inspections.append(dict(sim_time_s=r['sim_time'],valid=r['valid'],reason=r['reason'],
                                nearest_proposal_true_standoff_error_mm=closest['true_standoff_error_mm'],
                                verification_ms=r['verification_ms'],
                                identity_probability=r.get('spatial_identity_probability')))
    return dict(audit=str(audit),scene_group_id=data['scene_group_id'],
                selected_port=selected,proposals=len(proposals),
                closest=best,inspections=inspections,
                accepted=any(x['valid'] for x in inspections),
                final_scenes_opened=False,geometry_posthoc_only=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('audits',nargs='+',type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior score')
    results=[score(audit) for audit in a.audits]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(schema='sc_shadow_transport_recall_posthoc/v1',
                                        results=results),indent=2)+'\n')
    print(json.dumps([dict(audit=Path(x['audit']).name,
                           closest_mm=x['closest']['true_standoff_error_mm'],
                           accepted=x['accepted']) for x in results],indent=2))


if __name__=='__main__':main()
