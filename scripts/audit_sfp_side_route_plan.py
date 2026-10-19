#!/usr/bin/env python3
"""Rank one SFP side-route plan using a rigidly carried measured cable shape.

This proxy is uncalibrated against dynamic cable contact. It cannot certify safety.
"""
import argparse
import json
from pathlib import Path

from audit_sc_port_targets import matrix as calibration_matrix
from audit_sfp_requested_corridor_cable import (
    cable_samples_tcp, nic_solids, pose_matrix, signed_clearance, take_tf,
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous audit')
    plan=json.loads(a.plan.read_text());audit=Path(plan['audit'])
    cal=json.loads(Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())
    world_base=calibration_matrix(cal['world_to_base'])
    bag=next(audit.glob('results/bag_trial*/*.mcap'))
    tf_error,tf_time,edges=take_tf(bag,plan['takeover_sim_s'])
    start=world_base@pose_matrix(plan['start_tcp_base_m'],plan['start_tcp_quat_xyzw'])
    cable=cable_samples_tcp(edges,start)
    solids=nic_solids(audit/'eval_config.yaml')
    variants={}
    for name,route in plan['route_variants'].items():
        if not route['success']:
            variants[name]=dict(moveit_success=False)
            continue
        rows=[]
        for s in route['samples']:
            tcp=world_base@pose_matrix(s['tcp_base_m'],s['tcp_quat_xyzw'])
            proximal=signed_clearance(cable,tcp,solids,5)
            full=signed_clearance(cable,tcp,solids,20)
            rows.append(dict(proximal_mm=proximal[0],proximal_link=proximal[1],
                             proximal_object=proximal[2],full_mm=full[0],
                             full_link=full[1],full_object=full[2]))
        variants[name]=dict(moveit_success=True,planned_samples=len(rows),
                            minimum_proximal_mm=min(x['proximal_mm'] for x in rows),
                            minimum_full_mm=min(x['full_mm'] for x in rows),
                            minimum_full_at=next(i for i,x in enumerate(rows)
                                                 if x['full_mm']==min(y['full_mm'] for y in rows)),
                            trace=rows)
    out=dict(schema='aic_sfp_side_route_rigid_cable_proxy/v1',plan=str(a.plan),
             source_bag=str(bag),measured_cable_tf_s=tf_time,
             rigid_cable_proxy_only=True,contact_oracle=False,variants=variants)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:{x:y for x,y in v.items() if x!='trace'} for k,v in variants.items()},indent=2))


if __name__=='__main__':main()
