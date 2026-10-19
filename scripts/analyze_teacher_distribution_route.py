#!/usr/bin/env python3
"""Posthoc teacher/actor TCP route comparison on identical nonfinal resets."""
import argparse
from bisect import bisect_left
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from build_two_family_port_projection_labels import sfp_ports
from materialize_sc_scored_route import port_opening
from materialize_sfp_scored_transport import posematrix
from score_official_anchor_route_dev import score

ROOT=Path(__file__).resolve().parents[1]
SC_STREAM=ROOT/'outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized/streams'
CAL=json.loads((ROOT/'configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())


def opening_for(stream,audit,family):
    task=stream['task']
    if family=='sc':return port_opening(stream,int(task['target_module_name'].split('_')[-1]))
    cfg=yaml.safe_load((audit/'eval_config.yaml').read_text())
    trial=next(iter(cfg['trials']))
    card=int(task['target_module_name'].split('_')[-1]);port=int(task['port_name'].split('_')[-1])
    return {(c,p):m for c,p,m in sfp_ports(
        {'trial_yaml':str(audit/'eval_config.yaml'),'trial_id':trial},CAL)}[(card,port)]


def polar(tcp,opening):
    point=opening[:3,:3].T@(tcp-opening[:3,3])*1000
    return float(point[2]),float(np.linalg.norm(point[:2]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing analysis')
    plan=json.loads((a.comparison/'plan.json').read_text())
    progress=json.loads((a.comparison/'progress.json').read_text())
    rows=[]
    for spec in plan['scenes']:
        name=spec['name'];family=spec['family']
        if name not in progress:continue
        actor=Path(progress[name]['audit']);teacher=Path(spec['teacher'])
        if not (actor/'scored_stream.json').exists():continue
        if family=='sc':stream=json.loads((SC_STREAM/(name+'.json')).read_text())
        else:stream=json.loads((teacher/'scored_stream.json').read_text())
        opening=opening_for(stream,teacher,family)
        controls=stream['streams']['/aic_controller/controller_state']
        stamps=np.asarray([r[0] for r in controls],np.int64)
        proposals=[json.loads(line) for line in (actor/'runtime_trace.jsonl').open()
                   if json.loads(line).get('event')=='proposal']
        comparison=[]
        for row in proposals:
            stamp=int(round(float(row['sim_time'])*1e9))
            j=bisect_left(stamps,stamp)
            if j>=len(controls):continue
            if j and abs(stamps[j-1]-stamp)<abs(stamps[j]-stamp):j-=1
            if abs(stamps[j]-stamp)>200_000_000:continue
            ref=posematrix(controls[j][1]['tcp_pose'])
            obs=np.asarray(row['state'][:3],float)
            quat=np.asarray(row['state'][3:7],float)
            d=float(np.linalg.norm(obs-ref[:3,3])*1000)
            angle=float((Rotation.from_quat(quat).inv()*Rotation.from_matrix(ref[:3,:3])).magnitude()*180/np.pi)
            ax,lat=polar(obs,opening);teacher_ax,teacher_lat=polar(ref[:3,3],opening)
            target=np.asarray(row['targets'][0][:3],float)
            body=Rotation.from_quat(quat).apply(target)
            command=opening[:3,:3].T@body*1000
            relative=opening[:3,:3].T@(obs-opening[:3,3])*1000
            correction=-relative[:2]
            targetward=float(np.dot(command[:2],correction)/(np.linalg.norm(correction)+1e-9))
            comparison.append(dict(sim_s=float(row['sim_time']),tcp_teacher_mm=d,
                orientation_teacher_deg=angle,actor_axial_mm=ax,actor_lateral_mm=lat,
                teacher_axial_mm=teacher_ax,teacher_lateral_mm=teacher_lat,
                first_command_port_mm=command.tolist(),targetward_lateral_command_mm=targetward))
        departure=next((comparison[i] for i in range(max(0,len(comparison)-2))
                        if all(comparison[k]['tcp_teacher_mm']>20 for k in range(i,i+3))),None)
        band=[x for x in comparison if -100<=x['actor_axial_mm']<=-70]
        scored=score(actor,CAL)
        teacher_band=[]
        for _,msg in controls[::20]:
            ax,lat=polar(posematrix(msg['tcp_pose'])[:3,3],opening)
            if -100<=ax<=-70:teacher_band.append(lat)
        rows.append(dict(name=name,family=family,teacher=str(teacher),actor=str(actor),
            teacher_full_insertion=True,teacher_nearest_buffer_lateral_mm=min(teacher_band) if teacher_band else None,
            first_sustained_20mm_departure=departure,
            compared_proposals=len(comparison),actor_score=scored,
            actor_min_lateral_in_band_mm=min((x['actor_lateral_mm'] for x in band),default=None),
            matched_trace=comparison))
    result=dict(schema='aic_teacher_distribution_route_analysis/v1',rows=rows,
                actor_privileged_inputs=False,geometry_posthoc_only=True,final_scenes_opened=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:
        d=r['first_sustained_20mm_departure']
        print(json.dumps(dict(name=r['name'],teacher_buffer_mm=r['teacher_nearest_buffer_lateral_mm'],
            actor_buffer_mm=r['actor_min_lateral_in_band_mm'],departure_sim_s=d['sim_s'] if d else None,
            departure_mm=d['tcp_teacher_mm'] if d else None,contact=r['actor_score']['official_contact_message'])),flush=True)


if __name__=='__main__':main()
