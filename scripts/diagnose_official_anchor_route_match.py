#!/usr/bin/env python3
"""Posthoc measured target-frame trace diagnosis for frozen paired transport runs."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from build_two_family_port_projection_labels import sfp_ports
from materialize_sc_scored_route import port_opening


def opening_for(audit, calibration):
    stream=json.loads((audit/'scored_stream.json').read_text())
    task=stream['task']
    if task['plug_type']=='sc':
        return port_opening(stream,int(task['target_module_name'].split('_')[-1]))
    config=yaml.safe_load((audit/'eval_config.yaml').read_text())
    trial_id=next(iter(config['trials']))
    card=int(task['target_module_name'].split('_')[-1])
    port=int(task['port_name'].split('_')[-1])
    return {(c,p):matrix for c,p,matrix in sfp_ports(
        {'trial_yaml':str(audit/'eval_config.yaml'),'trial_id':trial_id},calibration)}[card,port]


def diagnose(audit,calibration):
    opening=opening_for(audit,calibration)
    trace=[json.loads(line) for line in (audit/'runtime_trace.jsonl').open()]
    proposals=[r for r in trace if r.get('event')=='proposal' and len(r.get('state',[]))==32]
    relative=[];commands=[];times=[]
    for row in proposals:
        state=np.asarray(row['state'],dtype=float)
        rel=opening[:3,:3].T@(state[:3]-opening[:3,3])*1000
        body=np.asarray(row['targets'][0][:3],dtype=float)
        world=Rotation.from_quat(state[3:7]).apply(body)
        cmd=opening[:3,:3].T@world*1000
        relative.append(rel);commands.append(cmd);times.append(float(row['sim_time']))
    rel=np.asarray(relative);cmd=np.asarray(commands);times=np.asarray(times)
    if not len(rel):raise ValueError(f'No actor proposals {audit}')
    lateral=np.linalg.norm(rel[:,:2],axis=1)
    in_band=np.flatnonzero((-150<=rel[:,2])&(rel[:,2]<=-70))
    toward_command=np.sum(-rel[:,:2]*cmd[:,:2],axis=1)/(lateral+1e-6)
    measured=np.diff(rel,axis=0)
    toward_measured=np.sum(-rel[:-1,:2]*measured[:,:2],axis=1)/(lateral[:-1]+1e-6)
    return dict(audit=str(audit),proposals=len(rel),duration_s=float(times[-1]-times[0]),
                start_target_frame_xyz_mm=rel[0].tolist(),end_target_frame_xyz_mm=rel[-1].tolist(),
                min_lateral_mm=float(lateral.min()),
                min_lateral_in_axial_minus150_to_minus70_mm=(float(lateral[in_band].min()) if len(in_band) else None),
                min_lateral_in_axial_minus100_to_minus70_mm=(float(lateral[(-100<=rel[:,2])&(rel[:,2]<=-70)].min())
                    if np.any((-100<=rel[:,2])&(rel[:,2]<=-70)) else None),
                max_axial_progress_mm=float(rel[:,2].max()-rel[0,2]),
                command_norm_mm_p50=float(np.median(np.linalg.norm(cmd,axis=1))),
                commanded_lateral_toward_port_mm_p50_in_near_band=(float(np.median(toward_command[in_band]))
                    if len(in_band) else None),
                measured_lateral_toward_port_mm_p50_in_near_band=(float(np.median(toward_measured[in_band[in_band<len(toward_measured)]]))
                    if len(in_band) else None),
                trajectory_target_frame_xyz_mm=rel.tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior diagnosis')
    plan=json.loads((a.comparison/'plan.json').read_text())
    progress=json.loads((a.comparison/'progress.json').read_text())
    calibration=json.loads(Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())
    rows=[]
    for scene in plan['scenes']:
        for arm in ('control','candidate'):
            item=progress[arm+'_'+scene['name']]
            row=diagnose(Path(item['audit']),calibration)
            row.update(arm=arm,scene=scene['name'],family=scene['family'])
            rows.append(row)
    result=dict(schema='aic_official_anchor_route_trace_diagnosis/v1',rows=rows,
                posthoc_geometry_only=True,final_scenes_opened=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=False,sharey=False)
    for ax,scene in zip(axes.flat,plan['scenes']):
        for arm,color in (('control','tab:gray'),('candidate','tab:blue')):
            row=next(r for r in rows if r['scene']==scene['name'] and r['arm']==arm)
            arr=np.asarray(row['trajectory_target_frame_xyz_mm'])
            ax.plot(arr[:,2],np.linalg.norm(arr[:,:2],axis=1),color=color,label=arm)
            ax.scatter([arr[0,2]],[np.linalg.norm(arr[0,:2])],color=color,marker='o',s=12)
        ax.axvspan(-100,-70,color='green',alpha=.1)
        ax.axhline(10,color='green',linestyle='--',linewidth=1)
        ax.set_title(scene['name'].replace('_validation',' dev'))
        ax.set_xlabel('Axial to port (mm)');ax.set_ylabel('Lateral from port (mm)')
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(a.output.with_suffix('.png'),dpi=150)
    plt.close(fig)
    print(json.dumps([{k:r[k] for k in ('arm','scene','min_lateral_in_axial_minus100_to_minus70_mm',
                                         'commanded_lateral_toward_port_mm_p50_in_near_band',
                                         'measured_lateral_toward_port_mm_p50_in_near_band')}
                      for r in rows],indent=2))


if __name__=='__main__':main()
