#!/usr/bin/env python3
"""Post-run diagnostics only. Planned geometry never reaches the live actor."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit',type=Path);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--labels',type=Path)
    p.add_argument('--episode',help='Optional recorded opening-frame label for posthoc transport-envelope scoring')
    a=p.parse_args()
    records=[json.loads(v) for v in (a.audit/'runtime_trace.jsonl').open()]
    proposals=[r for r in records if r['event']=='proposal']
    commands=[r for r in records if r['event']=='command']
    terminal=[r for r in records if r['event']=='terminal_before_return']
    if not commands:raise ValueError('No autonomous commands')
    waypoints=json.loads(a.plan.read_text())['waypoints']
    goal=waypoints[-1]['tcp_pose']
    xyz=np.array([r['state'][:3] for r in commands])
    distance=np.linalg.norm(xyz-goal['position'],axis=1)*1000
    final=terminal[-1]['state'] if terminal else commands[-1]['state']
    final_distance=float(np.linalg.norm(np.array(final[:3])-goal['position'])*1000)
    orientation=float((Rotation.from_quat(final[3:7]).inv()*Rotation.from_quat(goal['orientation_xyzw'])).magnitude()*180/np.pi)
    latencies=[r['inference_ms'] for r in proposals]
    score=yaml.safe_load((a.audit/'results/scoring.yaml').read_text())
    trial=next(v for k,v in score.items() if k.startswith('trial_'))
    forces=np.linalg.norm(np.array([r['state'][26:29] for r in commands]),axis=1)
    result=dict(schema='aic_clean_transport_autonomous/v1',commands=len(commands),proposals=len(proposals),
                simulation_duration_s=commands[-1]['sim_time']-commands[0]['sim_time'],
                final_planned_standoff_error_mm=final_distance,min_planned_standoff_error_mm=float(distance.min()),
                final_standoff_orientation_error_deg=orientation,
                measured_tcp_total_path_mm=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()*1000),
                final_tcp_base_m=final[:3],planned_standoff_base_m=goal['position'],
                raw_wrist_force_max_n=float(forces.max()),raw_wrist_force_p95_n=float(np.quantile(forces,.95)),
                inference_ms=dict(zip(('p50','p95','p99'),map(float,np.quantile(latencies,[.5,.95,.99])))),
                inference_max_ms=max(latencies),inference_includes_cold_start=True,
                dropped_stale_members=sum(r['event']=='stale_chunk_member_dropped' for r in records),
                terminal_retained=bool(terminal),official_tier3=trial['tier_3'],official_total=score['total'],
                geometry_live_input=False,teacher_or_planner_live=False,final_scene_opened=False,
                standoff_gate_distance_mm=10.,standoff_distance_gate=final_distance<=10.,
                note='Standoff error is post-run plan comparison, not perception error or full insertion. Force is raw wrist wrench including gravity.')
    if a.labels:
        if not a.episode:p.error('--labels requires --episode')
        row=next(json.loads(s) for s in a.labels.open() if json.loads(s)['episode_id']==a.episode)
        # Base_T_opening = Base_T_TCP @ inverse(Opening_T_TCP).
        def matrix(pose):
            m=np.eye(4);m[:3,:3]=Rotation.from_quat(pose[3:7]).as_matrix();m[:3,3]=pose[:3];return m
        opening=matrix(row['state'][:7])@np.linalg.inv(matrix(row['observed_tcp_pose_opening_frame']))
        err=opening[:3,:3].T@(np.array(final[:3])-goal['position'])
        lateral=float(np.linalg.norm(err[:2])*1000);axial=float(abs(err[2])*1000)
        result['transport_envelope']=dict(lateral_error_mm=lateral,axial_error_mm=axial,
            orientation_error_deg=orientation,lateral_threshold_mm=12,axial_threshold_mm=20,
            orientation_threshold_deg=3,geometry_only_pass=lateral<=12 and axial<=20 and orientation<=3,
            note='Posthoc teacher prealignment envelope; not an insertion gate. Route/blockage review is additionally required.')
    (a.audit/'autonomous_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    # Save measured path versus the teacher path for failure localization.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    teacher=np.array([r['tcp_pose']['position'] for r in waypoints])
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,(i,j) in zip(axes,[(0,1),(0,2),(1,2)]):
        ax.plot(teacher[:,i]*1000,teacher[:,j]*1000,label='teacher plan')
        ax.plot(xyz[:,i]*1000,xyz[:,j]*1000,label='autonomous measured')
        ax.scatter(xyz[-1,i]*1000,xyz[-1,j]*1000,marker='x',c='red')
        ax.set_xlabel('xyz'[i]+' (mm)');ax.set_ylabel('xyz'[j]+' (mm)');ax.axis('equal');ax.legend()
    fig.tight_layout();fig.savefig(a.audit/'measured_transport_path.png',dpi=140);plt.close(fig)
    print(json.dumps(result))


if __name__=='__main__':main()
