#!/usr/bin/env python3
"""Verify a route-only teacher at its retained terminal observation, before reset."""
import argparse,bisect,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def matrix(d):
    m=np.eye(4);m[:3,3]=[d['translation'][k] for k in 'xyz'];m[:3,:3]=Rotation.from_quat([d['rotation'][k] for k in 'xyzw']).as_matrix();return m


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stream',type=Path);a=p.parse_args();r=json.loads(a.stream.read_text())
    root=Path(__file__).resolve().parents[1];audit=Path(r['audit'].replace('/repo/',str(root)+'/'))
    events=[json.loads(l) for l in (audit/'runtime_trace.jsonl').open()];terminal=[e for e in events if e['event']=='transport_replay_completed'];assert len(terminal)==1
    end=round(terminal[0]['time_sec']*1e9);rows=r['streams']['/aic_controller/controller_state'];t,c=rows[bisect.bisect_right(rows,end,key=lambda x:x[0])-1];assert end-t<=50000000
    plan=json.loads((audit/'moveit_joint_trajectory.json').read_text())['waypoints'][-1]['tcp_pose'];actual=np.array([c['tcp_pose']['position'][k] for k in 'xyz']);delta=actual-np.array(plan['position'])
    def edge(k):
        rows=r['edges'][k];return matrix(rows[bisect.bisect_right(rows,end,key=lambda x:x[0])-1][1])
    selected=r['task']['target_module_name'];world_base=edge('world|tabletop')@edge('tabletop|base_link')
    world_port=edge('aic_world|task_board')@edge('task_board|task_board/'+selected)@edge('task_board/'+selected+'|task_board/'+selected+'/sc_port_base_link_entrance')
    opening=np.linalg.inv(world_base)@world_port;error=opening[:3,:3].T@delta*1000
    deg=float((Rotation.from_quat([c['tcp_pose']['orientation'][k] for k in 'xyzw']).inv()*Rotation.from_quat(plan['orientation_xyzw'])).magnitude()*180/np.pi)
    score=yaml.safe_load((audit/'results/scoring.yaml').read_text());trial=next(v for k,v in score.items() if k.startswith('trial_'));contacts=trial['tier_2']['categories']['contacts'];contact_free='No contact detected' in contacts['message']
    commands=[v for v in r['commands'] if v['sim_ns']<=end];joint_count=sum(v['kind']=='joint' for v in commands);pose_count=sum(v['kind']=='pose' for v in commands)
    result=dict(schema='aic_cartesian_route_terminal_audit/v1',joint_command_count=joint_count,pose_command_count=pose_count,terminal_sim_ns=end,terminal_before_reset=True,
        measured_vs_plan_final_tcp_mm=float(np.linalg.norm(delta)*1000),measured_final_tcp_base_m=actual.tolist(),planned_final_tcp_base_m=plan['position'],
        axial_tracking_error_mm=float(abs(error[2])),lateral_tracking_error_mm=float(np.linalg.norm(error[:2])),orientation_error_deg=deg,
        # These are transport-standoff gates from the teacher's prealign settings,
        # not the 0.5mm final-insertion corridor or perception accuracy thresholds.
        axial_gate_mm=20.,lateral_gate_mm=12.,orientation_gate_deg=3.,whole_run_offlimit_contact_free=contact_free,
        passed=bool(joint_count==0 and pose_count>0 and abs(error[2])<=20 and np.linalg.norm(error[:2])<=12 and deg<=3 and contact_free),
        admission='heldout_transport_validation_only',actor_geometry_input=False)
    (audit/'route_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
