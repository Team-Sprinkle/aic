#!/usr/bin/env python3
"""Explicit transport-only admission; failed local insertion never becomes BC."""
import argparse,bisect,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stream',type=Path);a=p.parse_args()
    r=json.loads(a.stream.read_text());root=Path(__file__).resolve().parents[1]
    audit=Path(r['audit'].replace('/repo/',str(root)+'/'))
    assert r['teacher_interface']['transport_action_mode']=='absolute_cartesian_pose_base_link'
    trace=[json.loads(l) for l in (audit/'runtime_trace.jsonl').open()]
    events=[e for e in trace if e['event']=='local_preinsert_align_started'];assert events,'No explicit local boundary'
    cutoff=round(min(e['time_sec'] for e in events)*1e9)
    states=r['streams']['/aic_controller/controller_state'];index=bisect.bisect_right(states,cutoff,key=lambda x:x[0])-1
    t,c=states[index];assert cutoff-t<=50000000
    plan=json.loads((audit/'moveit_joint_trajectory.json').read_text())['waypoints'][-1]['tcp_pose']
    actual=np.array([c['tcp_pose']['position'][k] for k in 'xyz']);goal=np.array(plan['position'])
    distance=float(np.linalg.norm(actual-goal)*1000)
    orientation=float((Rotation.from_quat([c['tcp_pose']['orientation'][k] for k in 'xyzw']).inv()*Rotation.from_quat(plan['orientation_xyzw'])).magnitude()*180/np.pi)
    commands=[x for x in r['commands'] if x['sim_ns']<cutoff]
    assert commands and all(c['kind']=='pose' and c['message']['header']['frame_id']=='base_link' for c in commands)
    # Whole-run scoring is stricter than prefix-only contact exclusion here.
    score=yaml.safe_load((audit/'results/scoring.yaml').read_text());trial=next(v for k,v in score.items() if k.startswith('trial_'))
    contacts=trial['tier_2']['categories']['contacts']
    contact_free='No contact detected' in contacts['message']
    result=dict(schema='aic_explicit_verified_transport_prefix/v1',cutoff_sim_ns=cutoff,teacher_interface='absolute_cartesian_pose_base_link',
        official_whole_episode_tier3=trial['tier_3'],full_success=False,
        purpose='control-matched Cartesian transport examples unavailable; do not include failed insertion',
        commanded_transport_poses=len(commands),measured_handoff_tcp_base_m=actual.tolist(),
        measured_vs_plan_final_tcp_mm=distance,orientation_error_deg=orientation,
        distance_gate_mm=10.,orientation_gate_deg=3.,whole_run_offlimit_contact_free=contact_free,
        includes_local_insertion=False,includes_supervisor_retreat=False,
        passed=distance<=10 and orientation<=3 and contact_free,video_review_required=True)
    (audit/'transport_prefix_admission.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
