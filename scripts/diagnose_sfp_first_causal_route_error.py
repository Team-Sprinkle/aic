#!/usr/bin/env python3
"""Join actor proposals to near-identical teacher TCP states before route drift.

Teacher geometry/actions are used only for offline diagnosis. Matching solely
by clock would mistake a slower execution for a wrong action.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def actor_rows(path):
    return [r for line in (path / 'runtime_trace.jsonl').open()
            if (r := json.loads(line)).get('event') == 'proposal']


def teacher_rows(path):
    data = json.loads((path / 'scored_stream.json').read_text())
    states = data['streams']['/aic_controller/controller_state']
    poses = np.asarray([[r['tcp_pose']['position'][k] for k in 'xyz']
                        for _, r in states])
    quat = np.asarray([[r['tcp_pose']['orientation'][k] for k in 'xyzw']
                       for _, r in states])
    times = np.asarray([t/1e9 for t, _ in states])
    commands = [r for r in data['commands'] if r['kind']=='pose']
    command_times = np.asarray([r['sim_ns']/1e9 for r in commands])
    command_xyz = np.asarray([[r['message']['pose']['position'][k] for k in 'xyz']
                              for r in commands])
    assert {r['message']['header']['frame_id'] for r in commands} == {'gripper/tcp'}
    return times, poses, quat, command_times, command_xyz


def diagnose(run, teacher, max_actor_sim_time_s=12):
    rows = actor_rows(run)
    t, xyz, quat, ct, cmd = teacher_rows(teacher)
    tree = cKDTree(xyz)
    joined = []
    for i, row in enumerate(rows):
        sim = float(row['sim_time'])
        if sim > max_actor_sim_time_s:
            break
        state = np.asarray(row['state'], float)
        actor_quat = Rotation.from_quat(state[3:7])
        distance, j = tree.query(state[:3])
        orientation = (actor_quat.inv()*Rotation.from_quat(quat[j])).magnitude()*180/np.pi
        # Teacher's command immediately after this measured state, not an
        # arbitrary nearest command from before it.
        k = int(np.searchsorted(ct, t[j], side='left'))
        if k >= len(ct):
            continue
        teacher_delta = cmd[k]*1000
        actor_delta = np.asarray(row['targets'][0][:3])*1000
        actor_base = actor_quat.apply(actor_delta)
        teacher_base = Rotation.from_quat(quat[j]).apply(teacher_delta)
        cosine = float(actor_base @ teacher_base / max(1e-9,
            np.linalg.norm(actor_base)*np.linalg.norm(teacher_base)))
        next_state = np.asarray(rows[min(i+1,len(rows)-1)]['state'][:3],float)
        motion = (next_state-state[:3])*1000
        age = {cam: round(sim-stamp/1e9,3)
               for cam,stamp in row.get('image_sim_ns',{}).items()}
        joined.append(dict(actor_time_s=sim,teacher_time_s=float(t[j]),
            position_gap_mm=float(distance*1000),orientation_gap_deg=float(orientation),
            teacher_command_time_s=float(ct[k]),
            actor_first_command_tcp_mm=actor_delta.tolist(),
            teacher_first_command_tcp_mm=teacher_delta.tolist(),
            actor_command_norm_mm=float(np.linalg.norm(actor_delta)),
            teacher_command_norm_mm=float(np.linalg.norm(teacher_delta)),
            command_base_cosine=cosine,
            measured_next_tcp_motion_mm=motion.tolist(),
            measured_next_motion_norm_mm=float(np.linalg.norm(motion)),
            command_motion_cosine=float(actor_base @ motion / max(1e-9,
                np.linalg.norm(actor_base)*np.linalg.norm(motion))),
            image_age_s=age,task_vector=row['task_vector']))
    close = [r for r in joined if r['position_gap_mm']<=3 and r['orientation_gap_deg']<=3
             and 0<=r['teacher_command_time_s']-r['teacher_time_s']<=.12]
    early = [r for r in close if r['actor_time_s']<=8]
    late = [r for r in close if 12<=r['actor_time_s']<=20]
    return dict(run=str(run),teacher=str(teacher),proposals_first12=len(joined),
        matched_3mm_3deg=len(close),matched_before8s=len(early),
        matched_actor_command_norm_mm_p50=float(np.median([r['actor_command_norm_mm'] for r in early]))
            if early else None,
        matched_teacher_command_norm_mm_p50=float(np.median([r['teacher_command_norm_mm'] for r in early]))
            if early else None,
        matched_command_cosine_p50=float(np.median([r['command_base_cosine'] for r in early]))
            if early else None,
        matched_next_motion_mm_p50=float(np.median([r['measured_next_motion_norm_mm'] for r in early]))
            if early else None,
        matched_image_age_ms_p95=float(np.percentile([age*1000 for r in early
            for age in r['image_age_s'].values()],95)) if early else None,
        matched_examples=early[::max(1,len(early)//12)],
        matched_late_12to20s=len(late),
        late_command_cosine_p50=(float(np.median([r['command_base_cosine'] for r in late]))
                                 if late else None),
        late_command_norm_mm_p50=(float(np.median([r['actor_command_norm_mm'] for r in late]))
                                  if late else None),
        late_teacher_command_norm_mm_p50=(float(np.median([r['teacher_command_norm_mm'] for r in late]))
                                          if late else None),
        late_examples=late[::max(1,len(late)//12)],rows=joined)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--posthoc',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-actor-sim-time-s',type=float,default=12)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous diagnosis')
    posthoc=json.loads(a.posthoc.read_text())
    out=[diagnose(Path(r['run']),Path(r['teacher']),a.max_actor_sim_time_s)
         for r in posthoc['runs']]
    a.output.write_text(json.dumps(dict(schema='aic_sfp_first_causal_route_error/v1',
        posthoc=str(a.posthoc),max_actor_sim_time_s=a.max_actor_sim_time_s,position_match_mm=3,
        orientation_match_deg=3,teacher_command_after_state_max_s=.12,
        geometry_actor_input=False,final_scenes_opened=False,runs=out),indent=2)+'\n')
    for r in out:
        print(json.dumps({k:r[k] for k in ('run','matched_before8s',
            'matched_actor_command_norm_mm_p50','matched_teacher_command_norm_mm_p50',
            'matched_command_cosine_p50','matched_next_motion_mm_p50',
            'matched_image_age_ms_p95')}))


if __name__=='__main__':main()
