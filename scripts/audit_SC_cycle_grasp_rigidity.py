#!/usr/bin/env python3
"""Offline causal SC grasp drift audit; no labels enter the running controller."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def matrix(value):
    result = np.eye(4)
    result[:3, 3] = [value['translation'][k] for k in 'xyz']
    result[:3, :3] = Rotation.from_quat([value['rotation'][k] for k in 'xyzw']).as_matrix()
    return result


def past(rows, stamp):
    index = bisect_right(rows, stamp, key=lambda row: row[0])-1
    if index < 0:
        raise ValueError('No causal observation; never substitute reset geometry')
    return rows[index]


def audit(path):
    stream = json.loads((path/'scored_stream.json').read_text())
    events = [json.loads(line) for line in (path/'runtime_trace.jsonl').open()]
    task = stream['task']
    assert task['plug_name'].startswith('sc')
    cable = task['cable_name']
    keys = ('aic_world|'+cable, cable+'|'+cable+'/'+task['plug_name']+'_link')
    edges = stream['edges']

    def truth(time_s):
        requested = round(time_s*1e9)
        stamp = min(past(edges[key], requested)[0] for key in keys)
        def edge(key):
            return matrix(past(edges[key], stamp)[1])
        tip = (np.linalg.inv(edge('world|tabletop')@edge('tabletop|base_link'))
               @edge('world|aic_world')@edge(keys[0])@edge(keys[1]))
        controller_stamp, controller = past(stream['streams']['/aic_controller/controller_state'], stamp)
        tcp = matrix(dict(translation=controller['tcp_pose']['position'],
                          rotation=controller['tcp_pose']['orientation']))
        ages = [stamp-past(edges[key], stamp)[0] for key in keys]+[stamp-controller_stamp]
        if max(ages) > 10_000_000 or requested-stamp > 100_000_000:
            raise ValueError('Stale causal geometry/controller join')
        return np.linalg.inv(tcp)@tip, dict(observation_to_TF_ms=(requested-stamp)/1e6,
                                          join_ages_ms=[age/1e6 for age in ages])

    update = next(row for row in events if row['event']=='close_grasp_update_accepted')
    initial, _ = truth(update['sim_time'])
    prediction = np.asarray(update['candidate']['rotation'])
    stamps = [('near_update', update['sim_time'])]
    steps = [row for row in events if row['event']=='visual_servo_step']
    for depth in (0., .010, .014):
        row = min(steps, key=lambda row: abs(row['estimated_tip_port_m'][2]-depth))
        stamps.append((f'insertion_{depth*1000:g}mm', row['sim_time']))
    terminal = next(row for row in reversed(events) if row['event']=='terminal_before_return'
                    and row.get('observation_source')=='retained_public_observation_before_return')
    stamps.append(('terminal', terminal['sim_time']))
    stages = []
    for name, time_s in stamps:
        grasp, ages = truth(time_s)
        stages.append(dict(phase=name, time_s=time_s, causal_join=ages,
            grasp_change_deg=float(np.degrees(Rotation.from_matrix(initial[:3,:3].T@grasp[:3,:3]).magnitude())),
            grasp_translation_change_mm=float(np.linalg.norm(grasp[:3,3]-initial[:3,3])*1000),
            prediction_error_deg=float(np.degrees(Rotation.from_matrix(prediction.T@grasp[:3,:3]).magnitude()))))
    return dict(audit=str(path), stages=stages)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--cycles', type=int, nargs='+', default=[1,2,3,4,5])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier diagnostic')
    rows = [dict(cycle=cycle, **audit(args.root/f'cycle_{cycle}/per_trial/trial_3')) for cycle in args.cycles]
    args.output.write_text(json.dumps(dict(geometry_offline_only=True, post_reset_used=False,
                                          reports=rows), indent=2)+'\n')
    print(json.dumps(rows), flush=True)
