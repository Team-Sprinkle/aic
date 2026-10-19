#!/usr/bin/env python3
"""Posthoc measured route quality against a recorded teacher plan."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('audit', type=Path)
    p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args()
    trace = [json.loads(s) for s in (a.audit/'runtime_trace.jsonl').open()]
    commands = [r for r in trace if r['event'] == 'command']
    terminal = [r for r in trace if r['event'] == 'terminal_before_return']
    plan = json.loads(a.plan.read_text())['waypoints']
    path = np.asarray([w['tcp_pose']['position'] for w in plan], dtype=float)
    goal = plan[-1]['tcp_pose']
    xyz = np.asarray([r['state'][:3] for r in commands], dtype=float)
    times = np.asarray([r['sim_time'] for r in commands], dtype=float)
    distances = []
    for u, v in zip(path[:-1], path[1:]):
        dv = v-u
        if dv@dv < 1e-14: continue
        f = np.clip(((xyz-u)@dv)/(dv@dv), 0, 1)
        distances.append(np.linalg.norm(xyz-(u+f[:, None]*dv), axis=1))
    route_error = np.min(np.stack(distances), axis=0)*1000
    goal_error = np.linalg.norm(xyz-goal['position'], axis=1)*1000
    goal_rotation = Rotation.from_quat(goal['orientation_xyzw'])
    angular_error = np.asarray([(Rotation.from_quat(r['state'][3:7]).inv()*goal_rotation).magnitude()*180/np.pi
                                for r in commands])
    gate = np.flatnonzero((goal_error<=10.) & (angular_error<=3.))
    first = int(gate[0]) if len(gate) else None
    nearest = int(goal_error.argmin())
    dt = np.diff(times)
    good = dt > 0
    velocity = np.diff(xyz, axis=0)[good]/dt[good, None]
    acc = np.diff(velocity, axis=0)/np.maximum((dt[good][:-1]+dt[good][1:])/2, 1e-3)[:, None]
    chosen = route_error[:first+1] if first is not None else route_error
    terminal_state = terminal[-1]['state'] if terminal else commands[-1]['state']
    result = {'schema': 'aic_transport_teacher_path_diagnostic/v1',
              'audit': str(a.audit.resolve()), 'plan': str(a.plan.resolve()),
              'first_10mm_3deg_gate_time_s': float(times[first]-times[0]) if first is not None else None,
              'first_gate_position_mm': float(goal_error[first]) if first is not None else None,
              'first_gate_orientation_deg': float(angular_error[first]) if first is not None else None,
              'route_deviation_before_first_gate_mm': {'median': float(np.median(chosen)),
                  'p95': float(np.quantile(chosen,.95)), 'max': float(chosen.max())},
              'route_deviation_full_mm_p95': float(np.quantile(route_error,.95)),
              'nearest_standoff_position_mm': float(goal_error[nearest]),
              'nearest_standoff_orientation_deg': float(angular_error[nearest]),
              'nearest_standoff_time_s': float(times[nearest]-times[0]),
              'route_deviation_before_nearest_mm_p95': float(np.quantile(route_error[:nearest+1],.95)),
              'measured_path_before_gate_mm': float(np.linalg.norm(np.diff(xyz[:first+1],axis=0),axis=1).sum()*1000)
                  if first is not None else None,
              'measured_speed_mm_s_p95': float(np.quantile(np.linalg.norm(velocity,axis=1)*1000,.95)),
              'measured_acceleration_mm_s2_p95': float(np.quantile(np.linalg.norm(acc,axis=1)*1000,.95)),
              'terminal_drift_after_gate_mm': float(np.linalg.norm(np.asarray(terminal_state[:3])-xyz[first])*1000)
                  if first is not None else None,
              'terminal_error_mm': float(np.linalg.norm(np.asarray(terminal_state[:3])-goal['position'])*1000),
              'posthoc_only': True, 'final_scenes_opened': False}
    (a.audit/'teacher_path_diagnostic.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
