#!/usr/bin/env python3
"""Posthoc route and smoothness diagnostics; never used by the actor."""
import argparse
import json
from pathlib import Path

import numpy as np


def segment_distance(points, a, b):
    delta = b-a
    fraction = np.clip(((points-a)@delta)/(delta@delta), 0, 1)
    return np.linalg.norm(points-(a+fraction[:, None]*delta), axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('audit', type=Path)
    p.add_argument('--route', type=Path, required=True)
    a = p.parse_args()
    trace = [json.loads(s) for s in (a.audit/'runtime_trace.jsonl').open()]
    commands = [r for r in trace if r['event'] == 'command']
    if len(commands) < 3: raise ValueError('Too few measured commands')
    route = json.loads(a.route.read_text())
    waypoints = [{'name': 'start', 'tcp_base_m': route['start_tcp_base_m']}] + route['waypoints']
    path = np.asarray([w['tcp_base_m'] for w in waypoints], dtype=float)
    xyz = np.asarray([r['state'][:3] for r in commands], dtype=float)
    times = np.asarray([r['sim_time'] for r in commands], dtype=float)
    distance = np.min(np.stack([segment_distance(xyz, u, v)
                                for u, v in zip(path[:-1], path[1:])]), axis=0)*1000
    nearest = []
    cursor = 0
    for w in waypoints[1:]:
        distances = np.linalg.norm(xyz[cursor:]-w['tcp_base_m'], axis=1)*1000
        j = cursor+int(distances.argmin())
        nearest.append({'name': w['name'], 'min_ordered_distance_mm': float(distances.min()),
                        'nearest_elapsed_s': float(times[j]-times[0])})
        cursor = j
    dt = np.diff(times)
    valid = dt > 0
    velocity = np.diff(xyz, axis=0)[valid]/dt[valid, None]
    velocity_dt = dt[valid]
    accel = np.diff(velocity, axis=0)/np.maximum((velocity_dt[1:]+velocity_dt[:-1])/2, 1e-3)[:, None]
    report = {'schema': 'aic_transport_route_reliability/v1',
              'audit': str(a.audit.resolve()), 'route': str(a.route.resolve()),
              'commands': len(commands), 'duration_s': float(times[-1]-times[0]),
              'polyline_deviation_mm': {'median': float(np.median(distance)),
                                        'p95': float(np.quantile(distance, .95)),
                                        'max': float(distance.max())},
              'ordered_waypoint_visits': nearest,
              'measured_path_mm': float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()*1000),
              'nominal_route_mm': float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()*1000),
              'measured_speed_mm_s_p95': float(np.quantile(np.linalg.norm(velocity, axis=1)*1000, .95)),
              'measured_acceleration_mm_s2_p95': float(np.quantile(np.linalg.norm(accel, axis=1)*1000, .95)),
              'posthoc_only': True, 'final_scenes_opened': False}
    (a.audit/'route_reliability.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'deviation': report['polyline_deviation_mm'],
                      'visits': nearest, 'path_mm': report['measured_path_mm']}, indent=2))


if __name__ == '__main__': main()
