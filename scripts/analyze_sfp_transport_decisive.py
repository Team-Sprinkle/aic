#!/usr/bin/env python3
"""Posthoc SFP port-distance audit; geometry never enters the live actor."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def transform(pose):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--waypoints', type=Path, required=True)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--teacher-standoff-time', type=float,
                        help='Recorded teacher pre-insertion TCP pose time, for posthoc transport comparison')
    args = parser.parse_args()
    teacher = []
    for line in args.waypoints.open():
        row = json.loads(line)
        if row['episode_index'] == args.episode:
            teacher.append(row)
    if not teacher:
        raise ValueError('Episode absent from waypoint file')
    base_port = (transform(teacher[0]['state'][:7]) @
                 np.linalg.inv(transform(teacher[0]['observed_tcp_pose_port_frame'])))
    samples = []
    latencies = []
    for line in args.trace.open():
        row = json.loads(line)
        if row['event'] == 'proposal' and 'inference_ms' in row:
            latencies.append(row['inference_ms'])
        if row['event'] not in ('proposal', 'terminal_before_return'):
            continue
        port_tcp = np.linalg.inv(base_port) @ transform(row['state'][:7])
        xyz = port_tcp[:3, 3]
        samples.append(dict(time_s=row['sim_time'], event=row['event'],
                            port_tcp_xyz_m=xyz.tolist(), distance_mm=float(np.linalg.norm(xyz)*1000)))
    nearest = min(samples, key=lambda sample: sample['distance_mm'])
    result = dict(schema='aic_sfp_transport_decisive/v1', episode=args.episode,
                  live_samples=len(samples),teacher_rows=len(teacher),
                  first=samples[0],nearest=nearest,last=samples[-1],
                  teacher_last_port_tcp_xyz_m=teacher[-1]['observed_tcp_pose_port_frame'][:3],
                  teacher_last_distance_mm=float(np.linalg.norm(
                      teacher[-1]['observed_tcp_pose_port_frame'][:3])*1000),
                  inference_ms={q:float(np.quantile(latencies,q/100)) for q in (50,95,99)},
                  port_geometry_use='posthoc analysis only; not passed to controller')
    if args.teacher_standoff_time is not None:
        reference=min(teacher,key=lambda row:abs(row['timestamp_s']-args.teacher_standoff_time))
        teacher_pose=transform(reference['observed_tcp_pose_port_frame'])
        terminal=next(row for row in reversed(list(args.trace.open()))
                      if '"event": "terminal_before_return"' in row)
        live_pose=np.linalg.inv(base_port)@transform(json.loads(terminal)['state'][:7])
        error=live_pose[:3,3]-teacher_pose[:3,3]
        orientation=(Rotation.from_matrix(teacher_pose[:3,:3]).inv()*
                     Rotation.from_matrix(live_pose[:3,:3])).magnitude()*180/np.pi
        result['teacher_standoff_comparison']=dict(
            teacher_time_s=reference['timestamp_s'],
            teacher_port_tcp_xyz_m=reference['observed_tcp_pose_port_frame'][:3],
            terminal_position_error_mm=float(np.linalg.norm(error)*1000),
            terminal_lateral_error_mm=float(np.linalg.norm(error[:2])*1000),
            terminal_axial_error_mm=float(abs(error[2])*1000),
            terminal_orientation_error_deg=float(orientation),
            note='The selected teacher timestamp defines a development transport standoff, not the insertion target.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
