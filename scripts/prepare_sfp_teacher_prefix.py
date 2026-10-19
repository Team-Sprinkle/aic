#!/usr/bin/env python3
"""Export a same-scene teacher controller-reference prefix for diagnosis only."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--duration-s', type=float, default=5.0)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve prior prefix')
    scored = json.loads((a.teacher / 'scored_stream.json').read_text())
    rows = scored['streams']['/aic_controller/controller_state']
    if not rows:
        raise ValueError('No controller state')
    # Match the first admitted causal teacher RGB/action observation, rather
    # than the controller's pre-policy initial message.
    first = next(t / 1e9 for t, row in rows
                 if row['reference_tcp_pose'] != rows[0][1]['reference_tcp_pose'])
    exported = []
    for t, row in rows:
        sim = t / 1e9
        if first - .1 <= sim <= first + a.duration_s + .3:
            pose = row['reference_tcp_pose']
            exported.append(dict(sim_time_s=sim,
                                 pose_xyz_xyzw=[pose['position'][k] for k in ('x','y','z')] +
                                                [pose['orientation'][k] for k in ('x','y','z','w')]))
    if len(exported) < 50:
        raise ValueError(f'Prefix too short: {len(exported)}')
    a.output.write_text(json.dumps(dict(schema='aic_sfp_privileged_teacher_prefix/v1',
        teacher=str(a.teacher),teacher_start_sim_s=first,duration_s=a.duration_s,
        reference_base_poses=exported,diagnostic_only=True,
        autonomous_success_claim=False),indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),teacher_start_sim_s=first,
                          duration_s=a.duration_s,reference_poses=len(exported))))


if __name__=='__main__':
    main()
