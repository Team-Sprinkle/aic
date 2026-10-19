#!/usr/bin/env python3
"""Compare early actor and same-scene teacher command vectors in base frame."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def summarize(values):
    if not values:
        return None
    arr = np.asarray(values, float)
    return dict(median=float(np.median(arr)), p95=float(np.quantile(arr, .95)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--progress', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve previous diagnosis')
    progress = json.loads(args.progress.read_text())
    reports = []
    for name in ('card0_port0', 'card2_port0', 'card4_port1'):
        entry = progress[name]
        run = Path(entry['output'])
        teacher = json.loads((Path(entry['teacher']) / 'scored_stream.json').read_text())
        control = teacher['streams']['/aic_controller/controller_state']
        control_times = [ns for ns, _ in control]
        commands = [r for r in teacher['commands'] if r['kind'] == 'pose']
        command_times = [r['sim_ns'] for r in commands]
        observed = [json.loads(line) for line in (run / 'runtime_trace.jsonl').open()]
        observed = [r for r in observed if r.get('event') == 'proposal' and r.get('targets')
                    and len(r.get('state', [])) == 32]
        rows = []
        for row in observed:
            stamp = int(row['sim_time'] * 1e9)
            ci = bisect_right(command_times, stamp) - 1
            si = bisect_right(control_times, stamp) - 1
            if ci < 0 or si < 0 or stamp - command_times[ci] > 300_000_000:
                continue
            command = commands[ci]['message']['pose']['position']
            teacher_delta = np.asarray([command[k] for k in ('x', 'y', 'z')], float)
            teacher_tcp = control[si][1]['tcp_pose']['orientation']
            teacher_quat = [teacher_tcp[k] for k in ('x', 'y', 'z', 'w')]
            actor_delta = np.asarray(row['targets'][0][:3], float)
            actor_quat = row['state'][3:7]
            teacher_world = Rotation.from_quat(teacher_quat).apply(teacher_delta)
            actor_world = Rotation.from_quat(actor_quat).apply(actor_delta)
            tn = float(np.linalg.norm(teacher_world))
            an = float(np.linalg.norm(actor_world))
            angle = (float(np.degrees(np.arccos(np.clip(
                np.dot(teacher_world, actor_world) / (tn * an), -1, 1))))
                if tn > 1e-4 and an > 1e-4 else None)
            rows.append(dict(sim_time_s=row['sim_time'], actor_target_mm=an * 1000,
                             teacher_target_mm=tn * 1000, angle_deg=angle,
                             actor_vector_base_mm=(actor_world * 1000).tolist(),
                             teacher_vector_base_mm=(teacher_world * 1000).tolist()))
        phases = {}
        for lo, hi in ((3, 5), (5, 10), (10, 15)):
            subset = [r for r in rows if lo <= r['sim_time_s'] < hi]
            phases[f'{lo}-{hi}s'] = dict(count=len(subset),
                direction_angle_deg=summarize([r['angle_deg'] for r in subset
                                               if r['angle_deg'] is not None]),
                actor_target_mm=summarize([r['actor_target_mm'] for r in subset]),
                teacher_target_mm=summarize([r['teacher_target_mm'] for r in subset]))
        reports.append(dict(name=name, run=str(run), teacher=entry['teacher'],
                            causal_teacher_command_join=True, phases=phases, rows=rows))
    args.output.write_text(json.dumps(dict(schema='aic_sfp_corrective_direction_diagnosis/v1',
        actor_geometry_input=False, posthoc_same_scene_teacher=True,
        final_scenes_opened=False, runs=reports), indent=2) + '\n')
    print(json.dumps([{k: v for k, v in r.items() if k in ('name', 'phases')}
                      for r in reports]))


if __name__ == '__main__':
    main()
