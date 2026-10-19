#!/usr/bin/env python3
"""Compare measured autonomous TCP routes with same-scene teacher routes posthoc."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def measured_teacher(path):
    stream = json.loads((path / 'scored_stream.json').read_text())['streams']['/aic_controller/controller_state']
    t, xyz = [], []
    for stamp, row in stream:
        p = row['tcp_pose']['position']
        t.append(stamp / 1e9)
        xyz.append([p[k] for k in ('x', 'y', 'z')])
    return np.asarray(t), np.asarray(xyz)


def measured_actor(path):
    rows = [json.loads(line) for line in (path / 'runtime_trace.jsonl').open()]
    rows = [r for r in rows if r.get('event') == 'proposal' and len(r.get('state', [])) == 32]
    return np.asarray([r['sim_time'] for r in rows]), np.asarray([r['state'][:3] for r in rows])


def actor_command_norms(path):
    rows = [json.loads(line) for line in (path / 'runtime_trace.jsonl').open()]
    rows = [r for r in rows if r.get('event') == 'proposal' and r.get('targets')]
    return np.asarray([r['sim_time'] for r in rows]), np.asarray([
        np.linalg.norm(np.asarray(r['targets'][0][:3]) * 1000) for r in rows])


def teacher_command_norms(path):
    rows = json.loads((path / 'scored_stream.json').read_text())['commands']
    rows = [r for r in rows if r['kind'] == 'pose']
    return np.asarray([r['sim_ns'] / 1e9 for r in rows]), np.asarray([
        np.linalg.norm([r['message']['pose']['position'][k] for k in ('x', 'y', 'z')]) * 1000
        for r in rows])


def phase_norms(times, norms):
    return {f'{lo}-{hi}s_median_mm': float(np.median(norms[(times >= lo) & (times < hi)]))
            for lo, hi in ((3, 5), (5, 10), (10, 15))
            if np.any((times >= lo) & (times < hi))}


def interp_xyz(t, x, query):
    return np.column_stack([np.interp(query, t, x[:, k]) for k in range(3)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--progress', type=Path, required=True)
    p.add_argument('--candidate-progress', type=Path,
                   help='Optional additional candidate on the same three development scenes')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve previous diagnosis')
    a.output.mkdir(parents=True)
    progress = json.loads(a.progress.read_text())
    candidate = json.loads(a.candidate_progress.read_text()) if a.candidate_progress else None
    results = []
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex='col')
    for i, (card, port) in enumerate(((0, 0), (2, 0), (4, 1))):
        variants = ('baseline', 'ordinary', 'zero_previous') + (('corrective',) if candidate else ())
        for variant in variants:
            run = (candidate[f'card{card}_port{port}'] if variant == 'corrective'
                   else progress[f'card{card}_port{port}_{variant}'])
            ta, xa = measured_actor(Path(run['output']))
            tt, xt = measured_teacher(Path(run['teacher']))
            lo = max(ta.min(), tt.min())
            hi = min(ta.max(), tt.max(), 18.0)
            mask = (ta >= lo) & (ta <= hi)
            t = ta[mask]
            actor = xa[mask]
            teacher = interp_xyz(tt, xt, t)
            err = np.linalg.norm(actor - teacher, axis=1) * 1000
            sampled = {}
            for second in (3, 5, 7, 9, 11, 13, 15, 17):
                if lo <= second <= hi:
                    aa = interp_xyz(ta, xa, [second])[0]
                    tr = interp_xyz(tt, xt, [second])[0]
                    sampled[str(second)] = dict(tcp_error_mm=float(np.linalg.norm(aa-tr)*1000),
                                                actor_xyz_m=aa.tolist(), teacher_xyz_m=tr.tolist(),
                                                delta_xyz_mm=((aa-tr)*1000).tolist())
            first = {}
            for threshold in (5, 10, 25, 50):
                hits = np.flatnonzero(err > threshold)
                first[f'gt_{threshold}mm_s'] = float(t[hits[0]]) if len(hits) else None
            results.append(dict(card=card, port=port, variant=variant,
                                matched_time_range_s=[float(lo), float(hi)],
                                first_divergence=first, by_second=sampled,
                                max_error_mm=float(err.max()),
                                median_error_mm=float(np.median(err)),
                                actor_command_norms=phase_norms(*actor_command_norms(Path(run['output']))),
                                teacher_command_norms=phase_norms(*teacher_command_norms(Path(run['teacher'])))))
            axes[i, 0].plot(t, err, label=variant)
            axes[i, 1].plot(actor[:, 1], actor[:, 2], label=variant)
        axes[i, 0].set_ylabel(f'card {card} port {port}\nTCP distance (mm)')
        axes[i, 0].grid(alpha=.25)
        axes[i, 1].plot(xt[:, 1], xt[:, 2], 'k--', label='teacher')
        axes[i, 1].set_ylabel('TCP z (m)')
        axes[i, 1].grid(alpha=.25)
        axes[i, 0].legend()
        axes[i, 1].legend()
    axes[2, 0].set_xlabel('Simulation time (s)')
    axes[2, 1].set_xlabel('TCP y (m)')
    fig.tight_layout()
    fig.savefig(a.output / 'route_divergence.png', dpi=150)
    (a.output / 'route_divergence.json').write_text(json.dumps(dict(
        schema='aic_sfp_balanced_route_divergence/v1',
        same_scene_measured_tcp_only=True, geometry_actor_input=False,
        final_scenes_opened=False, runs=results), indent=2) + '\n')
    print(json.dumps([{k: r[k] for k in ('card', 'port', 'variant', 'first_divergence')}
                      for r in results]))


if __name__ == '__main__':
    main()
