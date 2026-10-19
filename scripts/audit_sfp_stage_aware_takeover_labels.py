#!/usr/bin/env python3
"""Read-only stage-aware label audit on saved actor-to-teacher feature rows."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import yaml


def norm(x):
    return float(np.linalg.norm(x))


def cosine(a, b):
    den = norm(a) * norm(b)
    return float(np.dot(a, b) / den) if den > 1e-9 else None


def metadata(paths, scenes):
    found = {}
    for path in paths:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                scene = row.get('scene_sha256')
                if scene not in scenes or scene in found or not row.get('actor_prefix_context_only'):
                    continue
                found[scene] = {
                    'episode_id': row.get('episode_id'),
                    'source_scored_stream': row.get('source_scored_stream'),
                    'official_tier3': row.get('official_tier3'),
                    'selected_card': row.get('selected_card'),
                    'selected_port': row.get('selected_port'),
                }
                scored = Path(row.get('source_scored_stream', '')).parent / 'results/scoring.yaml'
                if scored.is_file():
                    report = yaml.safe_load(scored.read_text())
                    trials = [value for key, value in report.items() if key.startswith('trial_')]
                    if len(trials) == 1:
                        category = trials[0]['tier_2']['categories']
                        found[scene]['official_contact_score'] = category['contacts']['score']
                        found[scene]['official_insertion_force_score'] = category['insertion force']['score']
        if len(found) == len(scenes):
            break
    return found


def score_one(rows, meta):
    rows.sort(key=lambda row: row['time_s'])
    prefix = [i for i, row in enumerate(rows) if row.get('actor_prefix_context_only')]
    if not prefix:
        return {'usable': False, 'reason': 'no_actor_prefix'}
    boundary = prefix[-1]
    teacher = [row for row in rows[boundary + 1:] if row.get('corrective_teacher')]
    if len(teacher) < 5:
        return {'usable': False, 'reason': 'too_few_teacher_rows'}
    btime = rows[boundary]['time_s']
    t = np.array([row['time_s'] for row in teacher], float)
    xyz = np.array([row['state'][:3] for row in teacher], float)
    tier = float(meta.get('official_tier3') or 0)
    result = {
        'usable': False, 'reason': 'no_sustained_movement_window',
        'teacher_first_delay_s': float(t[0] - btime),
        'official_tier3': tier,
        'official_full_insertion': tier >= 75,
        'source_scored_stream': meta.get('source_scored_stream'),
        'episode_id': meta.get('episode_id'),
        'selected_card': meta.get('selected_card'),
        'selected_port': meta.get('selected_port'),
        'official_contact_score': meta.get('official_contact_score'),
        'official_insertion_force_score': meta.get('official_insertion_force_score'),
        'cable_point_force_measured': False,
    }
    if tier < 75:
        result['reason'] = 'not_official_full_insertion'
        return result
    for i in range(len(teacher) - 4):
        if t[i] - btime > 4:
            break
        candidates = [j for j in range(i + 1, len(teacher)) if .4 <= t[j] - t[i] <= .8]
        for j in candidates:
            delta = xyz[j] - xyz[i]
            net = norm(delta) * 1000
            path = float(np.linalg.norm(np.diff(xyz[i:j + 1], axis=0), axis=1).sum() * 1000)
            if net < 5 or path / net > 1.3:
                continue
            nexts = [k for k in range(j + 1, len(teacher)) if .3 <= t[k] - t[j] <= .6]
            if not nexts:
                continue
            k = nexts[0]
            follow = xyz[k] - xyz[j]
            cos = cosine(delta, follow)
            if cos is None or cos < .7:
                continue
            end = next((m for m in range(j, len(teacher))
                        if t[m] - t[i] <= 4 and norm(xyz[m] - xyz[i]) * 1000 >= 10), None)
            if end is None:
                continue
            route = xyz[i:end + 1] - xyz[i]
            route_net = norm(route[-1]) * 1000
            route_path = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum() * 1000)
            if route_path / route_net > 1.5:
                continue
            result.update(
                usable=True, reason='stage_label_found',
                onset_after_takeover_s=float(t[i] - btime),
                boundary_to_onset_mm=norm(xyz[i] - np.asarray(rows[boundary]['state'][:3], float)) * 1000,
                onset_tcp_quaternion_xyzw=teacher[i]['state'][3:7],
                first_window_s=float(t[j] - t[i]),
                first_window_net_mm=net, first_window_path_mm=path,
                initial_vs_follow_cosine=cos,
                route_duration_s=float(t[end] - t[i]),
                route_net_mm=route_net, route_path_mm=route_path,
                route_path_to_net_ratio=route_path / route_net,
                initial_direction_base=(delta / norm(delta)).tolist(),
                measured_route_polyline_base_mm=(route * 1000).tolist(),
                measured_route_times_s=(t[i:end + 1] - t[i]).tolist(),
                onset_cache_frame=int(teacher[i]['frame']),
                onset_cache_time_s=float(t[i]),
            )
            later = next((m for m in range(end + 1, len(teacher))
                          if t[m] - t[i] <= 5 and norm(xyz[m] - xyz[i]) * 1000 >= 20), None)
            result['independent_later_20mm_found'] = later is not None
            result['initial_vs_later_20mm_cosine'] = (
                cosine(delta, xyz[later] - xyz[i]) if later is not None else None)
            result['independent_later_20mm_delay_s'] = (
                float(t[later] - t[i]) if later is not None else None)
            return result
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--source', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Preserve earlier audit output')
    groups = defaultdict(list)
    with (args.cache / 'rows.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('actor_prefix_context_only') or row.get('corrective_teacher'):
                groups[(row['split'], int(row['episode']), row['scene'])].append(row)
    scenes = {key[2] for key in groups}
    provenance = metadata(args.source, scenes)
    records = []
    for (split, episode, scene), rows in sorted(groups.items()):
        rec = {'split': split, 'episode': episode, 'scene': scene,
               'provenance_joined': scene in provenance}
        rec.update(score_one(rows, provenance.get(scene, {})))
        records.append(rec)
    summary = {
        'schema': 'aic_sfp_stage_aware_takeover_label_audit/v1',
        'cache': str(args.cache), 'source_files': [str(x) for x in args.source],
        'train_groups': sum(x['split'] == 'train' for x in records),
        'development_groups': sum(x['split'] == 'validation' for x in records),
        'train_usable': sum(x['split'] == 'train' and x['usable'] for x in records),
        'development_usable': sum(x['split'] == 'validation' and x['usable'] for x in records),
        'development_direction_pass': sum(x['split'] == 'validation' and x['usable']
                                          and x['initial_vs_later_20mm_cosine'] is not None
                                          and x['initial_vs_later_20mm_cosine'] >= .7
                                          for x in records),
        'all_provenance_joined': all(x['provenance_joined'] for x in records),
        'cable_point_force_verified': False,
        'label_gate_passed': False,
        'policy_training_allowed': False,
        'records': records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}))


if __name__ == '__main__':
    main()
