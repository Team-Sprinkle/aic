#!/usr/bin/env python3
"""Audit request-expanded SFP RGB labels before fitting a camera check."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random

from PIL import Image


CAMERAS = ('center', 'left', 'right')


def read(path):
    return [json.loads(line) for line in path.open()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset', type=Path)
    p.add_argument('--image-audit-count', type=int, default=90)
    p.add_argument('--max-age-ms', type=float, default=10.0)
    p.add_argument('--max-spread-ms', type=float, default=10.0)
    a = p.parse_args()
    rows = read(a.dataset / 'all_requests.jsonl')
    blind = read(a.dataset / 'blind_review_queue.jsonl')
    sealed = read(a.dataset / 'pilot_sealed_geometry.jsonl')
    errors = []
    roles = defaultdict(set)
    frames = defaultdict(list)
    ids = set()
    for r in rows:
        sid = r['sample_id']
        if sid in ids:
            errors.append(f'duplicate sample {sid}')
        ids.add(sid)
        roles[r['source_role']].add(r['scene_group'])
        frames[(r['source_role'], r['episode'], r['time_s'])].append(r)
        if r['geometry_runtime_input'] or r['visual_identifiability_label'] is not None or r['safe_pause_label'] is not None:
            errors.append(f'geometry/visibility contract violated {sid}')
        if r['request_physically_present'] is False and any(v['projected_in_frame'] for v in r['projection_training_label_only'].values()):
            errors.append(f'absent port projected {sid}')
        for cam in CAMERAS:
            q = r['projection_training_label_only'][cam]
            if q['projected_in_frame']:
                x, y = q['pixel']
                if not (0 <= x < 1152 and 0 <= y < 1024 and q['depth_m'] > 0 and math.isfinite(x+y)):
                    errors.append(f'invalid projection {sid} {cam}')
            elif q['pixel'] is not None or q['depth_m'] is not None:
                errors.append(f'outside frame retained coordinates {sid} {cam}')
    for key, rs in frames.items():
        if len(rs) != 10 or len({r['requested'] for r in rs}) != 10:
            errors.append(f'wrong request expansion {key}: {len(rs)}')
    for role in ('development', 'diagnostic', 'actor_diagnostic'):
        if roles['train'] & roles[role]:
            errors.append(f'train reset leakage into {role}')
    if len(blind) != len(sealed) or {r['sample_id'] for r in blind} != {r['sample_id'] for r in sealed}:
        errors.append('blind/sealed pilot mismatch')
    if any('projection_training_label_only' in r or 'distance_mm_training_label_only' in r for r in blind):
        errors.append('blind queue leaks geometry')
    all_ages_ms = []
    all_spreads_ms = []
    for key, rs in frames.items():
        r = rs[0]
        try:
            stamps = [int(Path(r['images'][cam]).stem.split('_')[-2]) for cam in CAMERAS]
            all_ages_ms.append(max(abs(r['time_s'] * 1e9 - stamp) for stamp in stamps) / 1e6)
            all_spreads_ms.append((max(stamps) - min(stamps)) / 1e6)
        except (ValueError, IndexError):
            errors.append(f'bad camera simulation timestamp {key}')
    rng = random.Random(20261005)
    sampled = rng.sample(list(frames), min(a.image_audit_count, len(frames)))
    image_ages_ms = []
    camera_spreads_ms = []
    image_failures = []
    for key in sampled:
        r = frames[key][0]
        stamps = []
        for cam in CAMERAS:
            path = Path(r['images'][cam])
            try:
                with Image.open(path) as image:
                    if image.size != (1152, 1024) or image.mode != 'RGB':
                        image_failures.append(f'{path}: {image.size} {image.mode}')
                    image.verify()
                sim_ns = int(path.stem.split('_')[-2])
                stamps.append(sim_ns)
                image_ages_ms.append(abs(r['time_s'] * 1e9 - sim_ns) / 1e6)
            except (OSError, ValueError, IndexError) as e:
                image_failures.append(f'{path}: {e}')
        if len(stamps) == 3:
            camera_spreads_ms.append((max(stamps) - min(stamps)) / 1e6)
    errors.extend(image_failures[:10])
    report = {
        'request_rows': len(rows), 'unique_triplets': len(frames),
        'groups_by_role': {k: len(v) for k, v in roles.items()},
        'sampled_triplets': len(sampled),
        'sampled_rgb_files': len(sampled) * 3,
        'all_triplet_max_state_image_age_ms': max(all_ages_ms, default=None),
        'all_triplet_max_camera_sim_spread_ms': max(all_spreads_ms, default=None),
        'all_triplet_age_over_limit_count': sum(v > a.max_age_ms for v in all_ages_ms),
        'all_triplet_spread_over_limit_count': sum(v > a.max_spread_ms for v in all_spreads_ms),
        'max_state_image_age_ms': max(image_ages_ms, default=None),
        'max_camera_sim_spread_ms': max(camera_spreads_ms, default=None),
        'image_age_over_100ms_count': sum(v > 100 for v in image_ages_ms),
        'camera_spread_over_50ms_count': sum(v > 50 for v in camera_spreads_ms),
        'image_failures': image_failures,
        'errors': errors[:30], 'pass': not errors and not any(v > a.max_age_ms for v in all_ages_ms) and not any(v > a.max_spread_ms for v in all_spreads_ms),
    }
    (a.dataset / 'structural_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
