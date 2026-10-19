#!/usr/bin/env python3
"""Make all-request SFP camera-check labels and a blind RGB review queue.

Geometry creates supervision only. This script never calls projection a visual
visibility label, never labels safe pause from distance alone, and never lets
development/diagnostic episodes enter the training split.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


CAMERAS = ('center', 'left', 'right')
REQUESTS = tuple(f'sfp_card_{card}_port_{port}' for card in range(5) for port in range(2))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def rows_from(path):
    source = path / 'rows.jsonl'
    if not source.is_file():
        raise FileNotFoundError(source)
    return [json.loads(line) for line in source.open()], source


def image_age_ms(row):
    stamps = [int(Path(row['images'][cam]).stem.split('_')[-2]) for cam in CAMERAS]
    return (max(abs(float(row['time_s']) * 1e9 - stamp) for stamp in stamps) / 1e6,
            (max(stamps) - min(stamps)) / 1e6)


def camera_projection(target, cam):
    uv = target['cameras'][cam]
    if len(uv) < 4:
        raise ValueError('Expected pixel, depth, in-frame flag')
    inside = bool(uv[3])
    if inside and not (0 <= uv[0] < 1152 and 0 <= uv[1] < 1024 and uv[2] > 0):
        raise ValueError(f'Inconsistent projected pixel: {uv}')
    return {'pixel': [float(uv[0]), float(uv[1])] if inside else None,
            'depth_m': float(uv[2]) if inside else None,
            'projected_in_frame': inside}


def build_one(row, source_role, request):
    if row['family'] != 'sfp' or row['image_kind'] != 'native_1152x1024_rgb_jpeg':
        raise ValueError(f'Expected native SFP RGB: {row["episode"]}')
    if row['selected_identity'] not in REQUESTS:
        raise ValueError('Invalid recorded task request')
    images = {cam: str(Path(row['images'][cam]).resolve()) for cam in CAMERAS}
    if any(not Path(path).is_file() for path in images.values()):
        raise FileNotFoundError(images)
    targets = {t['identity']: t for t in row['targets']}
    if len(targets) != len(row['targets']):
        raise ValueError('Duplicate port identity')
    target = targets.get(request)
    projected = ({cam: camera_projection(target, cam) for cam in CAMERAS}
                 if target else {cam: {'pixel': None, 'depth_m': None,
                                      'projected_in_frame': False} for cam in CAMERAS})
    distance = (math.dist(row['state'][:3], target['position_base_m']) * 1000
                if target else None)
    key = '|'.join((row['scene_group'], row['episode'],
                    f'{float(row["time_s"]):.6f}', request))
    sid = hashlib.sha256(key.encode()).hexdigest()[:20]
    return {'sample_id': sid, 'source_role': source_role,
            'episode': row['episode'], 'scene_group': row['scene_group'],
            'time_s': float(row['time_s']), 'images': images,
            'recorded_request': row['selected_identity'], 'requested': request,
            'card_count': int(row['card_count']),
            'request_physically_present': target is not None,
            'projection_training_label_only': projected,
            'distance_mm_training_label_only': distance,
            'visual_identifiability_label': None,
            'safe_pause_label': None,
            'geometry_runtime_input': False}


def category(row):
    if not row['request_physically_present']:
        return 'physically_absent'
    visible = sum(row['projection_training_label_only'][c]['projected_in_frame']
                  for c in CAMERAS)
    if visible < 3:
        return 'projected_partial_or_outside'
    if row['distance_mm_training_label_only'] is not None and row['distance_mm_training_label_only'] < 100:
        return 'projected_all_three_near100'
    return 'projected_all_three_far100'


def pick_queue(rows, limits):
    rng = random.Random(20261005)
    by_role = defaultdict(list)
    for row in rows:
        by_role[row['source_role']].append(row)
    selected = []
    for role, limit in limits.items():
        options = by_role[role][:]
        rng.shuffle(options)
        chosen = []
        per_episode = Counter()
        per_request = Counter()
        per_category = Counter()
        per_count = Counter()
        used_frame = set()
        cap = 3 if role == 'train' else 2
        while len(chosen) < limit:
            admissible = [r for r in options if per_episode[r['episode']] < cap and
                          (r['episode'], r['time_s']) not in used_frame]
            if not admissible:
                break
            # Greedy coverage across request, case type, card count and reset.
            def score(r):
                return (per_request[r['requested']], per_category[category(r)],
                        per_count[r['card_count']], per_episode[r['episode']])
            item = min(admissible, key=score)
            chosen.append(item)
            per_episode[item['episode']] += 1
            per_request[item['requested']] += 1
            per_category[category(item)] += 1
            per_count[item['card_count']] += 1
            used_frame.add((item['episode'], item['time_s']))
        if len(chosen) < limit:
            raise ValueError(f'Cannot fill blind queue {role}: {len(chosen)}/{limit}')
        selected.extend(chosen)
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--extra-train', type=Path, action='append', default=[],
                   help='Additional train-only native projection directory; repeatable')
    p.add_argument('--development', type=Path, required=True)
    p.add_argument('--diagnostic', type=Path, required=True)
    p.add_argument('--actor', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-state-image-age-ms', type=float, default=10.0)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve previous dataset')
    sources = {'train': [a.train, *a.extra_train],
               'development': [a.development],
               'diagnostic': [a.diagnostic],
               'actor_diagnostic': [a.actor]}
    rows = []
    source_hashes = {}
    groups = defaultdict(set)
    seen = set()
    raw_counts = Counter()
    excluded_age = Counter()
    excluded_camera_spread = Counter()
    for role, paths in sources.items():
      source_hashes[role] = {}
      for path in paths:
        originals, source = rows_from(path)
        source_hashes[role][str(source)] = digest(source)
        for original in originals:
            # The canonical 66/10 labels intentionally share one source file.
            if role == 'train' and original['split'] != 'train':
                continue
            if role == 'development' and original['split'] != 'validation':
                continue
            age_ms, spread_ms = image_age_ms(original)
            if spread_ms > a.max_state_image_age_ms:
                excluded_camera_spread[role] += 1
                continue
            if age_ms > a.max_state_image_age_ms:
                excluded_age[role] += 1
                continue
            groups[role].add(original['scene_group'])
            raw_counts[role] += 1
            for request in REQUESTS:
                item = build_one(original, role, request)
                if item['sample_id'] in seen:
                    raise ValueError(f'Duplicate sample: {item["sample_id"]}')
                seen.add(item['sample_id'])
                rows.append(item)
    if groups['train'] & (groups['development'] | groups['diagnostic'] | groups['actor_diagnostic']):
        raise ValueError('Training reset leaked into evaluation/diagnostic source')
    queue = pick_queue(rows, {'train': 120, 'development': 20,
                              'diagnostic': 20, 'actor_diagnostic': 6})
    a.output.mkdir(parents=True)
    with (a.output / 'all_requests.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, separators=(',', ':')) + '\n')
    with (a.output / 'pilot_sealed_geometry.jsonl').open('w') as f:
        for row in queue:
            f.write(json.dumps(row, separators=(',', ':')) + '\n')
    with (a.output / 'blind_review_queue.jsonl').open('w') as f:
        for row in queue:
            blind = {k: row[k] for k in ('sample_id', 'source_role', 'episode',
                       'scene_group', 'time_s', 'images', 'requested', 'card_count')}
            blind['reviewed_visibility_by_camera'] = None
            blind['requested_slot_identifiable'] = None
            blind['reviewer_note'] = None
            f.write(json.dumps(blind, separators=(',', ':')) + '\n')
    summary = {'schema': 'sfp_camera_check_all_request_dataset/v1',
               'source_hashes': source_hashes,
               'source_triplets': dict(raw_counts),
               'excluded_stale_triplets': dict(excluded_age),
               'excluded_unsynchronized_triplets': dict(excluded_camera_spread),
               'max_state_image_age_ms': a.max_state_image_age_ms,
               'source_reset_groups': {k: len(v) for k, v in groups.items()},
               'request_rows': dict(Counter(r['source_role'] for r in rows)),
               'training_geometry_categories': dict(Counter(category(r) for r in rows
                                                            if r['source_role'] == 'train')),
               'pilot_rows': len(queue),
               'pilot_by_role': dict(Counter(r['source_role'] for r in queue)),
               'pilot_by_category_sealed': dict(Counter(category(r) for r in queue)),
               'true_visual_visibility_labels': 0,
               'safe_pause_labels': 0,
               'projection_is_not_visibility': True,
               'no_geometry_in_blind_review_queue': True,
               'final_scenes_opened': False}
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('source_triplets', 'request_rows',
          'training_geometry_categories', 'pilot_by_role')}))


if __name__ == '__main__':
    main()
