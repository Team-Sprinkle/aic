#!/usr/bin/env python3
"""Join unmarked wide-view RGB frames to held poses, before truth scoring."""
import argparse
import csv
import json
from pathlib import Path


CASES = (('card1_port0', 'card1_port0_v2'), ('card2_port0', 'card2_port0'),
         ('card3_port0', 'card3_port0'), ('card4_port1', 'card4_port1'))
CAMERAS = ('center', 'left', 'right')
STAGES = ('baseline_view', 'wide_view_hold')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Preserve existing raw manifest')
    plan = {row['name']: row for row in json.loads(args.source_plan.read_text())['scenes']}
    rows = []; episodes = []
    for name, folder in CASES:
        audit = args.root / folder
        if not (audit / 'wide_view_events.json').exists():
            episodes.append({'episode': name, 'complete': False, 'reason': 'no_event_record'})
            continue
        data = json.loads((audit / 'wide_view_events.json').read_text())
        events = {row['stage']: row for row in data['events'] if row['stage'] in STAGES}
        if set(events) != set(STAGES):
            episodes.append({'episode': name, 'complete': False,
                             'reason': 'missing_hold', 'stages': sorted(events)})
            continue
        frames = list(csv.DictReader((audit / 'frame_manifest.csv').open()))
        by_cam = {c: [row for row in frames if row['camera'] == c] for c in CAMERAS}
        baseline = events['baseline_view']['tcp_base_m']
        for stage in STAGES:
            event = events[stage]
            stamp = round(event['sim_time_s'] * 1e9)
            images = {}; gaps = {}
            for cam in CAMERAS:
                frame = min(by_cam[cam], key=lambda row: abs(int(row['sim_ns']) - stamp))
                gap = abs(int(frame['sim_ns']) - stamp) / 1e6
                if gap > 75: raise ValueError(f'{name} {stage} {cam}: stale image {gap:.1f} ms')
                path = audit / 'frames' / frame['file']
                if not path.is_file(): raise FileNotFoundError(path)
                images[cam] = str(path)
                gaps[cam] = gap
            row = {'episode': name, 'stage': stage,
                   'scene_group': plan[name]['reset_sha256'],
                   'requested': f"sfp_card_{plan[name]['card']}_port_{plan[name]['port']}",
                   'installed_card_count': plan[name]['card_count'],
                   'sim_time_s': event['sim_time_s'], 'images': images,
                   'camera_gap_ms': gaps, 'tcp_base_m': event['tcp_base_m'],
                   'tcp_offset_from_baseline_mm': [(event['tcp_base_m'][j] - baseline[j]) * 1000
                                                  for j in range(3)],
                   'force_n': event['force_n'], 'contact_pairs': event['contact_pairs'],
                   'simulator_port_pixels_in_manifest': False}
            rows.append(row)
        episodes.append({'episode': name, 'complete': True,
                         'unsafe_abort': any(row['stage'] == 'unsafe_abort' for row in data['events']),
                         'raw_images': 6, 'max_camera_gap_ms': max(max(r['camera_gap_ms'].values())
                                                                    for r in rows if r['episode'] == name)})
    args.output.mkdir(parents=True)
    with (args.output / 'rows.jsonl').open('w') as stream:
        for row in rows: stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    summary = {'schema': 'sfp_wide_unmarked_view_join/v1', 'rows': len(rows),
               'episodes': episodes, 'review_before_truth': True,
               'ground_truth_port_position_input': False, 'final_scenes_opened': False}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
