#!/usr/bin/env python3
"""Add measured-path-spaced native grasp views to an immutable quiet-view set.

Selection uses recorded TCP travel only, never plug geometry or predictions.
Each source stays in its original episode/reset split. Raw simulator TF is a
training label only. Existing label rows are reused without mutation.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from mcap.reader import make_reader
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from prepare_sfp_native_grasp_labels import label, matrix, stored_matrix


def path_indices(rows, count):
    if not rows:
        return []
    positions = np.asarray([r['state'][:3] for r in rows])
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    if distance[-1] < .001:
        return [0]
    return sorted(set(np.searchsorted(distance, np.linspace(0, distance[-1], count)).tolist()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('commands', 'sources', 'quiet-labels', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--path-samples', type=int, default=8)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=a.resume)
    original = json.loads((a.quiet_labels / 'summary.json').read_text())
    quiet = {r['source']: r for r in original['episodes']}
    commands = json.loads((a.commands / 'summary.json').read_text())
    sources = {r['source_stream']: r for r in map(json.loads, a.sources.open()) if r['family'] == 'sfp'}
    optics = {c: np.asarray(v) for c, v in original['tcp_to_optical'].items()}
    progress = a.output / 'progress.jsonl'
    done = list(map(json.loads, progress.open())) if a.resume and progress.exists() else []
    seen = {r['source'] for r in done}
    typ = get_message('tf2_msgs/msg/TFMessage')
    for e in commands['episodes']:
        if e['family'] != 'sfp' or e['source'] in seen:
            continue
        rows = list(map(json.loads, (a.commands / e['rows_file']).open()))
        retained = list(map(json.loads, (a.quiet_labels / quiet[e['source']]['rows_file']).open()))
        image_keys = {tuple(r['images'][c] for c in ('center','left','right')) for r in retained}
        selected = [rows[i] for i in path_indices(rows, a.path_samples)
                    if tuple(rows[i]['images'][c] for c in ('center','left','right')) not in image_keys]
        d = json.loads(Path(e['source']).read_text())
        wb = stored_matrix(d['edges']['world|tabletop'][0][1]) @ stored_matrix(d['edges']['tabletop|base_link'][0][1])
        bag = next((Path(sources[e['source']]['source_audit']) / 'results').glob('bag_*/*.mcap'))
        rejected = Counter()
        with bag.open('rb') as f:
            reader = make_reader(f)
            wa = None
            for _, _, message in reader.iter_messages(topics=['/tf_static']):
                for tf in deserialize_message(message.data, typ).transforms:
                    if (tf.header.frame_id,tf.child_frame_id) == ('world','aic_world'):
                        wa = matrix(tf.transform)
            if wa is None:
                raise ValueError('Missing explicit world/aic_world calibration')
            for row in selected:
                try:
                    r = label(reader, typ, d, row, wa, wb, optics)
                    r['sample_selection'] = 'measured TCP cumulative travel quantiles; no geometry/model-output selection'
                    retained.append(r)
                except (ValueError, KeyError) as exc:
                    rejected[str(exc)] += 1
        retained.sort(key=lambda r: r['time_s'])
        filename = quiet[e['source']]['rows_file']
        (a.output / filename).write_text(''.join(json.dumps(r)+'\n' for r in retained))
        result = dict(source=e['source'], episode=e['episode'], split=e['split'],
                      rows=len(retained), rows_file=filename, path_candidates=len(selected), rejected=dict(rejected))
        done.append(result)
        with progress.open('a') as f:
            f.write(json.dumps(result)+'\n')
        print(json.dumps(dict(completed=len(done), planned=len(quiet), last=result)), flush=True)
    report = dict(original, episodes=done, source_episodes=len(done), rows=sum(r['rows'] for r in done),
                  original_quiet_labels=str(a.quiet_labels), path_samples=a.path_samples,
                  selection='quiet originals plus measured TCP path quantiles', original_modified=False)
    (a.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
