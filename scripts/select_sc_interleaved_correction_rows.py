#!/usr/bin/env python3
"""Select actor pulse and immediate teacher-correction labels from scored SC runs.

The source must already have passed full-episode scoring, force, grasp, and
causal-label admission. This script does not turn failed episodes into BC data.
"""

import argparse
import bisect
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--after-pulse-seconds', type=float, default=.5)
    p.add_argument('--expected-episodes', type=int)
    a = p.parse_args()
    if not 0 < a.after_pulse_seconds <= 2:
        raise ValueError('Correction window must be in (0, 2] seconds')
    frames_cache = {}
    rows_by_episode = defaultdict(lambda: {'all': 0, 'actor': 0, 'after': 0})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.labels.open() as source, a.output.open('w') as dest:
        for line in source:
            row = json.loads(line)
            if (row['split'] != 'train' or not row['corrective_episode'] or
                    not row['teacher_rescued_direct_actor_takeover']):
                raise ValueError('Input contains nonadmitted or nonrescue label')
            folder = Path(row['images']['center']).parent
            if folder not in frames_cache:
                frames = [json.loads(s) for s in (folder / 'frames.jsonl').open()]
                nominal = {int(f['frame']): float(f['nominal_expert_time']) for f in frames}
                active = sorted(float(f['nominal_expert_time']) for f in frames
                                if f.get('student_active'))
                if not active:
                    raise ValueError(f'No actor pulse in {folder}')
                frames_cache[folder] = (nominal, active)
            nominal, active = frames_cache[folder]
            t = nominal[int(row['frame'])]
            previous = bisect.bisect_right(active, t + 1e-8) - 1
            is_active = bool(row['corrective_student_active_anchor'])
            after = (not is_active and previous >= 0 and
                     0 <= t - active[previous] <= a.after_pulse_seconds + 1e-8)
            entry = rows_by_episode[str(row['episode_index'])]
            entry['all'] += 1
            if is_active or after:
                dest.write(line)
                entry['actor' if is_active else 'after'] += 1
    report = {'schema': 'sc_interleaved_correction_row_selection/v1',
              'source': str(a.labels), 'source_sha256': digest(a.labels),
              'output': str(a.output), 'output_sha256': digest(a.output),
              'after_pulse_seconds': a.after_pulse_seconds,
              'complete_success_admission_required': True,
              'episodes': dict(rows_by_episode),
              'total_actor_rows': sum(x['actor'] for x in rows_by_episode.values()),
              'total_after_rows': sum(x['after'] for x in rows_by_episode.values())}
    if ((a.expected_episodes is not None and len(rows_by_episode) != a.expected_episodes) or
            not report['total_actor_rows'] or not report['total_after_rows']):
        raise ValueError('Unexpected selected scene or phase counts')
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'episodes': len(rows_by_episode),
                      'actor_rows': report['total_actor_rows'],
                      'after_rows': report['total_after_rows']}))


if __name__ == '__main__':
    main()
