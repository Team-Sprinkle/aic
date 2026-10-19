#!/usr/bin/env python3
"""Repair SC future targets from untouched command clocks; preserve other labels."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def pose_matrix(p):
    m = np.eye(4)
    m[:3, 3] = p[:3]
    m[:3, :3] = Rotation.from_quat(p[3:]).as_matrix()
    return m


def message_matrix(p):
    return pose_matrix([p['position'][k] for k in 'xyz']+[p['orientation'][k] for k in 'xyzw'])


def pose9(m):
    return np.r_[m[:3, 3], m[:3, :2].T.reshape(-1)].astype(np.float32)


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--raw-transport', type=Path, required=True)
    p.add_argument('--streams', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--slot-s', type=float, default=.1)
    p.add_argument('--max-command-age-s', type=float, default=.15)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output exists; keep data derivatives immutable.')
    rows = [json.loads(l) for l in (a.cache/'rows.jsonl').open()]
    targets = np.array(np.load(a.cache/'targets12.npy'), copy=True)
    valid = np.array(np.load(a.cache/'valid12.npy'), copy=True)
    raw = defaultdict(list)
    for line in a.raw_transport.open():
        r = json.loads(line)
        raw[r['episode_index']].append(r)
    cached = defaultdict(list)
    for i, row in enumerate(rows):
        if row['family'] == 'sc':
            cached[row['episode']].append(i)
    reasons = Counter()
    audits = []
    slot_provenance = []
    for episode, ids in cached.items():
        source = raw[episode]
        if not source:
            raise ValueError(f'No raw episode {episode}')
        name = source[0]['episode_id']
        stream = json.loads((a.streams/(name+'.json')).read_text())
        times = np.asarray([c['sim_ns']/1e9 for c in stream['commands']])
        if np.any(np.diff(times) < 0):
            raise ValueError(f'Clock decreased in {name}')
        events = [json.loads(l) for l in (Path(stream['audit'])/'runtime_trace.jsonl').open()]
        boundary = min(e['time_sec'] for e in events if e['event']=='local_preinsert_align_started')
        # FK-derived targets already materialized from the original commands
        # cover joint-mode commands without assuming measured pose is a target.
        absolute = {}
        for r in source:
            for idx, pose in zip(r['command_indices'], r['teacher_tcp_target_poses_base_frame']):
                m = pose_matrix(pose)
                if idx in absolute and not np.allclose(absolute[idx], m, atol=1e-8):
                    raise ValueError(f'Conflicting command {name}/{idx}')
                absolute[idx] = m
        for idx, c in enumerate(stream['commands']):
            if c['kind'] == 'pose' and c['message']['header']['frame_id']=='base_link':
                absolute[idx] = message_matrix(c['message']['pose'])
        old_full = int(valid[ids].all(1).sum())
        examples = []
        for i in ids:
            row = rows[i]
            anchor_row = next((r for r in source if r['frame']==row['frame']), None)
            if anchor_row is None:
                raise ValueError(f'Missing source anchor {name}/{row["frame"]}')
            if row['scene'] != anchor_row['scene_sha256'] or row['split'] != anchor_row['split']:
                raise ValueError(f'Split/reset mismatch in {name}')
            anchor = pose_matrix(row['state'][:7])
            inverse = np.linalg.inv(anchor)
            t0 = anchor_row['sim_time']
            targets[i] = 0
            valid[i] = False
            indices = []
            ages = []
            for j in range(12):
                wanted = t0+j*a.slot_s
                idx = bisect_right(times, wanted+1e-10)-1
                age = wanted-times[idx] if idx >= 0 else None
                indices.append(idx)
                ages.append(age)
                if wanted >= boundary or wanted > times[-1]+1e-10:
                    reasons['ownership_or_episode_end'] += 1
                    continue
                if idx < 0 or age > a.max_command_age_s+1e-10:
                    reasons['missing_or_stale_command'] += 1
                    continue
                if idx not in absolute:
                    reasons['unmaterialized_joint_target'] += 1
                    continue
                targets[i,j] = pose9(inverse@absolute[idx])
                valid[i,j] = True
            item = dict(cache_index=i, episode=name, frame=row['frame'], split=row['split'],
                        anchor_sim_s=t0, phase_boundary_s=boundary,
                        command_indices=indices, command_age_s=ages, valid=valid[i].tolist(),
                        handoff_label=row['handoff'])
            slot_provenance.append(item)
            if len(examples) < 2 or row['handoff'] or row['stage']==5 and len(examples)<4:
                examples.append(item)
        audits.append(dict(episode=name, split=source[0]['split'], rows=len(ids),
                           old_full12=old_full, repaired_full12=int(valid[ids].all(1).sum()),
                           rows_with_any_action=int(valid[ids].any(1).sum()),
                           stage5_rows=sum(rows[i]['stage']==5 for i in ids),
                           stage5_with_action=sum(rows[i]['stage']==5 and bool(valid[i].any()) for i in ids),
                           handoff_examples=sum(bool(rows[i]['handoff']) for i in ids),
                           boundary_sim_s=boundary, examples=examples))
    a.output.mkdir(parents=True)
    np.save(a.output/'targets12.npy', targets)
    np.save(a.output/'valid12.npy', valid)
    np.save(a.output/'handoff.npy', np.asarray([r['handoff'] for r in rows], np.float32))
    with (a.output/'slot_provenance.jsonl').open('w') as out:
        for item in slot_provenance:
            out.write(json.dumps(item)+'\n')
    manifest = dict(schema='aic_timestamp_route_targets/v1', rows=len(rows), slot_s=a.slot_s,
                    cache=str(a.cache), max_command_age_s=a.max_command_age_s,
                    source_sha256={str(q):digest(q) for q in [a.raw_transport,a.cache/'rows.jsonl']},
                    sc_episodes=len(audits), episodes=audits, invalid_reason=dict(reasons),
                    source_episodes_mutated=False, sfp_labels_unchanged=True,
                    terminal_mask=True, stop_supervision_independent_of_future_validity=True,
                    final_scenes_opened=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(sc_episodes=len(audits), old_full12=sum(r['old_full12'] for r in audits),
                          repaired_full12=sum(r['repaired_full12'] for r in audits),
                          stage5_with_action=sum(r['stage5_with_action'] for r in audits),
                          handoff_examples=sum(r['handoff_examples'] for r in audits), reasons=dict(reasons))), flush=True)


if __name__ == '__main__':
    main()
