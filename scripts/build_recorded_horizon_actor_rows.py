#!/usr/bin/env python3
"""Copy recorded teacher targets into causal4/8/12-step matched derivatives.

Extend from independently recorded, overlapping command labels in the SAME
episode. Never stretch a four-step chunk, bridge absent labels, or substitute
future measured TCP poses. All images/observations remain the anchor's inputs.
"""
import argparse
from collections import defaultdict, Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from shared_port_tcp_geometry import matrix_from_pose9, matrix_from_xyz_xyzw


def pose9(m):
    return [*m[:3, 3], *m[:3, :2].T.reshape(-1)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve the old derivative')
    a.output.mkdir(parents=True)
    rows = [json.loads(l) for l in (a.source/'rows.jsonl').open()]
    groups = defaultdict(list)
    for r in rows:
        if (r['action_owner'] != 'teacher' or not r['recorded_complete_targets'] or
            not r['post_completion_commands_excluded'] or r['true_geometry_actor_input']):
            raise ValueError('Unverified command ownership or geometry input')
        groups[r['family'], r['episode']].append(r)
    counts = Counter()
    admitted = []
    disagreements = []
    # Index past/future RECORDED COMMANDS, not future observations. A temporal
    # slot exists only where the original verified data actually has a label.
    for key, group in groups.items():
        group.sort(key=lambda r: (r['time_s'], r['frame']))
        timeline = {}
        inconsistent = set()
        for r in group:
            base_port = matrix_from_xyz_xyzw(r['state'][:7]) @ np.linalg.inv(matrix_from_pose9(r['predicted_port_tcp_pose9']))
            for j, value in enumerate(r['recorded_tcp_targets_predicted_port_pose9']):
                stamp = round(float(r['time_s'])*1e6) + (j+1)*50000
                target = base_port @ matrix_from_pose9(value)
                old = timeline.get(stamp)
                if old is not None:
                    position = np.linalg.norm(old[0][:3, 3]-target[:3, 3])*1000
                    cosine = (np.trace(old[0][:3, :3].T @ target[:3, :3])-1)/2
                    angle = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
                    if position > .05 or angle > .05:
                        inconsistent.add(stamp)
                        if len(disagreements) < 30:
                            disagreements.append(dict(episode=key, time_us=stamp, mm=position, deg=angle))
                else:
                    timeline[stamp] = (target, dict(episode=r['episode'], frame=r['frame'],
                        label_offset_s=(j+1)*.05, source_command_indices=r.get('source_command_indices')))
        for r in group:
            stamps = [round(float(r['time_s'])*1e6)+(j+1)*50000 for j in range(12)]
            if any(t in inconsistent for t in stamps):
                counts['conflicting_recorded_slot'] += 1
                continue
            if any(t not in timeline for t in stamps):
                counts['missing_recorded_future_slot'] += 1
                continue
            base_port = matrix_from_xyz_xyzw(r['state'][:7]) @ np.linalg.inv(matrix_from_pose9(r['predicted_port_tcp_pose9']))
            inverse = np.linalg.inv(base_port)
            body = np.asarray(r['predicted_body_tip_position_m'])
            tcp_targets = []
            point_targets = []
            provenance = []
            for t in stamps:
                target, source = timeline[t]
                tcp = inverse @ target
                point = tcp.copy()
                point[:3, 3] += tcp[:3, :3] @ body
                tcp_targets.append(pose9(tcp))
                point_targets.append(pose9(point))
                provenance.append(dict(absolute_command_time_us=t, **source))
            # Existing first four labels must agree with the reconstructed
            # record; a copied derivative must not silently change the action.
            if not np.allclose(np.asarray(tcp_targets[:4]), r['recorded_tcp_targets_predicted_port_pose9'], atol=2e-6):
                raise ValueError('Original recorded target roundtrip failed')
            if not np.allclose(np.asarray(point_targets[:4]), r['teacher_tcp_targets_predicted_port_pose9'], atol=2e-6):
                raise ValueError('Original plug-point encoding roundtrip failed')
            new = copy.copy(r)
            new.update(recorded_tcp_targets_predicted_port_pose9=tcp_targets,
                teacher_tcp_targets_predicted_port_pose9=point_targets,
                action_target_offsets_s=[round((j+1)*.05, 8) for j in range(12)],
                horizon_target_provenance=provenance, maximum_verified_action_horizon=12,
                extended_from='same_episode_recorded_command_timeline_not_measured_future_pose')
            admitted.append(new)
            counts['admitted'] += 1
    if not admitted:
        raise ValueError('No causally matched12-step anchors; inspect time/ownership provenance')
    for h in (4, 8, 12):
        folder = a.output/f'h{h}'
        folder.mkdir()
        with (folder/'rows.jsonl').open('w') as f:
            for row in admitted:
                r = dict(row)
                for k in ['recorded_tcp_targets_predicted_port_pose9', 'teacher_tcp_targets_predicted_port_pose9',
                          'action_target_offsets_s', 'horizon_target_provenance']:
                    r[k] = r[k][:h]
                f.write(json.dumps(r)+'\n')
        # Reuse the immutable image cache using original cache_index values.
        for filename in ['grasp_locks.jsonl']:
            src = a.source/filename
            if src.exists():
                (folder/filename).symlink_to(src.resolve())
    cells = Counter((r['family'], r['split']) for r in admitted)
    summary = dict(original_rows=len(rows), matched_rows=len(admitted), counts=dict(counts),
        cells=[dict(family=k[0], split=k[1], rows=v,
            episodes=len({r['episode'] for r in admitted if (r['family'], r['split']) == k})) for k, v in cells.items()],
        example_disagreements=disagreements, identical_anchor_sets=True, original_data_modified=False,
        true_geometry_actor_input=False, no_future_observation_as_actor_input=True,
        source_sha256=hashlib.sha256((a.source/'rows.jsonl').read_bytes()).hexdigest())
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
