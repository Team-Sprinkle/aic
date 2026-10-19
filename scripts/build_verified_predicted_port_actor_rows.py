#!/usr/bin/env python3
"""Re-express qualified recorded commands in a causal RGB port-lock frame.

This stage performs no perception inference and reads no simulator geometry.
It requires a separately frozen observation-only lock manifest. Pre-lock rows
are excluded; future measured robot motion never supplies an action label.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from shared_port_tcp_geometry import matrix_from_pose9


def pose9(matrix):
    return np.r_[matrix[:3, 3], matrix[:3, :2].T.reshape(-1)].tolist()


def convert_record(row, lock):
    """Return one row, or None before its causal lock; reject bad contracts."""
    from shared_port_tcp_geometry import matrix_from_xyz_xyzw
    if lock.get('geometry_input') is not False:
        raise ValueError('Port lock must declare observation-only inference')
    if any(row[k] != lock[k] for k in ('episode', 'scene', 'split', 'task')):
        raise ValueError('Port lock identity/reset/query mismatch')
    if lock['time_s'] > row['time_s']:
        return None
    if row.get('action_owner') != 'teacher':
        raise ValueError('Supervisor/reset commands are not actor labels')
    if not row.get('post_completion_commands_excluded'):
        raise ValueError('Completion cutoff must be audited before relabeling')
    if not row.get('labels_are_recorded_targets_not_measured_future_motion'):
        raise ValueError('Recorded targets required')
    if (row.get('action_target_offsets_s') != [.05, .10, .15, .20] or
            len(row['recorded_base_tcp_targets_pose9']) != 4):
        raise ValueError('Expected four genuine 50 ms future active targets')
    if not np.allclose(row['recorded_controller_stiffness'], [90,90,90,50,50,50]):
        raise ValueError('Insertion controller stiffness must match the expert')
    if not np.allclose(row['recorded_controller_damping'], [50,50,50,20,20,20]):
        raise ValueError('Insertion controller damping must match the expert')
    if (row.get('full_recorded_controller_contract_verified') is not True or
            not np.allclose(row.get('recorded_controller_wrench_feedback_gains',[]),[.5,.5,.5,0,0,0]) or
            not np.allclose(row.get('recorded_feedforward_wrench',[]),[0.]*6) or
            row.get('recorded_trajectory_mode')!=2):
        raise ValueError('Full recorded force/trajectory profile must match deployment')
    base_port = np.asarray(lock['base_port_matrix'], float)
    if (base_port.shape != (4,4) or not np.isfinite(base_port).all() or
            not np.allclose(base_port[3], [0,0,0,1]) or
            not np.allclose(base_port[:3,:3].T@base_port[:3,:3],np.eye(3),atol=1e-5) or
            np.linalg.det(base_port[:3,:3]) < .999):
        raise ValueError('Invalid rigid port lock')
    inverse = np.linalg.inv(base_port)
    result = {k:row[k] for k in ('family','episode','scene','split','frame','time_s',
        'images','image_kind','state','task','source_stream','source_command_indices',
        'action_target_offsets_s','recorded_controller_stiffness',
        'recorded_controller_damping','body_command_conversion')}
    result.update({k:row[k] for k in ('recorded_controller_wrench_feedback_gains',
        'recorded_feedforward_wrench','recorded_trajectory_mode','full_recorded_controller_contract_verified')})
    result.update(schema='aic_predicted_port_local_bc/v1',cache_index=None,
        predicted_port_tcp_pose9=pose9(inverse@matrix_from_xyz_xyzw(row['state'][:7])),
        teacher_tcp_targets_predicted_port_pose9=[pose9(inverse@matrix_from_pose9(t))
            for t in row['recorded_base_tcp_targets_pose9']],
        perception_lock_time_s=lock['time_s'],perception_age_s=row['time_s']-lock['time_s'],
        static_port_lock=True,true_geometry_actor_input=False,
        recorded_complete_targets=True,post_completion_commands_excluded=True,
        action_owner='teacher',actor_training_ready=True,
        label_frame_source='same causal RGB lock as actor input')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commands',type=Path,required=True)
    p.add_argument('--locks',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    locks={}
    for line in a.locks.open():
        lock=json.loads(line)
        if lock['episode'] in locks:raise ValueError('Ambiguous static episode lock')
        locks[lock['episode']]=lock
    report=json.loads((a.commands/'summary.json').read_text())
    counts=Counter();excluded=Counter();episodes=defaultdict(set);groups=defaultdict(set)
    envelope=defaultdict(list)
    with (a.output/'rows.jsonl').open('w') as writer:
        for episode in report['episodes']:
            for line in (a.commands/episode['rows_file']).open():
                row=json.loads(line);lock=locks.get(row['episode'])
                if lock is None:excluded['no_RGB_lock']+=1;continue
                result=convert_record(row,lock)
                if result is None:excluded['pre_lock_row']+=1;continue
                writer.write(json.dumps(result,separators=(',',':'))+'\n')
                key=f"{row['family']}/{row['split']}";counts[key]+=1
                episodes[key].add(row['episode']);groups[row['split']].add(row['scene'])
                envelope[key].append(result['predicted_port_tcp_pose9'][:3])
    if groups['train']&groups['validation']:raise ValueError('Physical reset split leakage')
    summary=dict(schema='verified_predicted_port_actor_rows/v1',rows=dict(counts),
        episodes={k:len(v) for k,v in episodes.items()},exclusions=dict(excluded),
        estimated_TCP_position_range_m={k:dict(min=np.min(v,axis=0).tolist(),
            max=np.max(v,axis=0).tolist()) for k,v in envelope.items()},
        command_summary_sha256=hashlib.sha256((a.commands/'summary.json').read_bytes()).hexdigest(),
        lock_manifest_sha256=hashlib.sha256(a.locks.read_bytes()).hexdigest(),
        no_true_geometry_read=True,original_data_modified=False,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
