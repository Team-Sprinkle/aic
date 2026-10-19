#!/usr/bin/env python3
"""Independently score pre-return SC/SFP teacher terminal TF from raw MCAP.

This is offline label qualification. Completion topics and scoring geometry
must never enter an autonomous actor's observation.
"""
import argparse
from bisect import bisect_right
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from audit_sc_port_targets import matrix
from audit_sfp_raw_bag_terminal_geometry import matrix as ros_matrix, stored_matrix


def past_transform(rows, stamp):
    i = bisect_right(rows, stamp, key=lambda r: r[0]) - 1
    if i < 0:
        raise ValueError('No causal transform; refusing a future/reset join')
    return stored_matrix(rows[i][1])


def audit(path, terminal_event='wide_teacher_terminal_before_return'):
    stream = json.loads((path / 'scored_stream.json').read_text())
    events = [json.loads(l) for l in (path / 'runtime_trace.jsonl').open()]
    terminal = next(e for e in reversed(events)
                    if e['event'] == terminal_event)
    if terminal_event != 'wide_teacher_terminal_before_return':
        if (terminal.get('observation_source') != 'retained_public_observation_before_return'
                or terminal.get('post_reset_geometry_used') is not False
                or terminal.get('observation_wall_time_ns') is None):
            raise ValueError('Autonomous terminal must retain a causal pre-return observation')
        terminal=dict(terminal,time_sec=terminal['sim_time'])
    stamp = round(terminal['time_sec'] * 1e9)
    commands = [c for c in stream['commands']
                if c['kind'] == 'pose' and c['sim_ns'] <= stamp]
    command = max(commands, key=lambda c: c['wall_ns'])
    wall = command['wall_ns']
    task = stream['task']; cable = task['cable_name']
    tip = f"{cable}/{task['plug_name']}_link"
    needed = {('aic_world', cable), (cable, tip)}
    latest = {}; world_aic = None
    bags = list((path / 'results').glob('bag_*/*.mcap'))
    if len(bags) != 1:
        raise ValueError(f'Expected one raw bag: {bags}')
    typ = get_message('tf2_msgs/msg/TFMessage')
    with bags[0].open('rb') as raw:
        reader = make_reader(raw)
        if terminal.get('observation_wall_time_ns') is not None:
            wall=int(terminal['observation_wall_time_ns'])
        elif stamp-command['sim_ns']>100_000_000:
            # New stop-on-event teachers hold without further commands. Use
            # a same-episode pre-return controller observation as wall bound,
            # never a post-reset pose or simply the bag's last geometry.
            observed=None
            for schema,_,message in reader.iter_messages(
                    topics=['/aic_controller/controller_state'],start_time=wall-100_000_000):
                c=deserialize_message(message.data,get_message(schema.name))
                ts=c.header.stamp.sec*1_000_000_000+c.header.stamp.nanosec
                if ts>stamp:break
                observed=(ts,message.log_time)
            if observed is None or stamp-observed[0]>100_000_000:
                raise ValueError('No causal terminal hold observation')
            wall=observed[1]
        for _, _, message in reader.iter_messages(topics=['/tf_static'], end_time=wall+1):
            for tf in deserialize_message(message.data, typ).transforms:
                if (tf.header.frame_id, tf.child_frame_id) == ('world', 'aic_world'):
                    world_aic = ros_matrix(tf.transform)
        for _, _, message in reader.iter_messages(
                topics=['/tf', '/scoring/tf'], start_time=wall-1_000_000_000,
                end_time=wall+1):
            for tf in deserialize_message(message.data, typ).transforms:
                key = tf.header.frame_id, tf.child_frame_id
                ts = tf.header.stamp.sec*1_000_000_000 + tf.header.stamp.nanosec
                if key in needed and ts <= stamp and (key not in latest or ts > latest[key][0]):
                    latest[key] = ts, ros_matrix(tf.transform)
    if set(latest) != needed or world_aic is None:
        raise ValueError('Incomplete raw causal TF chain')
    ages = [(stamp-latest[k][0])/1e6 for k in sorted(needed)]
    if max(ages) > 100:
        raise ValueError(f'Stale terminal TF: {ages}')
    family='sfp' if task['plug_name'].startswith('sfp') else 'sc'
    edges = stream['edges']; port = task['target_module_name']
    prefix = f'task_board/{port}'
    opening_offset = matrix(json.loads(Path(
        'configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json'
    ).read_text())['base_to_opening'])
    base_world = np.linalg.inv(past_transform(edges['world|tabletop'], stamp) @
                              past_transform(edges['tabletop|base_link'], stamp))
    # All port edges are same-episode past transforms, never post-reset poses.
    base_mount=(base_world @ world_aic @ past_transform(edges['aic_world|task_board'],stamp) @
                past_transform(edges[f'task_board|{prefix}'],stamp))
    if family=='sc':
        base_opening=base_mount@past_transform(edges[f'{prefix}|{prefix}/sc_port_base_link'],stamp)@opening_offset
    else:
        calibration=json.loads(Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())
        base_opening=base_mount@matrix(calibration['nic_mount_to_port_opening'][task['port_name'].rsplit('_',1)[1]])
    base_tip = base_world @ world_aic @ latest[('aic_world', cable)][1] @ latest[(cable, tip)][1]
    relative = np.linalg.inv(base_opening) @ base_tip
    depth = float(relative[2, 3]*1000)
    lateral = float(np.linalg.norm(relative[:2, 3])*1000)
    angle = float(np.degrees(Rotation.from_matrix(relative[:3, :3]).magnitude()))
    score = yaml.safe_load((path/'results/scoring.yaml').read_text())[stream['trial_id']]
    categories = score['tier_2']['categories']
    safe = all(categories[k]['score'] >= 0 for k in ('contacts', 'insertion force'))
    force = [np.linalg.norm([d['wrench']['force'][k] for k in 'xyz'])
             for t, d in stream['streams']['/fts_broadcaster/wrench'] if t <= stamp]
    local = [e['force_delta_n'] for e in events if e['event'] == 'wide_teacher_local_command'
             and e['time_sec'] <= terminal['time_sec'] and e.get('force_delta_n') is not None]
    minimum_depth=15.5 if family=='sc' else 44.
    return dict(audit=str(path),family=family,minimum_physical_depth_mm=minimum_depth,terminal_time_s=terminal['time_sec'],
                terminal_event=terminal_event,
                command_wall_cutoff_ns=wall, TF_age_ms=ages, depth_mm=depth,
                lateral_mm=lateral, orientation_error_deg=angle,
                official_tier3=score['tier_3']['score'], no_scored_harm=safe,
                raw_force_norm_p50_p95_max_n=np.percentile(force, [50,95,100]).tolist(),
                teacher_baseline_subtracted_force_max_n=float(max(local)) if local else None,
                qualified=depth >= minimum_depth and lateral <= .5 and angle <= 1 and
                          score['tier_3']['score'] == 75 and safe,
                measured_tip_port_m=relative[:3,3].tolist(),
                measured_base_opening_matrix=base_opening.tolist(),
                measured_base_tip_matrix=base_tip.tolist(),
                original_data_modified=False, geometry_posthoc_only=True,
                post_reset_geometry_used=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audits', nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--terminal-event',default='wide_teacher_terminal_before_return',
                   choices=['wide_teacher_terminal_before_return','terminal_before_return'])
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve previous audits')
    results = [audit(path,a.terminal_event) for path in a.audits]
    report = dict(schema=('wide_teacher_causal_terminal_audit/v1' if a.terminal_event=='wide_teacher_terminal_before_return'
                         else 'autonomous_shared_actor_causal_terminal_audit/v1'), results=results,
                  qualified=sum(r['qualified'] for r in results), final_scenes_opened=False,
                  caveat='Raw force includes gravity/load; it is not itself contact force. '
                         'Official harmful-contact/force scoring is checked separately.')
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
