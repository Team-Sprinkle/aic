#!/usr/bin/env python3
"""Prepare immutable, causal command supervision from admitted SC/SFP sources.

This stage does not create actor port inputs. It retains complete recorded
base-frame command targets; a separate RGB prediction pass must re-express
them in an observation-only estimated frame before training. Scene geometry
is used here only to select a generous training approach envelope.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import pose_matrix
from prepare_verified_insertion_source_index import source


def message_pose(d):
    return pose_matrix([*[d['position'][k] for k in 'xyz'],
                        *[d['orientation'][k] for k in 'xyzw']])


def pose9(t):
    return [*t[:3,3].tolist(), *t[:3,:2].T.reshape(-1).tolist()]


def episode_identity(source_path, name):
    # Different collection banks legitimately reuse names such as
    # sfp_trial1_train_00. Histories must never join those distinct episodes.
    digest=hashlib.sha256(str(Path(source_path).resolve()).encode()).hexdigest()[:12]
    return f'{name}__source{digest}'


def base_command(command, controllers, controller_times):
    t = message_pose(command['message']['pose'])
    frame = command['message']['header']['frame_id']
    if frame == 'base_link':
        return t
    if frame != 'gripper/tcp':
        raise ValueError(f'Unrecognized command frame {frame}')
    i = bisect_right(controller_times, command['sim_ns'])-1
    if i < 0 or command['sim_ns']-controller_times[i] > 20_000_000:
        raise ValueError('Body command has no causal measured TCP within20ms')
    return message_pose(controllers[i][1]['tcp_pose']) @ t


def task_identity(s):
    task = s['actual_public_task']
    port = int((task['port_name'] if s['family']=='sfp' else task['target_module_name']).rsplit('_',1)[1])
    return (f"sfp_card_{int(task['target_module_name'].rsplit('_',1)[1])}_port_{port}"
            if s['family']=='sfp' else f'sc_port_{port}')


def measured_state(stream, stamp):
    values = {}
    for name in ('/aic_controller/controller_state', '/joint_states', '/fts_broadcaster/wrench'):
        rows = stream['streams'][name]
        i = bisect_right(rows, stamp, key=lambda r:r[0])-1
        if i < 0 or stamp-rows[i][0] > 100_000_000:
            raise ValueError('Missing or stale causal robot state')
        values[name] = rows[i][1]
    c=values['/aic_controller/controller_state']; j=values['/joint_states']; w=values['/fts_broadcaster/wrench']['wrench']
    p=c['tcp_pose']; v=c['tcp_velocity']; joints=dict(zip(j['name'],j['position']))
    q=np.asarray([p['orientation'][k] for k in 'xyzw'],float)
    q/=np.linalg.norm(q)
    if q[3]<0:q=-q
    return [*[p['position'][k] for k in 'xyz'],*q.tolist(),
            *[v[k][axis] for k in ('linear','angular') for axis in 'xyz'],
            *c['tcp_error'],*[joints[k] for k in stream['joint_order']],
            joints['gripper/left_finger_joint']/2.,
            *[w[k][axis] for k in ('force','torque') for axis in 'xyz']]


def native_anchors(stream, template, last_command_ns, period_ns=200_000_000):
    """Retain5Hz native triplets instead of sparse perception-training samples.

    The template contributes reset identity and a dataset-only fixed-port
    envelope. Neither its TCP pose nor target position is an actor input.
    """
    cameras=stream['cameras']; result=[]; last=-period_ns
    for stamp,center in cameras['center']:
        if stamp>last_command_ns or stamp-last<period_ns:continue
        images={'center':center}; stamps={'center':stamp}
        for cam in ('left','right'):
            rows=cameras[cam]; i=bisect_right(rows,stamp,key=lambda r:r[0])-1
            if i<0 or stamp-rows[i][0]>1_000_000:break
            stamps[cam],images[cam]=rows[i]
        if len(images)!=3 or not all(Path(p).exists() for p in images.values()):continue
        if not result:
            if any(cv2.imread(p).shape[:2]!=(1024,1152) for p in images.values()):
                raise ValueError('Source camera stream is not original native RGB')
        r=dict(template,images=images,image_sim_ns=stamps,time_s=stamp/1e9)
        result.append(r);last=stamp
    return result


def prepare(s, anchors, output):
    d = json.loads(Path(s['source_stream']).read_text())
    if d['official_tier3'] != 75:
        raise ValueError('Source score changed after admission')
    commands = [c for c in d['commands'] if c['kind']=='pose']
    times = [c['sim_ns'] for c in commands]
    if times != sorted(times):
        raise ValueError('Unsorted commands')
    controllers = d['streams']['/aic_controller/controller_state']
    ct = [v[0] for v in controllers]
    joint_times = [c['sim_ns'] for c in d['commands'] if c['kind']=='joint']
    excluded = Counter(); emitted = 0; last_stationary = -float('inf')
    trace = Path(s['source_audit'])/'runtime_trace.jsonl'
    events = [json.loads(l) for l in trace.open()] if trace.exists() else []
    if any(e.get('command_owner')=='dataset_reset_supervisor' for e in events):
        raise ValueError('Wider reset sources require explicit interval-owned builder')
    starts = [e['time_sec'] for e in events if e['event']=='local_preinsert_align_started']
    if s['family']=='sc' and not starts:
        raise ValueError('Missing SC local teacher phase boundary')
    start = min(starts) if starts else -float('inf')
    converted = {}
    actual_identity = task_identity(s)
    actual_anchors=[r for r in anchors if r['selected_identity']==actual_identity]
    if not actual_anchors:
        raise ValueError('No native provenance with the actual public request')
    # Original camera captures supply more anchors without new collection.
    anchors=native_anchors(d,actual_anchors[0],times[-1])
    anchor_source='source_camera_stream_5Hz'
    if not anchors:
        # Some archived streams retain pre-move paths. The audited native
        # manifest already resolves these original captures; do not silently
        # lose otherwise qualified episodes or mutate the old stream paths.
        anchors=[r for r in actual_anchors if max(r['image_sim_ns'].values())<=times[-1]
                 and all(Path(p).exists() for p in r['images'].values())]
        anchor_source='audited_native_provenance_fallback'
        if not anchors:
            excluded['no_resolvable_original_native_images']+=1
    seen = set()
    with output.open('w') as writer:
        for r in sorted(anchors, key=lambda r:r['time_s']):
            if r['selected_identity'] != actual_identity:
                excluded['counterfactual_perception_request_not_actor_task'] += 1; continue
            stamps = r['image_sim_ns']; stamp = max(stamps.values())
            key = tuple(r['images'][c] for c in ('center','left','right'))
            if key in seen:
                excluded['duplicate_image_triplet'] += 1; continue
            seen.add(key)
            if max(stamps.values())-min(stamps.values()) > 1_000_000:
                excluded['asynchronous_triplet'] += 1; continue
            if r['time_s'] < start:
                excluded['SC_transport_prefix'] += 1; continue
            try:
                state=measured_state(d,stamp)
            except ValueError:
                excluded['missing_or_stale_measured_state'] += 1; continue
            # Dataset-only selection, never an actor feature or runtime stop.
            target = next(t for t in r['targets'] if t['identity']==actual_identity)
            port = pose_matrix([*target['position_base_m'],*target['orientation_base_xyzw']])
            tcp = pose_matrix(state[:7]); relative = np.linalg.inv(port) @ tcp
            if not (-.240 <= relative[2,3] <= .100 and np.linalg.norm(relative[:2,3]) <= .080):
                excluded['outside_generous_training_envelope'] += 1; continue
            selected = []
            for h in range(4):
                desired = stamp+(h+1)*50_000_000
                # Targets remain active between publications. Requiring four
                # distinct messages dropped most SC insertion holds. Resample
                # the recorded command signal, never future measured motion.
                i = bisect_right(times, desired)-1
                if desired>times[-1] or i<0 or desired-times[i]>1_200_000_000:
                    break
                selected.append(i)
            if len(selected)!=4:
                excluded['missing_causal_4x50ms_recorded_targets'] += 1; continue
            jt = bisect_left(joint_times, stamp)
            if jt<len(joint_times) and joint_times[jt] <= times[selected[-1]]:
                excluded['joint_control_intervenes'] += 1; continue
            transforms = []
            for i in selected:
                if i not in converted:
                    converted[i] = base_command(commands[i], controllers, ct)
                transforms.append(converted[i])
            # Remove prolonged static holds, never compress the time axis or
            # concatenate targets across a removed interval. Slow insertion
            # remains: 0.01mm over150ms is below the teacher's normal progress.
            span = np.linalg.norm(transforms[-1][:3,3]-transforms[0][:3,3])
            angle = Rotation.from_matrix(transforms[0][:3,:3].T@transforms[-1][:3,:3]).magnitude()
            stationary = span < .00001 and angle < np.deg2rad(.02) and np.linalg.norm(state[7:10]) < .0002
            if stationary:
                if r['time_s']-last_stationary < .5:
                    excluded['duplicate_static_hold_thinned_copy_only'] += 1; continue
                last_stationary = r['time_s']
            if len(state) != 32:
                raise ValueError('Actor requires32D measured state')
            row = dict(schema='verified_insertion_recorded_commands/v2',
                family=s['family'],episode=episode_identity(s['source_stream'],s['source_episode_id']),
                source_episode_name=s['source_episode_id'],scene=r['scene_group'],
                split=s['split'],frame=emitted,time_s=stamp/1e9,
                images=r['images'],image_kind='native_1152x1024_rgb_jpeg',
                state=state,task=s['task_vector'],
                recorded_base_tcp_targets_pose9=[pose9(t) for t in transforms],
                source_command_indices=selected,source_command_stamps_ns=[times[i] for i in selected],
                action_target_offsets_s=[.05,.10,.15,.20],
                target_sampling='recorded piecewise-constant active target at future50ms ticks',
                source_command_frames=[commands[i]['message']['header']['frame_id'] for i in selected],
                image_stamps_ns=stamps,source_stream=s['source_stream'],action_owner='teacher',
                labels_are_recorded_targets_not_measured_future_motion=True,
                static_hold_retained=bool(stationary),
                supervision_only_TCP_axial_m=float(relative[2,3]),
                predicted_port_input_ready=False,actor_training_ready=False)
            writer.write(json.dumps(row,separators=(',',':'))+'\n'); emitted += 1
    return dict(source=s['source_stream'],episode=episode_identity(s['source_stream'],s['source_episode_id']),source_episode_name=s['source_episode_id'],family=s['family'],split=s['split'],anchor_source=anchor_source,
                rows=emitted,rejected=dict(excluded),SC_local_start_s=start if starts else None,
                command_frames=dict(Counter(c['message']['header']['frame_id'] for c in commands)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sources=[json.loads(l) for l in a.sources.open()]
    if a.limit:sources=sources[:a.limit]
    admitted={str(Path(s['source_stream']).resolve()):s for s in sources}
    grouped=defaultdict(list)
    for l in a.labels.open():
        r=json.loads(l);path=source(r)
        if path in admitted:grouped[path].append(r)
    report=[];counts=Counter()
    for index,(path,s) in enumerate(admitted.items()):
        filename=f'{index:04d}_{s["family"]}_{s["source_episode_id"]}.jsonl'
        result=prepare(s,grouped[path],a.output/filename);result['rows_file']=filename
        report.append(result);counts[f'{s["family"]}/{s["split"]}']+=result['rows']
        with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        print(json.dumps(dict(completed=index+1,planned=len(sources),rows=result['rows'],family=s['family'])),flush=True)
    summary=dict(schema='verified_insertion_command_derivative/v1',sources=len(report),counts=dict(counts),
        source_sha256=hashlib.sha256(a.sources.read_bytes()).hexdigest(),episodes=report,
        original_data_modified=False,actor_training_ready=False,
        remaining_gate='RGB-only causal accepted port locks and pose-frame conversion required',
        final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
