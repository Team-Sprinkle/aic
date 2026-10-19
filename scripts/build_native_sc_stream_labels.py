#!/usr/bin/env python3
"""Causal native-image SC/SFP labels from new perception-only rollouts.

Failed policies may supply correct perception labels, never successful BC.
Scoring geometry is a label only. No post-return/reset observations admitted.
"""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from audit_sc_port_targets import pose_matrix
from analyze_local_actor_scored_tip import transform
from build_sc_native_pose_labels import project


def past(rows,stamp,max_age_ns=None):
    i=bisect_right(rows,stamp,key=lambda r:r[0])-1
    if i<0:raise ValueError('No past observation; future/reset joins forbidden')
    if max_age_ns is not None and stamp-rows[i][0]>max_age_ns:raise ValueError('Stale state')
    return rows[i]


def build(audit,split,output):
    d=json.loads((audit/'scored_stream.json').read_text());cfg=yaml.safe_load((audit/'eval_config.yaml').read_text())
    trial=cfg['trials'][d['trial_id']];board=trial['scene']['task_board'];task=d['task']
    family='sfp' if task['plug_name'].startswith('sfp') else 'sc'
    configured_ports=None
    if family=='sfp':
        from build_two_family_port_projection_labels import sfp_ports
        calibration=json.loads(Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())
        configured_ports=sfp_ports(dict(trial_yaml=str(audit/'eval_config.yaml'),trial_id=d['trial_id']),calibration)
        card=int(task['target_module_name'].rsplit('_',1)[1]);port=int(task['port_name'].rsplit('_',1)[1])
        selected_identity=f'sfp_card_{card}_port_{port}'
    else:selected_identity=task['target_module_name']
    physical=dict(scene=trial['scene'],robot=cfg['robot']);group=hashlib.sha256(json.dumps(physical,sort_keys=True).encode()).hexdigest()
    edges=d['edges'];public_cal=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics={c:np.asarray(t) for c,t in public_cal['tcp_to_optical'].items()}
    trace=[json.loads(l) for l in (audit/'runtime_trace.jsonl').open()]
    terminal=next(e for e in reversed(trace) if e['event']=='terminal_before_return')
    cutoff=min(round(terminal['sim_time']*1e9),max(c['sim_ns'] for c in d['commands'] if c['kind']=='pose'))
    counters=Counter();records=[];previous=-1e30;present=[p for p in (0,1) if board[f'sc_rail_{p}']['entity_present']]
    for stamp,image in d['cameras']['center']:
        if stamp>cutoff:counters['post_terminal_excluded']+=1;continue
        if stamp-previous<500_000_000:continue
        try:
            images={'center':image};stamps={'center':stamp}
            for c in ('left','right'):
                ts,file=past(d['cameras'][c],stamp,1_000_000);images[c]=file;stamps[c]=ts
            ct,controller=past(d['streams']['/aic_controller/controller_state'],stamp,10_000_000)
            if not all(Path(p).exists() for p in images.values()):raise ValueError('Missing original RGB')
            if not records and any(cv2.imread(p).shape[:2]!=(1024,1152) for p in images.values()):raise ValueError('Not original native RGB')
            pose=controller['tcp_pose'];state=[*[pose['position'][k] for k in 'xyz'],*[pose['orientation'][k] for k in 'xyzw']]
            tcp=pose_matrix(state)
            def edge(key):return transform(past(edges[key],stamp)[1])
            target_frames=[]
            if family=='sc':
                world_base=edge('world|tabletop')@edge('tabletop|base_link')
                world_aic=edge('world|aic_world')
                world_board=world_aic@edge('aic_world|task_board')
                for p in present:
                    parent=f'task_board/sc_port_{p}'
                    port=np.linalg.inv(world_base)@world_board@edge(f'task_board|{parent}')@edge(f'{parent}|{parent}/sc_port_base_link_entrance')
                    target_frames.append((f'sc_port_{p}',port))
            else:
                target_frames=[(f'sfp_card_{c}_port_{p}',t) for c,p,t in configured_ports]
            targets=[dict(identity=identity,position_base_m=port[:3,3].tolist(),orientation_base_xyzw=Rotation.from_matrix(port[:3,:3]).as_quat().tolist(),cameras={c:project(port,tcp@optics[c]) for c in images}) for identity,port in target_frames]
            if selected_identity not in {t['identity'] for t in targets}:raise ValueError('Requested target absent in scene configuration')
            records.append(dict(family=family,episode=audit.name,scene_group=group,split=split,time_s=stamp/1e9,state=state,images=images,image_sim_ns=stamps,
                image_kind='native_1152x1024_rgb_jpeg',source_stream=str(audit/'scored_stream.json'),source_scored_stream=str(audit/'scored_stream.json'),
                targets=targets,selected_identity=selected_identity,card_count=sum(board[f'nic_rail_{i}']['entity_present'] for i in range(5)),
                target_geometry_label_source='configured calibrated opening; not loaded-card pose' if family=='sfp' else 'causal scoring TF',
                image_pose_age_ms=(stamp-ct)/1e6,label_pose_join='latest preceding controller at synchronized native RGB',
                physical_reset_config=str(audit/'eval_config.yaml'),physical_visibility_certified=False,
                geometry_training_label_only=True,BC_eligible=False,perception_only=True,terminal_clipped_to_last_pose_command=True))
            previous=stamp
        except ValueError as exc:counters[str(exc)]+=1
    if not records:raise ValueError(f'No causally valid native rows: {counters}')
    output.mkdir(parents=True,exist_ok=False)
    (output/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    summary=dict(family=family,rows=len(records),rejected=dict(counters),split=split,physical_reset_group=group,selected_identity=selected_identity,present_ports=present if family=='sc' else [f'sfp_card_{c}_port_{p}' for c,p,t in configured_ports],
        native_image_shape=[1024,1152,3],post_reset_geometry_used=False,BC_eligible=False,final_scenes_opened=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('audit',type=Path);p.add_argument('--split',choices=('train','validation'),required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.audit,a.split,a.output)),flush=True)


if __name__=='__main__':main()
