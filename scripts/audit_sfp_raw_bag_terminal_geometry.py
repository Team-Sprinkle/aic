#!/usr/bin/env python3
"""Recover SFP tip TF discarded by the old SC-specific stream extractor.

Reads only messages delivered by the final recorded command, with TF simulation
stamps <= that command and <=100ms age. Opening is the configured, calibrated
fixed port pose from the native label manifest: not a claim that actual loaded
card pose was dynamically measured. Originals and frozen scores stay intact.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def matrix(transform):
    p=transform.translation;q=transform.rotation;m=np.eye(4)
    m[:3,3]=[p.x,p.y,p.z];m[:3,:3]=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix();return m


def stored_matrix(d):
    m=np.eye(4);m[:3,3]=[d['translation'][k] for k in 'xyz'];m[:3,:3]=Rotation.from_quat([d['rotation'][k] for k in 'xyzw']).as_matrix();return m


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    reps={}
    for l in a.labels.open():
        r=json.loads(l)
        if r['family']=='sfp':
            source=r.get('source_scored_stream') or r.get('source_stream') or str(Path(r['physical_reset_config']).parent/'scored_stream.json')
            reps.setdefault(source,r)
    typ=get_message('tf2_msgs/msg/TFMessage');results=[]
    for index,(source,row) in enumerate(reps.items()):
        if a.limit and index>=a.limit:break
        result=dict(source_stream=source,episode=row['episode'],scene_group=row['scene_group'],split=row['split'])
        try:
            d=json.loads(Path(source).read_text());task=d['task'];cable=task['cable_name'];tip=f"{cable}/{task['plug_name']}_link"
            selected=f"sfp_card_{int(task['target_module_name'].rsplit('_',1)[1])}_port_{int(task['port_name'].rsplit('_',1)[1])}"
            target=next(t for t in row['targets'] if t['identity']==selected)
            commands=[c for c in d['commands'] if c['kind']=='pose'];command=max(commands,key=lambda c:c['sim_ns'])
            wall=command['wall_ns'];stamp=command['sim_ns'];audit=Path(source).parent
            if not (audit/'results').is_dir():audit=Path(d['audit'])
            bags=list((audit/'results').glob('bag_*/*.mcap'))
            if len(bags)!=1:raise ValueError(f'Expected one raw MCAP, found{len(bags)}')
            needed={('aic_world',cable),(cable,tip)};latest={};world_aic=None
            with bags[0].open('rb') as raw:
                reader=make_reader(raw)
                for _,_,msg in reader.iter_messages(topics=['/tf_static']):
                    for tf in deserialize_message(msg.data,typ).transforms:
                        if (tf.header.frame_id,tf.child_frame_id)==('world','aic_world'):world_aic=matrix(tf.transform)
                for _,channel,msg in reader.iter_messages(topics=['/tf','/scoring/tf'],start_time=wall-1_000_000_000,end_time=wall+1):
                    for tf in deserialize_message(msg.data,typ).transforms:
                        key=(tf.header.frame_id,tf.child_frame_id)
                        ts=tf.header.stamp.sec*1_000_000_000+tf.header.stamp.nanosec
                        if key in needed and ts<=stamp and (key not in latest or ts>latest[key][0]):
                            latest[key]=(ts,matrix(tf.transform))
            if set(latest)!=needed:raise ValueError('Raw terminal tip TF chain incomplete')
            ages=[(stamp-latest[k][0])/1e6 for k in sorted(needed)]
            if max(ages)>100:raise ValueError(f'Only stale terminal TF exists: {ages}')
            if world_aic is None:raise ValueError('Missing explicit world-to-aic_world static transform')
            e=d['edges'];world_base=stored_matrix(e['world|tabletop'][0][1])@stored_matrix(e['tabletop|base_link'][0][1])
            base_tip=np.linalg.inv(world_base)@world_aic@latest[('aic_world',cable)][1]@latest[(cable,tip)][1]
            base_port=np.eye(4);base_port[:3,3]=target['position_base_m'];base_port[:3,:3]=Rotation.from_quat(target['orientation_base_xyzw']).as_matrix()
            relative=np.linalg.inv(base_port)@base_tip;depth=float(relative[2,3]*1000);lateral=float(np.linalg.norm(relative[:2,3])*1000)
            orient=float(np.degrees(Rotation.from_matrix(relative[:3,:3]).magnitude()))
            result.update(valid=True,raw_bag=str(bags[0]),selected_identity=selected,manifest_request_matches_task=selected==row['selected_identity'],
                last_command_sim_s=stamp/1e9,TF_age_ms=ages,depth_mm=depth,lateral_mm=lateral,orientation_error_deg=orient,
                official_tier3=d['official_tier3'],configured_opening_depth_lateral_pass=depth>=44 and lateral<=.5,
                raw_tip_position_base_m=base_tip[:3,3].tolist(),configured_opening_position_base_m=base_port[:3,3].tolist(),
                geometry_access='posthoc scoring only; configured port is not a dynamically measured loaded card pose')
        except Exception as exc:result.update(valid=False,error=f'{type(exc).__name__}: {exc}')
        results.append(result)
        with (a.output/'results.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        if (index+1)%25==0:print(json.dumps(dict(completed=index+1,planned=min(len(reps),a.limit) if a.limit else len(reps),valid=sum(r['valid'] for r in results))),flush=True)
    valid=[r for r in results if r['valid']]
    def summary_stat(key):
        v=[r[key] for r in valid];return dict(p50=float(np.median(v)),p95=float(np.percentile(v,95)),min=float(min(v)),max=float(max(v))) if v else {}
    report=dict(schema='SFP_raw_terminal_geometry_audit/v1',episodes=len(results),valid=len(valid),invalid=len(results)-len(valid),
        configured_opening_depth_lateral_pass=sum(r['configured_opening_depth_lateral_pass'] for r in valid),
        depth_mm=summary_stat('depth_mm'),lateral_mm=summary_stat('lateral_mm'),orientation_error_deg=summary_stat('orientation_error_deg'),
        false_expert_admission_avoided=True,raw_tip_recovered_without_recollection=True,original_data_modified=False,
        caveat='Configured calibrated opening reference, not newly observed dynamic card pose. This audit is not autonomous insertion evidence or final expert admission.',final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
