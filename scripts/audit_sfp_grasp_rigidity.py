#!/usr/bin/env python3
"""Check whether per-episode SFP grasp translation labels transfer across time."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from audit_sfp_raw_bag_terminal_geometry import matrix, stored_matrix
from prepare_verified_insertion_command_rows import message_pose


def calibrate(s,cutoffs,typ):
    d=json.loads(Path(s['source_stream']).read_text());task=d['task'];cable=task['cable_name'];tip=f"{cable}/{task['plug_name']}_link";needed={('aic_world',cable),(cable,tip)}
    commands=[c for c in d['commands'] if c['kind']=='pose' and (cutoffs is None or c['sim_ns']<=cutoffs[s['source_stream']])];controllers=d['streams']['/aic_controller/controller_state'];times=[r[0] for r in controllers]
    wb=stored_matrix(d['edges']['world|tabletop'][0][1])@stored_matrix(d['edges']['tabletop|base_link'][0][1]);samples=[]
    bag=next((Path(s['source_audit'])/'results').glob('bag_*/*.mcap'))
    with bag.open('rb') as f:
        reader=make_reader(f);wa=None
        for _,_,message in reader.iter_messages(topics=['/tf_static']):
            for tf in deserialize_message(message.data,typ).transforms:
                if (tf.header.frame_id,tf.child_frame_id)==('world','aic_world'):wa=matrix(tf.transform)
        if wa is None:raise ValueError('Missing explicit world/aic_world calibration')
        for fraction in (.1,.5,.9):
            c=commands[int(fraction*(len(commands)-1))];observed={k:[] for k in needed}
            for _,_,message in reader.iter_messages(topics=['/tf','/scoring/tf'],start_time=c['wall_ns']-1_000_000_000,end_time=c['wall_ns']+1):
                for tf in deserialize_message(message.data,typ).transforms:
                    key=tf.header.frame_id,tf.child_frame_id;ts=tf.header.stamp.sec*1_000_000_000+tf.header.stamp.nanosec
                    if key in needed and ts<=c['sim_ns']:observed[key].append((ts,matrix(tf.transform)))
            if any(not v for v in observed.values()):raise ValueError('Missing causal grasp geometry')
            common=min(max(t for t,_ in v) for v in observed.values());transforms={}
            ages=[]
            for key,v in observed.items():
                ts,t=max((x for x in v if x[0]<=common),key=lambda x:x[0]);transforms[key]=t;ages.append((common-ts)/1e6)
            if max(ages)>2:raise ValueError('Asynchronous cable/tip transforms')
            j=bisect_right(times,common)-1
            if j<0 or common-times[j]>4_000_000:raise ValueError('Stale TCP for grasp check')
            bt=np.linalg.inv(wb)@wa@transforms[('aic_world',cable)]@transforms[(cable,tip)];tcp=message_pose(controllers[j][1]['tcp_pose'])
            relative=np.linalg.inv(tcp)@bt;samples.append(dict(fraction=fraction,sim_time_s=common/1e9,body_translation_m=relative[:3,3].tolist(),body_rotation_xyzw=Rotation.from_matrix(relative[:3,:3]).as_quat().tolist(),TCP_age_ms=(common-times[j])/1e6))
    x=np.asarray([r['body_translation_m'] for r in samples]);spread=float(np.max(np.linalg.norm(x[:,None]-x[None],axis=-1))*1000)
    rotations=Rotation.from_quat([v['body_rotation_xyzw'] for v in samples]);mean_rotation=rotations.mean()
    angle=float(np.max([np.degrees((x.inv()*y).magnitude()) for x in rotations for y in rotations]))
    result=dict(source=s['source_stream'],source_episode_id=s['source_episode_id'],split=s['split'],samples=samples,max_pairwise_body_translation_mm=spread,max_pairwise_body_rotation_deg=angle,
        mean_body_translation_m=np.median(x,axis=0).tolist(),mean_body_rotation_xyzw=mean_rotation.as_quat().tolist(),
        passed=spread<=.25 and angle<=.5,geometry_label_only=True,post_completion_excluded=cutoffs is not None)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=6)
    p.add_argument('--completion-cutoffs',type=Path,help='Restrict calibration to audited pre-completion commands')
    p.add_argument('--resume',action='store_true',help='Keep audited source records and continue after an interrupted audit')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=a.resume)
    sources=[json.loads(l) for l in a.sources.open() if json.loads(l)['family']=='sfp']
    if a.limit:sources=sources[:a.limit]
    cutoffs=({r['source_stream']:r['conservative_BC_cutoff_sim_ns'] for r in map(json.loads,a.completion_cutoffs.open()) if r['valid']} if a.completion_cutoffs else None)
    progress=a.output/'progress.jsonl'
    results=[json.loads(l) for l in progress.open()] if a.resume and progress.exists() else []
    seen={r['source'] for r in results};typ=get_message('tf2_msgs/msg/TFMessage')
    for s in sources:
        if s['source_stream'] in seen:continue
        try:result=calibrate(s,cutoffs,typ)
        except (ValueError,KeyError,StopIteration) as exc:
            result=dict(source=s['source_stream'],source_episode_id=s['source_episode_id'],split=s['split'],passed=False,calibration_label_failure=True,error=str(exc),expert_episode_admission_unchanged=True)
        results.append(result);print(json.dumps(result),flush=True)
        with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    summary=dict(episodes=len(results),passed=sum(r['passed'] for r in results),results=results,gate_mm=.25,gate_orientation_deg=.5,calibration_failures=sum(r.get('calibration_label_failure',False) for r in results),post_completion_excluded=cutoffs is not None,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
