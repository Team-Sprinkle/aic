#!/usr/bin/env python3
"""Bound body-target conversion timing error against raw controller samples.

The controller resolves body deltas at its hardware update, not by looking up
the header time. First post-message hardware state is a diagnostic proxy, not
an exact RT-consumption log. No future measured motion is an actor input.
"""
import argparse
from bisect import bisect_right, bisect_left
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from prepare_verified_insertion_command_rows import message_pose


def stats(x):
    return dict(n=len(x),p50=float(np.median(x)),p95=float(np.percentile(x,95)),maximum=float(max(x))) if x else dict(n=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=4)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sources=[json.loads(l) for l in a.sources.open() if json.loads(l)['family']=='sfp'][:a.limit];reports=[]
    for s in sources:
        d=json.loads(Path(s['source_stream']).read_text());bag=next((Path(s['source_audit'])/'results').glob('bag_*/*.mcap'))
        commands=[c for c in d['commands'] if c['kind']=='pose' and c['message']['header']['frame_id']=='gripper/tcp']
        rows=[];walls=[]
        with bag.open('rb') as f:
            reader=make_reader(f)
            for schema,_,message in reader.iter_messages(topics=['/aic_controller/controller_state']):
                msg=deserialize_message(message.data,get_message(schema.name))
                stamp=msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
                p=msg.tcp_pose.position;q=msg.tcp_pose.orientation;t=np.eye(4)
                t[:3,3]=[p.x,p.y,p.z];t[:3,:3]=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
                rows.append((stamp,t));walls.append(message.log_time)
        times=[v[0] for v in rows];errors=[];rotations=[];ages=[];count=Counter();gains=Counter();samples=[]
        for c in commands:
            before=bisect_right(times,c['sim_ns'])-1;after=bisect_left(walls,c['wall_ns'])
            if before<0 or after>=len(rows):count['no_bracketing_state']+=1;continue
            if c['message']['trajectory_generation_mode']['mode']!=2:count['not_position_mode']+=1;continue
            # Exclude a stale ROS delivery proxy, rather than treating it as
            # exact consumption. Report how often the proxy cannot be used.
            age=(rows[after][0]-c['sim_ns'])/1e6
            if not 0<=age<=20:count['proxy_stamp_outside0to20ms']+=1;continue
            old=rows[before][1]@message_pose(c['message']['pose']);new=rows[after][1]@message_pose(c['message']['pose'])
            err=float(np.linalg.norm(old[:3,3]-new[:3,3])*1000);angle=float(np.degrees(Rotation.from_matrix(old[:3,:3].T@new[:3,:3]).magnitude()))
            errors.append(err);rotations.append(angle);ages.append(age)
            stiffness=c['message']['target_stiffness'];gains[str([stiffness[i*7] for i in range(6)])]+=1
            samples.append(dict(command_sim_s=c['sim_ns']/1e9,proxy_after_header_ms=age,target_difference_mm=err,target_difference_deg=angle,geometry_input=False))
        report=dict(source=s['source_stream'],body_commands=len(commands),bounded_proxy_rows=len(errors),rejections=dict(count),
                    complete_target_translation_difference_mm=stats(errors),orientation_difference_deg=stats(rotations),proxy_after_header_ms=stats(ages),stiffness_profiles=dict(gains),
                    exact_consumption_recovered=False)
        reports.append(report)
        with (a.output/'command_comparisons.jsonl').open('a') as f:
            for sample in samples:f.write(json.dumps(dict(source=s['source_stream'],**sample))+'\n')
        print(json.dumps(report),flush=True)
    summary=dict(results=reports,controller_semantics='current_tool_state at RT consumption, not header TF lookup',
        caveat='Post-message state is a bounded diagnostic proxy. It cannot certify exact target recovery.',future_measurement_actor_input=False,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
