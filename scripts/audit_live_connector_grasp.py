#!/usr/bin/env python3
"""Posthoc raw-bag labels for causal RGB grasp acquisition; never policy input."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from mcap.reader import make_reader
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from prepare_sfp_native_grasp_labels import label,matrix,stored_matrix


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();d=json.loads((a.run/'scored_stream.json').read_text())
    trace=list(map(json.loads,(a.run/'runtime_trace.jsonl').open()))
    events=[t for t in trace if t['event']=='connector_rotation_acquisition' and t['accepted']]
    terminal=next(t for t in reversed(trace) if t['event']=='terminal_before_return')
    if not events:return
    bag=next((a.run/'results').glob('bag_*/*.mcap'));typ=get_message('tf2_msgs/msg/TFMessage')
    optics=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())['tcp_to_optical']
    wb=stored_matrix(d['edges']['world|tabletop'][0][1])@stored_matrix(d['edges']['tabletop|base_link'][0][1]);reports=[]
    bound=next(t for t in trace if t['event']=='full_connector_grasp_bound')
    estimated=Rotation.from_matrix(np.asarray(bound['rotation_tcp']))
    with bag.open('rb') as f:
        reader=make_reader(f);wa=None
        for _,_,message in reader.iter_messages(topics=['/tf_static']):
            for tf in deserialize_message(message.data,typ).transforms:
                if (tf.header.frame_id,tf.child_frame_id)==('world','aic_world'):wa=matrix(tf.transform)
        if wa is None:raise ValueError('No explicit world/aic_world calibration')
        for event in [events[-1],terminal]:
            time_s=event['sim_time'];stamp=int(time_s*1e9);images={};stamps={}
            for c in ['center','left','right']:
                rows=d['cameras'][c];j=bisect_right(rows,stamp,key=lambda x:x[0])-1
                if j<0:raise ValueError('Missing causal image')
                stamps[c],images[c]=rows[j]
            row=dict(episode='posthoc_live',source_episode_name=a.run.name,scene='frozen-development',split='diagnostic',
                time_s=time_s,images=images,image_stamps_ns=stamps,source_stream=str(a.run/'scored_stream.json'))
            truth=label(reader,typ,d,row,np.eye(4) if wa is None else wa,wb,{c:np.asarray(x) for c,x in optics.items()})
            body=np.asarray(truth['body_tip_matrix']);error=float((Rotation.from_matrix(body[:3,:3]).inv()*estimated).magnitude()*180/np.pi)
            reports.append(dict(event=event['event'],time_s=time_s,image_to_label_age_ms=truth['image_to_label_age_ms'],
                orientation_error_deg=error,true_body_matrix=body.tolist(),
                position_error_mm=float(np.linalg.norm(body[:3,3]-bound['position_tcp_m'])*1000),
                original_rgb=images))
    report=dict(samples=reports,geometry_posthoc_only=True,no_post_reset_join=True,
        bound_rotation_estimated_from_RGB=bound['rotation_tcp'],world_aic_rotation_degrees=float(Rotation.from_matrix(wa[:3,:3]).magnitude()*180/np.pi))
    (a.run/'live_connector_grasp_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
