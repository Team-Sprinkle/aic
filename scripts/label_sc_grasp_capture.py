#!/usr/bin/env python3
"""Strict image-time raw-bag connector pose labels for a public observation-only hold."""
import cv2
cv2.setNumThreads(2)
import argparse,bisect,json
from pathlib import Path
import numpy as np
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from prepare_sfp_native_grasp_labels import label,qa,matrix,stored_matrix


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--case',type=Path,required=True);a=p.parse_args();case=json.loads(a.case.read_text())
 d=json.loads((a.run/'scored_stream.json').read_text());trace=[json.loads(l) for l in (a.run/'runtime_trace.jsonl').open()];terminal=next(t for t in trace if t['event']=='grasp_capture_terminal')
 controls=d['streams']['/aic_controller/controller_state'];times=[x[0] for x in controls];cams=d['cameras'];quiet=[]
 for stamp,path in cams['center']:
  if not terminal['sim_time']-case.get('sampling_window_s',2.5)<=stamp/1e9<=terminal['sim_time']-.1:continue
  j=bisect.bisect_right(times,stamp)-1
  if j<0 or stamp-times[j]>4_000_000:continue
  v=controls[j][1]['tcp_velocity']
  if np.linalg.norm([v['linear'][k] for k in 'xyz'])<=.01 and np.linalg.norm([v['angular'][k] for k in 'xyz'])<=.1:quiet.append((stamp,path))
 chosen=[quiet[i] for i in np.linspace(0,len(quiet)-1,min(12,len(quiet)),dtype=int)]
 wb=stored_matrix(d['edges']['world|tabletop'][0][1])@stored_matrix(d['edges']['tabletop|base_link'][0][1]);typ=get_message('tf2_msgs/msg/TFMessage');results=[];errors=[]
 with next((a.run/'results').glob('bag_*/*.mcap')).open('rb') as f:
  reader=make_reader(f);wa=None
  for _,_,message in reader.iter_messages(topics=['/tf_static']):
   for tf in deserialize_message(message.data,typ).transforms:
    if (tf.header.frame_id,tf.child_frame_id)==('world','aic_world'):wa=matrix(tf.transform)
  if wa is None:raise ValueError('Missing explicit calibration')
  for stamp,_ in chosen:
   images={};stamps={}
   for c in ['center','left','right']:
    j=bisect.bisect_right(cams[c],stamp,key=lambda x:x[0])-1
    if j<0:raise ValueError('No causal RGB')
    stamps[c],images[c]=cams[c][j]
   item=dict(family=case.get('family','sc'),episode=case['case'],source_episode_name=case['case'],scene=d['scene_group_id'],split=case['split'],time_s=stamp/1e9,images=images,image_stamps_ns=stamps,source_stream=str(a.run/'scored_stream.json'))
   optics=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())['tcp_to_optical']
   try:result=label(reader,typ,d,item,wa,wb,{c:np.asarray(x) for c,x in optics.items()})
   except ValueError as exc:errors.append(str(exc));continue
   body=np.asarray(result['body_tip_matrix']);pose=np.r_[body[:3,3],body[:3,:2].T.reshape(-1)]
   role=('Near25mm SC grasp capture; perception only' if case.get('viewpoint')=='near' else 'Predeadline acquisition-view SC grasp capture; perception only' if case.get('viewpoint')=='acquisition'
         else 'Nominal-centred predeadline-sample-derived SC grasp capture; perception only')
   role=case.get('source_role',role)
   results.append(dict(item,tip_in_tcp_pose9=pose.tolist(),geometry_label_only=True,GT_crop_selection=False,source_role=role,image_to_label_age_ms=result['image_to_label_age_ms']))
   if len(results)==1:qa(result,a.run/'grasp_label_QA.jpg')
 (a.run/'grasp_labels.json').write_text(json.dumps(results)+'\n');(a.run/'grasp_labels_summary.json').write_text(json.dumps(dict(samples=len(results),errors=errors,split=case['split'],pose_only=True,BC_expert=False),indent=2)+'\n')
 print(json.dumps(dict(samples=len(results),errors=errors)),flush=True)


if __name__=='__main__':main()
