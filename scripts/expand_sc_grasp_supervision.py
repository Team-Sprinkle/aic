#!/usr/bin/env python3
"""Image-time SC grasp labels from the existing audited BC corpus; no new scenes."""
import cv2
cv2.setNumThreads(2)
import argparse,json,os
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from prepare_sfp_native_grasp_labels import label,qa,matrix,stored_matrix


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8]);a.output.mkdir(exist_ok=False,parents=True)
 source=Path('/home/nas/chmin/aic_archives/2026-10-17_recorded_horizon_actor_rows_v1/h4/rows.jsonl')
 groups=defaultdict(list)
 for l in source.open():
  r=json.loads(l)
  if r['family']!='sc' or r['predicted_tip_port_pose9'][2]>-.05:continue
  if np.linalg.norm(r['state'][7:10])>.010 or np.linalg.norm(r['state'][10:13])>.10:continue
  v=groups[r['episode']]
  if len(v)<12 and (not v or r['time_s']-v[-1]['time_s']>=.15):v.append(r)
 old=Path('/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/sc_grasp_joint_geometry_v6_continuation/joint_geometry/best.pt')
 checkpoint=torch.load(old,map_location='cpu',weights_only=False);centres=checkpoint['plan']['crop_centres_train_only'];origins=np.stack([np.rint(np.asarray(centres[c])-96) for c in ['center','left','right']])
 optics={c:np.asarray(v) for c,v in json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())['tcp_to_optical'].items()}
 typ=get_message('tf2_msgs/msg/TFMessage');manifest=[];poses=[];pixels=[];failures=[]
 for n,(episode,rows) in enumerate(groups.items()):
  stream=Path(rows[0]['source_stream']);d=json.loads(stream.read_text());bag=Path(d['bag'])
  if not bag.exists():bag=stream.parent/'results'/bag.name
  if not bag.exists():raise FileNotFoundError(bag)
  wb=stored_matrix(d['edges']['world|tabletop'][0][1])@stored_matrix(d['edges']['tabletop|base_link'][0][1])
  with next(bag.glob('*.mcap')).open('rb') as f:
   reader=make_reader(f);wa=None
   for _,_,m in reader.iter_messages(topics=['/tf_static']):
    for tf in deserialize_message(m.data,typ).transforms:
     if (tf.header.frame_id,tf.child_frame_id)==('world','aic_world'):wa=matrix(tf.transform)
   if wa is None:raise ValueError('Missing world/aic_world calibration')
   for r in rows:
    item=dict(episode=episode,source_episode_name=episode,scene=r['scene'],split=r['split'],time_s=r['time_s'],images=r['images'],source_stream=str(stream),image_stamps_ns={c:int(Path(v).stem.split('_')[1]) for c,v in r['images'].items()})
    try:result=label(reader,typ,d,item,wa,wb,optics)
    except ValueError as e:failures.append(dict(episode=episode,time_s=r['time_s'],reason=str(e)));continue
    body=np.asarray(result['body_tip_matrix']);pose=np.r_[body[:3,3],body[:3,:2].T.reshape(-1)]
    one=[]
    for ci,c in enumerate(['center','left','right']):
     im=cv2.cvtColor(cv2.imread(r['images'][c]),cv2.COLOR_BGR2RGB);x,y=origins[ci]
     one.append(cv2.warpAffine(im,np.array([[1,0,-x],[0,1,-y]],np.float32),(192,192)))
    manifest.append(dict(item,tip_in_tcp_pose9=pose.tolist(),geometry_label_only=True,GT_crop_selection=False,image_to_label_age_ms=result['image_to_label_age_ms']))
    poses.append(pose);pixels.append(np.stack(one))
    if len(manifest) in [1,12,60]:qa(result,a.output/f'label_QA_{len(manifest)}.jpg')
  (a.output/'progress.json').write_text(json.dumps(dict(episodes=n+1,planned=len(groups),samples=len(manifest),failures=len(failures)))+'\n')
 for key in ['episode','scene']:
  assert not ({r[key] for r in manifest if r['split']=='train'}&{r[key] for r in manifest if r['split']=='validation'})
 (a.output/'manifest.json').write_text(json.dumps(manifest)+'\n');np.save(a.output/'labels.npy',np.asarray(poses));np.save(a.output/'native_crops.npy',np.stack(pixels))
 train=np.stack([np.asarray(r['tip_in_tcp_pose9'])[3:] for r in manifest if r['split']=='train'])
 def rotation6(x):
  first=x[:3];second=x[3:];return np.stack([first,second,np.cross(first,second)],1)
 rotations=Rotation.from_matrix(np.stack([rotation6(x) for x in train]));angles=(rotations.mean().inv()*rotations).magnitude()*180/np.pi
 summary=dict(samples=len(manifest),episodes=len({r['episode'] for r in manifest}),splits={s:dict(samples=sum(r['split']==s for r in manifest),episodes=len({r['episode'] for r in manifest if r['split']==s})) for s in ['train','validation']},training_rotation_distance_to_mean_p50_p95_max_deg=np.percentile(angles,[50,95,100]).tolist(),source=str(source),source_modified=False,final_scenes_opened=False,failures=failures)
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
