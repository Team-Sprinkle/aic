#!/usr/bin/env python3
"""Reuse six predeadline-sample development acquisitions, grouped by complete reset."""
import cv2
cv2.setNumThreads(2)
import argparse,bisect,json,os
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from prepare_sfp_native_grasp_labels import label,qa,matrix,stored_matrix
from shared_port_tcp_geometry import matrix_from_pose9


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8]);a.output.mkdir(parents=True,exist_ok=False)
 manifest=json.loads((a.source/'manifest.json').read_text());poses=list(np.load(a.source/'labels.npy'));pixels=list(np.load(a.source/'native_crops.npy',mmap_mode='r'))
 old=Path('/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/sc_grasp_joint_geometry_v6_continuation/joint_geometry/best.pt')
 saved=torch.load(old,map_location='cpu',weights_only=False);centres=saved['plan']['crop_centres_train_only'];origins={c:np.rint(np.asarray(centres[c])-96) for c in ['center','left','right']}
 optics={c:np.asarray(v) for c,v in json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())['tcp_to_optical'].items()}
 root=Path('/home/nas/chmin/aic_archives/2026-10-16_sample_conservative_development_v3/SC_adaptive_route_all7_v7')
 runs={r['case']:Path(r['audit']) for r in json.loads((root/'summary.json').read_text())['results']}
 plan=json.loads(Path('/home/nas/chmin/aic_archives/2026-10-16_sample_conservative_development_v3/plan.json').read_text())
 cases=[c for c in plan['cases'] if c['family']=='sc' and c['split']=='development'];assert len(cases)==6
 typ=get_message('tf2_msgs/msg/TFMessage');added=[]
 for case in cases:
  run=runs[case['case']];d=json.loads((run/'scored_stream.json').read_text());assert d['task']['plug_type']=='sc'
  split='validation' if (case['card_count'],case['port']) in [(0,1),(1,0)] else 'train'
  episode='sampledev_grasp_'+case['case'];scene=d['scene_group_id']
  assert not any(r['scene']==scene for r in manifest),'Existing reset must preserve its split'
  trace=[json.loads(l) for l in (run/'runtime_trace.jsonl').open()];terminal=next(e['sim_time'] for e in reversed(trace) if e['event']=='terminal_before_return')
  controls=d['streams']['/aic_controller/controller_state'];times=[x[0] for x in controls];cameras=d['cameras'];quiet=[]
  for stamp,path in cameras['center']:
   if not terminal-4<=stamp/1e9<=terminal:continue
   j=bisect.bisect_right(times,stamp)-1
   if j<0:continue
   v=controls[j][1]['tcp_velocity'];linear=np.linalg.norm([v['linear'][k] for k in 'xyz']);angular=np.linalg.norm([v['angular'][k] for k in 'xyz'])
   if linear<=.01 and angular<=.1:quiet.append((stamp,path))
  chosen=[quiet[i] for i in np.linspace(0,len(quiet)-1,min(12,len(quiet)),dtype=int)];assert len(chosen)>=3
  wb=stored_matrix(d['edges']['world|tabletop'][0][1])@stored_matrix(d['edges']['tabletop|base_link'][0][1]);bag=Path(d['bag']);counts=0
  with next(bag.glob('*.mcap')).open('rb') as f:
   reader=make_reader(f);wa=None
   for _,_,m in reader.iter_messages(topics=['/tf_static']):
    for tf in deserialize_message(m.data,typ).transforms:
     if (tf.header.frame_id,tf.child_frame_id)==('world','aic_world'):wa=matrix(tf.transform)
   if wa is None:raise ValueError('Missing calibration')
   for stamp,_ in chosen:
    images={};stamps={}
    for c in ['center','left','right']:
     j=bisect.bisect_right(cameras[c],stamp,key=lambda x:x[0])-1
     if j<0:raise ValueError('No causal image')
     stamps[c],images[c]=cameras[c][j]
    item=dict(family='sc',episode=episode,source_episode_name=episode,scene=scene,split=split,time_s=stamp/1e9,images=images,image_stamps_ns=stamps,source_stream=str(run/'scored_stream.json'))
    try:truth=label(reader,typ,d,item,wa,wb,optics)
    except ValueError:continue
    body=np.asarray(truth['body_tip_matrix']);pose=np.r_[body[:3,3],body[:3,:2].T.reshape(-1)];one=[]
    for c in ['center','left','right']:
     rgb=cv2.cvtColor(cv2.imread(images[c]),cv2.COLOR_BGR2RGB);x,y=origins[c];one.append(cv2.warpAffine(rgb,np.array([[1,0,-x],[0,1,-y]],np.float32),(192,192)))
    manifest.append(dict(item,tip_in_tcp_pose9=pose.tolist(),geometry_label_only=True,GT_crop_selection=False,image_to_label_age_ms=truth['image_to_label_age_ms'],source_role='Predeadline sample-derived development acquisition; never reserved final or released-evaluation scene'))
    poses.append(pose);pixels.append(np.stack(one));counts+=1
    if counts==1:qa(truth,a.output/(case['case']+'_label_QA.jpg'))
  added.append(dict(case=case['case'],split=split,scene=scene,samples=counts));assert counts>=3
 (a.output/'manifest.json').write_text(json.dumps(manifest)+'\n');np.save(a.output/'labels.npy',np.asarray(poses));np.save(a.output/'native_crops.npy',np.stack(pixels))
 rotations=Rotation.from_matrix(np.stack([matrix_from_pose9(r['tip_in_tcp_pose9'])[:3,:3] for r in manifest if r['split']=='train']));angles=(rotations.mean().inv()*rotations).magnitude()*180/np.pi
 for key in ['episode','scene']:
  assert not ({r[key] for r in manifest if r['split']=='train'}&{r[key] for r in manifest if r['split']=='validation'})
 report=dict(samples=len(manifest),episodes=len({r['episode'] for r in manifest}),added=added,train_rotation_distance_to_mean_p50_p95_max_deg=np.percentile(angles,[50,95,100]).tolist(),reserved_final_scenes_opened=False,postrelease_data_used=False,BC_expert_admission=False)
 (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
