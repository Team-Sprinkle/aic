#!/usr/bin/env python3
"""Add causal image-estimated plug positions; preserve expert action labels."""
import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np,torch
from sc_native_grasp_observation import SCNativeGraspObservation
from shared_port_tcp_geometry import matrix_from_pose9,matrix_from_xyz_xyzw

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--transport',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('Preserve source data')
 a.output.mkdir();torch.set_num_threads(4);cv2.setNumThreads(2);rows=[json.loads(l) for l in (a.dataset/'rows.jsonl').open()];groups=defaultdict(list)
 for r in map(json.loads,a.transport.open()):groups[r['episode_id']].append(r)
 model=SCNativeGraspObservation(a.checkpoint);locks={};raw_times={}
 for episode in sorted({r['episode'] for r in rows if r['family']=='sc'}):
  candidates=groups[episode];end=max(r['sim_time'] for r in candidates);chosen=[max((r for r in candidates if r['sim_time']<=end-age),key=lambda x:x['sim_time']) for age in [.4,.2,0]];pred=[]
  for r in chosen:
   images={c:cv2.cvtColor(cv2.imread(r['images'][c]),cv2.COLOR_BGR2RGB) for c in ['center','left','right']};pred.append(model.predict(images,already_training_jpeg=True))
  position=np.median([p['predicted_tip_position_tcp_m'] for p in pred],axis=0);locks[episode]={'body_tip_position_m':position.tolist(),'prediction_time_s':max(min(r['image_sim_ns'].values()) for r in chosen)/1e9,'samples':[{'frame':r['frame'],'command_time_s':r['sim_time'],'image_sim_ns':r['image_sim_ns'],'images':r['images'],'prediction':p} for r,p in zip(chosen,pred)],'geometry_input':False};print(episode,'locked',flush=True)
 output=[]
 for r in rows:
  value=dict(r)
  if r['family']=='sc':
   lock=locks[r['episode']];image_stamp=min(int(Path(p).stem.split('_')[-2]) for p in r['images'].values())/1e9
   assert lock['prediction_time_s']<=image_stamp+1e-9,'Future image enters actor input'
   current=matrix_from_pose9(r['predicted_port_tcp_pose9']);position=current[:3,:3]@np.asarray(lock['body_tip_position_m'])+current[:3,3];tip=[*position,1.,0.,0.,0.,1.,0.];value.update(predicted_tip_port_pose9=tip,predicted_tip_valid=True,tip_orientation_input=False,tip_lock_image_time_s=lock['prediction_time_s'])
   base_tcp=matrix_from_xyz_xyzw(r['state'][:7]);base_port=base_tcp@np.linalg.inv(current);assert np.linalg.norm((base_port@np.r_[position,1])[:3]-(base_tcp@np.r_[lock['body_tip_position_m'],1])[:3])<1e-10
  else:value.update(predicted_tip_port_pose9=[0.,0.,0.,1.,0.,0.,0.,1.,0.],predicted_tip_valid=False,tip_orientation_input=False)
  assert value['teacher_tcp_targets_predicted_port_pose9']==r['teacher_tcp_targets_predicted_port_pose9'];output.append(value)
 with (a.output/'rows.jsonl').open('w') as f:
  for r in output:f.write(json.dumps(r,separators=(',',':'))+'\n')
 (a.output/'grasp_locks.json').write_text(json.dumps(locks,indent=2)+'\n');(a.output/'summary.json').write_text(json.dumps({'schema':'aic_translation_conditioned_local_bc/v1','original_dataset':str(a.dataset),'source_rows_sha256':hashlib.sha256((a.dataset/'rows.jsonl').read_bytes()).hexdigest(),'grasp_checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'rows':len(output),'sc_episodes':len(locks),'tip_inputs':'SCimage-estimatedtranslationonly; SFPmaskfalse. No trueTIPpose or orientation enters actor.','actions_unchanged':True,'image_locks_causal':True,'frame_backconversion_verified':True,'full_grasp_orientation_gate_passed':False,'final_scenes_opened':False},indent=2)+'\n')
if __name__=='__main__':main()
