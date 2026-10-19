#!/usr/bin/env python3
"""Re-express recorded SC TCP target translations as plug-tip targets.

True grasp translation is label-only. Inference converts with the camera
estimate. TCP orientation targets remain unchanged; no residual actor exists.
"""
import argparse,json,hashlib
from pathlib import Path
from bisect import bisect_right
from collections import defaultdict
import numpy as np
from train_sc_grasp_pose_probe import interpolate
from analyze_local_actor_scored_tip import record_at
from shared_port_tcp_geometry import matrix_from_xyz_xyzw,matrix_from_pose9

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--raw-local',type=Path,required=True);p.add_argument('--streams',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('Preserve earlier action labels')
 a.output.mkdir();first={}
 for row in map(json.loads,a.raw_local.open()):first.setdefault(row['episode_id'],row)
 body={}
 for episode,row in sorted(first.items()):
  stream=json.loads((a.streams/f'{episode}.json').read_text());e=stream['edges'];controllers=stream['streams']['/aic_controller/controller_state'];image_ns=min(row['image_sim_ns'].values());j=bisect_right([x[0] for x in controllers],image_ns)-1;t=controllers[j][0];wb=record_at(e['world|tabletop'],0)@record_at(e['tabletop|base_link'],0);tcp=matrix_from_xyz_xyzw(row['state'][:7]);tip=np.linalg.inv(wb@tcp)@interpolate(e['aic_world|cable_1'],t)@interpolate(e['cable_1|cable_1/sc_tip_link'],t);body[episode]={'translation':tip[:3,3].tolist(),'controller_stamp_ns':t,'label_only':True};print(episode,'label calibrated',flush=True)
 old=[];new=[];by_episode=defaultdict(lambda:[[],[]]);roundtrip=0;rows=[]
 for row in map(json.loads,(a.source/'rows.jsonl').open()):
  r=dict(row)
  if r['family']=='sc':
   offset=np.asarray(body[r['episode']]['translation']);targets=[]
   for pose in r['teacher_tcp_targets_predicted_port_pose9']:
    tcp=matrix_from_pose9(pose);point=tcp[:3,3]+tcp[:3,:3]@offset;value=list(pose);value[:3]=point.tolist();targets.append(value);roundtrip=max(roundtrip,float(np.linalg.norm(point-tcp[:3,:3]@offset-tcp[:3,3])))
   if abs(r['predicted_port_tcp_pose9'][2])<.03:
    old.append(r['teacher_tcp_targets_predicted_port_pose9'][0][:2]);new.append(targets[0][:2]);by_episode[r['episode']][0].append(old[-1]);by_episode[r['episode']][1].append(new[-1])
   r['recorded_tcp_targets_predicted_port_pose9']=r['teacher_tcp_targets_predicted_port_pose9'];r['teacher_tcp_targets_predicted_port_pose9']=targets;r['action_reference']='SCplug_tip_translation_and_TCP_orientation';r['action_label_true_grasp_translation_only']=True
  else:r['action_reference']='SFP_TCP_translation_and_TCP_orientation'
  rows.append(r)
 with (a.output/'rows.jsonl').open('w') as f:
  for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
 (a.output/'label_grasp_calibrations.json').write_text(json.dumps(body,indent=2)+'\n')
 # Only input predictions, never label calibrations, are used by the actor.
 (a.output/'grasp_locks.json').write_text((a.source/'grasp_locks.json').read_text())
 summary={'schema':'aic_sc_tip_target_action_labels/v1','source':str(a.source),'source_rows_sha256':hashlib.sha256((a.source/'rows.jsonl').read_bytes()).hexdigest(),'rows':len(rows),'episodes':len(body),'reference_change':'SC targettranslation is plugtip, rotation stillTCP. SFP unchanged. Completefuturetargets, not delta/residual actions.','label_grasp_geometry_input':False,'runtime_inverse':'TCP_target_xyz = tip_target_xyz - TCP_target_rotation @ RGB_body_tip_position','max_recorded_TCP_backconversion_error_m':roundtrip,'near_target_lateral_std_mm':{'TCP':(np.std(old,axis=0)*1000).tolist(),'plug_tip':(np.std(new,axis=0)*1000).tolist()},'episode_median_lateral_std_mm':{'TCP':(np.std([np.median(v[0],axis=0) for v in by_episode.values()],axis=0)*1000).tolist(),'plug_tip':(np.std([np.median(v[1],axis=0) for v in by_episode.values()],axis=0)*1000).tolist()},'final_scenes_opened':False};assert roundtrip<1e-12;(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
