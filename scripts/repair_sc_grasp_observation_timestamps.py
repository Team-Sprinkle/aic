#!/usr/bin/env python3
"""Correct command/observation timestamp joins in a versioned grasp probe copy."""
import argparse,json,hashlib
from pathlib import Path
from bisect import bisect_right
import numpy as np
from scipy.spatial.transform import Rotation
from train_sc_grasp_pose_probe import interpolate,pose9
from analyze_local_actor_scored_tip import record_at
from shared_port_tcp_geometry import matrix_from_xyz_xyzw,matrix_from_pose9

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--raw-chunks',type=Path,required=True);p.add_argument('--streams',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('Preserve original and earlier repairs')
 a.output.mkdir();manifest=json.loads((a.source/'manifest.json').read_text());original=np.load(a.source/'labels.npy');wanted={(r['episode'],r['frame']) for r in manifest};raw={}
 for r in map(json.loads,a.raw_chunks.open()):
  k=(r['episode_id'],r['frame'])
  if k in wanted:raw[k]=r
 groups={}
 for i,r in enumerate(manifest):groups.setdefault(r['episode'],[]).append(i)
 updated=manifest.copy();labels=original.copy();audit=[]
 for episode,ids in groups.items():
  stream=json.loads((a.streams/f'{episode}.json').read_text());e=stream['edges'];wb=record_at(e['world|tabletop'],0)@record_at(e['tabletop|base_link'],0);controllers=stream['streams']['/aic_controller/controller_state'];stamps=[x[0] for x in controllers]
  for i in ids:
   row=manifest[i];source=raw[episode,row['frame']];image_ns=min(source['image_sim_ns'].values());j=bisect_right(stamps,image_ns)-1
   if j<0:raise ValueError('No causal controller observation')
   state_ns,message=controllers[j];assert image_ns-state_ns<=100_000_000
   tcp=matrix_from_xyz_xyzw(source['state'][:7]);p=message['tcp_pose']['position'];q=message['tcp_pose']['orientation'];recorded=matrix_from_xyz_xyzw([p[k] for k in 'xyz']+[q[k] for k in 'xyzw']);assert np.max(abs(tcp-recorded))<1e-7,'State provenance does not match adapter join'
   wt=interpolate(e['aic_world|cable_1'],state_ns)@interpolate(e['cable_1|cable_1/sc_tip_link'],state_ns);correct=np.linalg.inv(wb@tcp)@wt;labels[i]=pose9(correct);old=matrix_from_pose9(original[i]);audit.append({'episode':episode,'frame':row['frame'],'split':row['split'],'command_time_s':source['sim_time'],'image_time_s':image_ns/1e9,'controller_time_s':state_ns/1e9,'command_minus_controller_ms':(source['sim_time']*1e9-state_ns)/1e6,'image_minus_controller_ms':(image_ns-state_ns)/1e6,'label_change_mm':float(np.linalg.norm(correct[:3,3]-old[:3,3])*1000),'label_change_deg':float((Rotation.from_matrix(old[:3,:3]).inv()*Rotation.from_matrix(correct[:3,:3])).magnitude()*180/np.pi)})
   updated[i]={**row,'command_time_s':row['time_s'],'time_s':state_ns/1e9,'observation_image_sim_ns':image_ns,'observation_controller_sim_ns':state_ns,'tip_in_tcp_pose9':labels[i].tolist(),'label_join':'TIP TF interpolated at recorded controller timestamp; identical causal state matched explicitly'}
  print(episode,'repaired',flush=True)
 np.save(a.output/'labels.npy',labels);(a.output/'manifest.json').write_text(json.dumps(updated,indent=2)+'\n');plan=json.loads((a.source/'frozen_plan.json').read_text());plan.update(schema='aic_grasp_observation_timestamp_repair/v3',parent=str(a.source),parent_manifest_sha256=hashlib.sha256((a.source/'manifest.json').read_bytes()).hexdigest(),label_contract='Grasp in measured controller TCP at its own timestamp; command time stored separately; no postreset observations')
 (a.output/'frozen_plan.json').write_text(json.dumps(plan,indent=2)+'\n');within=[]
 for episode,ids in groups.items():within.extend(np.linalg.norm(labels[ids,:3]-labels[ids,:3].mean(0),axis=1)*1000)
 summary={'rows':audit,'samples':len(audit),'original_preserved':True,'command_minus_controller_ms_p50_p95_max':np.percentile([r['command_minus_controller_ms'] for r in audit],[50,95,100]).tolist(),'label_change_mm_p50_p95_max':np.percentile([r['label_change_mm'] for r in audit],[50,95,100]).tolist(),'corrected_within_episode_translation_mm_p50_p95':np.percentile(within,[50,95]).tolist(),'final_scenes_opened':False};(a.output/'timestamp_audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
