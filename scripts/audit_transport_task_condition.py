#!/usr/bin/env python3
"""Offline same-observation selected-port swap for a frozen transport actor."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import torch
from train_shared_port_tcp_bc import SharedPortTCPActor
from shared_bc_camera_preprocessing import transport_rgb_tensor
p=argparse.ArgumentParser(description=__doc__);p.add_argument('audit',type=Path);p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args();torch.set_num_threads(8)
trace=[json.loads(s) for s in (a.audit/'runtime_trace.jsonl').open()];v=[r for r in trace if r['event']=='proposal'];frames=list(csv.DictReader((a.audit/'frame_manifest.csv').open()));ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);stats={k:ck['model'][k].numpy() for k in ('state_mean','state_std','pose_mean','pose_std','target_mean','target_std')};m=SharedPortTCPActor(stats,pretrained=False,use_time=False,condition_on_pose=False);m.load_state_dict(ck['model']);m.masked_state_indices=ck.get('masked_state_indices',[]);m.eval()
out=[]
for elapsed in (0,20,35,45,60):
 row=min(v,key=lambda x:abs(x['sim_time']-v[0]['sim_time']-elapsed));paths={c:min((f for f in frames if f['camera']==c),key=lambda f:abs(int(f['sim_ns'])/1e9-row['sim_time'])) for c in ('center','left','right')};offsets={c:int(f['sim_ns'])/1e9-row['sim_time'] for c,f in paths.items()};assert max(abs(x) for x in offsets.values())<=.1
 image=torch.stack([transport_rgb_tensor(np.asarray(Image.open(a.audit/'frames'/paths[c]['file']).convert('RGB')),native_jpeg_quality=86,native_already_encoded=True) for c in ('center','left','right')]);task0=[0.,1.,1.,0.,0.,0.,0.,0.,0.,0.];task1=[0.,1.,0.,1.,0.,0.,0.,0.,0.,0.]
 with torch.inference_mode():pred,_=m(torch.stack((image,image)),torch.tensor([row['state']]*2,dtype=torch.float32),torch.tensor([task0,task1],dtype=torch.float32))
 d=pred[:,0,:3].numpy();r=Rotation.from_quat(row['state'][3:7]);base=r.apply(d);out.append(dict(elapsed_s=elapsed,sim_time=row['sim_time'],camera_offset_s=offsets,tcp_base=row['state'][:3],body_first_target_mm=(d*1000).tolist(),base_first_target_mm=(base*1000).tolist(),port_task_effect_mm=float(np.linalg.norm(base[0]-base[1])*1000),base_y_effect_mm=float((base[1,1]-base[0,1])*1000)))
result=dict(schema='aic_transport_task_condition_audit/v1',offline_only=True,inputs='same saved RGB/robot state, only selected SC port code switched',checkpoint=str(a.checkpoint),samples=out,exact_rgb_join=False,final_scenes_opened=False);(a.audit/'task_condition_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
