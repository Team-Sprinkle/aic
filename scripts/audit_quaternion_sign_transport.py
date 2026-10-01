#!/usr/bin/env python3
"""Test physical-quaternion sign invariance of a frozen actor offline."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from train_shared_port_tcp_bc import SharedPortTCPActor
from shared_bc_camera_preprocessing import transport_rgb_tensor
p=argparse.ArgumentParser(description=__doc__);p.add_argument('audit',type=Path);p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args();torch.set_num_threads(8)
r=[json.loads(s) for s in (a.audit/'runtime_trace.jsonl').open()];r=[x for x in r if x['event']=='proposal'];q=np.array([x['state'][3:7] for x in r]);flip=np.where(np.sum(q[1:]*q[:-1],axis=1)<0)[0]
assert len(flip), 'No observed sign flip in this rollout'
indices=sorted(set([0,int(flip[0]),int(flip[0]+1),int(flip[0]+2)]))
manifest=list(csv.DictReader((a.audit/'frame_manifest.csv').open()));ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
stats={k:ck['model'][k].numpy() for k in ('state_mean','state_std','pose_mean','pose_std','target_mean','target_std')}
model=SharedPortTCPActor(stats,pretrained=False,use_time=False,condition_on_pose=False);model.load_state_dict(ck['model']);model.masked_state_indices=ck.get('masked_state_indices',[]);model.eval();reports=[]
for i in indices:
 x=r[i];paths={c:min((f for f in manifest if f['camera']==c),key=lambda f:abs(int(f['sim_ns'])/1e9-x['sim_time'])) for c in ('center','left','right')}
 offsets={c:int(f['sim_ns'])/1e9-x['sim_time'] for c,f in paths.items()};assert max(abs(v) for v in offsets.values())<=.1
 image=torch.stack([transport_rgb_tensor(np.asarray(Image.open(a.audit/'frames'/paths[c]['file']).convert('RGB')),native_jpeg_quality=86,native_already_encoded=True) for c in ('center','left','right')])
 s=np.asarray(x['state'],dtype=np.float32);alt=s.copy();alt[3:7]*=-1
 with torch.inference_mode():v,_=model(torch.stack((image,image)),torch.from_numpy(np.stack((s,alt))),torch.tensor([x['task_vector']]*2,dtype=torch.float32))
 reports.append(dict(sim_time=x['sim_time'],q_xyzw=s[3:7].tolist(),camera_offset_s=offsets,raw_first_target_mm=(v[0,0,:3].numpy()*1000).tolist(),sign_flipped_first_target_mm=(v[1,0,:3].numpy()*1000).tolist(),translation_output_difference_mm=float(torch.linalg.norm(v[0,0,:3]-v[1,0,:3])*1000)))
out=dict(schema='aic_quaternion_sign_invariance_audit/v1',signs_represent_identical_physical_orientation=True,offline_only=True,checkpoint=str(a.checkpoint),observed_sign_flip_count=len(flip),samples=reports,exact_rgb_join=False,geometry_input=False,final_scenes_opened=False)
(a.audit/'quaternion_sign_invariance.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
