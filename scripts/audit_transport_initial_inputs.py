#!/usr/bin/env python3
"""Bounded OFFLINE input substitutions; not privileged autonomous control."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from train_shared_port_tcp_bc import SharedPortTCPActor,relative_tcp_target9
from shared_bc_camera_preprocessing import transport_rgb_tensor

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('audit',type=Path);p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--labels',type=Path,required=True);p.add_argument('--episode',required=True)
a=p.parse_args();torch.set_num_threads(8)
teacher=next(r for s in a.labels.open() if (r:=json.loads(s))['episode_id']==a.episode)
live=next(r for s in (a.audit/'runtime_trace.jsonl').open() if (r:=json.loads(s))['event']=='proposal')
manifest=list(csv.DictReader((a.audit/'frame_manifest.csv').open()))
frames={c:min((r for r in manifest if r['camera']==c),key=lambda r:abs(int(r['sim_ns'])/1e9-live['sim_time'])) for c in ('center','left','right')}
offsets={c:int(r['sim_ns'])/1e9-live['sim_time'] for c,r in frames.items()}
assert max(abs(v) for v in offsets.values())<=.1,'No approximate initial RGB join within 100ms'
def images(paths):
    return torch.stack([transport_rgb_tensor(np.asarray(Image.open(paths[c]).convert('RGB')),
        native_jpeg_quality=86,native_already_encoded=True) for c in ('center','left','right')])
ti=images(teacher['images']);li=images({c:a.audit/'frames'/r['file'] for c,r in frames.items()})
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
stats={k:ck['model'][k].numpy() for k in ('state_mean','state_std','pose_mean','pose_std','target_mean','target_std')}
model=SharedPortTCPActor(stats,pretrained=False,use_time=False,condition_on_pose=False);model.load_state_dict(ck['model']);model.eval()
cases=[('teacher_RGB_teacher_state',ti,teacher['state']),('live_RGB_live_state',li,live['state']),
       ('teacher_RGB_live_state',ti,live['state']),('live_RGB_teacher_state',li,teacher['state'])]
for name,start,end in (('TCP_pose',0,7),('velocity',7,13),('controller_error',13,19),('joints',19,26),('wrench',26,32)):
    state=list(live['state']);state[start:end]=teacher['state'][start:end]
    cases.append(('live_RGB_teacher_'+name,li,state))
with torch.inference_mode():
    pred,_=model(torch.stack([r[1] for r in cases]),torch.tensor([r[2] for r in cases],dtype=torch.float32),
        torch.tensor([teacher['task_vector']]*len(cases),dtype=torch.float32))
outputs={r[0]:pred[i,0].tolist() for i,r in enumerate(cases)}
base_target=teacher['teacher_tcp_target_poses_base_frame'][0]
expect={name:relative_tcp_target9(np.asarray(state[:7]),base_target).tolist() for name,_,state in cases}
result=dict(schema='aic_transport_initial_input_substitution/v1',offline_only=True,
    reset_scene=a.episode,teacher_frame=teacher['frame'],approximate_live_image_offsets_s=offsets,
    live_RGB_join_exact=False,teacher_states_already_in_motion=True,
    targets_predicted=outputs,recorded_first_teacher_target_in_each_state_frame=expect,
    translation_error_to_recorded_target_mm={name:float(np.linalg.norm(np.array(v[:3])-expect[name][:3])*1000) for name,v in outputs.items()},
    reconstructed_live_first_target_difference_mm=float(np.linalg.norm(pred[1,0,:3].numpy()-np.array(live['targets'][0][:3]))*1000),
    caveat='Offline state/image substitution is not a matched physical counterfactual or full route metric. Approximate saved RGB join must be disclosed.',
    new_data_collected=False,training_updates=0,final_scenes_opened=False)
(a.audit/'initial_input_substitution.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
