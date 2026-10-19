#!/usr/bin/env python3
"""Observation-only shadow replay of a frozen SFP pause head on saved actor runs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torchvision.models import resnet18
import yaml

from build_two_family_port_projection_labels import sfp_ports
from handoff_region_prior import allows, load_prior
from shared_bc_camera_preprocessing import transport_rgb_tensor
from train_temporal_transport_probe import TemporalTransport, sha

CAMERAS=('center','left','right')


def tensor(values,device):
    return torch.as_tensor(np.asarray(values),dtype=torch.float32,device=device)[None]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--imagenet',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--run',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--external-head',type=Path,
                   help='Optional nonlinear head on frozen hidden plus public state/task')
    p.add_argument('--prior',type=Path,
                   help='Train-derived region veto; required for external head')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior replay')
    torch.set_num_threads(8)
    device=torch.device(a.device)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    manifest=saved['source_manifest']
    if manifest['feature_dim']!=1536 or not manifest['full_state_skip'] or manifest['route_conditioning']:
        raise ValueError('Unsupported frozen-head actor')
    base=torch.load(a.base,map_location='cpu',weights_only=False)
    state_mean=base['model']['state_mean'].numpy()
    state_std=base['model']['state_std'].numpy()
    net=resnet18(weights=None);net.fc=torch.nn.Identity()
    weights=torch.load(a.imagenet,map_location='cpu',weights_only=True)
    net.load_state_dict({k:v for k,v in weights.items() if not k.startswith('fc.')})
    net=net.to(device).eval()
    model=TemporalTransport(saved['model']['target_mean'].numpy(),saved['model']['target_std'].numpy(),
                            raw_state_skip=True,feature_dim=1536,state_task_dim=42,
                            route_conditioning=False).to(device).eval()
    model.load_state_dict(saved['model'])
    external=None;external_saved=None
    if a.external_head:
        external_saved=torch.load(a.external_head,map_location='cpu',weights_only=False)
        if external_saved.get('actor_sha256')!=sha(a.checkpoint):
            raise ValueError('External head frozen actor hash mismatch')
        dim=external_saved['input_dim']
        external=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,128),nn.GELU(),
                               nn.Linear(128,1)).to(device).eval()
        external.load_state_dict(external_saved['model'])
    target_mean=model.target_mean.cpu().numpy()
    target_std=model.target_std.cpu().numpy()
    prior_path=(a.prior or (Path(external_saved['prior']) if external_saved else
                            Path(manifest['handoff_prior_json'])))
    prior=load_prior(prior_path)
    cal=json.loads(a.calibration.read_text())
    results=[]
    for run in a.run:
        trace=[json.loads(s) for s in (run/'runtime_trace.jsonl').open()]
        proposals=sorted((r for r in trace if r.get('event')=='proposal' and len(r.get('state',[]))==32),
                         key=lambda r:r['sim_time'])
        frames=list(csv.DictReader((run/'frame_manifest.csv').open()))
        lookup={(r['camera'],int(r['sim_ns'])):run/'frames'/r['file'] for r in frames}
        stamp_lists={c:np.asarray(sorted(int(r['sim_ns']) for r in frames if r['camera']==c))
                     for c in CAMERAS}
        cfg=yaml.safe_load((run/'eval_config.yaml').read_text())
        tid=next(iter(cfg['trials']))
        truth_ports={(c,q):m for c,q,m in sfp_ports({'trial_yaml':str(run/'eval_config.yaml'),
                                                      'trial_id':tid},cal)}
        records=[];scored=[];previous_state=None;previous_target=None;imputed_frames=0
        for row in proposals:
            sim=float(row['sim_time']);state=np.asarray(row['state'],np.float32)
            task=np.asarray(row['task_vector'],np.float32)
            images=[];image_ages=[]
            for camera in CAMERAS:
                stamp=int(row['image_sim_ns'][camera])
                path=lookup.get((camera,stamp))
                if path is None:
                    available=stamp_lists[camera]
                    closest=int(available[np.argmin(abs(available-stamp))])
                    if abs(closest-stamp)>100_000_000:
                        raise FileNotFoundError(f'No frame within 100 ms: {camera}/{stamp} in {run}')
                    path=lookup[camera,closest];imputed_frames+=1
                images.append(transport_rgb_tensor(np.asarray(Image.open(path).convert('RGB')),
                                                   native_already_encoded=True))
                image_ages.append(max(0.,sim-stamp/1e9)/.05)
            visual=torch.stack(images).to(device).float()/255.
            visual=(visual-visual.new_tensor((.485,.456,.406))[None,:,None,None]) / \
                    visual.new_tensor((.229,.224,.225))[None,:,None,None]
            with torch.inference_mode():feature=net(visual).reshape(1536).float()
            normalized=(state-state_mean)/state_std
            normalized[row['masked_state_indices']]=0
            prev=np.zeros(36,np.float32) if previous_target is None else \
                 ((previous_target-target_mean)/target_std).reshape(-1)
            motion=np.zeros(6,np.float32)
            if previous_state is not None:
                motion[:3]=state[:3]-previous_state[:3]
                motion[3:]=(Rotation.from_quat(previous_state[3:7]).inv()*
                            Rotation.from_quat(state[3:7])).as_rotvec()
            records.append(dict(time=sim,feature=feature,previous=prev,motion=motion,
                                state_task=np.r_[normalized,task],
                                camera_clock=np.r_[image_ages,0.].astype(np.float32)))
            previous_state=state.copy()
            previous_target=np.asarray(row['targets'],np.float32)
            times=np.asarray([r['time'] for r in records]);chosen=[]
            for age in manifest['history_ages_s']:
                j=np.searchsorted(times,sim-age,side='right')-1
                if j>=0 and (age==0 or sim-times[j]<=60) and j not in chosen:chosen.append(int(j))
            chosen.sort();items=[records[j] for j in chosen]
            ages=np.asarray([(sim-r['time'])/4 for r in items],np.float32)
            gaps=np.asarray([0.]+[(items[j]['time']-items[j-1]['time'])/4
                                   for j in range(1,len(items))],np.float32)
            with torch.inference_mode():
                output=model(torch.stack([r['feature'] for r in items])[None],
                    tensor([r['previous'] for r in items],device),
                    tensor([r['motion'] for r in items],device),tensor(ages,device),
                    tensor(gaps,device),tensor([r['camera_clock'] for r in items],device),
                    torch.ones((1,len(items)),device=device,dtype=torch.bool),
                    tensor([r['state_task'] for r in items],device),
                    return_hidden=bool(external))
                action=output[0]
                handoff=(external(torch.cat((output[4],tensor(items[-1]['state_task'],device)),dim=1))[:,0]
                         if external else output[3])
            pred=(action[0].cpu().numpy()*target_std+target_mean).reshape(4,9)
            recorded=np.asarray(row['targets'],np.float32)
            fidelity_mm=float(np.linalg.norm(pred[0,:3]-recorded[0,:3])*1000)
            card=int(task[4:9].argmax());port=int(task[2:4].argmax())
            truth=truth_ports[card,port]
            relative=truth[:3,:3].T@(state[:3]-truth[:3,3])
            axial=float(relative[2]*1000);lateral=float(np.linalg.norm(relative[:2])*1000)
            label=bool(-130<=axial<=-60 and lateral<=30)
            prob=float(torch.sigmoid(handoff)[0].item())
            permitted=bool(allows(prior,task,state[:3]))
            scored.append(dict(time_s=sim,probability=prob,prior_permits=permitted,
                               posthoc_label=label,posthoc_axial_mm=axial,
                               posthoc_lateral_mm=lateral,actor_replay_first_mm=fidelity_mm))
        streak=0;trigger=None
        for row in scored:
            streak=streak+1 if row['probability']>=.95 and row['prior_permits'] else 0
            if streak>=3 and trigger is None:trigger=row
        fidelity=np.asarray([r['actor_replay_first_mm'] for r in scored])
        results.append(dict(run=str(run),proposals=len(scored),
            imputed_frames=imputed_frames,
            trigger=trigger,raw_false_positive_rows=sum(r['probability']>=.95 and
                not r['posthoc_label'] for r in scored),
            accepted_false_positive_rows=sum(r['probability']>=.95 and r['prior_permits'] and
                not r['posthoc_label'] for r in scored),
            true_region_rows=sum(r['posthoc_label'] for r in scored),
            replay_actor_first_mm_p50_p95=np.percentile(fidelity,[50,95]).tolist(),
            rows=scored))
        print(json.dumps({k:v for k,v in results[-1].items() if k!='rows'}),flush=True)
    a.output.write_text(json.dumps(dict(schema='aic_sfp_pause_head_saved_actor_replay/v1',
        checkpoint=str(a.checkpoint),external_head=str(a.external_head) if a.external_head else None,
        prior=str(prior_path),observation_only_inference=True,
        truth_used_posthoc_only=True,final_scenes_opened=False,runs=results),indent=2)+'\n')


if __name__=='__main__':main()
