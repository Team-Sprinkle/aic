#!/usr/bin/env python3
"""Predict plug tip and both SC openings, then select by public task code.

Gazebo TF creates paired training labels only. Validation is grouped by reset
configuration, and no TF is a runtime input.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from build_sc_native_pose_labels import WIDTH,HEIGHT
from train_sc_native_keypoints import PretrainedKeypointRefiner


CAMERAS=('center','left','right')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=2200)
    p.add_argument('--budget-minutes',type=float,default=20.)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing run')
    a.output.mkdir(parents=True)
    torch.set_num_threads(8);random.seed(20261001);np.random.seed(20261001);torch.manual_seed(20261001)
    records=[json.loads(s) for s in a.labels.open()]
    pairs=defaultdict(dict)
    for r in records:
        original=r['trial'].removesuffix('_alternate')
        pairs[(original,r['frame'])]['alternate' if r['trial'].endswith('_alternate') else 'selected']=r
    images=[];targets=[];masks=[];splits=[];near=[];keys=[]
    for (case,frame),pair in pairs.items():
        selected=pair['selected'];alternate=pair.get('alternate')
        if selected['split']=='train' and alternate is None:
            raise ValueError(f'Missing alternate-port training label: {case} {frame}')
        by_port={int(np.argmax(r['task_vector'][2:4])):r for r in pair.values()}
        selected_port=int(np.argmax(selected['task_vector'][2:4]))
        tip_distance=np.linalg.norm(selected['observed_sc_tip_pose_opening_frame'][:3])
        for cam in CAMERAS:
            with Image.open(selected['images'][cam]) as im:
                rgb=im.convert('RGB')
                if rgb.size!=(WIDTH,HEIGHT):raise ValueError(rgb.size)
                small=rgb.resize((288,256),Image.Resampling.BILINEAR)
                images.append(torch.from_numpy(np.asarray(small).copy()).permute(2,0,1))
            points=[selected['projected_training_pixels'][cam]['tip']]
            for port in (0,1):
                points.append(by_port[port]['projected_training_pixels'][cam]['opening']
                              if port in by_port else [None,None,None,False])
            target=[];mask=[]
            for point in points:
                target.extend([(point[j] if point[j] is not None else 0)/
                               ((WIDTH-1) if j==0 else (HEIGHT-1)) for j in (0,1)])
                mask.extend([float(point[3])]*2)
            targets.append(target);masks.append(mask);splits.append(selected['split'])
            near.append(tip_distance<.2);keys.append((case,frame,cam,selected_port))
    x=torch.stack(images);y=torch.as_tensor(np.asarray(targets),dtype=torch.float32)
    mask=torch.as_tensor(np.asarray(masks),dtype=torch.float32)
    train=torch.as_tensor([i for i,s in enumerate(splits) if s=='train'])
    validation=torch.as_tensor([i for i,s in enumerate(splits) if s=='validation'])
    if not len(train) or not len(validation):raise ValueError('Need grouped train and validation')
    train_scenes={r['scene_sha256'] for r in records if r['split']=='train'}
    val_scenes={r['scene_sha256'] for r in records if r['split']=='validation'}
    if train_scenes&val_scenes:raise ValueError('Scene/reset leakage')
    model=PretrainedKeypointRefiner(None,False,point_count=3).cuda()
    old=torch.load(a.initial,map_location='cpu',weights_only=False)['model']
    state=model.state_dict()
    for name,value in state.items():
        if name=='decoder.0.weight':value.copy_(old[name][:,:320])
        elif name=='decoder.4.weight':
            value[0].copy_(old[name][0]);value[1:].copy_(old[name][1])
        elif name=='decoder.4.bias':
            value[0].copy_(old[name][0]);value[1:].copy_(old[name][1])
        elif name in old and old[name].shape==value.shape:value.copy_(old[name])
        else:raise ValueError(f'Unmapped pretrained weight {name}')
    model.load_state_dict(state)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    scale=torch.as_tensor([WIDTH-1,HEIGHT-1],device='cuda')
    @torch.inference_mode()
    def evaluate():
        model.eval();pred=[]
        for ids in validation.split(128):
            pred.append(model(x[ids].cuda().float()/255).float().cpu())
        pred=torch.cat(pred)
        errors=[];by_case=defaultdict(list);all_errors=[]
        for j,i in enumerate(validation.tolist()):
            port=keys[i][3];start=2+2*port
            if mask[i,start]>.5:
                err=float(((pred[j,start:start+2]-y[i,start:start+2])*scale.cpu()).norm())
                all_errors.append(err);by_case[keys[i][0]].append(err)
                if near[i]:errors.append(err)
        return {'near_visible_views':len(errors),'near_selected_opening_px_p50':float(np.median(errors)),
                'near_selected_opening_px_p95':float(np.quantile(errors,.95)),
                'all_selected_opening_px_p95':float(np.quantile(all_errors,.95)),
                'per_case_px_p95':{k:float(np.quantile(v,.95)) for k,v in by_case.items()}}
    history=[];baseline=evaluate();print(json.dumps({'update':0,'validation':baseline}),flush=True)
    best=float('inf');best_step=0;bad=0;started=time.monotonic();reason='update_cap'
    for step in range(1,a.updates+1):
        if time.monotonic()-started>a.budget_minutes*60:reason='wall_budget';break
        ids=train[torch.randint(len(train),(32,))]
        model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(x[ids].cuda().float()/255)
            loss=(F.smooth_l1_loss(out,y[ids].cuda(),beta=.01,reduction='none')*
                  mask[ids].cuda()).sum()/mask[ids].sum().clamp_min(1)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%100==0 or step==a.updates:
            result=evaluate();row={'update':step,'train_loss':float(loss.detach()),'validation':result}
            history.append(row);print(json.dumps(row),flush=True)
            score=result['near_selected_opening_px_p95']
            if score<best-.25:
                best=score;best_step=step;bad=0
                torch.save({'model':model.state_dict(),'point_count':3,
                            'architecture':'dual_sc_port_spatial_heatmaps',
                            'selected_task_code_only_after_both_predictions':True,
                            'final_scenes_opened':False},a.output/'best.pt')
            else:bad+=1
            if step>=500 and bad>=7:reason='validation_plateau';break
    report={'schema':'aic_sc_dual_port_locator/v1','train_scenes':len(train_scenes),
            'validation_scenes':len(val_scenes),'train_views':len(train),
            'validation_views':len(validation),'baseline':baseline,
            'best_step':best_step,'updates_completed':step,'stop_reason':reason,
            'selected':next((h['validation'] for h in history if h['update']==best_step),None),
            'near_definition':'3D plug-tip/opening distance <200 mm; offline checkpoint selection only',
            'teacher_geometry_runtime_input':False,'final_scenes_opened':False}
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
