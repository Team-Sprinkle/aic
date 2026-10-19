#!/usr/bin/env python3
"""Multi-target RGB projection locator for SFP-to-NIC and SC-to-SC.

The presence head learns only *projected in-frame*, never occlusion visibility.
Each image supervises every physically present candidate port, making the
nonselected ports explicit counterfactual targets. Grouped scene splits are
fixed by the label manifest. No geometry is a model input.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from torchvision import models

from act_cache import load_cached_images


CAMERAS=('center','left','right')
WIDTH,HEIGHT=1152,1024
NAMES=[f'sc_port_{p}' for p in (0,1)]+[
    f'sfp_card_{c}_port_{p}' for c in range(5) for p in (0,1)]
LOOKUP={name:i for i,name in enumerate(NAMES)}


class Locator(nn.Module):
    def __init__(self,weights):
        super().__init__()
        self.backbone=models.resnet18(weights=None)
        if weights is not None:
            self.backbone.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
        for name,param in self.backbone.named_parameters():
            param.requires_grad_(name.startswith('layer3.'))
        self.decoder=nn.Sequential(nn.Conv2d(64+256+5,128,3,padding=1),nn.GELU(),
                                   nn.Conv2d(128,64,3,padding=1),nn.GELU(),nn.Conv2d(64,12,1))
        self.inframe=nn.Sequential(nn.Linear(256+5,128),nn.GELU(),nn.Linear(128,12))
        self.register_buffer('mean',torch.tensor((.485,.456,.406))[None,:,None,None])
        self.register_buffer('std',torch.tensor((.229,.224,.225))[None,:,None,None])

    def train(self,mode=True):
        super().train(mode)
        self.backbone.bn1.eval();self.backbone.layer1.eval();self.backbone.layer2.eval()
        return self

    def forward(self,rgb,meta,return_features=False,return_heatmap=False,
                return_spatial=False):
        x=(rgb-self.mean)/self.std
        x=self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        early=self.backbone.layer1(x)
        deep=self.backbone.layer3(self.backbone.layer2(early))
        joined=torch.cat((early,F.interpolate(deep,size=early.shape[-2:],mode='bilinear',align_corners=False),
                          meta[:,:,None,None].expand(-1,-1,*early.shape[-2:])),dim=1)
        heat=self.decoder(joined)
        b,k,h,w=heat.shape
        probs=F.softmax(heat.reshape(b,k,-1).float()*4,dim=-1)
        y,x=torch.meshgrid(torch.linspace(0,1,h,device=rgb.device),
                           torch.linspace(0,1,w,device=rgb.device),indexing='ij')
        coords=torch.stack(((probs*x.flatten()).sum(-1),(probs*y.flatten()).sum(-1)),-1)
        flag=self.inframe(torch.cat((deep.mean((-1,-2)),meta),-1))
        if return_features:
            feature=torch.cat((deep.mean((-1,-2)),self.backbone.layer4(deep).mean((-1,-2))),dim=-1)
            if return_spatial:return coords,flag,feature,deep
            return coords,flag,feature
        if return_heatmap:return coords,flag,heat
        return coords,flag


class Views(Dataset):
    def __init__(self,rows,cache,indices,training):
        self.rows=rows;self.indices=indices;self.training=training
        self.cached=None
        if cache is not None:
            info=json.loads((cache/'cache.json').read_text())
            self.cached=load_cached_images(cache,info)

    def __len__(self):return len(self.indices)

    def __getitem__(self,ordinal):
        index,forced_cam=self.indices[ordinal];r=self.rows[index]
        cam=random.randrange(3) if forced_cam is None else forced_cam
        name=CAMERAS[cam]
        if r.get('image_kind') in ('native_1152x1024_rgb_jpeg','agent_288x256_rgb_jpeg') or r['family']=='sc':
            image=cv2.imread(r['images'][name],cv2.IMREAD_COLOR)
            if image is None:raise FileNotFoundError(r['images'][name])
            image=cv2.resize(image,(288,256),interpolation=cv2.INTER_AREA)
            if r['family']=='sfp':
                ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])
                if not ok:raise RuntimeError('SFP runtime JPEG preprocessing failed')
                image=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        elif r['family']=='sfp':
            if self.cached is None:raise ValueError('Low-resolution SFP images require --sfp-cache')
            key=f'observation.images.{name}_camera'
            image=np.ascontiguousarray(self.cached[key][r['images'][name]])
        points=np.zeros((12,2),np.float32);present=np.zeros(12,np.float32);inframe=np.zeros(12,np.float32)
        for t in r['targets']:
            j=LOOKUP[t['identity']];v=t['cameras'][name];present[j]=1.
            if v[3]:
                inframe[j]=1.;points[j]=[v[0]/(WIDTH-1),v[1]/(HEIGHT-1)]
        family=0 if r['family']=='sc' else 1
        meta=np.zeros(5,np.float32);meta[family]=1.;meta[2+cam]=1.
        selected=LOOKUP[r['selected_identity']]
        distance=np.linalg.norm(np.asarray(next(t for t in r['targets'] if t['identity']==r['selected_identity'])['position_base_m'])-r['state'][:3])
        return (torch.from_numpy(image.copy()).permute(2,0,1),torch.from_numpy(meta),
                torch.from_numpy(points),torch.from_numpy(present),torch.from_numpy(inframe),
                selected,float(distance),index,cam)


@torch.inference_mode()
def evaluate(model,loader,device,max_batches=None):
    model.eval();errors=defaultdict(list);flags=defaultdict(list);wrong=[];confused=defaultdict(list);total=0
    for batchno,batch in enumerate(loader):
        if max_batches is not None and batchno>=max_batches:break
        image,meta,y,present,inframe,selected,distance,*_=batch
        image=image.to(device).float()/255.;meta=meta.to(device)
        pred,logit=model(image,meta)
        pred=pred.cpu().numpy();logit=logit.sigmoid().cpu().numpy()
        y=y.numpy();present=present.numpy();inframe=inframe.numpy();selected=selected.numpy();distance=distance.numpy()
        families=np.argmax(meta.cpu().numpy()[:,:2],axis=1)
        for i in range(len(pred)):
            family='sc' if families[i]==0 else 'sfp';j=int(selected[i]);total+=1
            for k in np.flatnonzero(present[i]):
                flags[(family,'positive' if inframe[i,k] else 'negative')].append((logit[i,k],inframe[i,k]))
                if not inframe[i,k]:continue
                error=float(np.linalg.norm((pred[i,k]-y[i,k])*[WIDTH-1,HEIGHT-1]))
                typ='selected' if k==j else 'alternate'
                errors[(family,typ,'near' if distance[i]<.2 else 'far')].append(error)
                if k==j:
                    errors[(family,'selected','all')].append(error)
                    others=[np.linalg.norm((pred[i,j]-y[i,o])*[WIDTH-1,HEIGHT-1]) for o in np.flatnonzero(inframe[i]) if o!=j]
                    if others:
                        wrong.append(float(min(others)))
                        confused[family].append(float(min(others)<error))
    summary={str(k):{'n':len(v),'p50_px':float(np.median(v)),'p95_px':float(np.quantile(v,.95))} for k,v in errors.items() if v}
    cls={}
    for family in ('sc','sfp'):
        pairs=[v for key,values in flags.items() if key[0]==family for v in values]
        if not pairs:continue
        prob=np.asarray([x[0] for x in pairs]);truth=np.asarray([x[1] for x in pairs])>.5
        predicted=prob>=.5
        cls[family]={'rows':len(pairs),'precision':float(np.sum(predicted&truth)/max(np.sum(predicted),1)),
                     'recall':float(np.sum(predicted&truth)/max(np.sum(truth),1)),
                     'false_positive_rate':float(np.sum(predicted&~truth)/max(np.sum(~truth),1))}
    return {'images':total,'errors':summary,'projected_inframe_classification':cls,
            'selected_point_closer_to_alternate_fraction':{
                fam:float(np.mean(values)) for fam,values in confused.items()},
            'wrong_target_distance_from_selected_prediction_px_p50':float(np.median(wrong)) if wrong else None}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--sfp-cache',type=Path,required=True)
    p.add_argument('--imagenet',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=3500)
    p.add_argument('--minutes',type=float,default=60.)
    p.add_argument('--batch-size',type=int,default=24)
    p.add_argument('--extra-labels',type=Path,help='Grouped native RGB rows; retained separately for checkpoint selection')
    p.add_argument('--init-checkpoint',type=Path,help='Continue the shared locator from a prior checkpoint')
    p.add_argument('--native-source-fraction',type=float,default=1/3,
                   help='Fraction of minibatch views drawn from new native SFP scenes')
    p.add_argument('--native-selected-weight',type=float,default=1.,
                   help='Extra loss weight on the public-task-selected port in native SFP scenes')
    p.add_argument('--heatmap-loss-weight',type=float,default=0.,
                   help='Spatial cross-entropy on labeled openings; zero preserves coordinate-only baseline')
    p.add_argument('--structured-loss-weight',type=float,default=0.,
                   help='Pairwise relative-position loss for neighboring ports and cards')
    p.add_argument('--supervise-absent-identity',action='store_true',
                   help='Train in-frame flags to reject absent card identities within each task family')
    p.add_argument('--learning-rate',type=float,
                   help='Override optimizer learning rate for a bounded checkpoint continuation')
    p.add_argument('--focus-episode-substring',type=str,
                   help='Oversample a prespecified training episode group with a specific visual failure')
    p.add_argument('--focus-fraction',type=float,default=0.,
                   help='Minibatch probability allocated to matching training episodes')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing run')
    a.output.mkdir(parents=True)
    random.seed(20261002);np.random.seed(20261002);torch.manual_seed(20261002);torch.set_num_threads(8)
    rows=[json.loads(line) for line in (a.labels/'rows.jsonl').open()]
    native_start=len(rows)
    if a.extra_labels:
        rows.extend(json.loads(line) for line in (a.extra_labels/'rows.jsonl').open())
    if len({r['scene_group'] for r in rows if r['split']=='train'} &
           {r['scene_group'] for r in rows if r['split']=='validation'}):
        raise ValueError('Train/validation reset group leakage')
    train_indices=[(i,None) for i,r in enumerate(rows) if r['split']=='train']
    validation_indices=[(i,c) for i,r in enumerate(rows) if r['split']=='validation'
                        and i<native_start for c in range(3)]
    family_counts=Counter(rows[i]['family'] for i,_ in train_indices)
    def source(i):
        if i<native_start:return rows[i]['family']
        return 'agent_sfp' if rows[i].get('image_kind')=='agent_288x256_rgb_jpeg' else 'native_sfp'
    source_counts=Counter(source(i) for i,_ in train_indices)
    if not 0<a.native_source_fraction<1:raise ValueError('Invalid native source fraction')
    if a.native_selected_weight<1:raise ValueError('Selected-port weight must be at least one')
    legacy_fraction=(1-a.native_source_fraction)/2
    extra_share=a.native_source_fraction/(2 if source_counts['agent_sfp'] else 1)
    weights=np.asarray([(extra_share/source_counts[source(i)] if i>=native_start else
                        legacy_fraction/source_counts[source(i)]) for i,_ in train_indices])
    focus_count=0
    if a.focus_episode_substring:
        if not 0<a.focus_fraction<1:raise ValueError('Focus fraction must be between zero and one')
        focus=np.asarray([a.focus_episode_substring in rows[i]['episode'] for i,_ in train_indices])
        focus_count=int(focus.sum())
        if not focus_count:raise ValueError('No matching training episodes for focus')
        weights[focus]*=a.focus_fraction/weights[focus].sum()
        weights[~focus]*=(1-a.focus_fraction)/weights[~focus].sum()
    elif a.focus_fraction:raise ValueError('Focus fraction requires an episode substring')
    random.Random(20261002).shuffle(validation_indices)
    native_val_episodes=sorted({rows[i]['episode'] for i in range(native_start,len(rows))
                                if rows[i]['split']=='validation'})
    native_selection_episodes={e for i,e in enumerate(native_val_episodes)
                               if ((i//2)+(i%2))%2==0}
    native_test_episodes=set(native_val_episodes)-native_selection_episodes
    native_val_indices=[(i,c) for i in range(native_start,len(rows))
                        if rows[i]['split']=='validation'
                        and rows[i]['episode'] in native_selection_episodes for c in range(3)]
    native_test_indices=[(i,c) for i in range(native_start,len(rows))
                         if rows[i]['split']=='validation'
                         and rows[i]['episode'] in native_test_episodes for c in range(3)]
    sampler=WeightedRandomSampler(weights,num_samples=a.updates*a.batch_size,replacement=True)
    train=DataLoader(Views(rows,a.sfp_cache,train_indices,True),batch_size=a.batch_size,
                     sampler=sampler,num_workers=4,pin_memory=True,persistent_workers=True)
    val=DataLoader(Views(rows,a.sfp_cache,validation_indices,False),batch_size=96,shuffle=False,
                   num_workers=4,pin_memory=True,persistent_workers=True)
    native_val=(DataLoader(Views(rows,a.sfp_cache,native_val_indices,False),batch_size=96,
                           shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True)
                if native_val_indices else None)
    native_test=(DataLoader(Views(rows,a.sfp_cache,native_test_indices,False),batch_size=96,
                           shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True)
                 if native_test_indices else None)
    model=Locator(a.imagenet).cuda()
    if a.init_checkpoint:
        model.load_state_dict(torch.load(a.init_checkpoint,map_location='cpu',weights_only=False)['model'])
    learning_rate=a.learning_rate or (1e-4 if a.init_checkpoint else 2e-4)
    if learning_rate<=0:raise ValueError('Learning rate must be positive')
    opt=torch.optim.AdamW((v for v in model.parameters() if v.requires_grad),lr=learning_rate,weight_decay=1e-4)
    history=[];best=float('inf');best_step=0;stale=0;t0=time.monotonic();reason='update_cap'
    for step,batch in enumerate(train,start=1):
        if step>a.updates:break
        if time.monotonic()-t0>a.minutes*60:reason='wall_budget';break
        image,meta,points,present,inframe,selected,_,row_indices,_=batch
        image=image.cuda(non_blocking=True).float()/255.;meta=meta.cuda(non_blocking=True)
        points=points.cuda(non_blocking=True);present=present.cuda(non_blocking=True);inframe=inframe.cuda(non_blocking=True)
        selected=selected.cuda(non_blocking=True)
        model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred,flag,heat=model(image,meta,return_heatmap=True)
        error=F.smooth_l1_loss(pred.float(),points,beta=.01,reduction='none').sum(-1)
        point_weights=inframe.clone()
        native=torch.as_tensor(row_indices>=native_start,device='cuda')
        if native.any() and a.native_selected_weight>1:
            point_weights[native,selected[native]]*=a.native_selected_weight
        loss=(error*point_weights).sum()/point_weights.sum().clamp_min(1)
        if a.structured_loss_weight:
            pair_errors=[]
            # Adjacent card openings and the two ports of each present card
            # have a consistent geometric relationship across scene resets.
            pairs=[(0,1)]+[(2+2*c,3+2*c) for c in range(5)]
            pairs += [(2+2*c+p,2+2*(c+1)+p) for c in range(4) for p in (0,1)]
            for left,right in pairs:
                valid=inframe[:,left]*inframe[:,right]
                if valid.any():
                    delta=pred[:,left]-pred[:,right]
                    expected=points[:,left]-points[:,right]
                    mismatch=F.smooth_l1_loss(delta.float(),expected,beta=.02,reduction='none').sum(-1)
                    pair_errors.append((mismatch*valid).sum()/valid.sum().clamp_min(1))
            if pair_errors:loss+=a.structured_loss_weight*torch.stack(pair_errors).mean()
        if a.heatmap_loss_weight:
            height,width=heat.shape[-2:]
            cell_x=(points[...,0]*(width-1)).round().long().clamp(0,width-1)
            cell_y=(points[...,1]*(height-1)).round().long().clamp(0,height-1)
            cell=cell_y*width+cell_x
            spatial=F.cross_entropy((heat.float()*4).flatten(0,1).flatten(1),
                                    cell.flatten(),reduction='none').view_as(inframe)
            loss+=a.heatmap_loss_weight*(spatial*point_weights).sum()/point_weights.sum().clamp_min(1)
        flag_mask=present
        if a.supervise_absent_identity:
            flag_mask=torch.zeros_like(present)
            flag_mask[:,:2]=meta[:,0:1]
            flag_mask[:,2:]=meta[:,1:2]
        loss+=.1*(F.binary_cross_entropy_with_logits(flag.float(),inframe,reduction='none')*flag_mask).sum()/flag_mask.sum().clamp_min(1)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%100==0:
            report=evaluate(model,val,'cuda',max_batches=12)
            sc=report['errors'].get("('sc', 'selected', 'near')",{}).get('p95_px',1e6)
            sfp=report['errors'].get("('sfp', 'selected', 'near')",{}).get('p95_px',1e6)
            native_report=evaluate(model,native_val,'cuda') if native_val is not None else None
            native_sfp=(native_report['errors'].get("('sfp', 'selected', 'near')",{}).get('p95_px',1e6)
                        if native_report is not None else 0.)
            score=max(sc,sfp,native_sfp)
            entry={'update':step,'train_loss':float(loss.detach()),'validation_sample':report,
                   'native_validation':native_report,'selection_score':score}
            history.append(entry);print(json.dumps({'update':step,'loss':entry['train_loss'],'sc_near_p95':sc,'sfp_near_p95':sfp}),flush=True)
            if score<best-.25:
                best=score;best_step=step;stale=0
                torch.save({'model':model.state_dict(),'update':step,'names':NAMES,
                            'trained_projection_only':True,'certified_visibility':False,
                            'selected_target_from_public_task_code':True,'runtime_geometry':False},a.output/'best.pt')
            else:stale+=1
            if step>=600 and stale>=8:reason='validation_plateau';break
    if best_step:
        model.load_state_dict(torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)['model'])
        full=evaluate(model,val,'cuda')
    else:full=None
    summary={'schema':'two_family_port_locator/v1','best_update':best_step,
             'updates_completed':min(step,a.updates),'stop_reason':reason,
             'train_episodes':len({rows[i]['episode'] for i,_ in train_indices}),
             'validation_episodes':len({rows[i]['episode'] for i,_ in validation_indices}),
             'train_images':len(train_indices),'validation_views':len(validation_indices),
             'full_validation':full,'native_validation':evaluate(model,native_val,'cuda') if native_val is not None else None,
             'native_test':evaluate(model,native_test,'cuda') if native_test is not None else None,
             'native_selection_episodes':sorted(native_selection_episodes),
             'native_test_episodes':sorted(native_test_episodes),
             'native_train_episodes':len({rows[i]['episode'] for i,_ in train_indices if i>=native_start}),
             'native_source_fraction':a.native_source_fraction,
             'native_selected_weight':a.native_selected_weight,
             'heatmap_loss_weight':a.heatmap_loss_weight,
             'structured_loss_weight':a.structured_loss_weight,
             'supervise_absent_identity':a.supervise_absent_identity,
             'focus_episode_substring':a.focus_episode_substring,
             'focus_fraction':a.focus_fraction,'focus_train_images':focus_count,
             'learning_rate':learning_rate,
             'native_validation_episodes':len({rows[i]['episode'] for i,_ in native_val_indices}),
             'projection_only':True,'visibility_certified':False,
             'geometry_input_runtime':False,'final_scenes_opened':False}
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
