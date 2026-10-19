#!/usr/bin/env python3
"""Train one task-conditioned port keypoint head from grouped RGB labels.

Every present alternate port is a supervised candidate. Simulator geometry
supplies labels only; inference uses RGB, camera identity, and public task code.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
import time

import cv2  # Load pixi's JPEG/OpenCV libraries before torch and torchvision.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler
from torchvision import models

from train_two_family_port_locator import Views,NAMES,WIDTH,HEIGHT,Locator


class ConditionalLocator(nn.Module):
    def __init__(self,imagenet):
        super().__init__()
        self.backbone=models.resnet18(weights=None)
        if imagenet is not None:
            self.backbone.load_state_dict(torch.load(imagenet,map_location='cpu',weights_only=True))
        for name,param in self.backbone.named_parameters():
            param.requires_grad_(name.startswith('layer3.'))
        self.decoder=nn.Sequential(nn.Conv2d(64+256+5+16,128,3,padding=1),nn.GELU(),
                                   nn.Conv2d(128,64,3,padding=1),nn.GELU(),nn.Conv2d(64,1,1))
        self.identity=nn.Embedding(len(NAMES),16)
        self.inframe=nn.Sequential(nn.Linear(256+5+16,128),nn.GELU(),nn.Linear(128,1))
        self.register_buffer('mean',torch.tensor((.485,.456,.406))[None,:,None,None])
        self.register_buffer('std',torch.tensor((.229,.224,.225))[None,:,None,None])

    def train(self,mode=True):
        super().train(mode)
        self.backbone.bn1.eval();self.backbone.layer1.eval();self.backbone.layer2.eval()
        return self

    def forward(self,rgb,meta,target):
        x=(rgb-self.mean)/self.std
        x=self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        early=self.backbone.layer1(x)
        deep=self.backbone.layer3(self.backbone.layer2(early))
        conditioned=torch.cat((meta,self.identity(target)),-1)
        joined=torch.cat((early,F.interpolate(deep,size=early.shape[-2:],mode='bilinear',align_corners=False),
                          conditioned[:,:,None,None].expand(-1,-1,*early.shape[-2:])),dim=1)
        heat=self.decoder(joined).squeeze(1)
        b,h,w=heat.shape
        probs=F.softmax(heat.reshape(b,-1).float()*4,dim=-1)
        y,x=torch.meshgrid(torch.linspace(0,1,h,device=rgb.device),
                           torch.linspace(0,1,w,device=rgb.device),indexing='ij')
        xy=torch.stack(((probs*x.flatten()).sum(-1),(probs*y.flatten()).sum(-1)),-1)
        flag=self.inframe(torch.cat((deep.mean((-1,-2)),conditioned),-1)).squeeze(-1)
        return xy,flag,heat


class ResidualConditionalLocator(nn.Module):
    """Frozen proven 12-port detector plus a shared task-conditioned heatmap."""
    def __init__(self,imagenet,old_checkpoint):
        super().__init__()
        self.base=Locator(imagenet)
        self.base.load_state_dict(torch.load(old_checkpoint,map_location='cpu',weights_only=False)['model'])
        for param in self.base.parameters():param.requires_grad_(False)
        self.identity=nn.Embedding(len(NAMES),16)
        self.residual=nn.Sequential(nn.Conv2d(64+256+5+16,64,3,padding=1),nn.GELU(),
                                    nn.Conv2d(64,1,1))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
        self.flag_residual=nn.Sequential(nn.Linear(256+5+16,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.flag_residual[-1].weight);nn.init.zeros_(self.flag_residual[-1].bias)

    def train(self,mode=True):
        super().train(mode);self.base.eval();return self

    def forward(self,rgb,meta,target):
        with torch.no_grad():
            x=(rgb-self.base.mean)/self.base.std
            x=self.base.backbone.maxpool(self.base.backbone.relu(self.base.backbone.bn1(self.base.backbone.conv1(x))))
            early=self.base.backbone.layer1(x)
            deep=self.base.backbone.layer3(self.base.backbone.layer2(early))
            joined=torch.cat((early,F.interpolate(deep,size=early.shape[-2:],mode='bilinear',align_corners=False),
                              meta[:,:,None,None].expand(-1,-1,*early.shape[-2:])),dim=1)
            old_heat=self.base.decoder(joined)
            old_flag=self.base.inframe(torch.cat((deep.mean((-1,-2)),meta),-1))
        batch=torch.arange(len(target),device=target.device)
        embedding=self.identity(target)
        conditioned=torch.cat((joined,embedding[:,:,None,None].expand(-1,-1,*joined.shape[-2:])),dim=1)
        heat=old_heat[batch,target]+self.residual(conditioned).squeeze(1)
        flag=old_flag[batch,target]+self.flag_residual(torch.cat((deep.mean((-1,-2)),meta,embedding),-1)).squeeze(1)
        b,h,w=heat.shape
        probs=F.softmax(heat.reshape(b,-1).float()*4,dim=-1)
        y,x=torch.meshgrid(torch.linspace(0,1,h,device=rgb.device),
                           torch.linspace(0,1,w,device=rgb.device),indexing='ij')
        xy=torch.stack(((probs*x.flatten()).sum(-1),(probs*y.flatten()).sum(-1)),-1)
        return xy,flag,heat


class CandidateViews(Dataset):
    def __init__(self,rows,cache,indices,training):
        self.views=Views(rows,cache,indices,training);self.training=training
    def __len__(self):return len(self.views)
    def __getitem__(self,i):
        image,meta,points,present,inframe,selected,distance,row,cam=self.views[i]
        if self.training and random.random()<.7:
            choices=torch.nonzero(present>.5).flatten().tolist()
            selected=random.choice(choices)
        return image,meta,points[selected],inframe[selected],selected,distance,row,cam


@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval();errors=defaultdict(list);flags=defaultdict(list);total=0
    for image,meta,point,inframe,target,distance,*_ in loader:
        pred,flag,_=model(image.to(device).float()/255.,meta.to(device),target.to(device))
        pred=pred.cpu().numpy();flag=flag.sigmoid().cpu().numpy();point=point.numpy()
        family=np.argmax(meta.numpy()[:,:2],axis=1)
        for i in range(len(pred)):
            fam='sc' if family[i]==0 else 'sfp';total+=1
            flags[fam].append((flag[i],float(inframe[i])))
            if not inframe[i]:continue
            err=float(np.linalg.norm((pred[i]-point[i])*[WIDTH-1,HEIGHT-1]))
            errors[(fam,'near' if distance[i]<.2 else 'far')].append(err)
    report={str(k):{'n':len(v),'p50_px':float(np.median(v)),
                    'p95_px':float(np.quantile(v,.95))} for k,v in errors.items() if v}
    for fam,pairs in flags.items():
        p=np.array([x[0] for x in pairs]);y=np.array([x[1] for x in pairs])>.5
        report[f'{fam}_inframe_accuracy']=float(np.mean((p>.5)==y))
    return {'images':total,'errors':report}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--extra-labels',type=Path,required=True)
    p.add_argument('--sfp-cache',type=Path,required=True)
    p.add_argument('--imagenet',type=Path,required=True)
    p.add_argument('--old-locator',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=3500)
    p.add_argument('--minutes',type=float,default=90)
    p.add_argument('--batch-size',type=int,default=24)
    p.add_argument('--residual-from-old',action='store_true',
                   help='Keep the proven multiport detector frozen and train only a shared target correction')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing output')
    a.output.mkdir(parents=True)
    random.seed(20261002);np.random.seed(20261002);torch.manual_seed(20261002);torch.set_num_threads(8)
    rows=[json.loads(x) for x in (a.labels/'rows.jsonl').open()]
    native_start=len(rows)
    rows.extend(json.loads(x) for x in (a.extra_labels/'rows.jsonl').open())
    if {r['scene_group'] for r in rows if r['split']=='train'} & {r['scene_group'] for r in rows if r['split']=='validation'}:
        raise ValueError('Scene/reset leakage')
    native_val=sorted({r['episode'] for r in rows[native_start:] if r['split']=='validation'})
    selection={e for i,e in enumerate(native_val) if ((i//2)+(i%2))%2==0}
    test=set(native_val)-selection
    train=[(i,None) for i,r in enumerate(rows) if r['split']=='train']
    old_val=[]
    for family in ('sc','sfp'):
        candidates=[(i,c) for i,r in enumerate(rows[:native_start])
                    if r['split']=='validation' and r['family']==family for c in range(3)]
        old_val.extend(random.Random(20261002).sample(candidates,min(600,len(candidates))))
    native_selection=[(i,c) for i in range(native_start,len(rows)) if rows[i]['episode'] in selection for c in range(3)]
    native_test=[(i,c) for i in range(native_start,len(rows)) if rows[i]['episode'] in test for c in range(3)]
    def source(i):
        if i<native_start:return rows[i]['family']
        return 'agent_sfp' if rows[i].get('image_kind')=='agent_288x256_rgb_jpeg' else 'native_sfp'
    count=Counter(source(i) for i,_ in train)
    desired={'agent_sfp':.35,'native_sfp':.35,'sc':.15,'sfp':.15}
    total_mix=sum(desired[key] for key in count)
    weights=[desired[source(i)]/total_mix/count[source(i)] for i,_ in train]
    train_loader=DataLoader(CandidateViews(rows,a.sfp_cache,train,True),batch_size=a.batch_size,
        sampler=WeightedRandomSampler(weights,num_samples=a.updates*a.batch_size,replacement=True),
        num_workers=4,pin_memory=True,persistent_workers=True)
    def val_loader(indices):
        return DataLoader(CandidateViews(rows,a.sfp_cache,indices,False),batch_size=96,
                          num_workers=4,pin_memory=True,persistent_workers=True)
    old_loader=val_loader(old_val);select_loader=val_loader(native_selection);test_loader=val_loader(native_test)
    if a.residual_from_old:
        model=ResidualConditionalLocator(a.imagenet,a.old_locator).cuda()
    else:
        model=ConditionalLocator(a.imagenet).cuda()
        old=torch.load(a.old_locator,map_location='cpu',weights_only=False)['model']
        model.backbone.load_state_dict({k.removeprefix('backbone.'):v for k,v in old.items() if k.startswith('backbone.')})
    opt=torch.optim.AdamW((v for v in model.parameters() if v.requires_grad),lr=1.5e-4,weight_decay=1e-4)
    best=float('inf');best_step=0;stale=0;history=[];start=time.monotonic();reason='update_cap'
    for step,batch in enumerate(train_loader,start=1):
        if step>a.updates:break
        if time.monotonic()-start>a.minutes*60:reason='wall_budget';break
        image,meta,point,inframe,target,*_=batch
        image=image.cuda(non_blocking=True).float()/255.;meta=meta.cuda(non_blocking=True)
        point=point.cuda(non_blocking=True);inframe=inframe.cuda(non_blocking=True)
        target=target.cuda(non_blocking=True)
        model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred,flag,heat=model(image,meta,target)
        pixel=F.smooth_l1_loss(pred.float(),point,beta=.01,reduction='none').sum(-1)
        h,w=heat.shape[-2:]
        cell=(point[:,1]*(h-1)).round().long().clamp(0,h-1)*w+(point[:,0]*(w-1)).round().long().clamp(0,w-1)
        spatial=F.cross_entropy((heat.float()*4).flatten(1),cell,reduction='none')
        loss=((pixel+.02*spatial)*inframe).sum()/inframe.sum().clamp_min(1)
        loss+=.1*F.binary_cross_entropy_with_logits(flag.float(),inframe)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%100==0:
            old_report=evaluate(model,old_loader,'cuda')
            native_report=evaluate(model,select_loader,'cuda')
            sc=old_report['errors'].get("('sc', 'near')",{}).get('p95_px',1e6)
            sfp=old_report['errors'].get("('sfp', 'near')",{}).get('p95_px',1e6)
            native=native_report['errors'].get("('sfp', 'near')",{}).get('p95_px',1e6)
            score=max(native,sc*8,sfp*8)
            history.append({'update':step,'loss':float(loss.detach()),'selection_score':score,
                            'old_validation':old_report,'native_selection':native_report})
            print(json.dumps({'update':step,'loss':float(loss.detach()),'native_p95_px':native,
                              'sc_p95_px':sc,'old_sfp_p95_px':sfp}),flush=True)
            if score<best-1:
                best=score;best_step=step;stale=0
                torch.save({'model':model.state_dict(),'update':step,'names':NAMES,
                            'architecture':'residual_conditional_resnet18_keypoint' if a.residual_from_old
                                            else 'conditional_resnet18_keypoint',
                            'true_geometry_runtime':False,'literal_visibility_certified':False},a.output/'best.pt')
            else:stale+=1
            if step>=800 and stale>=10:reason='validation_plateau';break
    if not best_step:raise RuntimeError('No selected checkpoint')
    model.load_state_dict(torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)['model'])
    summary={'schema':'aic_conditional_port_locator/v1','best_update':best_step,
             'updates_completed':min(step,a.updates),'stop_reason':reason,
             'train_episodes':len({rows[i]['episode'] for i,_ in train}),
             'train_source_rows':dict(count),
             'native_selection_episodes':sorted(selection),'native_test_episodes':sorted(test),
             'old_validation':evaluate(model,old_loader,'cuda'),
             'native_selection':evaluate(model,select_loader,'cuda'),
             'native_test':evaluate(model,test_loader,'cuda'),
             'alternate_ports_supervised':True,'runtime_uses_task_code':True,
             'visibility_is_projected_inframe_only':True,'final_scenes_opened':False}
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
