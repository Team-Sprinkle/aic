#!/usr/bin/env python3
"""One bounded SFP all-slot internal map, returning only requested slot at eval.

Every present card/port opening is a training label. The model gets RGB and
camera identity; the task request selects one output channel at runtime.
Gazebo geometry never enters its forward pass or acceptance rule.
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset

from train_sfp_balanced_anchor import BalancedViews,evaluate
from train_two_family_port_locator import CAMERAS,LOOKUP,Locator


class AllSlots(Dataset):
    def __init__(self,rows,indices):self.rows=rows;self.indices=indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        r=self.rows[self.indices[i]];cam_idx=random.randrange(3);cam=CAMERAS[cam_idx]
        image=cv2.imread(r['images'][cam],cv2.IMREAD_COLOR)
        if image is None:raise FileNotFoundError(r['images'][cam])
        image=cv2.cvtColor(cv2.resize(image,(576,512),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        meta=np.zeros(5,np.float32);meta[1]=1;meta[2+cam_idx]=1
        points=np.zeros((12,2),np.float32);present=np.zeros(12,np.float32);inframe=np.zeros(12,np.float32)
        for target in r['targets']:
            j=LOOKUP[target['identity']];present[j]=1.;uv=target['cameras'][cam]
            if uv[3]:points[j]=[uv[0]/1151,uv[1]/1023];inframe[j]=1.
        return (torch.from_numpy(image.copy()).permute(2,0,1),torch.from_numpy(meta),
                torch.from_numpy(points),torch.from_numpy(present),torch.from_numpy(inframe))


class RequestedSlot(nn.Module):
    def __init__(self,base):super().__init__();self.base=base
    def forward(self,image,meta,target):
        xy,flag=self.base(image,meta)
        i=torch.arange(len(target),device=target.device)
        return xy[i,target],flag[i,target],None


def coverage(rows,predictions):
    by=defaultdict(list);wrong=0;cards=defaultdict(lambda:{'rows':0,'accepted':0})
    for item in predictions:
        by[item['episode']].append(item)
        c=int(item['requested'].split('_')[2]);cards[c]['rows']+=1;cards[c]['accepted']+=bool(item['accepted'])
        wrong+=bool(item['accepted'] and item['nearest_identity_label_only']!=item['requested'])
    early=0;any_episode=0
    for items in by.values():
        items.sort(key=lambda x:x['time_s']);first=next((x for x in items if x['accepted']),None)
        any_episode+=first is not None
        early+=first is not None and first['time_s']-items[0]['time_s']<=3
    return {'episode_with_any_anchor':any_episode,'episode_with_anchor_within_3s':early,
            'wrong':wrong,'by_card':dict(cards)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--init',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--minutes',type=float,default=120)
    p.add_argument('--batch-size',type=int,default=6)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous output')
    if a.updates>2000:p.error('Bounded update cap is 2000')
    a.output.mkdir(parents=True)
    rows=[json.loads(s) for s in (a.labels/'rows.jsonl').open()]
    train=[i for i,r in enumerate(rows) if r['split']=='train']
    val=[(i,c) for i,r in enumerate(rows) if r['split']=='validation' for c in range(3)]
    if {rows[i]['scene_group'] for i in train}&{rows[i]['scene_group'] for i,_ in val}:
        raise ValueError('Reset group leak')
    random.seed(20261004);np.random.seed(20261004);torch.manual_seed(20261004);torch.set_num_threads(8)
    sample=torch.utils.data.RandomSampler(AllSlots(rows,train),replacement=True,num_samples=a.updates*a.batch_size)
    loader=DataLoader(AllSlots(rows,train),batch_size=a.batch_size,sampler=sample,
                      num_workers=4,pin_memory=True,persistent_workers=True)
    vloader=DataLoader(BalancedViews(rows,val,False),batch_size=8,num_workers=4,
                       pin_memory=True,persistent_workers=True)
    cal=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics={c:np.asarray(cal['tcp_to_optical'][c],float) for c in CAMERAS}
    model=Locator(None).cuda()
    model.load_state_dict(torch.load(a.init,map_location='cpu',weights_only=False)['model'])
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=5e-5,weight_decay=1e-4)
    start=time.monotonic();history=[];best=float('inf');best_step=0;completed=0;reason='update_cap'
    for step,batch in enumerate(loader,1):
        if step>a.updates:break
        if time.monotonic()-start>a.minutes*60:reason='wall_budget';break
        image,meta,points,present,inframe=batch
        image=image.cuda(non_blocking=True).float()/255.;meta=meta.cuda(non_blocking=True)
        points=points.cuda(non_blocking=True);present=present.cuda(non_blocking=True)
        inframe=inframe.cuda(non_blocking=True)
        model.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):xy,flag,heat=model(image,meta,return_heatmap=True)
        xy=xy.float();flag=flag.float();heat=heat.float()
        mask=inframe[:,2:];target=points[:,2:];pred=xy[:,2:];b,h,w=heat.shape[0],heat.shape[-2],heat.shape[-1]
        coord=F.smooth_l1_loss(pred,target,beta=.01,reduction='none').sum(-1)
        cell=(target[:,:,1]*(h-1)).round().long().clamp(0,h-1)*w+(target[:,:,0]*(w-1)).round().long().clamp(0,w-1)
        spatial=F.cross_entropy(heat[:,2:].reshape(-1,h*w)*4,cell.reshape(-1),reduction='none').reshape(b,10)
        dist=torch.cdist(pred,target)
        same=torch.diagonal(dist,dim1=1,dim2=2)
        other=dist.masked_fill(~mask[:,None,:].bool(),float('inf'))
        other=other.masked_fill(torch.eye(10,device=other.device,dtype=torch.bool)[None],float('inf'))
        nearest=other.min(-1).values
        identity=F.relu(.01+same-nearest).masked_fill(~torch.isfinite(nearest),0)
        loss=((coord+.02*spatial+.2*identity)*mask).sum()/mask.sum().clamp_min(1)
        loss+=.1*F.binary_cross_entropy_with_logits(flag[:,2:],inframe[:,2:])
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();completed=step
        if step%200==0:
            path=a.output/f'predictions_step{step}.jsonl'
            report=evaluate(RequestedSlot(model),rows,vloader,'cuda',optics,path)
            predrows=[json.loads(s) for s in path.open()];cov=coverage(rows,predrows)
            p95=report['accepted_3d_mm_p95'] if report['accepted_3d_mm_p95'] is not None else 1e6
            score=p95+200*cov['wrong']+40*(10-cov['episode_with_anchor_within_3s'])
            rec={'step':step,'loss':float(loss.detach()),'score':score,'validation':report,'coverage':cov}
            history.append(rec);print(json.dumps(rec),flush=True)
            if score<best:
                best=score;best_step=step
                torch.save({'model':model.state_dict(),'step':step,'architecture':'resnet18_all_slot_native576_identity_margin',
                            'runtime_geometry_input':False},a.output/'best.pt')
    if not best_step:raise RuntimeError('No checkpoint')
    selected=next(x for x in history if x['step']==best_step)
    summary={'schema':'sfp_structured_slot_locator/v1','train_episodes':50,'development_episodes':10,
             'train_observations':len(train),'development_observations':len(val)//3,
             'best_step':best_step,'completed_updates':completed,'stop_reason':reason,
             'init':str(a.init),'image_network_size':[576,512],
             'all_present_port_heatmaps_supervised':True,'identity_margin_loss':True,
             'runtime_output':'requested_port_only','runtime_geometry_input':False,
             'final_scenes_opened':False,'selected':selected,
             'numeric_gate_passed':bool(selected['coverage']['wrong']==0 and
                 selected['validation']['accepted_3d_mm_p95'] is not None and
                 selected['validation']['accepted_3d_mm_p95']<=20),
             'coverage_gate_passed':bool(selected['coverage']['episode_with_anchor_within_3s']>=9 and
                 all(selected['coverage']['by_card'][c]['accepted']>0 for c in range(1,4)))}
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
