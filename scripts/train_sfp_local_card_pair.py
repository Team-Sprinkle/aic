#!/usr/bin/env python3
"""Train a requested-card native crop head for rail anchor and both port mouths."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
import time

import cv2
cv2.setNumThreads(1)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate,reprojection_residual_px
from train_sfp_observed_rail_crop import crop_native
from train_target_conditioned_port_locator import ConditionalLocator
from train_two_family_port_locator import CAMERAS,LOOKUP


def read(path):return [json.loads(s) for s in (path/'rows.jsonl').open()]


class PairLocator(nn.Module):
    def __init__(self,init):
        super().__init__()
        old=ConditionalLocator(None)
        old.load_state_dict(torch.load(init,map_location='cpu',weights_only=False)['model'])
        self.backbone=old.backbone;self.identity=old.identity
        self.register_buffer('mean',old.mean.clone());self.register_buffer('std',old.std.clone())
        self.decoder=nn.Sequential(*list(old.decoder.children())[:-1],nn.Conv2d(64,3,1))
        self.inframe=nn.Sequential(*list(old.inframe.children())[:-1],nn.Linear(128,3))
        with torch.no_grad():
            for idx in range(3):
                self.decoder[-1].weight[idx].copy_(old.decoder[-1].weight[0]);
                self.decoder[-1].bias[idx].copy_(old.decoder[-1].bias[0])
                self.inframe[-1].weight[idx].copy_(old.inframe[-1].weight[0]);
                self.inframe[-1].bias[idx].copy_(old.inframe[-1].bias[0])
        for name,p in self.backbone.named_parameters():
            p.requires_grad_(name.startswith('layer3.'))
    def train(self,mode=True):
        super().train(mode)
        self.backbone.bn1.eval();self.backbone.layer1.eval();self.backbone.layer2.eval()
        return self
    def forward(self,rgb,meta,card_target):
        x=(rgb-self.mean)/self.std
        x=self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        early=self.backbone.layer1(x)
        deep=self.backbone.layer3(self.backbone.layer2(early))
        condition=torch.cat((meta,self.identity(card_target)),-1)
        joined=torch.cat((early,F.interpolate(deep,size=early.shape[-2:],mode='bilinear',align_corners=False),
                          condition[:,:,None,None].expand(-1,-1,*early.shape[-2:])),dim=1)
        heat=self.decoder(joined)
        b,k,h,w=heat.shape
        probs=F.softmax(heat.reshape(b,k,-1).float()*4,dim=-1)
        yy,xx=torch.meshgrid(torch.linspace(0,1,h,device=rgb.device),
                             torch.linspace(0,1,w,device=rgb.device),indexing='ij')
        xy=torch.stack(((probs*xx.flatten()).sum(-1),(probs*yy.flatten()).sum(-1)),-1)
        flag=self.inframe(torch.cat((deep.mean((-1,-2)),condition),-1))
        return xy,flag,heat


class Views(Dataset):
    def __init__(self,rows,indices,training):self.rows=rows;self.indices=indices;self.training=training
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        ri,ci=self.indices[index]
        if ci is None:ci=random.randrange(3)
        row=self.rows[ri];cam=CAMERAS[ci];center=np.asarray(row['crop']['center_px'][cam],float)
        im=cv2.imread(row['images'][cam])
        if im is None:raise FileNotFoundError(row['images'][cam])
        crop=crop_native(im,center)
        meta=np.zeros(5,np.float32);meta[1]=1;meta[2+ci]=1
        card=row['card'];target=LOOKUP[f'sfp_card_{card}_port_0']
        rail=row['rail_projections_training_label_only'][cam][card]
        port_points=[next(t for t in row['targets'] if t['identity']==f'sfp_card_{card}_port_{p}')['cameras'][cam]
                     for p in range(2)]
        points=np.zeros((3,2),np.float32);inside=np.zeros(3,np.float32)
        for j,uv in enumerate([rail]+port_points):
            points[j]=.5+(np.asarray(uv[:2],np.float32)-center)/480
            inside[j]=float(bool(uv[3] and np.all((points[j]>=0)&(points[j]<=1))))
        return (torch.from_numpy(crop.copy()).permute(2,0,1),torch.from_numpy(meta),
                target,torch.from_numpy(points),torch.from_numpy(inside),ri,ci)


def materialize(labels,board,proposals):
    labels={(r['episode'],r['time_s']):r for r in read(labels)}
    boards={(r['episode'],r['time_s']):r for r in read(board)}
    rows=[];seen=set()
    for item in read(proposals):
        card=int(item['requested'].split('_')[2]);key=(item['episode'],item['time_s'],card)
        if key in seen:continue
        seen.add(key)
        label=labels[(item['episode'],item['time_s'])];ref=boards[(item['episode'],item['time_s'])]
        assert label['scene_group']==item['scene_group']==ref['scene_group']
        rows.append({'episode':item['episode'],'time_s':item['time_s'],'scene_group':item['scene_group'],
                     'images':label['images'],'state':label['state'],'targets':label['targets'],
                     'card':card,'crop':item['crop'],
                     'rail_projections_training_label_only':ref['rail_projections_training_label_only']})
    return rows


@torch.inference_mode()
def evaluate(model,rows,loader,optics,save=None):
    model.eval();views=defaultdict(dict)
    for image,meta,target,_,_,ri,ci in loader:
        xy,flag,_=model(image.cuda().float()/255.,meta.cuda(),target.cuda())
        xy=xy.float().cpu().numpy();flag=flag.float().sigmoid().cpu().numpy()
        for k in range(len(image)):
            j=int(ri[k]);cam=CAMERAS[int(ci[k])];center=np.asarray(rows[j]['crop']['center_px'][cam])
            views[j][cam]={'xy':(center+(xy[k]-.5)*480).tolist(),'flag':flag[k].tolist()}
    result=[];byep=defaultdict(list);errors=[];wrong=0
    for j,row in enumerate(rows):
        card=row['card'];port=row['requested_port']
        camposes={c:pose_matrix(row['state'])@optics[c] for c in CAMERAS}
        points={};reproj={}
        for channel in range(3):
            if not all(views[j][c]['flag'][channel]>=.5 for c in CAMERAS):continue
            pix={c:views[j][c]['xy'][channel] for c in CAMERAS}
            try:
                xyz=triangulate(pix,camposes);r=reprojection_residual_px(xyz,pix,camposes)
            except np.linalg.LinAlgError:continue
            if r<=5:points[channel]=xyz;reproj[channel]=float(r)
        selected=port+1;baseline=selected in points
        # The card anchor is RGB-predicted, while the crop center comes from a
        # separately frozen RGB rail head. Use their agreement as a card check.
        rail_agree=bool(sum(np.linalg.norm(np.asarray(views[j][c]['xy'][0])-
                                          np.asarray(row['crop']['center_px'][c]))<=40
                            for c in CAMERAS)>=2)
        pair=1 in points and 2 in points
        pair_gap=float(np.linalg.norm(points[2]-points[1])*1000) if pair else None
        guarded=bool(baseline and 0 in points and pair and rail_agree and 15<=pair_gap<=35)
        rec={'episode':row['episode'],'time_s':row['time_s'],'card':card,'port':port,
             'requested':f'sfp_card_{card}_port_{port}',
             'selected_accepted':baseline,'guarded_accepted':guarded,
             'rail_agree_2view_40px':rail_agree,'pair_gap_mm':pair_gap,
             'selected_reprojection_px':reproj.get(selected),
             'predicted_base_m':points[selected].tolist() if baseline else None,
             'error_mm_label_only':None,'nearest_identity_label_only':None}
        if baseline:
            target=next(t for t in row['targets'] if t['identity']==rec['requested'])
            err=float(np.linalg.norm(points[selected]-np.asarray(target['position_base_m']))*1000)
            nearest=min(row['targets'],key=lambda t:np.linalg.norm(points[selected]-np.asarray(t['position_base_m'])))
            rec['error_mm_label_only']=err;rec['nearest_identity_label_only']=nearest['identity']
            if guarded:errors.append(err);wrong+=nearest['identity']!=rec['requested']
        result.append(rec);byep[row['episode']].append(rec)
    early=0;anygroup=0
    for items in byep.values():
        items.sort(key=lambda r:r['time_s']);first=next((r for r in items if r['guarded_accepted']),None)
        anygroup+=first is not None;early+=bool(first and first['time_s']-items[0]['time_s']<=3)
    summary={'rows':len(rows),'accepted':len(errors),'wrong':wrong,'early_groups':early,
             'any_groups':anygroup,'accepted_3d_p95_mm':float(np.quantile(errors,.95)) if errors else None}
    if save:
        with save.open('w') as f:
            for row in result:f.write(json.dumps(row,separators=(',',':'))+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--board-labels',type=Path,required=True)
    p.add_argument('--train-proposals',type=Path,required=True)
    p.add_argument('--development-proposals',type=Path,required=True)
    p.add_argument('--init',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--minutes',type=float,default=120)
    p.add_argument('--batch-size',type=int,default=12)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve output')
    train=materialize(a.labels,a.board_labels,a.train_proposals)
    val=materialize(a.labels,a.board_labels,a.development_proposals)
    labels={(r['episode'],r['time_s']):r for r in read(a.labels)}
    for row in val:row['requested_port']=int(labels[(row['episode'],row['time_s'])]['selected_identity'].split('_')[-1])
    if {r['scene_group'] for r in train}&{r['scene_group'] for r in val}:raise ValueError('Reset split overlap')
    if len({r['scene_group'] for r in train})!=70 or len({r['scene_group'] for r in val})!=10:
        raise ValueError('Unexpected grouped split')
    a.output.mkdir(parents=True)
    random.seed(20261006);np.random.seed(20261006);torch.manual_seed(20261006);torch.set_num_threads(8)
    counts=Counter(r['card'] for r in train)
    weights=[1/counts[r['card']] for r in train]
    sampler=WeightedRandomSampler(weights,a.updates*a.batch_size,replacement=True)
    loader=DataLoader(Views(train,[(i,None) for i in range(len(train))],True),batch_size=a.batch_size,
                      sampler=sampler,num_workers=4,pin_memory=True,persistent_workers=True,multiprocessing_context='spawn')
    vloader=DataLoader(Views(val,[(i,c) for i in range(len(val)) for c in range(3)],False),batch_size=24,
                       num_workers=4,pin_memory=True,persistent_workers=True,multiprocessing_context='spawn')
    cal=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics={c:np.asarray(cal['tcp_to_optical'][c],float) for c in CAMERAS}
    model=PairLocator(a.init).cuda();opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=6e-5,weight_decay=1e-4)
    start=time.monotonic();history=[];best=float('inf');best_step=0;completed=0;reason='update_cap'
    for step,batch in enumerate(loader,1):
        if step>a.updates:break
        if time.monotonic()-start>a.minutes*60:reason='wall_budget';break
        image,meta,target,truth,inside,*_=batch
        image=image.cuda(non_blocking=True).float()/255.;meta=meta.cuda(non_blocking=True)
        target=target.cuda(non_blocking=True);truth=truth.cuda(non_blocking=True);inside=inside.cuda(non_blocking=True)
        model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):xy,flag,heat=model(image,meta,target)
        b,k,h,w=heat.shape
        cells=(truth[:,:,1]*(h-1)).round().long().clamp(0,h-1)*w+(truth[:,:,0]*(w-1)).round().long().clamp(0,w-1)
        loc=F.smooth_l1_loss(xy.float(),truth,beta=.01,reduction='none').sum(-1)
        spatial=F.cross_entropy((heat.float()*4).reshape(-1,h*w),cells.reshape(-1),reduction='none').reshape(b,k)
        loss=(((loc+.02*spatial)*inside).sum()/inside.sum().clamp_min(1)+
              .1*F.binary_cross_entropy_with_logits(flag.float(),inside))
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();completed=step
        if step%200==0:
            report=evaluate(model,val,vloader,optics,a.output/f'predictions_step{step}.jsonl')
            score=(report['accepted_3d_p95_mm'] if report['accepted_3d_p95_mm'] is not None else 1e6)
            score+=2000*report['wrong']+200*(10-report['early_groups'])+5*(len(val)-report['accepted'])
            item={'step':step,'loss':float(loss.detach()),'score':score,'development':report}
            history.append(item);print(json.dumps(item),flush=True)
            if score<best:
                best=score;best_step=step
                torch.save({'model':model.state_dict(),'step':step,'init':str(a.init),
                            'runtime_true_geometry_input':False},a.output/'best.pt')
    if not best_step:raise RuntimeError('No checkpoint')
    selected=next(x for x in history if x['step']==best_step)
    summary={'schema':'sfp_local_card_pair/v1','train_rows':len(train),'development_rows':len(val),
             'train_groups':len({r['scene_group'] for r in train}),
             'development_groups':len({r['scene_group'] for r in val}),
             'train_card_counts':dict(counts),'updates_completed':completed,'stop_reason':reason,
             'best_step':best_step,'selected':selected,'runtime_true_geometry_input':False,
             'final_scenes_opened':False}
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
