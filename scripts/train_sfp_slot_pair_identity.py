#!/usr/bin/env python3
"""Bounded global-slot and paired-port SFP identity model.

Simulator geometry is used for training targets and posthoc evaluation only.
At inference the model reads original RGB, camera identity, and requested
card/port code. Acceptance uses predicted 3D slot/port geometry and CAD
spacing, never the true Gazebo port transform.
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
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate,reprojection_residual_px
from train_two_family_port_locator import CAMERAS,Locator


class SlotPairModel(nn.Module):
    def __init__(self,init):
        super().__init__()
        old=Locator(None)
        old.load_state_dict(torch.load(init,map_location='cpu',weights_only=False)['model'])
        self.backbone=old.backbone
        for name,p in self.backbone.named_parameters():
            p.requires_grad_(name.startswith('layer3.') or name.startswith('layer4.'))
        self.decoder=nn.Sequential(nn.Conv2d(256+512+5+2,192,3,padding=1),nn.GELU(),
                                   nn.Conv2d(192,96,3,padding=1),nn.GELU(),
                                   nn.Conv2d(96,15,1))
        self.flags=nn.Sequential(nn.Linear(512+5,160),nn.GELU(),nn.Linear(160,15))
        self.register_buffer('mean',torch.tensor((.485,.456,.406))[None,:,None,None])
        self.register_buffer('std',torch.tensor((.229,.224,.225))[None,:,None,None])

    def train(self,mode=True):
        super().train(mode)
        self.backbone.bn1.eval();self.backbone.layer1.eval();self.backbone.layer2.eval()
        return self

    def forward(self,rgb,meta,return_heat=False):
        x=(rgb-self.mean)/self.std
        x=self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        x=self.backbone.layer1(x)
        x=self.backbone.layer2(x)
        mid=self.backbone.layer3(x)
        deep=self.backbone.layer4(mid)
        b,_,h,w=mid.shape
        y,x=torch.meshgrid(torch.linspace(0,1,h,device=rgb.device),
                           torch.linspace(0,1,w,device=rgb.device),indexing='ij')
        coordinates=torch.stack((x,y))[None].expand(b,-1,-1,-1)
        joined=torch.cat((mid,F.interpolate(deep,size=(h,w),mode='bilinear',align_corners=False),
                          meta[:,:,None,None].expand(-1,-1,h,w),coordinates),dim=1)
        heat=self.decoder(joined)
        probability=F.softmax(heat.reshape(b,15,-1).float()*4,dim=-1)
        xy=torch.stack(((probability*x.flatten()).sum(-1),
                        (probability*y.flatten()).sum(-1)),-1)
        flags=self.flags(torch.cat((deep.mean((-1,-2)),meta),-1))
        return (xy,flags,heat) if return_heat else (xy,flags)


class Views(Dataset):
    def __init__(self,rows,indices,training):
        self.rows=rows;self.indices=indices;self.training=training
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        ri,ci=self.indices[i];r=self.rows[ri]
        if ci is None:ci=random.randrange(3)
        cam=CAMERAS[ci]
        image=cv2.imread(r['images'][cam],cv2.IMREAD_COLOR)
        if image is None:raise FileNotFoundError(r['images'][cam])
        image=cv2.cvtColor(cv2.resize(image,(576,512),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        points=np.zeros((15,2),np.float32);mask=np.zeros(15,np.float32)
        for slot in r['slot_targets']:
            j=slot['card'];uv=slot['cameras'][cam]
            # Empty rail positions have a nominal geometry coordinate, but no
            # visible physical slot to localize in RGB. Never supervise them
            # as image detections.
            if slot['present'] and uv[3]:
                points[j]=[uv[0]/1151,uv[1]/1023];mask[j]=1
        for target in r['targets']:
            name=target['identity'];bits=name.split('_');j=5+2*int(bits[2])+int(bits[-1])
            uv=target['cameras'][cam]
            if uv[3]:points[j]=[uv[0]/1151,uv[1]/1023];mask[j]=1
        meta=np.zeros(5,np.float32);meta[1]=1;meta[2+ci]=1
        return (torch.from_numpy(image.copy()).permute(2,0,1),torch.from_numpy(meta),
                torch.from_numpy(points),torch.from_numpy(mask),ri,ci)


def posthoc(rows,records,save):
    wrong=[];errors=[];byepisode=defaultdict(list)
    for rec in records:
        r=rows[rec['row']]
        requested=next(t for t in r['targets'] if t['identity']==r['selected_identity'])
        if rec['accepted']:
            pred=np.asarray(rec['predicted_base_m'])
            nearest=min(r['targets'],key=lambda t:np.linalg.norm(pred-np.asarray(t['position_base_m'])))
            rec['error_mm']=float(np.linalg.norm(pred-np.asarray(requested['position_base_m']))*1000)
            rec['nearest_identity_label_only']=nearest['identity']
            errors.append(rec['error_mm'])
            if nearest['identity']!=r['selected_identity']:wrong.append(rec)
        byepisode[r['episode']].append(rec)
    early=0;anyx=0;episodes=[]
    for ep,items in sorted(byepisode.items()):
        items.sort(key=lambda x:x['time_s']);first=next((x for x in items if x['accepted']),None)
        anyx+=first is not None
        early+=bool(first and first['time_s']-items[0]['time_s']<=3)
        episodes.append(dict(episode=ep,request=items[0]['requested'],accepted=sum(x['accepted'] for x in items),
                             wrong=sum(x in wrong for x in items),
                             first_accept_s=first['time_s']-items[0]['time_s'] if first else None))
    report=dict(rows=len(records),accepted=len(errors),wrong=len(wrong),early=early,
                any_episode=anyx,episodes=episodes,
                accepted_3d_p95_mm=float(np.quantile(errors,.95)) if errors else None)
    if save:
        with save.open('w') as f:
            for rec in records:f.write(json.dumps(rec,separators=(',',':'))+'\n')
    return report


@torch.inference_mode()
def evaluate(model,rows,loader,optics,save=None):
    model.eval();views=defaultdict(dict)
    for image,meta,_,_,ri,ci in loader:
        xy,logits=model(image.cuda().float()/255.,meta.cuda())
        xy=xy.float().cpu().numpy();prob=logits.float().sigmoid().cpu().numpy()
        for i in range(len(xy)):
            views[int(ri[i])][CAMERAS[int(ci[i])]]=dict(xy=xy[i].tolist(),prob=prob[i].tolist())
    records=[]
    for j,v in sorted(views.items()):
        r=rows[j];card=int(r['selected_identity'].split('_')[2]);port=int(r['selected_identity'].split('_')[-1])
        rec=dict(row=j,episode=r['episode'],scene_group=r['scene_group'],time_s=r['time_s'],
                 requested=r['selected_identity'],accepted=False,predicted_base_m=None,
                 error_mm=None,nearest_identity_label_only=None,reason=None)
        if len(v)!=3:
            rec['reason']='missing_view';records.append(rec);continue
        poses={c:pose_matrix(r['state'])@optics[c] for c in CAMERAS}
        points={};reprojections={}
        for index in range(15):
            if not all(v[c]['prob'][index]>=.5 for c in CAMERAS):continue
            pixels={c:(np.asarray(v[c]['xy'][index])*[1151,1023]).tolist() for c in CAMERAS}
            try:
                p=triangulate(pixels,poses);rp=reprojection_residual_px(p,pixels,poses)
                if np.isfinite(p).all() and rp<=5:points[index]=p;reprojections[index]=float(rp)
            except np.linalg.LinAlgError:pass
        a=5+2*card;b=a+1;selected=a+port
        if a not in points or b not in points:
            rec['reason']='missing_pair';records.append(rec);continue
        slot_ids=sorted(k for k in points if k<5)
        if card not in slot_ids or len(slot_ids)<2:
            rec['reason']='missing_slot';records.append(rec);continue
        loc=np.asarray([points[k] for k in slot_ids]);indices=np.asarray(slot_ids,float)
        centered=indices-indices.mean();slope=(centered[:,None]*loc).sum(0)/(centered**2).sum()
        gap=float(np.linalg.norm(slope))*1000
        if not 30<=gap<=50:
            rec['reason']='slot_spacing';records.append(rec);continue
        row_axis=slope/np.linalg.norm(slope)
        slot_y=np.dot(loc-loc.mean(0),row_axis)
        residual=np.max(np.abs(slot_y-centered*np.linalg.norm(slope)))*1000
        if residual>10:
            rec['reason']='slot_order';records.append(rec);continue
        pair=points[b]-points[a];pair_gap=float(np.linalg.norm(pair))*1000
        if not 15<=pair_gap<=35:
            rec['reason']='pair_spacing';records.append(rec);continue
        x_axis=np.cross(row_axis,[0.,0.,1.]);x_axis/=max(np.linalg.norm(x_axis),1e-8)
        if float(np.dot(pair,x_axis))>=-.010:
            rec['reason']='pair_order';records.append(rec);continue
        row_offset=float(np.dot(points[selected]-points[card],row_axis))*1000
        if abs(row_offset)>20:
            rec['reason']='wrong_card_slot';records.append(rec);continue
        rec.update(accepted=True,predicted_base_m=points[selected].tolist(),reason='accepted',
                   slot_spacing_mm=gap,pair_spacing_mm=pair_gap,row_offset_mm=row_offset,
                   reprojection_px=reprojections[selected])
        records.append(rec)
    return posthoc(rows,records,save)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--labels',type=Path,required=True)
    ap.add_argument('--init',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--updates',type=int,default=2000)
    ap.add_argument('--minutes',type=float,default=120)
    ap.add_argument('--batch-size',type=int,default=6)
    args=ap.parse_args()
    if args.output.exists():ap.error('Preserve previous attempt')
    if args.updates>2000 or args.minutes>120:ap.error('Bounded 2,000-update / 120-minute attempt')
    rows=[json.loads(s) for s in (args.labels/'rows.jsonl').open()]
    train=[i for i,r in enumerate(rows) if r['fit_split']=='train']
    val=[(i,c) for i,r in enumerate(rows) if r['fit_split']=='validation' for c in range(3)]
    if len({rows[i]['scene_group'] for i in train})!=59 or len({rows[i]['scene_group'] for i,_ in val})!=10:
        raise ValueError('Unexpected grouped fit split')
    if {rows[i]['scene_group'] for i in train}&{rows[i]['scene_group'] for i,_ in val}:
        raise ValueError('Complete reset leakage')
    args.output.mkdir(parents=True)
    random.seed(20261004);np.random.seed(20261004);torch.manual_seed(20261004);torch.set_num_threads(4)
    group_counts=Counter(rows[i]['scene_group'] for i in train)
    weights=[1/group_counts[rows[i]['scene_group']] for i in train]
    sampler=WeightedRandomSampler(weights,args.updates*args.batch_size,replacement=True)
    loader=DataLoader(Views(rows,[(i,None) for i in train],True),batch_size=args.batch_size,
                      sampler=sampler,num_workers=0,pin_memory=True)
    vloader=DataLoader(Views(rows,val,False),batch_size=12,num_workers=0,pin_memory=True)
    calibration=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics={c:np.asarray(calibration['tcp_to_optical'][c],float) for c in CAMERAS}
    model=SlotPairModel(args.init).cuda()
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=4e-5,weight_decay=1e-4)
    start=time.monotonic();history=[];best=float('inf');best_step=0;completed=0;reason='update_cap'
    for step,batch in enumerate(loader,1):
        if step>args.updates:break
        if time.monotonic()-start>args.minutes*60:reason='wall_budget';break
        image,meta,points,mask,*_=batch
        image=image.cuda(non_blocking=True).float()/255.;meta=meta.cuda(non_blocking=True)
        points=points.cuda(non_blocking=True);mask=mask.cuda(non_blocking=True)
        model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred,flags,heat=model(image,meta,return_heat=True)
        pred=pred.float();flags=flags.float();heat=heat.float()
        b,k,h,w=heat.shape
        coord=F.smooth_l1_loss(pred,points,beta=.01,reduction='none').sum(-1)
        cells=(points[:,:,1]*(h-1)).round().long().clamp(0,h-1)*w+\
              (points[:,:,0]*(w-1)).round().long().clamp(0,w-1)
        spatial=F.cross_entropy(heat.reshape(-1,h*w)*4,cells.reshape(-1),reduction='none').reshape(b,k)
        loc=((coord+.02*spatial)*mask).sum()/mask.sum().clamp_min(1)
        visible=F.binary_cross_entropy_with_logits(flags,mask)
        pair_loss=0.;pair_count=0
        for card in range(5):
            a=5+2*card;bidx=a+1
            valid=mask[:,a]*mask[:,bidx]*mask[:,card]
            if valid.sum()>0:
                direction=F.smooth_l1_loss(pred[:,bidx]-pred[:,a],
                                            points[:,bidx]-points[:,a],beta=.01,reduction='none').sum(-1)
                pair_loss=pair_loss+(direction*valid).sum();pair_count+=valid.sum()
        loss=loc+.1*visible+.5*pair_loss/(pair_count+1e-6)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();completed=step
        if step%100==0:
            print(json.dumps(dict(step=step,train_loss=float(loss.detach()),
                                  elapsed_s=round(time.monotonic()-start,1))),flush=True)
        if step%400==0:
            report=evaluate(model,rows,vloader,optics,args.output/f'predictions_step{step}.jsonl')
            p95=report['accepted_3d_p95_mm'] if report['accepted_3d_p95_mm'] is not None else 1e4
            score=300*report['wrong']+40*(10-report['early'])+p95
            entry=dict(step=step,loss=float(loss.detach()),score=score,validation=report)
            history.append(entry);print(json.dumps({k:v for k,v in entry.items() if k!='validation'}|
                                             {'validation':{k:v for k,v in report.items() if k!='episodes'}}),flush=True)
            if score<best:
                best=score;best_step=step
                torch.save(dict(model=model.state_dict(),step=step,
                                architecture='resnet18_global_layer4_five_slots_ten_ports',
                                geometry_runtime_input=False,
                                acceptance='three_view_slot_pair_cad_spacing'),args.output/'best.pt')
    if best_step==0:raise RuntimeError('No checkpoint produced within bounded budget')
    bestreport=next(r['validation'] for r in history if r['step']==best_step)
    summary=dict(schema='sfp_slot_pair_identity_fit/v1',train_groups=59,validation_groups=10,
                 updates_completed=completed,stop_reason=reason,best_step=best_step,
                 best_validation=bestreport,one_gpu=True,final_scenes_opened=False,
                 runtime_geometry_input=False,validation_used_for_model_selection=True)
    (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='best_validation'}|
                     {'best_validation':{k:v for k,v in bestreport.items() if k!='episodes'}}),flush=True)


if __name__=='__main__':main()
