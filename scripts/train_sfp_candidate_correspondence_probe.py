#!/usr/bin/env python3
"""Bounded SFP point-to-request verifier on frozen RGB features and grouped labels.

The candidate point is an image-derived proposal at deployment. Scored geometry
creates positive and neighboring-port negative training pairs only. This is a
diagnostic, not a visibility or safe-handoff model.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from act_cache import load_cached_images
from train_two_family_port_locator import Locator


CAMERAS=('center','left','right')
NAMES=[f'sfp_card_{card}_port_{port}' for card in range(5) for port in range(2)]


class Correspondence(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(256+22+1,64,3,padding=1),nn.GELU(),
                                nn.Conv2d(64,32,3,padding=1),nn.GELU())
        self.final=nn.Sequential(nn.Linear(256+64+22,128),nn.GELU(),nn.Linear(128,1))

    def forward(self,visual,condition,point):
        b,_,h,w=visual.shape
        xx=torch.linspace(0,1,w,device=visual.device)[None,None,None,:]
        yy=torch.linspace(0,1,h,device=visual.device)[None,None,:,None]
        marker=torch.exp(-((xx-point[:,0,None,None,None]).square()+
                            (yy-point[:,1,None,None,None]).square())/(2*.065**2))
        field=torch.cat((visual,condition[:,:,None,None].expand(-1,-1,h,w),
                         marker.expand(-1,-1,h,w)),1)
        x=self.conv(field)
        grid=(point*2-1)[:,None,None,:]
        local=F.grid_sample(visual,grid,align_corners=True).flatten(1)
        joined=torch.cat((local,x.mean((-1,-2)),x.amax((-1,-2)),condition),1)
        return self.final(joined).squeeze(-1)


def spread(rows,cap):
    rows=sorted(rows,key=lambda x:x['time_s'])
    return [rows[int(i)] for i in np.linspace(0,len(rows)-1,min(len(rows),cap),dtype=int)]


def load_labeled_views(path,cache_root,frames_per_group):
    all_rows=defaultdict(list)
    for line in path.open():
        row=json.loads(line)
        if row['family']=='sfp':all_rows[(row['split'],row['scene_group'])].append(row)
    cache_info=json.loads((cache_root/'cache.json').read_text())
    cache=load_cached_images(cache_root,cache_info)
    records=[];images=[]
    for (split,group),episode in sorted(all_rows.items()):
        for r in spread(episode,frames_per_group):
            selected=r['selected_identity']
            if selected not in NAMES:continue
            for cam in CAMERAS:
                targets={t['identity']:t['cameras'][cam] for t in r['targets']
                         if t['identity'] in NAMES and bool(t['cameras'][cam][3])}
                if selected not in targets or len(targets)<2:continue
                true=np.asarray(targets[selected][:2],np.float32)
                other=min((name for name in targets if name!=selected),
                          key=lambda name:np.linalg.norm(np.asarray(targets[name][:2])-true))
                wrong=np.asarray(targets[other][:2],np.float32)
                if np.linalg.norm(wrong-true)<24:continue
                absent=next((name for name in NAMES if name not in targets and name!=selected),other)
                image=np.ascontiguousarray(cache[f'observation.images.{cam}_camera'][r['images'][cam]])
                if image.shape[:2]!=(256,288):
                    image=cv2.resize(image,(288,256),interpolation=cv2.INTER_AREA)
                records.append({'split':split,'group':group,'episode':r['episode'],
                    'camera':cam,'selected':selected,'negative_identity':other,
                    'absent_request_identity':absent,
                    'state':r['state'][:7],'true_xy':(true/[1151,1023]).tolist(),
                    'negative_xy':(wrong/[1151,1023]).tolist(),
                    'image_key':int(r['images'][cam])})
                images.append(image)
    if not records:raise ValueError('No valid labeled camera views')
    return records,np.stack(images)


def load_offscreen_views(path,cap_per_group):
    """Native RGB with requested target outside a view and a distractor inside."""
    grouped=defaultdict(list)
    for line in path.open():
        r=json.loads(line)
        if r['family']=='sfp':grouped[(r['split'],r['scene_group'])].append(r)
    records=[];images=[]
    for (split,group),episode in sorted(grouped.items()):
        valid=[]
        for r in episode:
            for cam in CAMERAS:
                all_targets={t['identity']:t['cameras'][cam] for t in r['targets']
                             if t['identity'] in NAMES}
                requested=r['selected_identity']
                if requested not in all_targets or bool(all_targets[requested][3]):continue
                visible=[name for name,point in all_targets.items()
                         if name!=requested and bool(point[3])]
                if not visible:continue
                intended=np.asarray(all_targets[requested][:2],np.float32)
                other=min(visible,key=lambda name:np.linalg.norm(
                          np.asarray(all_targets[name][:2])-intended))
                valid.append((r,cam,other,all_targets[other][:2]))
        if cap_per_group and len(valid)>cap_per_group:
            valid=[valid[int(i)] for i in np.linspace(0,len(valid)-1,cap_per_group,dtype=int)]
        for r,cam,other,point in valid:
            image=cv2.imread(r['images'][cam])
            if image is None:raise FileNotFoundError(r['images'][cam])
            image=cv2.cvtColor(cv2.resize(image,(288,256),interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)
            records.append({'split':split,'group':group,'episode':r['episode'],
                'camera':cam,'selected':r['selected_identity'],'negative_identity':other,
                'absent_request_identity':other,'state':r['state'][:7],
                'true_xy':(np.asarray(point)/[1151,1023]).tolist(),
                'negative_xy':(np.asarray(point)/[1151,1023]).tolist(),
                'offscreen_only':True,'image':r['images'][cam]})
            images.append(image)
    if not records:raise ValueError('No actual offscreen-request examples')
    return records,np.stack(images)


def load_native_onscreen_views(path,cap_per_group):
    """Matched positive views from the same native reset groups as offscreen negatives."""
    grouped=defaultdict(list)
    for line in path.open():
        r=json.loads(line)
        if r['family']=='sfp':grouped[(r['split'],r['scene_group'])].append(r)
    records=[];images=[]
    for (split,group),episode in sorted(grouped.items()):
        valid=[]
        for r in episode:
            for cam in CAMERAS:
                targets={t['identity']:t['cameras'][cam] for t in r['targets']
                         if t['identity'] in NAMES and bool(t['cameras'][cam][3])}
                requested=r['selected_identity']
                if requested not in targets or len(targets)<2:continue
                true=np.asarray(targets[requested][:2],np.float32)
                other=min((name for name in targets if name!=requested),
                          key=lambda name:np.linalg.norm(np.asarray(targets[name][:2])-true))
                if np.linalg.norm(np.asarray(targets[other][:2])-true)<24:continue
                valid.append((r,cam,other,targets[requested][:2],targets[other][:2]))
        if cap_per_group and len(valid)>cap_per_group:
            valid=[valid[int(i)] for i in np.linspace(0,len(valid)-1,cap_per_group,dtype=int)]
        for r,cam,other,true,wrong in valid:
            image=cv2.imread(r['images'][cam])
            if image is None:raise FileNotFoundError(r['images'][cam])
            image=cv2.cvtColor(cv2.resize(image,(288,256),interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)
            records.append({'split':split,'group':group,'episode':r['episode'],
                'camera':cam,'selected':r['selected_identity'],'negative_identity':other,
                'absent_request_identity':next((name for name in NAMES if name not in targets),other),
                'state':r['state'][:7],
                'true_xy':(np.asarray(true)/[1151,1023]).tolist(),
                'negative_xy':(np.asarray(wrong)/[1151,1023]).tolist(),
                'native_onscreen':True,'image':r['images'][cam]})
            images.append(image)
    if not records:raise ValueError('No native onscreen requested-port examples')
    return records,np.stack(images)


@torch.inference_mode()
def frozen_maps(locator,images,device):
    out=[];locator.eval()
    for i in range(0,len(images),48):
        x=torch.from_numpy(np.ascontiguousarray(images[i:i+48])).permute(0,3,1,2).to(device).float()/255
        x=(x-locator.mean)/locator.std
        x=locator.backbone.maxpool(locator.backbone.relu(locator.backbone.bn1(locator.backbone.conv1(x))))
        x=locator.backbone.layer3(locator.backbone.layer2(locator.backbone.layer1(x)))
        out.append(x.half().cpu())
    return torch.cat(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--sfp-cache',type=Path,required=True)
    p.add_argument('--locator',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames-per-group',type=int,default=4)
    p.add_argument('--updates',type=int,default=800)
    p.add_argument('--counterfactual-requests',action='store_true',
                   help='Add visible-neighbor and absent/offscreen request swaps on the same RGB')
    p.add_argument('--offscreen-labels',type=Path,
                   help='Native RGB labels with actual requested-port offscreen cases')
    p.add_argument('--offscreen-cap-per-group',type=int,default=120)
    p.add_argument('--offscreen-loss-weight',type=float,default=.5)
    p.add_argument('--balanced-native-onscreen',action='store_true')
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior result')
    a.output.mkdir(parents=True)
    torch.manual_seed(20261003);torch.set_num_threads(4);cv2.setNumThreads(1)
    records,images=load_labeled_views(a.labels,a.sfp_cache,a.frames_per_group)
    ordinary_count=len(records)
    if a.offscreen_labels:
        offscreen_records,offscreen_images=load_offscreen_views(
            a.offscreen_labels,a.offscreen_cap_per_group)
        overlap={r['group'] for r in records}&{r['group'] for r in offscreen_records}
        if overlap:raise ValueError(f'Group overlap with original source: {overlap}')
        records.extend(offscreen_records)
        images=np.concatenate((images,offscreen_images),0)
        if a.balanced_native_onscreen:
            onscreen_records,onscreen_images=load_native_onscreen_views(
                a.offscreen_labels,a.offscreen_cap_per_group)
            records.extend(onscreen_records)
            images=np.concatenate((images,onscreen_images),0)
    locator=Locator(None).to(a.device).eval()
    locator.load_state_dict(torch.load(a.locator,map_location='cpu',weights_only=False)['model'])
    visual=frozen_maps(locator,images,a.device).to(a.device).float()
    state=torch.tensor([r['state'] for r in records],device=a.device,dtype=torch.float32)
    train_mask=torch.tensor([r['split']=='train' for r in records],device=a.device)
    state_mean=state[train_mask].mean(0);state_std=state[train_mask].std(0).clamp_min(.02)
    state=(state-state_mean)/state_std
    camera=torch.tensor([[float(r['camera']==c) for c in CAMERAS] for r in records],
                        device=a.device)
    request=torch.tensor([[float(r['selected']==name) for name in NAMES] for r in records],
                         device=a.device)
    condition=torch.cat((request,camera,state,torch.zeros((len(records),2),device=a.device)),1)
    true=torch.tensor([r['true_xy'] for r in records],device=a.device)
    wrong=torch.tensor([r['negative_xy'] for r in records],device=a.device)
    train_indices=np.asarray([i for i,r in enumerate(records)
                              if r['split']=='train' and not r.get('offscreen_only')])
    offscreen_train_indices=np.asarray([i for i,r in enumerate(records)
                                        if r['split']=='train' and r.get('offscreen_only')])
    if not len(train_indices):raise ValueError('No train split')
    model=Correspondence().to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01)
    rng=np.random.default_rng(20261003);history=[]
    neighbor_request=torch.tensor([NAMES.index(r['negative_identity']) for r in records],device=a.device)
    absent_request=torch.tensor([NAMES.index(r['absent_request_identity']) for r in records],device=a.device)
    for step in range(1,a.updates+1):
        choices=32 if a.counterfactual_requests else 48
        base=torch.as_tensor(rng.choice(train_indices,choices,replace=True),device=a.device)
        variants=4 if a.counterfactual_requests else 2
        index=base.repeat_interleave(variants)
        variant=torch.arange(len(index),device=a.device)%variants
        label=variant==0
        xy=torch.where((variant==1)[:,None],wrong[index],true[index])
        xy=(xy+torch.randn_like(xy)*(.008 if step%2 else .004)).clamp(0,1)
        meta=condition[index].clone();meta[:,-2:]=xy
        if a.counterfactual_requests:
            swapped=(variant==2)|(variant==3)
            chosen=torch.where(variant==2,neighbor_request[index],absent_request[index])
            meta[swapped,:10]=F.one_hot(chosen[swapped],10).float()
        logits=model(visual[index],meta,xy)
        loss=F.binary_cross_entropy_with_logits(logits,label.float(),
             pos_weight=torch.tensor(variants-1.,device=a.device))
        if len(offscreen_train_indices):
            off_index=torch.as_tensor(rng.choice(offscreen_train_indices,32,replace=True),
                                      device=a.device)
            off_xy=(wrong[off_index]+torch.randn_like(wrong[off_index])*.004).clamp(0,1)
            off_meta=condition[off_index].clone();off_meta[:,-2:]=off_xy
            off_logits=model(visual[off_index],off_meta,off_xy)
            loss+=a.offscreen_loss_weight*F.binary_cross_entropy_with_logits(off_logits,
                                                         torch.zeros_like(off_logits))
        opt.zero_grad();loss.backward();opt.step()
        if step%200==0:history.append({'update':step,'train_loss':float(loss.detach())})
    model.eval();probs={}
    with torch.inference_mode():
        for kind,points in (('correct',true),('neighbor',wrong)):
            values=[]
            for i in range(0,len(records),128):
                xy=points[i:i+128];meta=condition[i:i+128].clone();meta[:,-2:]=xy
                values.extend(model(visual[i:i+128],meta,xy).sigmoid().cpu().tolist())
            probs[kind]=values
    if a.counterfactual_requests:
        counter_values=[]
        with torch.inference_mode():
            for i in range(0,len(records),128):
                xy=true[i:i+128];meta=condition[i:i+128].clone();meta[:,-2:]=xy
                meta[:,:10]=F.one_hot(absent_request[i:i+128],10).float()
                counter_values.extend(model(visual[i:i+128],meta,xy).sigmoid().cpu().tolist())
    else:counter_values=[None]*len(records)
    metrics={};by_group=defaultdict(lambda:{'correct_total':0,'correct_accepted':0,
        'wrong_total':0,'wrong_accepted':0,'counterfactual_total':0,
        'counterfactual_accepted':0})
    for i,r in enumerate(records[:ordinary_count]):
        m=by_group[(r['split'],r['group'])]
        m['correct_total']+=1;m['wrong_total']+=1
        m['correct_accepted']+=probs['correct'][i]>=.5
        m['wrong_accepted']+=probs['neighbor'][i]>=.5
        if a.counterfactual_requests:
            m['counterfactual_total']+=1
            m['counterfactual_accepted']+=counter_values[i]>=.5
    for split in ('train','validation'):
        groups=[v for (s,_),v in by_group.items() if s==split]
        metrics[split]={'groups':len(groups),**{k:int(sum(v[k] for v in groups))
            for k in ('correct_total','correct_accepted','wrong_total','wrong_accepted',
                      'counterfactual_total','counterfactual_accepted')}}
    if a.offscreen_labels:
        for split in ('train','development','validation'):
            indices=[i for i,r in enumerate(records) if r.get('offscreen_only') and r['split']==split]
            metrics[f'{split}_actual_offscreen']={'groups':len({records[i]['group'] for i in indices}),
                'views':len(indices),'false_accepted':sum(probs['neighbor'][i]>=.5 for i in indices)}
            indices=[i for i,r in enumerate(records) if r.get('native_onscreen') and r['split']==split]
            metrics[f'{split}_native_onscreen']={'groups':len({records[i]['group'] for i in indices}),
                'views':len(indices),'correct_accepted':sum(probs['correct'][i]>=.5 for i in indices),
                'wrong_accepted':sum(probs['neighbor'][i]>=.5 for i in indices)}
    torch.save({'model':model.state_dict(),'state_mean':state_mean.cpu(),
        'state_std':state_std.cpu(),'locator':str(a.locator),'names':NAMES,
        'label_kind':'candidate point association; simulator projection training labels only'},
        a.output/'correspondence.pt')
    summary={'schema':'sfp_candidate_correspondence_probe/v1',
        'hypothesis':'condition proposed point on requested card/port identity and spatial image evidence',
        'threshold':.5,'updates':a.updates,'rows':len(records),'metrics':metrics,
        'counterfactual_requests':a.counterfactual_requests,
        'offscreen_labels':str(a.offscreen_labels) if a.offscreen_labels else None,
        'offscreen_cap_per_group':a.offscreen_cap_per_group,
        'offscreen_loss_weight':a.offscreen_loss_weight,
        'balanced_native_onscreen':a.balanced_native_onscreen,
        'validation_by_group':{g:v for (s,g),v in by_group.items() if s=='validation'},
        'history':history,'labels':str(a.labels),'sfp_cache':str(a.sfp_cache),
        'checkpoint':str(a.output/'correspondence.pt'),
        'geometry_runtime_input':False,'final_scenes_opened':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'rows':len(records),'metrics':metrics},indent=2))


if __name__=='__main__':main()
