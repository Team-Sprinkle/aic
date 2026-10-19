#!/usr/bin/env python3
"""Fit only two SC in-frame output rows using one-port and two-port RGB.

This is a bounded presence diagnostic. Projection labels are supervision only;
no true geometry enters the actor or this model at inference.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from train_two_family_port_locator import Locator


CAMERAS = ('center','left','right')


def spread(rows, cap):
    rows = sorted(rows, key=lambda r:r['time_s'])
    return [rows[int(i)] for i in np.linspace(0,len(rows)-1,min(len(rows),cap),dtype=int)]


def load_rows(args):
    two = defaultdict(list)
    for line in args.two_port_labels.open():
        r = json.loads(line)
        if r['family'] == 'sc':
            two[(r['split'],r['scene_group'])].append(r)
    result = []
    for (split,group),rows in two.items():
        for r in spread(rows,args.two_port_frames_per_group):
            for cam in CAMERAS:
                bits = [next((bool(t['cameras'][cam][3]) for t in r['targets']
                              if t['identity']==f'sc_port_{p}'),False) for p in (0,1)]
                result.append({'split':split,'group':group,'source':'two_port',
                               'camera':cam,'image':r['images'][cam],
                               'state':r['state'][:7],
                               'labels':[float(v) for v in bits]})
    for split,paths in (('train',args.one_port_train),('development',args.one_port_development)):
        for path in paths:
            rows = [json.loads(x) for x in path.open()]
            by_camera = {cam:sorted((r for r in rows if r['camera']==cam),
                                    key=lambda r:r['sim_ns']) for cam in CAMERAS}
            if any(not values for values in by_camera.values()):
                raise ValueError(f'Missing camera: {path}')
            for camera in CAMERAS:
                values = by_camera[camera]
                sampled = [values[int(i)] for i in np.linspace(0,len(values)-1,
                           min(len(values),args.one_port_frames_per_group),dtype=int)]
                for r in sampled:
                    bits = [0.,0.]
                    bits[int(r['target_port'])] = float(r['projected_inframe'])
                    result.append({'split':split,'group':r['scene_group_id'],
                                   'source':'one_port','camera':r['camera'],
                                   'image':r['image'],'state':r['measured_tcp_base_pose'],
                                   'labels':bits,
                                   'relative_opening_tcp_m':r['relative_opening_tcp_m']})
    by_group = defaultdict(set)
    for row in result:by_group[row['group']].add(row['split'])
    leakage = {g:s for g,s in by_group.items() if len(s)>1}
    if leakage:raise ValueError(f'Scene-group leakage: {leakage}')
    return result


@torch.inference_mode()
def feature_cache(model,rows,device):
    model.eval()
    features=[]
    labels=[]
    for i in range(0,len(rows),24):
        batch=rows[i:i+24]
        rgb=[]
        meta=[]
        for row in batch:
            image=cv2.imread(row['image'])
            if image is None:raise FileNotFoundError(row['image'])
            image=cv2.cvtColor(cv2.resize(image,(288,256),interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)
            rgb.append(image)
            meta.append([1.,0.,*[float(row['camera']==c) for c in CAMERAS]])
        x=torch.from_numpy(np.stack(rgb)).permute(0,3,1,2).to(device).float()/255
        m=torch.tensor(meta,device=device)
        x=(x-model.mean)/model.std
        x=model.backbone.maxpool(model.backbone.relu(model.backbone.bn1(model.backbone.conv1(x))))
        x=model.backbone.layer3(model.backbone.layer2(model.backbone.layer1(x)))
        hidden=F.gelu(model.inframe[0](torch.cat((x.mean((-1,-2)),m),-1)))
        features.append(hidden.cpu())
        labels.extend(r['labels'] for r in batch)
    return torch.cat(features),torch.tensor(labels,dtype=torch.float32)


def score(rows,logits):
    groups=defaultdict(list)
    for row,logit in zip(rows,logits):
        groups[(row['split'],row['source'])].append((row,logit))
    result={}
    for key,items in groups.items():
        values=np.asarray([v for _,v in items]);truth=np.asarray([r['labels'] for r,_ in items])
        pred=values>=0
        result['/'.join(key)]={'views':len(items),'groups':len({r['group'] for r,_ in items}),
            'absent_false_positive':int(((truth==0)&pred).sum()),
            'absent_total':int((truth==0).sum()),
            'present_recalled':int(((truth==1)&pred).sum()),
            'present_total':int((truth==1).sum())}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--two-port-labels',type=Path,required=True)
    p.add_argument('--one-port-train',nargs='+',type=Path,required=True)
    p.add_argument('--one-port-development',nargs='+',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--two-port-frames-per-group',type=int,default=6)
    p.add_argument('--one-port-frames-per-group',type=int,default=32)
    p.add_argument('--updates',type=int,default=500)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior run')
    a.output.mkdir(parents=True)
    cv2.setNumThreads(1);torch.set_num_threads(4);torch.manual_seed(20261002)
    rows=load_rows(a)
    model=Locator(None).to(a.device).eval()
    baseline=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model']
    model.load_state_dict(baseline)
    features,labels=feature_cache(model,rows,a.device)
    initial_weight=model.inframe[2].weight.detach().cpu().clone()
    initial_bias=model.inframe[2].bias.detach().cpu().clone()
    base=features@initial_weight[:2].T+initial_bias[:2]
    correction_weight=torch.zeros_like(initial_weight[:2],requires_grad=True)
    correction_bias=torch.zeros_like(initial_bias[:2],requires_grad=True)
    opt=torch.optim.AdamW([correction_weight,correction_bias],lr=.003,weight_decay=.01)
    train_groups=defaultdict(list)
    for i,r in enumerate(rows):
        if r['split']=='train':train_groups[r['source']].append(i)
    if not all(train_groups.values()):raise ValueError('Missing source class')
    rng=np.random.default_rng(20261002)
    history=[]
    for step in range(1,a.updates+1):
        indices=np.r_[rng.choice(train_groups['one_port'],64,replace=True),
                      rng.choice(train_groups['two_port'],64,replace=True)]
        index=torch.tensor(indices,dtype=torch.long)
        logits=base[index]+features[index]@correction_weight.T+correction_bias
        loss=F.binary_cross_entropy_with_logits(logits,labels[index])
        loss+=.001*correction_weight.square().mean()
        opt.zero_grad();loss.backward();opt.step()
        if step%100==0:history.append({'step':step,'loss':float(loss.detach())})
    state={k:v.clone() for k,v in baseline.items()}
    state['inframe.2.weight'][:2]+=correction_weight.detach()
    state['inframe.2.bias'][:2]+=correction_bias.detach()
    for key in state:
        if key not in ('inframe.2.weight','inframe.2.bias') and not torch.equal(state[key],baseline[key]):
            raise AssertionError(key)
    if not torch.equal(state['inframe.2.weight'][2:],baseline['inframe.2.weight'][2:]):
        raise AssertionError('SFP output rows changed')
    torch.save({'model':state,'parent_checkpoint':str(a.checkpoint),
                'SC_presence_head_only':True,'projection_not_true_visibility':True},a.output/'head_only.pt')
    old=score(rows,base.numpy())
    new=score(rows,(base+features@correction_weight.detach().T+correction_bias.detach()).numpy())
    summary={'schema':'aic_sc_presence_head_ablation/v1','rows':len(rows),
             'train_groups':len({r['group'] for r in rows if r['split']=='train'}),
             'development_groups':len({r['group'] for r in rows if r['split']=='development'}),
             'two_port_validation_groups':len({r['group'] for r in rows if r['split']=='validation'}),
             'updates':a.updates,'feature_extractor_and_pose_decoder_frozen':True,
             'only_two_SC_presence_output_rows_changed':True,
             'SFP_outputs_bitwise_unchanged':True,
             'before':old,'after':new,'history':history,
             'one_port_train':[str(x) for x in a.one_port_train],
             'one_port_development':[str(x) for x in a.one_port_development],
             'two_port_labels':str(a.two_port_labels),
             'labels_are_projected_inframe_not_true_visibility':True,
             'final_scenes_opened':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'before':old,'after':new},indent=2))


if __name__=='__main__':main()
