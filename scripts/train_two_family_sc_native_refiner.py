#!/usr/bin/env python3
"""Refine an RGB-selected SC aperture crop at native camera resolution.

The frozen coarse RGB model chooses every crop in both splits. Projected TF
coordinates supervise and score the refiner, but never choose its crop.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler

from train_sc_native_keypoints import PretrainedKeypointRefiner


CAMERAS=('center','left','right')
SIZE=160
NETWORK_INPUT_SIZE=160


def stats(x):
    a=np.asarray(x,float)
    return {'n':len(a),'median':float(np.median(a)) if len(a) else None,
            'p95':float(np.quantile(a,.95)) if len(a) else None}


class Crops(Dataset):
    def __init__(self,examples,training):self.examples=examples;self.training=training
    def __len__(self):return len(self.examples)
    def __getitem__(self,i):
        example=self.examples[i]
        row,cam,coarse,truth=example[:4]
        port=int(example[4]) if len(example)>4 else 0
        rgb=cv2.imread(row['images'][cam],cv2.IMREAD_COLOR)
        if rgb is None:raise FileNotFoundError(row['images'][cam])
        if rgb.shape[:2]!=(1024,1152):raise ValueError(rgb.shape)
        jitter=np.random.uniform(-12,12,2) if self.training else np.zeros(2)
        center=np.asarray(coarse)+jitter
        left=int(round(center[0]-SIZE/2));top=int(round(center[1]-SIZE/2))
        crop=cv2.warpAffine(rgb,np.array([[1,0,-left],[0,1,-top]],np.float32),
                            (SIZE,SIZE),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        if NETWORK_INPUT_SIZE != SIZE:
            crop=cv2.resize(crop,(NETWORK_INPUT_SIZE,NETWORK_INPUT_SIZE),interpolation=cv2.INTER_AREA)
        crop=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
        point=(np.asarray(truth)-[left,top])/(SIZE-1)
        return (torch.from_numpy(crop.copy()).permute(2,0,1),torch.from_numpy(point.astype('float32')),
                torch.tensor(coarse,dtype=torch.float32),torch.tensor(truth,dtype=torch.float32),
                row['episode'],cam,torch.tensor([float(port==0),float(port==1)]))


@torch.inference_mode()
def evaluate(model,loader):
    model.eval();errors=[];coarse=[];inside=[];by_episode=defaultdict(list)
    selected=[]
    for image,truth,center,full_truth,episode,cam,port in loader:
        pred=model(image.cuda(non_blocking=True).float()/255.,
                   port.cuda(non_blocking=True) if model.condition_target_port else None).float().cpu().numpy()
        pred_px=pred.reshape(-1,2)*(SIZE-1)+np.round(center.numpy()-SIZE/2)
        target=full_truth.numpy()
        err=np.linalg.norm(pred_px-target,axis=1)
        errors.extend(err.tolist());coarse.extend(np.linalg.norm(center.numpy()-target,axis=1).tolist())
        for j,e in enumerate(err):
            example=loader.dataset.examples[len(errors)-len(err)+j]
            if len(example)<6 or example[5]:selected.append(float(e))
        in_crop=np.max(np.abs(center.numpy()-target),axis=1)<SIZE/2-16
        inside.extend(err[in_crop].tolist())
        for j,e in enumerate(err):by_episode[episode[j]].append(float(e))
    return {'refined_px':stats(errors),'selected_refined_px':stats(selected),'coarse_px':stats(coarse),
            'inside_crop_refined_px':stats(inside),
            'inside_crop_fraction':len(inside)/len(errors) if errors else None,
            'per_episode_px_p95':{k:float(np.quantile(v,.95)) for k,v in by_episode.items()}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--coarse-predictions',type=Path,required=True)
    p.add_argument('--imagenet',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=2200)
    p.add_argument('--minutes',type=float,default=45)
    p.add_argument('--family',choices=('sc','sfp'),default='sc')
    p.add_argument('--crop-size',type=int,default=160,
                   help='Native-pixel crop width; frozen coarse RGB estimate picks its center')
    p.add_argument('--network-input-size',type=int,default=160,
                   help='Input width after cropping native pixels')
    p.add_argument('--workers',type=int,default=6,
                   help='Use zero to avoid OpenCV/PyTorch loader forks when they stall')
    p.add_argument('--selection-metric',choices=('all_p95','inside_crop_p95'),default='all_p95')
    p.add_argument('--selection-episodes',type=int,
                   help='Use this many validation episodes for checkpoint selection; hold the rest for one final check')
    p.add_argument('--selection-manifest',type=Path,
                   help='Locator summary containing native_selection_episodes; share its untouched test groups')
    p.add_argument('--condition-port',action='store_true',
                   help='Condition the broad SFP crop on the requested port side')
    p.add_argument('--include-sibling',action='store_true',
                   help='Train both port targets from the same RGB-selected card crop')
    a=p.parse_args()
    if a.include_sibling and (a.family!='sfp' or not a.condition_port):
        p.error('Sibling supervision requires SFP and port conditioning')
    global SIZE,NETWORK_INPUT_SIZE
    SIZE=a.crop_size;NETWORK_INPUT_SIZE=a.network_input_size
    if not 160<=SIZE<=640 or not 160<=NETWORK_INPUT_SIZE<=320:
        p.error('Invalid crop or network size')
    if a.output.exists():p.error('Preserve existing run')
    a.output.mkdir(parents=True)
    torch.set_num_threads(8);torch.manual_seed(20261002);np.random.seed(20261002);random.seed(20261002)
    rows=[json.loads(l) for l in (a.labels/'rows.jsonl').open()]
    pred={(r['row'],r['camera']):r for r in map(json.loads,a.coarse_predictions.open())}
    grouped=defaultdict(list);rejected=defaultdict(int)
    for i,r in enumerate(rows):
        if r['family']!=a.family:continue
        target=next(t for t in r['targets'] if t['identity']==r['selected_identity'])
        if np.linalg.norm(np.asarray(target['position_base_m'])-r['state'][:3])>=.2:
            rejected['tcp_beyond_200mm']+=3;continue
        for cam in CAMERAS:
            gt=target['cameras'][cam]
            if not gt[3]:rejected['projected_offscreen']+=1;continue
            coarse=pred[(i,cam)]['predicted'][r['selected_identity']]['xy']
            if pred[(i,cam)]['predicted'][r['selected_identity']]['projected_inframe_probability']<.5:
                rejected['coarse_inframe_negative']+=1;continue
            if max(abs(gt[0]-coarse[0]),abs(gt[1]-coarse[1]))>SIZE/2-16:
                rejected['outside_observation_selected_crop']+=1
                if r['split']=='train':continue
            port=int(r['selected_identity'].split('_')[-1]) if a.condition_port else 0
            grouped[r['split']].append((r,cam,coarse,gt[:2],port,True))
            if a.include_sibling:
                card=int(r['selected_identity'].split('_')[2])
                sibling=next((t for t in r['targets'] if t['identity']==f'sfp_card_{card}_port_{1-port}'),None)
                if sibling is not None:
                    other=sibling['cameras'][cam]
                    if other[3] and max(abs(other[0]-coarse[0]),abs(other[1]-coarse[1]))<=SIZE/2-16:
                        grouped[r['split']].append((r,cam,coarse,other[:2],1-port,False))
    train=grouped['train'];all_val=grouped['validation']
    if a.selection_manifest:
        selector=set(json.loads(a.selection_manifest.read_text())['native_selection_episodes'])
        names={e[0]['episode'] for e in all_val}
        if not selector or not names-selector or not selector<=names:
            raise ValueError('Selection manifest does not partition the validation episodes')
        val=[e for e in all_val if e[0]['episode'] in selector]
        test=[e for e in all_val if e[0]['episode'] not in selector]
    elif a.selection_episodes:
        names=sorted({e[0]['episode'] for e in all_val})
        if not 0<a.selection_episodes<len(names):raise ValueError('Need selection and held-out test episodes')
        selector=set(names[:a.selection_episodes])
        val=[e for e in all_val if e[0]['episode'] in selector]
        test=[e for e in all_val if e[0]['episode'] not in selector]
    else:val=all_val;test=[]
    if not train or not val:raise ValueError('Empty grouped split')
    scenes={s:{e[0]['scene_group'] for e in grouped[s]} for s in ('train','validation')}
    if scenes['train']&scenes['validation']:raise ValueError('Scene group leakage')
    counts=defaultdict(int)
    for r,*_ in train:counts[r['episode']]+=1
    weights=[1/counts[r['episode']] for r,*_ in train]
    sampler=WeightedRandomSampler(weights,num_samples=a.updates*32,replacement=True)
    loader_options={'num_workers':a.workers,'persistent_workers':a.workers>0,'pin_memory':True}
    train_loader=DataLoader(Crops(train,True),batch_size=32,sampler=sampler,**loader_options)
    val_loader=DataLoader(Crops(val,False),batch_size=96,shuffle=False,**loader_options)
    test_loader=(DataLoader(Crops(test,False),batch_size=96,shuffle=False,
                            **loader_options) if test else None)
    model=PretrainedKeypointRefiner(a.imagenet,a.condition_port,point_count=1).cuda()
    optimizer=torch.optim.AdamW((w for w in model.parameters() if w.requires_grad),lr=2e-4,weight_decay=1e-4)
    history=[];best=float('inf');best_step=0;stale=0;started=time.monotonic();reason='update_cap'
    for step,(image,target,_,_,_,_,port) in enumerate(train_loader,start=1):
        if step>a.updates:break
        if time.monotonic()-started>a.minutes*60:reason='wall_budget';break
        model.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(image.cuda(non_blocking=True).float()/255.,
                         port.cuda(non_blocking=True) if a.condition_port else None)
            loss=F.smooth_l1_loss(output.float(),target.cuda(non_blocking=True),beta=.01)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%100==0:
            report=evaluate(model,val_loader)
            score=(report['selected_refined_px']['p95'] if a.include_sibling else report['refined_px']['p95']) \
                if a.selection_metric=='all_p95' else report['inside_crop_refined_px']['p95']
            history.append({'step':step,'loss':float(loss.detach()),'validation':report})
            print(json.dumps({'step':step,'loss':float(loss.detach()),'coarse_p95':report['coarse_px']['p95'],
                              'refined_p95':score}),flush=True)
            if score<best-.05:
                best=score;best_step=step;stale=0
                torch.save({'model':model.state_dict(),'coarse_checkpoint':str(a.coarse_predictions),
                            'crop_size':SIZE,'network_input_size':NETWORK_INPUT_SIZE,
                            'condition_target_port':a.condition_port,'include_sibling':a.include_sibling,
                            'runtime_geometry':False,'visibility_certified':False},a.output/'best.pt')
            else:stale+=1
            if step>=600 and stale>=7:reason='validation_plateau';break
    model.load_state_dict(torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)['model'])
    final=evaluate(model,val_loader)
    fresh=evaluate(model,test_loader) if test_loader else None
    with (a.output/'validation_predictions.jsonl').open('w') as file:
        with torch.inference_mode():
            for split_name,examples,loader in [('selection',val,val_loader),('test',test,test_loader)]:
                if loader is None:continue
                offset=0
                for image,_,center,full_truth,episode,camera,port in loader:
                    output=model(image.cuda(non_blocking=True).float()/255.,
                                 port.cuda(non_blocking=True) if a.condition_port else None).float().cpu().numpy()
                    pixel=output.reshape(-1,2)*(SIZE-1)+np.round(center.numpy()-SIZE/2)
                    for j in range(len(episode)):
                        row=examples[offset+j][0]
                        record={'episode':episode[j],'scene_group':row['scene_group'],
                                'time_s':row['time_s'],'camera':camera[j],
                                'target_port':int(port[j].argmax()),
                                'predicted_xy':pixel[j].tolist(),
                                'truth_xy_scoring_only':full_truth[j].numpy().tolist(),
                                'split_within_validation':split_name,
                                'runtime_crop_from_frozen_rgb':True}
                        file.write(json.dumps(record,separators=(',',':'))+'\n')
                    offset+=len(episode)
    report={'schema':'two_family_native_refiner/v1','family':a.family,'train_episodes':len({e[0]['episode'] for e in train}),
            'validation_episodes':len({e[0]['episode'] for e in val}),'train_views':len(train),
            'validation_views':len(val),'unseen_to_refiner_selection_episodes':len({e[0]['episode'] for e in test}),
            'unseen_to_refiner_selection_views':len(test),'unseen_to_refiner_selection':fresh,
            'rejected':dict(rejected),'best_step':best_step,
            'updates_completed':step,'stop_reason':reason,'validation':final,
            'runtime_crop_selected_from_frozen_rgb_only_model':True,'visibility_certified':False,
            'crop_size':SIZE,'network_input_size':NETWORK_INPUT_SIZE,
            'workers':a.workers,
            'selection_metric':a.selection_metric,
            'condition_target_port':a.condition_port,'include_sibling':a.include_sibling,
            'reserved_final_opened':False}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
