#!/usr/bin/env python3
"""Native-image same-card SFP port correspondence diagnostic.

The crop uses RGB-predicted pair locations; scored port locations provide only
labels and audits. The reserved final configurations are not read.
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

from train_sc_native_keypoints import PretrainedKeypointRefiner
from train_sfp_candidate_correspondence_probe import Correspondence, frozen_maps
from train_two_family_port_locator import Locator

CAMERAS=('center','left','right')


class NativeEncoder(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.model=PretrainedKeypointRefiner(None,False,point_count=1)
        self.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'])
        self.model.eval()

    def forward(self,image,points):
        m=self.model
        x=(image-m.mean)/m.std
        x=m.backbone.maxpool(m.backbone.relu(m.backbone.bn1(m.backbone.conv1(x))))
        deep=m.backbone.layer3(m.backbone.layer2(m.backbone.layer1(x)))
        local=[]
        for k in (0,1):
            grid=(points[:,k]*2-1)[:,None,None,:]
            local.append(F.grid_sample(deep,grid,align_corners=True).flatten(1))
        return torch.cat((deep.mean((-1,-2)),*local),1)


class Match(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean=nn.Parameter(torch.zeros(774),requires_grad=False)
        self.std=nn.Parameter(torch.ones(774),requires_grad=False)
        self.net=nn.Sequential(nn.Linear(774,128),nn.SiLU(),nn.Linear(128,64),
                               nn.SiLU(),nn.Linear(64,1))

    def forward(self,x):
        return self.net((x-self.mean)/self.std).squeeze(-1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--labels',type=Path,required=True)
    ap.add_argument('--predictions',type=Path,required=True)
    ap.add_argument('--refiner',type=Path,required=True)
    ap.add_argument('--baseline',type=Path,required=True,
                    help='Frozen low-resolution correspondence checkpoint for matched proposals')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',default='cuda:2')
    ap.add_argument('--frames-per-episode',type=int,default=30)
    ap.add_argument('--updates',type=int,default=1000)
    a=ap.parse_args()
    if a.output.exists():ap.error('Preserve previous experiment')
    a.output.mkdir(parents=True)
    torch.manual_seed(20261003);np.random.seed(20261003);torch.set_num_threads(4)
    cv2.setNumThreads(1)
    rows=[json.loads(x) for x in a.labels.open()]
    pred={(q['row'],q['camera']):q for q in map(json.loads,a.predictions.open())}
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        target=next(t for t in r['targets'] if t['identity']==r['selected_identity'])
        if np.linalg.norm(np.asarray(target['position_base_m'])-r['state'][:3])>=.1:continue
        card=int(r['selected_identity'].split('_')[2]);names=[f'sfp_card_{card}_port_{p}' for p in (0,1)]
        by_name={t['identity']:t for t in r['targets']}
        if not all(name in by_name and all(by_name[name]['cameras'][c][3] for c in CAMERAS)
                   for name in names):continue
        groups[r['episode']].append(i)
    selected=[]
    for ep,ids in sorted(groups.items()):
        selected += [ids[j] for j in np.unique(np.linspace(0,len(ids)-1,
                     min(a.frames_per_episode,len(ids)),dtype=int))]
    enc=NativeEncoder(a.refiner).to(a.device).eval()
    records=[];tensors=[];locations=[];full_low=[]
    for i in selected:
        r=rows[i];card=int(r['selected_identity'].split('_')[2])
        names=[f'sfp_card_{card}_port_{p}' for p in (0,1)]
        target={t['identity']:t for t in r['targets']}
        for cam in CAMERAS:
            candidate=np.asarray([pred[i,cam]['predicted'][n]['xy'] for n in names],np.float32)
            truth=np.asarray([target[n]['cameras'][cam][:2] for n in names],np.float32)
            center=candidate.mean(0);left,top=np.round(center-160).astype(int)
            bgr=cv2.imread(r['images'][cam],cv2.IMREAD_COLOR)
            if bgr is None:raise FileNotFoundError(r['images'][cam])
            full_low.append(cv2.cvtColor(cv2.resize(bgr,(288,256),
                            interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
            crop=cv2.warpAffine(bgr,np.asarray([[1,0,-left],[0,1,-top]],np.float32),
                                (320,320),borderMode=cv2.BORDER_CONSTANT)
            crop=cv2.resize(crop,(224,224),interpolation=cv2.INTER_AREA)
            tensors.append(torch.from_numpy(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1))
            locations.append((candidate-np.asarray([left,top]))/320)
            records.append({'episode':r['episode'],'group':r['scene_group'],'split':r['split'],
                            'camera':cam,'row':i,'card':card,'candidate_xy':candidate.tolist(),
                            'true_xy':truth.tolist(),'crop_center_xy':center.tolist(),
                            'image':r['images'][cam]})
    native=[]
    with torch.inference_mode():
        for start in range(0,len(records),64):
            image=torch.stack(tensors[start:start+64]).to(a.device).float()/255
            points=torch.tensor(np.asarray(locations[start:start+64]),device=a.device,dtype=torch.float32)
            native.append(enc(image,points).float().cpu().numpy())
    native=np.concatenate(native)
    x=[];y=[];meta=[];ambiguous=0
    for j,r in enumerate(records):
        candidate=np.asarray(r['candidate_xy']);truth=np.asarray(r['true_xy'])
        d=np.linalg.norm(candidate[:,None,:]-truth[None,:,:],axis=-1)
        for request in (0,1):
            for slot in (0,1):
                match=d[slot,request]<=30 and d[slot,request]+15<d[slot,1-request]
                wrong=d[slot,1-request]<=30 and d[slot,1-request]+15<d[slot,request]
                far=d[slot,request]>=60
                if not (match or wrong or far):ambiguous+=1;continue
                order=[slot,1-slot]
                # Keep candidate/other features and point locations in the same order;
                # never pass the slot's semantic name or scored target coordinate.
                local=np.r_[native[j,:256],native[j,256+256*slot:512+256*slot],
                            native[j,256+256*(1-slot):512+256*(1-slot)]]
                xy=np.asarray(locations[j])[order].reshape(-1)
                code=np.asarray([float(request==0),float(request==1)])
                x.append(np.r_[local,xy,code].astype(np.float32))
                y.append(float(match))
                meta.append({'record':j,'episode':r['episode'],'group':r['group'],
                             'split':r['split'],'request_port':request,'candidate_slot':slot,
                             'label':int(match),'distance_requested_px':float(d[slot,request]),
                             'distance_sibling_px':float(d[slot,1-request])})
    x=np.stack(x);y=np.asarray(y,np.float32)
    baseline_saved=torch.load(a.baseline,map_location='cpu',weights_only=False)
    low_locator=Locator(None).to(a.device).eval()
    low_locator.load_state_dict(torch.load(baseline_saved['locator'],map_location='cpu',weights_only=False)['model'])
    maps=frozen_maps(low_locator,np.stack(full_low),a.device).to(a.device).float()
    low_model=Correspondence().to(a.device).eval()
    low_model.load_state_dict(baseline_saved['model'])
    low_scores=[]
    with torch.inference_mode():
        for start in range(0,len(meta),128):
            block=meta[start:start+128]
            rec_ids=[q['record'] for q in block]
            request=torch.tensor([[float(f'sfp_card_{records[q["record"]]["card"]}_port_{q["request_port"]}'==name)
                                   for name in baseline_saved['names']] for q in block],device=a.device)
            camera=torch.tensor([[float(records[q['record']]['camera']==cam) for cam in CAMERAS]
                                 for q in block],device=a.device)
            state=torch.tensor([rows[records[j]['row']]['state'][:7] for j in rec_ids],
                               device=a.device,dtype=torch.float32)
            state=(state-baseline_saved['state_mean'].to(a.device))/baseline_saved['state_std'].to(a.device)
            point=torch.tensor([np.asarray(records[q['record']]['candidate_xy'][q['candidate_slot']])/
                                [1151,1023] for q in block],device=a.device,dtype=torch.float32)
            condition=torch.cat((request,camera,state,point),1)
            score=low_model(maps[rec_ids],condition,point).sigmoid().cpu().numpy()
            low_scores.extend(score.tolist())
    low_scores=np.asarray(low_scores)
    fit=np.asarray([i for i,q in enumerate(meta) if q['split']=='train'])
    val_episodes={f'card{card}_port{card%2}_cards3_validation' for card in range(5)}
    select=np.asarray([i for i,q in enumerate(meta) if q['episode'] in val_episodes])
    test=np.asarray([i for i,q in enumerate(meta) if q['split']=='validation'
                     and q['episode'] not in val_episodes])
    if not len(fit) or not len(select) or not len(test):raise ValueError('Missing grouped split')
    if {meta[i]['group'] for i in fit}&{meta[i]['group'] for i in np.r_[select,test]}:
        raise ValueError('Reset leakage')
    model=Match().to(a.device)
    model.mean.data=torch.from_numpy(x[fit].mean(0)).to(a.device)
    model.std.data=torch.from_numpy(np.maximum(x[fit].std(0),.03)).to(a.device)
    xx=torch.from_numpy(x).to(a.device);yy=torch.from_numpy(y).to(a.device)
    opt=torch.optim.AdamW(model.net.parameters(),lr=2e-4,weight_decay=1e-3)
    rng=np.random.default_rng(20261003)
    pos=fit[y[fit]==1];neg=fit[y[fit]==0]
    def evaluate(ids,probabilities=None):
        if probabilities is None:
            with torch.inference_mode():
                prob=model(xx[ids]).sigmoid().cpu().numpy()
        else:
            prob=probabilities[ids]
        target=y[ids];choice=prob>=.5
        episodes=sorted({meta[i]['episode'] for i in ids})
        by_triplet=defaultdict(list)
        for k,i in enumerate(ids):
            q=meta[i];r=records[q['record']]
            by_triplet[(r['row'],q['request_port'],q['candidate_slot'])].append((k,r['camera']))
        triplets=[]
        for key,items in by_triplet.items():
            if len(items)!=3 or {cam for _,cam in items}!=set(CAMERAS):continue
            places=[p for p,_ in items]
            truth_all=bool(np.all(target[places]==1))
            accept_all=bool(np.all(choice[places]))
            triplets.append((truth_all,accept_all))
        tp3=sum(true and accept for true,accept in triplets)
        fp3=sum(not true and accept for true,accept in triplets)
        fn3=sum(true and not accept for true,accept in triplets)
        return {'examples':len(ids),'episodes':len(episodes),'positive':int(target.sum()),
                'negative':int((1-target).sum()),'tp':int(np.sum(choice&(target==1))),
                'fp':int(np.sum(choice&(target==0))),
                'fn':int(np.sum((~choice)&(target==1))),
                'complete_triplets':len(triplets),'triplet_positive':sum(t for t,_ in triplets),
                'triplet_negative':sum(not t for t,_ in triplets),
                'triplet_tp':tp3,'triplet_fp':fp3,'triplet_fn':fn3,
                'by_episode':{ep:{'positive':int(sum(y[i]==1 for i in ids if meta[i]['episode']==ep)),
                                  'negative':int(sum(y[i]==0 for i in ids if meta[i]['episode']==ep))}
                              for ep in episodes}}
    history=[];best=-1e9;best_step=0
    for step in range(1,a.updates+1):
        batch=np.r_[rng.choice(pos,64),rng.choice(neg,64)]
        model.train();opt.zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(model(xx[batch]),yy[batch]);loss.backward();opt.step()
        if step%100==0:
            model.eval();res=evaluate(select)
            quality=res['tp']/(res['positive'] or 1)-2*res['fp']/(res['negative'] or 1)
            history.append({'step':step,'loss':float(loss.detach()),'selection':{k:res[k] for k in
                            ('positive','negative','tp','fp','fn')}})
            print(json.dumps(history[-1]),flush=True)
            if quality>best:
                best=quality;best_step=step
                torch.save({'model':model.state_dict(),'step':step,'schema':'sfp_native_pair_match/v1',
                            'runtime_geometry_input':False},a.output/'best.pt')
    model.load_state_dict(torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)['model'])
    model.eval()
    audit={'schema':'sfp_native_pair_crop_identity_audit/v1','source_rows':len(selected),
           'camera_views':len(records),'training_groups':len({meta[i]['group'] for i in fit}),
           'selection_groups':len({meta[i]['group'] for i in select}),
           'test_groups':len({meta[i]['group'] for i in test}),
           'excluded_ambiguous_pairs':ambiguous,'crop_size_original_px':320,
           'original_resolution': [1152,1024], 'network_input':[224,224],
           'opening_inside_crop_fraction':float(np.mean([np.max(np.abs(np.asarray(r['true_xy'])-
                                        np.asarray(r['crop_center_xy'])))<160 for r in records]))}
    summary={'schema':'sfp_native_pair_crop_identity_ablation/v1','audit':audit,
             'best_step':best_step,'selection':evaluate(select),'development_test':evaluate(test),
             'matched_frozen_lowres_baseline':{'selection':evaluate(select,low_scores),
                                              'development_test':evaluate(test,low_scores)},
             'history':history,'runtime_true_geometry_input':False,
             'visibility_certified':False,'final_scenes_opened':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.output/'manifest.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in records))
    with (a.output/'predictions.jsonl').open('w') as out:
        with torch.inference_mode():scores=model(xx).sigmoid().cpu().numpy()
        for i,q in enumerate(meta):
            out.write(json.dumps({**q,'crop_score':float(scores[i]),
                                  'frozen_lowres_score':float(low_scores[i]),
                                  'camera':records[q['record']]['camera']})+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
