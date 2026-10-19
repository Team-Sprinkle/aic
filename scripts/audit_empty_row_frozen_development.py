#!/usr/bin/env python3
"""Fixed-threshold replay on all four independent zero/five-card SC starts."""
import cv2
cv2.setNumThreads(2)
import fcntl
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

root=Path('/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1');folder=root/'empty_row_visual_ablation';output=folder/'frozen_development_review.json';assert not output.exists()
assert os.environ['CUDA_VISIBLE_DEVICES']=='3';lock=open('/var/tmp/chmin_aic_gpu3.lock','w');fcntl.flock(lock,fcntl.LOCK_EX)
torch.set_num_threads(3);data=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);backbone=resnet18(weights=None)
encoder=nn.Sequential(backbone.conv1,backbone.bn1,backbone.relu,backbone.maxpool,backbone.layer1,backbone.layer2,backbone.layer3,nn.AdaptiveAvgPool2d((2,2))).cuda().eval();encoder.load_state_dict(data['encoder'])
head=nn.Sequential(nn.Linear(3072,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1)).cuda().eval();head.load_state_dict(data['head'])
mu=torch.tensor(data['feature_mean'],device='cuda');scale=torch.tensor(data['feature_scale'],device='cuda');mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
manifest=json.loads((folder/'manifest.json').read_text());train_scenes={r['scene'] for r in manifest if r['split']=='train'};cases=[]
old=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1')
for case in json.loads((old/'V17_fresh_SC_development/evaluation_cases.json').read_text())['cases']:
    base=old/'V34_SC_matched_coverage'/case['case']/'runs';audit=next(p for p in base.iterdir() if p.is_dir() and (p/'scored_stream.json').exists())
    stream=json.loads((audit/'scored_stream.json').read_text());assert stream['scene_group_id'] not in train_scenes
    rows=[json.loads(line) for line in (audit/'runtime_trace.jsonl').open()];start=next(r['sim_time'] for r in rows if r['event']=='initial_force_reference_frozen');values=[]
    with torch.inference_mode():
        for stamp in np.arange(start,start+5,.2):
            chosen={c:max((r for r in stream['cameras'][c] if r[0]<=round(stamp*1e9)),key=lambda x:x[0]) for c in ('center','left','right')}
            images=np.stack([cv2.resize(cv2.cvtColor(cv2.imread(chosen[c][1]),cv2.COLOR_BGR2RGB),(224,224)) for c in ('center','left','right')]);x=torch.from_numpy(images).permute(0,3,1,2).cuda().float()/255
            feature=encoder((x-mean)/std).reshape(1,-1);prob=float(torch.sigmoid(head((feature-mu)/scale)).item());values.append(dict(time_s=float(stamp),probability=prob,clear=prob>=data['threshold'],images={c:r[1] for c,r in chosen.items()}))
    positive=case['card_count']==0
    result=dict(case=case['case'],scene=stream['scene_group_id'],empty=positive,samples=len(values),false_clear=sum(r['clear'] for r in values) if not positive else 0,true_clear=sum(r['clear'] for r in values) if positive else 0,records=values)
    cases.append(result)
base=json.loads((folder/'summary.json').read_text());passed=(sum(c['false_clear'] for c in cases)==0 and all(c['true_clear']>=3 for c in cases if c['empty']) and base['reports']['held_out']['false_clear']==0 and base['plateau_reached'])
report=dict(cases=cases,threshold_unchanged=data['threshold'],train_scene_disjoint=True,combined_held_out_positive_episodes=base['reports']['held_out']['positive_episodes']+sum(c['empty'] for c in cases),expanded_offline_gate_passed=passed,live_gate_passed=False,automatic_deployment=False,final_scenes_sealed=True)
output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)
print([(c['case'],c['false_clear'],c['true_clear']) for c in cases],flush=True)
