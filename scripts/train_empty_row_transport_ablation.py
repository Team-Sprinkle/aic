#!/usr/bin/env python3
"""Bounded full-image empty-row gate from existing training corpora only."""
import cv2
cv2.setNumThreads(2)
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18
import yaml

root=Path('/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1')
out=root/'empty_row_visual_ablation';out.mkdir(exist_ok=False)
physical=os.environ['CUDA_VISIBLE_DEVICES'];assert physical=='3'
lock=open('/var/tmp/chmin_aic_gpu3.lock','w');fcntl.flock(lock,fcntl.LOCK_EX)
torch.set_num_threads(3);torch.manual_seed(20261018103);rng=np.random.default_rng(20261018103)
source_paths=[Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1/spacefilled_near_pose_dataset/manifest.json'),Path('/home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1/SFP_native_grasp_crops_v2/manifest.json')]
selected=[];rejected={};groups={}
for source in source_paths:
    for row in json.loads(source.read_text()):groups.setdefault((str(source),row['episode']),[]).append(row)
for (_,episode),rows in sorted(groups.items()):
    row=rows[0];stream=Path(row['source_stream']);audit=Path(row['images']['center']).parent.parent
    try:
        with stream.open() as f:prefix=f.read(20000)
        match=re.search(r'"trial_id"\s*:\s*"([^"]+)"',prefix);assert match
        cfg=yaml.safe_load((audit/'eval_config.yaml').read_text());board=cfg['trials'][match[1]]['scene']['task_board']
        assert all('entity_present' in board.get(f'nic_rail_{i}',{}) for i in range(5))
        count=sum(bool(board[f'nic_rail_{i}']['entity_present']) for i in range(5))
        for index in np.unique(np.linspace(0,len(rows)-1,min(2,len(rows)),dtype=int)):
            r=rows[index];assert all(Path(x).exists() for x in r['images'].values())
            selected.append(dict(episode=episode,scene=r['scene'],split=r['split'],images=r['images'],label=int(count==0),card_count=count,GT_label_only=True))
    except (OSError,KeyError,AssertionError,TypeError) as exc:
        key=type(exc).__name__;rejected[key]=rejected.get(key,0)+1
assert selected
for key in ['episode','scene']:
    assert not ({r[key] for r in selected if r['split']=='train'}&{r[key] for r in selected if r['split']=='validation'})
weights_path=Path('/var/tmp/aic_sc_fix_20260924/resnet18-f37072fd.pth');backbone=resnet18(weights=None);backbone.load_state_dict(torch.load(weights_path,map_location='cpu',weights_only=True))
encoder=nn.Sequential(backbone.conv1,backbone.bn1,backbone.relu,backbone.maxpool,backbone.layer1,backbone.layer2,backbone.layer3,nn.AdaptiveAvgPool2d((2,2))).cuda().eval()
mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
features=[];labels=[];manifest=[]
with torch.inference_mode():
    for row in selected:
        images=np.stack([cv2.resize(cv2.cvtColor(cv2.imread(row['images'][c]),cv2.COLOR_BGR2RGB),(224,224)) for c in ['center','left','right']])
        inputs=torch.from_numpy(images).permute(0,3,1,2).cuda().float()/255
        feature=encoder((inputs-mean)/std).reshape(-1).cpu().numpy();features.append(feature);labels.append(row['label']);manifest.append(dict(row,variant='full'))
        # Deterministic robot-only / missing-context negative. No GT crop.
        unknown=np.stack([cv2.resize(x[150:,:,:],(224,224)) for x in images])
        inputs=torch.from_numpy(unknown).permute(0,3,1,2).cuda().float()/255
        features.append(encoder((inputs-mean)/std).reshape(-1).cpu().numpy());labels.append(0);manifest.append(dict(row,label=0,variant='insufficient_context_bottom_third'))
features=np.asarray(features,np.float32);labels=np.asarray(labels,np.float32)
np.save(out/'features.npy',features);(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
fit=np.array([r['split']=='train' for r in manifest]);cal=np.array([r['split']=='validation' and int(hashlib.sha256(r['scene'].encode()).hexdigest()[:2],16)%2==0 for r in manifest]);test=~fit&~cal
assert fit.any() and cal.any() and test.any()
mu=features[fit].mean(0);scale=features[fit].std(0).clip(.05);x=torch.tensor((features-mu)/scale,device='cuda');y=torch.tensor(labels,device='cuda')
head=nn.Sequential(nn.Linear(features.shape[1],64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1)).cuda();opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=.01)
train_ids=np.flatnonzero(fit);cal_ids=np.flatnonzero(cal);best=float('inf');best_step=0;history=[]
for step in range(1,4001):
    head.train();ids=rng.choice(train_ids,min(128,len(train_ids)),replace=False);logits=head(x[ids]).flatten();loss=nn.functional.binary_cross_entropy_with_logits(logits,y[ids]);opt.zero_grad();loss.backward();opt.step()
    if step%50==0:
        head.eval()
        with torch.no_grad():value=float(nn.functional.binary_cross_entropy_with_logits(head(x[cal_ids]).flatten(),y[cal_ids]))
        history.append(dict(step=step,calibration_BCE=value))
        if value<best*.995:best=value;best_step=step;torch.save(head.state_dict(),out/'head_best.pt')
        if step>=500 and step-best_step>=400:break
head.load_state_dict(torch.load(out/'head_best.pt',weights_only=True));head.eval()
with torch.no_grad():prob=torch.sigmoid(head(x).flatten()).cpu().numpy()
threshold=min(.9999,max(.95,float(prob[cal&(labels==0)].max())+.02))
reports={}
for name,mask in [('calibration',cal),('held_out',test)]:
    predicted=prob>=threshold;positive=mask&(labels==1);negative=mask&(labels==0)
    reports[name]=dict(positive_samples=int(positive.sum()),negative_samples=int(negative.sum()),positive_episodes=len({manifest[i]['episode'] for i in np.flatnonzero(positive)}),false_clear=int((predicted&negative).sum()),true_clear=int((predicted&positive).sum()),clear_recall=float(predicted[positive].mean()) if positive.any() else None)
passed=reports['held_out']['positive_episodes']>=3 and reports['held_out']['false_clear']==0 and reports['held_out']['clear_recall']>=.67 and reports['calibration']['false_clear']==0
checkpoint=out/'best.pt';torch.save(dict(encoder=encoder.state_dict(),head=head.state_dict(),feature_mean=mu,feature_scale=scale,threshold=threshold,architecture='imagenet_layer3_spatial2x2_three_camera_MLP64',input='full224_RGB_triplets_only',card_count_runtime_input=False),checkpoint)
summary=dict(reports=reports,threshold=threshold,offline_gate_passed=passed,live_gate_passed=False,automatic_deployment=False,steps=step,plateau_reached=step-best_step>=400,source_episodes=len(groups),admitted_samples=len(selected),rejected=rejected,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),final_scenes_sealed=True,warning='An empty-row classifier is not a collision planner. A live shortcut additionally requires accepted requested-port vision, bounded path and unchanged force guards.')
(out/'history.json').write_text(json.dumps(history,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
