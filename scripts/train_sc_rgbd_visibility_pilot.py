#!/usr/bin/env python3
"""Four-scene grouped pilot for selected-opening visibility from RGB crops.

The frozen coarse RGB locator chooses the crop. Label-only depth supplies
geometric visibility targets, and depth is never an inference input. Four
scenes are too few to certify a deployable visibility gate.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler
from torchvision import models


SIZE=160


class Examples(Dataset):
    def __init__(self,rows,training):self.rows=rows;self.training=training
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];rgb=cv2.imread(r['image'],cv2.IMREAD_COLOR)
        if rgb is None:raise FileNotFoundError(r['image'])
        x,y=r['predicted_selected_xy_native']
        if self.training:
            x+=np.random.uniform(-5,5);y+=np.random.uniform(-5,5)
        left=int(round(x-SIZE/2));top=int(round(y-SIZE/2))
        crop=cv2.warpAffine(rgb,np.array([[1,0,-left],[0,1,-top]],np.float32),
                            (SIZE,SIZE),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        crop=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
        return torch.from_numpy(crop.copy()).permute(2,0,1),torch.tensor(float(r['geometric_visible_fraction']>=.7))


class Net(nn.Module):
    def __init__(self,path):
        super().__init__();self.encoder=models.resnet18(weights=None)
        self.encoder.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        self.encoder.fc=nn.Linear(512,1)
        for name,w in self.encoder.named_parameters():w.requires_grad_(name.startswith('layer4.') or name.startswith('fc.'))
        self.register_buffer('mean',torch.tensor((.485,.456,.406))[None,:,None,None])
        self.register_buffer('std',torch.tensor((.229,.224,.225))[None,:,None,None])
    def forward(self,image):return self.encoder((image-self.mean)/self.std).squeeze(-1)


@torch.inference_mode()
def score(net,loader):
    net.eval();prob=[];truth=[]
    for image,label in loader:
        value=net(image.cuda().float()/255.).float().sigmoid().cpu().numpy()
        prob.extend(value.tolist());truth.extend(label.numpy().tolist())
    prob=np.asarray(prob);truth=np.asarray(truth)>0.5;accepted=prob>=.5
    return {'n':len(prob),'positive':int(truth.sum()),'precision_at_half':float(np.sum(accepted&truth)/max(1,accepted.sum())),
            'recall_at_half':float(np.sum(accepted&truth)/max(1,truth.sum())),
            'false_visible_rate_at_half':float(np.sum(accepted&~truth)/max(1,(~truth).sum())),
            'brier':float(np.mean((prob-truth)**2)),'probability':prob.tolist(),'truth':truth.astype(int).tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--imagenet',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=500)
    a=p.parse_args();torch.set_num_threads(8);torch.manual_seed(20261002);np.random.seed(20261002)
    if a.output.exists():p.error('Preserve existing run')
    a.output.mkdir(parents=True)
    source=json.loads(a.audit.read_text())['records']
    rows=[r for r in source if r['inframe_fraction']>=.7]
    episodes=sorted({r['episode'] for r in rows})
    if len(episodes)!=4:raise ValueError(f'Expected four grouped pilot scenes, got {episodes}')
    report=[]
    for heldout in episodes:
        train=[r for r in rows if r['episode']!=heldout]
        test=[r for r in rows if r['episode']==heldout]
        classes=Counter(r['geometric_visible_fraction']>=.7 for r in train)
        weights=[1/classes[r['geometric_visible_fraction']>=.7] for r in train]
        sampler=WeightedRandomSampler(weights,num_samples=a.updates*24,replacement=True)
        loader=DataLoader(Examples(train,True),batch_size=24,sampler=sampler,num_workers=4,
                          persistent_workers=True,pin_memory=True)
        test_loader=DataLoader(Examples(test,False),batch_size=64,shuffle=False,num_workers=2,pin_memory=True)
        net=Net(a.imagenet).cuda();opt=torch.optim.AdamW((w for w in net.parameters() if w.requires_grad),lr=1e-4)
        for step,(image,label) in enumerate(loader,start=1):
            net.train();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):logit=net(image.cuda().float()/255.)
            loss=F.binary_cross_entropy_with_logits(logit.float(),label.cuda().float())
            loss.backward();opt.step()
            if step==a.updates:break
        result=score(net,test_loader)
        coarse=np.asarray([r['predicted_projected_inframe_probability'] for r in test]);truth=np.asarray(result['truth'])>0
        result['coarse_false_visible_rate_at_half']=float(np.mean(coarse[~truth]>=.5)) if (~truth).any() else None
        result['episode']=heldout;report.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('probability','truth')}),flush=True)
    all_prob=np.asarray([p for r in report for p in r['probability']]);all_truth=np.asarray([x for r in report for x in r['truth']])>0
    accept=all_prob>=.5
    summary={'n':len(all_prob),'positive':int(all_truth.sum()),
             'precision_at_half':float(np.sum(accept&all_truth)/max(1,accept.sum())),
             'recall_at_half':float(np.sum(accept&all_truth)/max(1,all_truth.sum())),
             'false_visible_rate_at_half':float(np.sum(accept&~all_truth)/max(1,(~all_truth).sum())),
             'brier':float(np.mean((all_prob-all_truth)**2))}
    out={'schema':'sc_rgbd_visibility_four_scene_pilot/v1','episodes':episodes,'summary':summary,
         'folds':report,'depth_inference_input':False,'native_crop_selected_by_rgb_locator':True,
         'geometric_visibility_not_human_identifiability':True,'insufficient_scenes_for_deployment':True,
         'reserved_final_opened':False}
    (a.output/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
