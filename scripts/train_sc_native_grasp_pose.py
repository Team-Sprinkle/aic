#!/usr/bin/env python3
"""Train a native-crop 3D grasp probe on immutable grouped triplets.

Fixed crop centres derive only from training data. TF labels never choose
held-out crops. This probe has no deployment authority unless its gate passes.
"""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import cv2  # Load the pixi image libraries before torchvision/Pillow.
import torch
from torch import nn
from torchvision.models import resnet18
from train_sc_grasp_pose_probe import errors

class NativePose(nn.Module):
    def __init__(self,imagenet):
        super().__init__();base=resnet18(weights=None)
        base.load_state_dict(torch.load(imagenet,map_location='cpu',weights_only=True))
        self.encoder=nn.Sequential(*list(base.children())[:6],nn.AdaptiveAvgPool2d((8,8)))
        for name,p in self.encoder.named_parameters():p.requires_grad_(name.startswith('5.'))
        self.project=nn.Conv2d(128,24,1)
        self.head=nn.Sequential(nn.LayerNorm(3*24*64),nn.Linear(3*24*64,256),nn.GELU(),nn.Linear(256,64),nn.GELU(),nn.Linear(64,9))
        self.register_buffer('rgb_mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('rgb_std',torch.tensor([.229,.224,.225])[None,:,None,None])
    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self
    def forward(self,x):
        n=x.shape[0];x=x.reshape(-1,3,192,192).float()/255
        x=(x-self.rgb_mean)/self.rgb_std
        return self.head(self.project(self.encoder(x)).reshape(n,-1))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--native-crops',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--imagenet',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Preserve existing run')
    a.output.mkdir();torch.set_num_threads(4);torch.manual_seed(20261014);rng=np.random.default_rng(20261014)
    rows=json.loads((a.source/'manifest.json').read_text());fit=np.array([r['split']=='train' for r in rows]);ids=np.flatnonzero(fit);val=np.flatnonzero(~fit);y=np.load(a.source/'labels.npy');mean=y[fit].mean(0);std=y[fit].std(0).clip(min=1e-4)
    pixels=np.load(a.native_crops);assert pixels.shape==(len(rows),3,192,192,3);x=torch.from_numpy(pixels).permute(0,1,4,2,3).contiguous();yy=torch.from_numpy((y-mean)/std).cuda()
    assert not {r['scene'] for r in rows if r['split']=='train'}&{r['scene'] for r in rows if r['split']=='validation'}
    plan={'schema':'aic_native_grasp_pose/v1','hypothesis':'Frozen low-resolution spatial features and independent pixel targets leave grasp-depth errors; adapt layer2 on native192 triplets with direct 3D supervision and denser8x8 spatial features. This is a combined representation/objective repair, not an isolated resolution claim.','source':str(a.source),'manifest_sha256':hashlib.sha256((a.source/'manifest.json').read_bytes()).hexdigest(),'pixels_sha256':hashlib.sha256(a.native_crops.read_bytes()).hexdigest(),'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'train_episodes':83,'heldout_episodes':21,'samples':len(rows),'inputs':'Three original192 crops, fixed training-only centres; no geometry or state input','budget':'6000 updates / 2 GPU hours; minimum2000, plateau2000 updates without0.5% gate-score improvement','gate':{'translation_p95_mm':.4,'orientation_p95_deg':1},'final_scenes_opened':False}
    (a.output/'frozen_plan.json').write_text(json.dumps(plan,indent=2)+'\n');model=NativePose(a.imagenet).cuda();parameters=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=1e-4);history=[];best=float('inf');best_step=0;reason='update_cap';start=time.monotonic()
    def predict():
        model.eval();pred=[]
        with torch.inference_mode():
            for j in range(0,len(val),32):pred.append(model(x[val[j:j+32]].cuda()).cpu().numpy()*std+mean)
        return np.concatenate(pred)
    for step in range(1,6001):
        if time.monotonic()-start>7200:reason='time_cap';break
        selected=ids[rng.integers(len(ids),size=16)];model.train();opt.zero_grad(set_to_none=True);out=model(x[selected].cuda());loss=nn.functional.smooth_l1_loss(out,yy[selected],beta=.2);loss.backward();nn.utils.clip_grad_norm_(parameters,1);opt.step()
        if step%100==0:
            metric=errors(predict(),y[val]);score=max(metric['translation_p50_p95_mm'][1]/.4,metric['orientation_p50_p95_deg'][1]);history.append({'step':step,'metrics':metric,'score':score})
            if score<best*.995:
                best=score;best_step=step;torch.save({'model':model.state_dict(),'mean':mean,'std':std,'plan':plan,'update':step},a.output/'best.pt')
            if step>=2000 and step-best_step>=2000:reason='validation_plateau';break
    torch.save({'model':model.state_dict(),'mean':mean,'std':std,'plan':plan,'update':step},a.output/'last.pt');selected=torch.load(a.output/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(selected['model']);pred=predict();np.save(a.output/'heldout_predictions.npy',pred);metrics=errors(pred,y[val]);summary={'best_step':best_step,'updates_completed':step,'stop_reason':reason,'metrics':metrics,'passed':metrics['translation_p50_p95_mm'][1]<=.4 and metrics['orientation_p50_p95_deg'][1]<=1,'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in parameters),'wall_seconds':time.monotonic()-start,'final_scenes_opened':False};(a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n');(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
