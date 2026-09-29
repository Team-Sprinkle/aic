#!/usr/bin/env python3
"""Estimate SC port-relative orientation directly from native RGB crops.

The three crops are selected by a frozen RGB locator. Simulator orientation is
a supervised label only. No target quaternion or port TF enters inference.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CAMERAS=('center','left','right')


class CropFrames(Dataset):
    def __init__(self,rows,crops,mean,std):
        self.rows=rows;self.crops=crops;self.mean=mean;self.std=std
        self.transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((.485,.456,.406),(.229,.224,.225))])

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];crop=self.crops[index]
        images=[]
        for cam in CAMERAS:
            with Image.open(crop['images'][cam]) as image:
                rgb=image.convert('RGB')
                if rgb.size!=(224,224):raise ValueError('Expected original-resolution 224px crop')
                images.append(self.transform(rgb))
        state=(torch.tensor(row['state'],dtype=torch.float32)-self.mean)/self.std
        state=torch.cat((state,torch.tensor(row['task_vector'],dtype=torch.float32)))
        rotvec=Rotation.from_quat(row['observed_sc_tip_pose_opening_frame'][3:7]).as_rotvec()
        return torch.stack(images),state,torch.tensor(rotvec,dtype=torch.float32),index


class CropOrientation(nn.Module):
    def __init__(self,pretrained):
        super().__init__()
        self.backbone=models.resnet18(weights=None)
        if pretrained is not None:
            self.backbone.load_state_dict(torch.load(pretrained,map_location='cpu',weights_only=True))
        self.backbone.fc=nn.Identity()
        for name,parameter in self.backbone.named_parameters():
            parameter.requires_grad_(name.startswith('layer4.'))
        self.state_net=nn.Sequential(nn.Linear(42,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(3*512+128,256),nn.ReLU(),nn.Linear(256,128),
                                nn.ReLU(),nn.Linear(128,3))

    def forward(self,images,state):
        batch=images.shape[0]
        visual=self.backbone(images.reshape(batch*3,3,224,224)).reshape(batch,-1)
        return .2*torch.tanh(self.head(torch.cat((visual,self.state_net(state)),dim=-1)))


def predict(model,loader,device,rows):
    model.eval();out=[None]*len(rows)
    with torch.inference_mode():
        for images,state,_,indices in loader:
            values=model(images.to(device),state.to(device)).cpu().numpy()
            for i,value in zip(indices.tolist(),values):out[i]=value.tolist()
    errors=[];episode={}
    for row,value in zip(rows,out):
        if value is None:
            errors.append(None)
            continue
        truth=Rotation.from_quat(row['observed_sc_tip_pose_opening_frame'][3:7])
        error=(truth.inv()*Rotation.from_rotvec(value)).magnitude()*180/np.pi
        record={'trial':row['trial'],'frame':row['frame'],'near':abs(row['observed_sc_tip_pose_opening_frame'][2])<.03,
                'error_deg':float(error),'predicted_rotvec_rad':value}
        errors.append(record);episode.setdefault(row['trial'],[]).append(record)
    near=[r['error_deg'] for r in errors if r is not None and r['near']]
    stats=lambda values:{'count':len(values),'median':float(np.median(values)) if values else None,
                         'p95':float(np.percentile(values,95)) if values else None}
    report={'near_orientation_deg':stats(near),'all_orientation_deg':stats([r['error_deg'] for r in errors if r is not None]),
            'episodes':len(episode),'per_episode_near_p95_deg':{
                key:stats([r['error_deg'] for r in group if r['near']])['p95'] for key,group in episode.items()}}
    return report,errors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--crop-labels',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--pretrained-resnet',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=20260924)
    args=p.parse_args()
    if not args.checkpoint and not args.pretrained_resnet:raise ValueError('ImageNet initialization required')
    torch.set_num_threads(8);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    rows=[json.loads(x) for x in args.labels.open()]
    crops=[json.loads(x) for x in args.crop_labels.open()]
    if [(r['trial'],r['frame']) for r in rows]!=[(r['trial'],r['frame']) for r in crops]:
        raise ValueError('Unmatched crop/pose rows')
    # This local controller component is used within 80 mm of the opening.
    selected=[i for i,r in enumerate(rows) if abs(r['observed_sc_tip_pose_opening_frame'][2])<.08]
    rows=[rows[i] for i in selected];crops=[crops[i] for i in selected]
    train=[i for i,r in enumerate(rows) if r['split']=='train']
    val=[i for i,r in enumerate(rows) if r['split']=='validation']
    if {rows[i]['scene_sha256'] for i in train}&{rows[i]['scene_sha256'] for i in val}:
        raise ValueError('Scene leakage')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False) if args.checkpoint else None
    if saved:
        mean,std=saved['state_mean'],saved['state_std']
    else:
        state=torch.tensor([rows[i]['state'] for i in train],dtype=torch.float32)
        mean=state.mean(0);std=state.std(0).clamp_min(1e-3)
    ds=CropFrames(rows,crops,mean,std)
    train_loader=DataLoader(torch.utils.data.Subset(ds,train),batch_size=args.batch_size,
        shuffle=bool(train),num_workers=args.workers,pin_memory=True)
    val_loader=DataLoader(torch.utils.data.Subset(ds,val),batch_size=args.batch_size,
        num_workers=args.workers,pin_memory=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=CropOrientation(args.pretrained_resnet if not saved else None).to(device)
    if saved:model.load_state_dict(saved['model'],strict=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if saved:
        result,pred=predict(model,val_loader,device,rows)
        # Validation loader returns original subset indices, so report only val.
        val_rows=[rows[i] for i in val]
        val_pred=[pred[i] for i in val]
        result={'near_orientation_deg':{'count':sum(x['near'] for x in val_pred),
                 'median':float(np.median([x['error_deg'] for x in val_pred if x['near']])),
                 'p95':float(np.percentile([x['error_deg'] for x in val_pred if x['near']],95))},
                'per_episode_near_p95_deg':{t:float(np.percentile([x['error_deg'] for x in val_pred if x['trial']==t and x['near']],95))
                                            for t in sorted({r['trial'] for r in val_rows}) if any(x['trial']==t and x['near'] for x in val_pred)}}
        (args.output_dir/'metrics.json').write_text(json.dumps({'schema':'sc_native_orientation/v1','eval_only':True,
            'checkpoint':str(args.checkpoint),'validation':result},indent=2)+'\n')
        print(json.dumps(result));return
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=1e-4)
    best=float('inf');history=[]
    for epoch in range(1,args.epochs+1):
        model.train();losses=[]
        for images,state,target,_ in train_loader:
            prediction=model(images.to(device),state.to(device))
            loss=((prediction-target.to(device))**2).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();losses.append(float(loss))
        result,pred=predict(model,val_loader,device,rows)
        errors=[pred[i]['error_deg'] for i in val if pred[i]['near']]
        score=float(np.percentile(errors,95))
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'val_near_orientation_p95_deg':score})
        print(json.dumps(history[-1]),flush=True)
        if score<best:
            best=score
            torch.save({'model':copy.deepcopy(model.state_dict()),'state_mean':mean,'state_std':std,
                        'epoch':epoch,'validation_p95_deg':score},args.output_dir/'best.pt')
    (args.output_dir/'metrics.json').write_text(json.dumps({'schema':'sc_native_orientation/v1','history':history,
        'best_val_near_orientation_p95_deg':best,'train_episodes':len({rows[i]['trial'] for i in train}),
        'validation_episodes':len({rows[i]['trial'] for i in val})},indent=2)+'\n')


if __name__=='__main__':main()
