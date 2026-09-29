#!/usr/bin/env python3
"""Train a native-crop spatial heatmap refiner for SC plug/opening pixels.

The coarse crop center comes from a frozen RGB-only locator. Projected TF
landmarks supervise this model but are never used to choose evaluation crops.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import models


CAMERAS=("center","left","right")


class KeypointRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,24,5,2,2),nn.GELU(),
            nn.Conv2d(24,48,3,2,1),nn.GELU(),
            nn.Conv2d(48,64,3,2,1),nn.GELU(),
            nn.Conv2d(64,64,3,1,1),nn.GELU(),
            nn.Conv2d(64,2,1))

    def forward(self,image):
        heatmap=self.features(image)
        batch,points,height,width=heatmap.shape
        prob=torch.softmax(heatmap.reshape(batch,points,-1)*4,dim=-1)
        y,x=torch.meshgrid(torch.linspace(0,1,height,device=image.device),
                           torch.linspace(0,1,width,device=image.device),indexing="ij")
        x=x.reshape(-1);y=y.reshape(-1)
        return torch.stack(((prob*x).sum(-1),(prob*y).sum(-1)),dim=-1).reshape(batch,4)


class PretrainedKeypointRefiner(nn.Module):
    """ImageNet ResNet-18 spatial features with a compact 56x56 heatmap head."""

    def __init__(self, pretrained_path, condition_target_port=False, point_count=2,
                 offscreen_opening_head=False):
        super().__init__()
        self.condition_target_port=condition_target_port
        self.point_count=point_count
        self.offscreen_opening_head=offscreen_opening_head
        if offscreen_opening_head and (not condition_target_port or point_count < 2):
            raise ValueError('Offscreen opening head requires a target port and two points')
        self.backbone=models.resnet18(weights=None)
        if pretrained_path is not None:
            self.backbone.load_state_dict(torch.load(pretrained_path,map_location='cpu',weights_only=True))
        for name,parameter in self.backbone.named_parameters():
            parameter.requires_grad_(name.startswith('layer3.'))
        self.decoder=nn.Sequential(nn.Conv2d(64+256+(2 if condition_target_port else 0),128,3,padding=1),nn.GELU(),
                                   nn.Conv2d(128,64,3,padding=1),nn.GELU(),nn.Conv2d(64,point_count,1))
        if offscreen_opening_head:
            self.opening_offset=nn.Sequential(nn.Linear(256+2,128),nn.GELU(),nn.Linear(128,2))
            nn.init.zeros_(self.opening_offset[-1].weight)
            nn.init.zeros_(self.opening_offset[-1].bias)
        self.register_buffer('mean',torch.tensor((.485,.456,.406))[None,:,None,None])
        self.register_buffer('std',torch.tensor((.229,.224,.225))[None,:,None,None])

    def train(self,mode=True):
        super().train(mode)
        self.backbone.bn1.eval();self.backbone.layer1.eval();self.backbone.layer2.eval()
        return self

    def forward(self,image,target_port=None):
        x=(image-self.mean)/self.std
        x=self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        early=self.backbone.layer1(x)
        middle=self.backbone.layer2(early)
        deep=self.backbone.layer3(middle)
        features=[early,F.interpolate(deep,size=early.shape[-2:],mode='bilinear',align_corners=False)]
        if self.condition_target_port:
            if target_port is None or target_port.shape!=(image.shape[0],2):
                raise ValueError('Two-element target-port one-hot required')
            features.append(target_port[:,:,None,None].expand(-1,-1,*early.shape[-2:]))
        heatmap=self.decoder(torch.cat(features,dim=1))
        batch,points,height,width=heatmap.shape
        prob=torch.softmax(heatmap.reshape(batch,points,-1)*4,dim=-1)
        y,x=torch.meshgrid(torch.linspace(0,1,height,device=image.device),
                           torch.linspace(0,1,width,device=image.device),indexing='ij')
        x=x.reshape(-1);y=y.reshape(-1)
        coordinates=torch.stack(((prob*x).sum(-1),(prob*y).sum(-1)),dim=-1).reshape(batch,2*self.point_count)
        if self.offscreen_opening_head:
            offset=self.opening_offset(torch.cat((deep.mean((-1,-2)),target_port),dim=-1))
            coordinates=torch.cat((coordinates[:,:2],coordinates[:,2:4]+offset,
                                   coordinates[:,4:]),dim=-1)
        return coordinates


def percentile(values,p):return float(np.percentile(values,p)) if values else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-labels",type=Path,required=True)
    p.add_argument("--crop-labels",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--updates",type=int,default=2500)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--crop-size",type=int,default=224)
    p.add_argument("--seed",type=int,default=20260924)
    p.add_argument("--pretrained-resnet",type=Path,
                   help="Use an ImageNet ResNet-18 spatial heatmap backbone instead of the small scratch CNN")
    p.add_argument("--condition-target-port",action="store_true",
                   help="Condition target-opening heatmap on the ordinary SC port task one-hot")
    args=p.parse_args()
    torch.set_num_threads(8);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    truth=[json.loads(line) for line in args.native_labels.open()]
    crops=[json.loads(line) for line in args.crop_labels.open()]
    if [(r['trial'],r['frame']) for r in truth]!=[(r['trial'],r['frame']) for r in crops]:
        raise ValueError("Unmatched native/crop rows")
    images=[];labels=[];tasks=[];splits=[];keys=[];rejected={"far":0,"out_of_crop":0,"out_of_frame":0}
    for row,crop in zip(truth,crops):
        if abs(row['observed_sc_tip_pose_opening_frame'][2])>=.08:
            rejected['far']+=3;continue
        for cam in CAMERAS:
            points=row['projected_training_pixels'][cam]
            if not points['tip'][3] or not points['opening'][3]:
                rejected['out_of_frame']+=1;continue
            cx,cy=crop['predicted_crop_centers'][cam]
            left=round(cx-args.crop_size/2);top=round(cy-args.crop_size/2)
            coordinates=[(points[name][0]-left,points[name][1]-top) for name in ('tip','opening')]
            if not all(0<=x<args.crop_size and 0<=y<args.crop_size for x,y in coordinates):
                rejected['out_of_crop']+=1;continue
            with Image.open(crop['images'][cam]) as im:
                rgb=im.convert('RGB')
                if rgb.size!=(args.crop_size,args.crop_size):raise ValueError('Crop shape mismatch')
                images.append(torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1))
            labels.append([v/(args.crop_size-1) for pair in coordinates for v in pair])
            tasks.append(row['task_vector'][2:4])
            splits.append(row['split']);keys.append((row['trial'],row['frame'],cam))
    x=torch.stack(images)
    y=torch.tensor(labels,dtype=torch.float32)
    task=torch.tensor(tasks,dtype=torch.float32)
    train=torch.tensor([i for i,s in enumerate(splits) if s=='train'])
    val=torch.tensor([i for i,s in enumerate(splits) if s=='validation'])
    if not len(train) or not len(val):raise ValueError('Empty grouped split')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    architecture='pretrained_resnet18' if args.pretrained_resnet else 'small_cnn'
    if args.condition_target_port and not args.pretrained_resnet:
        raise ValueError('Target-port conditioning currently requires the pretrained backbone')
    model=(PretrainedKeypointRefiner(args.pretrained_resnet,args.condition_target_port) if args.pretrained_resnet
           else KeypointRefiner()).to(device)
    def predict(indices):
        image=x[indices].to(device).float()/255
        return model(image,task[indices].to(device)) if args.condition_target_port else model(image)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    gen=torch.Generator().manual_seed(args.seed)
    best,best_metric,best_step,history=None,float('inf'),0,[]
    weight=torch.tensor([1.,1.,2.,2.],device=device)
    for step in range(1,args.updates+1):
        idx=train[torch.randint(len(train),(args.batch_size,),generator=gen)]
        pred=predict(idx)
        loss=(F.smooth_l1_loss(pred,y[idx].to(device),beta=.01,reduction='none')*weight).mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step==1 or step%100==0:
            model.eval();outputs=[]
            with torch.inference_mode():
                for batch in val.split(128):outputs.append(predict(batch).cpu())
            pred_val=torch.cat(outputs)
            target_error=((pred_val[:,2:]-y[val,2:])*(args.crop_size-1)).norm(dim=1).numpy()
            metric=float(np.percentile(target_error,95))
            history.append({"step":step,"train_loss":float(loss),"validation_target_p95_px":metric})
            print(json.dumps(history[-1]),flush=True)
            if metric<best_metric:
                best,best_metric,best_step=copy.deepcopy(model.state_dict()),metric,step
            model.train()
            if step-best_step>=700:break
    model.load_state_dict(best);model.eval()
    predictions=[]
    with torch.inference_mode():
        for batch in torch.arange(len(x)).split(128):
            predictions.append(predict(batch).cpu())
    pred=torch.cat(predictions)
    errors={split:{'tip':[],'opening':[]} for split in ('train','validation')}
    predicted_rows=[]
    for i,(key,split) in enumerate(zip(keys,splits)):
        pixel_error=(pred[i]-y[i])*(args.crop_size-1)
        for point,offset in (('tip',0),('opening',2)):
            errors[split][point].append(float(pixel_error[offset:offset+2].norm()))
        predicted_rows.append({"trial":key[0],"frame":key[1],"camera":key[2],
                               "crop_prediction_xy":(pred[i]*(args.crop_size-1)).tolist(),
                               "crop_truth_xy_training_only":(y[i]*(args.crop_size-1)).tolist()})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.cpu().state_dict(),'crop_size':args.crop_size,
                'architecture':architecture,'condition_target_port':args.condition_target_port},args.output_dir/'refiner.pt')
    with (args.output_dir/'predictions.jsonl').open('w') as out:
        for row in predicted_rows:out.write(json.dumps(row,separators=(',',':'))+'\n')
    report={'schema':'sc_native_heatmap_refiner/v1','train_views':len(train),'validation_views':len(val),
            'architecture':architecture,'condition_target_port':args.condition_target_port,
            'pretrained_resnet':str(args.pretrained_resnet) if args.pretrained_resnet else None,
            'train_episodes':len({keys[i][0] for i in train.tolist()}),
            'validation_episodes':len({keys[i][0] for i in val.tolist()}),
            'rejected':rejected,'best_step':best_step,'completed_step':step,'history':history,
            'errors_px':{split:{point:{'median':percentile(values,50),'p95':percentile(values,95)}
                                for point,values in item.items()} for split,item in errors.items()},
            'runtime_crop_selection':'frozen RGB-only locator; no TF or true pixel labels'}
    (args.output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
