#!/usr/bin/env python3
"""Train five virtual SC opening-axis landmark heatmaps on RGB-selected crops."""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from train_sc_native_keypoints import PretrainedKeypointRefiner


CAMERAS=('center','left','right')
POINTS=('opening','x_plus','x_minus','y_plus','y_minus')


def percentile(values,p):return float(np.percentile(values,p)) if len(values) else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--crop-labels',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--pretrained-resnet',type=Path,required=True)
    p.add_argument('--updates',type=int,default=2200)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--seed',type=int,default=20260924)
    args=p.parse_args()
    torch.set_num_threads(8);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    rows=[json.loads(x) for x in args.labels.open()]
    crops=[json.loads(x) for x in args.crop_labels.open()]
    if [(r['trial'],r['frame']) for r in rows]!=[(r['trial'],r['frame']) for r in crops]:
        raise ValueError('Unmatched crop/label identity')
    images=[];targets=[];tasks=[];splits=[];keys=[]
    rejected={'far':0,'out_of_frame':0,'out_of_crop':0}
    for row,crop in zip(rows,crops):
        if abs(row['observed_sc_tip_pose_opening_frame'][2])>=.08:
            rejected['far']+=3;continue
        for cam in CAMERAS:
            points=row['projected_orientation_landmarks'][cam]
            if not all(points[name][3] for name in POINTS):
                rejected['out_of_frame']+=1;continue
            cx,cy=crop['predicted_crop_centers'][cam]
            left=round(cx-112);top=round(cy-112)
            coords=[(points[name][0]-left,points[name][1]-top) for name in POINTS]
            if not all(0<=x<224 and 0<=y<224 for x,y in coords):
                rejected['out_of_crop']+=1;continue
            with Image.open(crop['images'][cam]) as image:
                rgb=image.convert('RGB')
                if rgb.size!=(224,224):raise ValueError('Not native 224px crop')
                images.append(torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1))
            targets.append([v/223 for pair in coords for v in pair])
            tasks.append(row['task_vector'][2:4]);splits.append(row['split'])
            keys.append((row['trial'],row['frame'],cam))
    x=torch.stack(images);y=torch.tensor(targets,dtype=torch.float32)
    task=torch.tensor(tasks,dtype=torch.float32)
    train=torch.tensor([i for i,s in enumerate(splits) if s=='train'])
    val=torch.tensor([i for i,s in enumerate(splits) if s=='validation'])
    if not len(train) or not len(val):raise ValueError('Empty episode-grouped split')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=PretrainedKeypointRefiner(args.pretrained_resnet,True,point_count=len(POINTS)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    gen=torch.Generator().manual_seed(args.seed)
    best,best_metric,best_step,history=None,float('inf'),0,[]
    def estimate(indices):
        return model(x[indices].to(device).float()/255,task[indices].to(device))
    for step in range(1,args.updates+1):
        idx=train[torch.randint(len(train),(args.batch_size,),generator=gen)]
        pred=estimate(idx)
        loss=F.smooth_l1_loss(pred,y[idx].to(device),beta=.01)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step==1 or step%100==0:
            model.eval();pieces=[]
            with torch.inference_mode():
                for batch in val.split(128):pieces.append(estimate(batch).cpu())
            pred_val=torch.cat(pieces).reshape(-1,len(POINTS),2)
            truth=y[val].reshape(-1,len(POINTS),2)
            error=((pred_val-truth)*223).norm(dim=-1).numpy()
            metric=float(np.percentile(error,95))
            history.append({'step':step,'loss':float(loss),'validation_landmark_p95_px':metric})
            print(json.dumps(history[-1]),flush=True)
            if metric<best_metric:
                best,best_metric,best_step=copy.deepcopy(model.state_dict()),metric,step
            model.train()
            if step-best_step>=700:break
    model.load_state_dict(best);model.eval()
    metrics={}
    with torch.inference_mode():
        for split,indices in (('train',train),('validation',val)):
            pieces=[estimate(batch).cpu() for batch in indices.split(128)]
            pred=torch.cat(pieces).reshape(-1,len(POINTS),2)
            truth=y[indices].reshape(-1,len(POINTS),2)
            error=((pred-truth)*223).norm(dim=-1).numpy()
            metrics[split]={point:{'median':percentile(error[:,i],50),'p95':percentile(error[:,i],95)}
                            for i,point in enumerate(POINTS)}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.cpu().state_dict(),'point_count':len(POINTS),'points':POINTS,
                'condition_target_port':True,'crop_size':224},args.output_dir/'landmarks.pt')
    report={'schema':'sc_orientation_landmarks/v1','train_views':len(train),'validation_views':len(val),
            'train_episodes':len({keys[i][0] for i in train.tolist()}),
            'validation_episodes':len({keys[i][0] for i in val.tolist()}),
            'rejected':rejected,'best_step':best_step,'completed_step':step,
            'pixel_error':metrics,'history':history,
            'runtime_inputs':'RGB-selected crops and task one-hot; no simulator geometry'}
    (args.output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
