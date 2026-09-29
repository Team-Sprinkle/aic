#!/usr/bin/env python3
"""Run a frozen RGB-only SC port-axis landmark model on all predicted crops."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train_sc_native_keypoints import PretrainedKeypointRefiner


CAMERAS=('center','left','right')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--crop-labels',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(8)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    names=tuple(saved['points']);size=int(saved['crop_size'])
    if not saved['condition_target_port']:raise ValueError('Missing task conditioning')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=PretrainedKeypointRefiner(None,True,point_count=len(names)).to(device).eval()
    model.load_state_dict(saved['model'],strict=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with args.output.open('w') as out,torch.inference_mode():
        for line in args.crop_labels.open():
            row=json.loads(line);images=[]
            for cam in CAMERAS:
                with Image.open(row['images'][cam]) as image:
                    rgb=image.convert('RGB')
                    if rgb.size!=(size,size):raise ValueError('Wrong crop size')
                    images.append(torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1))
            tasks=torch.tensor(row['task_vector'][2:4],dtype=torch.float32,device=device).repeat(3,1)
            local=model(torch.stack(images).to(device).float()/255,tasks).cpu().numpy().reshape(3,len(names),2)* (size-1)
            result={'trial':row['trial'],'frame':row['frame'],'episode_id':row['episode_id'],
                    'sim_time':row['sim_time'],'cameras':{}}
            for cam,coords in zip(CAMERAS,local):
                cx,cy=row['predicted_crop_centers'][cam]
                left=round(cx-size/2);top=round(cy-size/2)
                result['cameras'][cam]={name:[float(x+left),float(y+top)]
                                        for name,(x,y) in zip(names,coords)}
            out.write(json.dumps(result,separators=(',',':'))+'\n');count+=1
    print(json.dumps({'rows':count,'output':str(args.output),'runtime_inputs':'RGB-selected crops and task one-hot only'}))


if __name__=='__main__':main()
