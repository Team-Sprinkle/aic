#!/usr/bin/env python3
"""Run a frozen SC heatmap refiner on all observation-selected native crops.

No projected landmark, scored TF, or success label is read. Predictions for
every view are kept, including targets outside the crop or under occlusion.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train_sc_native_keypoints import CAMERAS, KeypointRefiner, PretrainedKeypointRefiner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-labels",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(8)
    rows=[json.loads(line) for line in args.crop_labels.open()]
    saved=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    size=int(saved['crop_size'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    architecture=saved.get('architecture','small_cnn')
    conditioned=bool(saved.get('condition_target_port',False))
    model=(PretrainedKeypointRefiner(None,conditioned) if architecture=='pretrained_resnet18'
           else KeypointRefiner()).to(device).eval()
    model.load_state_dict(saved['model'],strict=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with args.output.open('w') as out, torch.inference_mode():
        for row in rows:
            images=[]
            for cam in CAMERAS:
                with Image.open(row['images'][cam]) as im:
                    rgb=im.convert('RGB')
                    if rgb.size!=(size,size):raise ValueError('Crop-size mismatch')
                    images.append(torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1))
            stacked=torch.stack(images).to(device).float()/255
            if conditioned:
                onehot=torch.tensor(row['task_vector'][2:4],dtype=torch.float32,device=device).repeat(len(CAMERAS),1)
                local=model(stacked,onehot).cpu().numpy()*(size-1)
            else:
                local=model(stacked).cpu().numpy()*(size-1)
            result={"trial":row['trial'],"frame":row['frame'],"episode_id":row['episode_id'],
                    "sim_time":row['sim_time'],"cameras":{}}
            for cam,pixels in zip(CAMERAS,local):
                cx,cy=row['predicted_crop_centers'][cam]
                left=round(cx-size/2);top=round(cy-size/2)
                result['cameras'][cam]={"tip_xy":[float(pixels[0]+left),float(pixels[1]+top)],
                                        "opening_xy":[float(pixels[2]+left),float(pixels[3]+top)]}
            out.write(json.dumps(result,separators=(',',':'))+'\n')
            count+=1
    print(json.dumps({"rows":count,"output":str(args.output),
                      "runtime_inputs":"RGB-selected crops only; no projected labels or TF"}))


if __name__=='__main__':main()
