#!/usr/bin/env python3
"""Apply a frozen native crop refiner to RGB-selected port crops in every split.

This cache uses no target projection to choose a crop or accept a prediction.
Ground-truth labels, if present elsewhere in the row, are not read here.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from train_sc_native_keypoints import PretrainedKeypointRefiner


CAMERAS=('center','left','right')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--coarse-predictions',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=64)
    a=p.parse_args();torch.set_num_threads(6)
    rows=[json.loads(s) for s in (a.labels/'rows.jsonl').open()]
    coarse={(r['row'],r['camera']):r for r in map(json.loads,a.coarse_predictions.open())}
    model=PretrainedKeypointRefiner(None,False,point_count=1).cuda().eval()
    model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model'])
    items=[];skipped={}
    for i,row in enumerate(rows):
        if row.get('image_kind')!='native_1152x1024_rgb_jpeg':continue
        for cam in CAMERAS:
            prediction=coarse[(i,cam)]['predicted'][row['selected_identity']]
            if prediction['projected_inframe_probability']<.5:
                skipped['coarse_inframe_negative']=skipped.get('coarse_inframe_negative',0)+1;continue
            xy=prediction['xy'];left=int(round(xy[0]-80));top=int(round(xy[1]-80))
            image=cv2.imread(row['images'][cam],cv2.IMREAD_COLOR)
            if image is None:raise FileNotFoundError(row['images'][cam])
            crop=cv2.warpAffine(image,np.array([[1,0,-left],[0,1,-top]],np.float32),
                (160,160),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            items.append((i,row['episode'],row['time_s'],cam,xy,[left,top],
                          cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w') as file,torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,len(items),a.batch_size):
            batch=items[start:start+a.batch_size]
            image=torch.from_numpy(np.stack([r[-1] for r in batch])).permute(0,3,1,2).cuda().float()/255.
            pred=model(image).float().cpu().numpy().reshape(-1,2)
            for item,point in zip(batch,pred):
                i,episode,t,cam,coarse_xy,offset,_=item
                xy=(point*159+offset).tolist()
                file.write(json.dumps({'row':i,'episode':episode,'time_s':t,'camera':cam,
                    'predicted_xy':xy,'coarse_xy':coarse_xy,
                    'crop_selected_from_frozen_rgb':True,'runtime_geometry_input':False},
                    separators=(',',':'))+'\n')
    print(json.dumps({'rows':len(items),'skipped':skipped,'output':str(a.output)}))


if __name__=='__main__':main()
