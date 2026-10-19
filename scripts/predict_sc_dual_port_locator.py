#!/usr/bin/env python3
"""Apply frozen two-port locator to RGB and select with the public task code."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import cv2

from build_sc_native_pose_labels import WIDTH,HEIGHT
from train_sc_native_keypoints import PretrainedKeypointRefiner


CAMERAS=('center','left','right')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cv2-area-resize',action='store_true')
    a=p.parse_args();torch.set_num_threads(8)
    rows=[json.loads(s) for s in a.labels.open()]
    if len({(r['trial'],r['frame']) for r in rows})!=len(rows):
        raise ValueError('Only one selected-port record per frame is allowed')
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if saved.get('architecture')!='dual_sc_port_spatial_heatmaps':
        raise ValueError('Expected explicit dual-port checkpoint')
    model=PretrainedKeypointRefiner(None,False,point_count=3).cuda().eval()
    model.load_state_dict(saved['model'])
    predictions=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,len(rows),32):
            group=rows[start:start+32];images=[]
            for row in group:
                for cam in CAMERAS:
                    with Image.open(row['images'][cam]) as im:
                        rgb=np.asarray(im.convert('RGB'))
                        small=(cv2.resize(rgb,(288,256),interpolation=cv2.INTER_AREA)
                               if a.cv2_area_resize else
                               np.asarray(Image.fromarray(rgb).resize((288,256),Image.Resampling.BILINEAR)))
                        images.append(torch.from_numpy(np.asarray(small).copy()).permute(2,0,1))
            pred=model(torch.stack(images).cuda().float()/255).float().cpu().numpy()
            pred=pred.reshape(len(group),3,6)
            pred[:,:,0::2]*=WIDTH-1;pred[:,:,1::2]*=HEIGHT-1
            for row,views in zip(group,pred):
                selected=int(np.argmax(row['task_vector'][2:4]))
                cameras={}
                for cam,points in zip(CAMERAS,views):
                    cameras[cam]={'tip_xy':points[:2].tolist(),
                                  'opening_xy':points[2+2*selected:4+2*selected].tolist(),
                                  'port0_xy':points[2:4].tolist(),
                                  'port1_xy':points[4:6].tolist()}
                predictions.append({'trial':row['trial'],'frame':row['frame'],
                                    'episode_id':row['episode_id'],'sim_time':row['sim_time'],
                                    'cameras':cameras,'task_selected_port':selected,
                                    'runtime_inputs':'RGB and public task code only'})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w') as f:
        for row in predictions:f.write(json.dumps(row,separators=(',',':'))+'\n')
    print(json.dumps({'rows':len(predictions),'output':str(a.output)}))


if __name__=='__main__':main()
