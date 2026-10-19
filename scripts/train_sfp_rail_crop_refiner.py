#!/usr/bin/env python3
"""Bounded original-resolution SFP rail endpoint refinement diagnostic.

The fixed coarse RGB prediction alone centers every crop. Projected geometry
is used for supervision and post-hoc scoring, never crop selection at inference.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_two_family_port_locator import Locator


CAMS = ('center', 'left', 'right')
SCALE = np.array([1151., 1023.], dtype=np.float32)
CROP = 160
HALF = (CROP - 1) / 2


def read(path):
    return [json.loads(line) for line in path.open()]


def join(pred, row, split):
    if row['split'] != split or pred['scene_group'] != row['scene_group'] or \
            pred['episode'] != row['episode'] or abs(pred['time_s']-row['time_s']) > 1e-6:
        raise ValueError('Prediction/row causal group join mismatch')


def prepare(predictions, rows, split, require_label, cache_path):
    samples = []
    crops = []
    last_image = None
    last_path = None
    for pi, prediction in enumerate(predictions):
        row = rows[prediction['row']]
        join(prediction, row, split)
        if row['autonomous_actor_diagnostic']:
            raise ValueError('Actor diagnostic row in supervised fit')
        for ci, cam in enumerate(CAMS):
            path = row['images'][cam]
            if path != last_path:
                last_image = cv2.imread(path, cv2.IMREAD_COLOR)
                if last_image is None or last_image.shape[:2] != (1024, 1152):
                    raise ValueError(f'Invalid original-resolution image: {path}')
                last_path = path
            for end, slot in enumerate((0, 4)):
                if prediction['endpoint_inframe_prob'][ci][end] < .5:
                    continue
                label = row['rail_projections_training_label_only'][cam][slot]
                center = np.asarray(prediction['endpoint_xy_normalized'][ci][end],
                                    dtype=np.float32) * SCALE
                truth = np.asarray(label[:2], dtype=np.float32)
                inside = bool(label[3] and np.all(np.abs(truth-center) <= HALF))
                if require_label and not inside:
                    continue
                crop = cv2.getRectSubPix(last_image, (CROP, CROP),
                                         tuple(float(x) for x in center))
                crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                samples.append({'pi': pi, 'cam': cam, 'ci': ci, 'end': end,
                                'center': center.tolist(), 'truth': truth.tolist(),
                                'label_inframe': bool(label[3]), 'inside': inside,
                                'scene_group': row['scene_group']})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, np.asarray(crops, dtype=np.uint8))
    return samples


class Refiner(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        for name, param in self.backbone.named_parameters():
            param.requires_grad_(name.startswith('layer2.') or name.startswith('layer3.'))
        self.decoder = nn.Sequential(nn.Conv2d(64+256+5, 128, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(64, 1, 1))
        self.register_buffer('mean', torch.tensor((.485,.456,.406))[None,:,None,None])
        self.register_buffer('std', torch.tensor((.229,.224,.225))[None,:,None,None])

    def train(self, mode=True):
        super().train(mode)
        self.backbone.bn1.eval()
        self.backbone.layer1.eval()
        return self

    def forward(self, rgb, code):
        x = (rgb-self.mean)/self.std
        x = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        early = self.backbone.layer1(x)
        deep = self.backbone.layer3(self.backbone.layer2(early))
        heat = self.decoder(torch.cat((early, F.interpolate(deep, size=early.shape[-2:],
                    mode='bilinear', align_corners=False),
                    code[:,:,None,None].expand(-1,-1,*early.shape[-2:])), 1))[:,0]
        b, h, w = heat.shape
        probs = F.softmax(heat.flatten(1).float(), -1).reshape(b,h,w)
        yy, xx = torch.meshgrid(torch.arange(h, device=rgb.device),
                                torch.arange(w, device=rgb.device), indexing='ij')
        xy = torch.stack(((probs*xx).sum((1,2)), (probs*yy).sum((1,2))), -1)
        return xy * ((CROP-1)/(w-1)), heat


def codes(samples, device):
    code = torch.zeros((len(samples),5), device=device)
    for i, sample in enumerate(samples):
        code[i, CAMS.index(sample['cam'])] = 1
        code[i, 3+sample['end']] = 1
    return code


@torch.inference_mode()
def evaluate(model, array, samples, device, batch_size):
    model.eval()
    xy = []
    for start in range(0,len(samples),batch_size):
        chunk = samples[start:start+batch_size]
        image = torch.from_numpy(np.array(array[start:start+batch_size])).permute(0,3,1,2).to(device).float()/255
        pred,_ = model(image,codes(chunk,device))
        xy.extend(pred.cpu().numpy())
    pixel = np.asarray([np.asarray(s['center'])+v-HALF for s,v in zip(samples,xy)])
    valid = np.asarray([i for i,s in enumerate(samples) if s['label_inframe']])
    errors = np.linalg.norm(pixel[valid]-np.asarray([samples[i]['truth'] for i in valid]),axis=1)
    return pixel, {'inframe_count':len(valid), 'median_px':float(np.median(errors)),
                   'p95_px':float(np.percentile(errors,95)),
                   'inside_count':int(sum(samples[i]['inside'] for i in valid))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--train-predictions',type=Path,required=True)
    p.add_argument('--development-predictions',type=Path,required=True)
    p.add_argument('--init',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--minutes',type=float,default=30)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--train-groups-manifest',type=Path,
                   help='Optional nested complete-group subset for learning-curve audit')
    args=p.parse_args()
    if args.output.exists():p.error('Preserve prior output')
    args.output.mkdir(parents=True)
    random.seed(17);np.random.seed(17);torch.manual_seed(17)
    cv2.setNumThreads(1);torch.set_num_threads(4)
    rows=read(args.labels/'rows.jsonl')
    train_rows=[r for r in rows if r['split']=='train']
    tr=read(args.train_predictions);dv=read(args.development_predictions)
    if len({r['scene_group'] for r in train_rows})!=70 or len({rows[x['row']]['scene_group'] for x in dv})!=10:
        raise ValueError('Unexpected grouped split')
    if args.train_groups_manifest:
        selected=set(json.loads(args.train_groups_manifest.read_text())['scene_groups'])
        if not selected or not selected.issubset({r['scene_group'] for r in train_rows}):
            raise ValueError('Unknown or empty train group subset')
        tr=[x for x in tr if x['scene_group'] in selected]
        if {x['scene_group'] for x in tr} != selected:
            raise ValueError('Incomplete selected train groups')
    else:
        selected={r['scene_group'] for r in train_rows}
    (args.output/'training_groups.json').write_text(json.dumps(sorted(selected),indent=2)+'\n')
    train_samples=prepare(tr,train_rows,'train',True,args.output/'train_crops.npy')
    dev_samples=prepare(dv,rows,'validation',False,args.output/'development_crops.npy')
    train_array=np.load(args.output/'train_crops.npy',mmap_mode='r')
    dev_array=np.load(args.output/'development_crops.npy',mmap_mode='r')
    print(json.dumps({'train_crops':len(train_samples),'development_crops':len(dev_samples)}),flush=True)
    base=Locator(None)
    base.load_state_dict(torch.load(args.init,map_location='cpu',weights_only=False)['model'])
    model=Refiner(base.backbone).cuda()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=1e-4)
    history=[];best=float('inf');best_step=None;start=time.monotonic()
    for step in range(1,args.updates+1):
        if time.monotonic()-start>=args.minutes*60:break
        ids=np.random.randint(0,len(train_samples),size=args.batch_size)
        batch=[train_samples[i] for i in ids]
        image=torch.from_numpy(np.array(train_array[ids])).permute(0,3,1,2).cuda().float()/255
        target=torch.as_tensor(np.asarray([np.asarray(s['truth'])-s['center']+HALF for s in batch]),device='cuda',dtype=torch.float32)
        model.train();opt.zero_grad(set_to_none=True)
        xy,heat=model(image,codes(batch,'cuda'))
        h,w=heat.shape[-2:]
        cell=(target[:,1]*(h-1)/(CROP-1)).round().long().clamp(0,h-1)*w+\
             (target[:,0]*(w-1)/(CROP-1)).round().long().clamp(0,w-1)
        loss=F.smooth_l1_loss(xy,target,beta=3.)+\
             .25*F.cross_entropy(heat.flatten(1),cell)
        loss.backward();opt.step()
        if step==1 or step%250==0 or step==args.updates:
            pixel,metrics=evaluate(model,dev_array,dev_samples,'cuda',args.batch_size)
            metrics.update({'step':step,'loss':float(loss.detach()),
                            'elapsed_s':round(time.monotonic()-start,1)})
            history.append(metrics)
            print(json.dumps(metrics),flush=True)
            if metrics['p95_px']<best:
                best=metrics['p95_px'];best_step=step
                torch.save({'model':model.state_dict(),'step':step,
                            'init':str(args.init)},args.output/'best.pt')
                output=[dict(x) for x in dv]
                for sample,point in zip(dev_samples,pixel):
                    output[sample['pi']]['endpoint_xy_normalized'][sample['ci']][sample['end']]=(point/SCALE).tolist()
                with (args.output/'development_predictions.jsonl').open('w') as f:
                    for r in output:f.write(json.dumps(r)+'\n')
    summary={'schema':'sfp_rail_original_crop_refiner/v1','train_groups':len(selected),
             'development_groups':10,'train_labeled_crops':len(train_samples),
             'development_proposal_crops':len(dev_samples),'best_step':best_step,
             'best_development_p95_px':best,'history':history,
             'original_rgb_crop':True,'observation_only_crop_center':True,
             'projected_label_is_not_visibility':True,'final_scenes_opened':False}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
