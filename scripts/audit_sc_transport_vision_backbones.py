#!/usr/bin/env python3
"""Frozen, episode-grouped SC route probe for ResNet18 and DINOv2-S features.

This is an offline representation diagnostic, not an autonomous policy test.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
from torch.nn import functional as F

from shared_bc_camera_preprocessing import transport_rgb_tensor


def route_targets(rows):
    target = np.zeros((len(rows), 3), np.float32)
    valid = np.zeros(len(rows), bool)
    episodes = defaultdict(list)
    for i, row in enumerate(rows):
        if row['family'] == 'sc':
            episodes[(row['split'], row['episode'])].append(i)
    for ids in episodes.values():
        ids.sort(key=lambda i: rows[i]['time_s'])
        times = np.asarray([rows[i]['time_s'] for i in ids])
        for k, i in enumerate(ids):
            at = np.searchsorted(times, times[k] + 1.)
            candidates = [j for j in (at-1, at) if k < j < len(ids)
                          and .8 <= times[j]-times[k] <= 1.25]
            if not candidates:
                continue
            j = min(candidates, key=lambda j: abs(times[j]-times[k]-1.))
            current = np.asarray(rows[i]['state'][:3])
            future = np.asarray(rows[ids[j]]['state'][:3])
            target[i] = Rotation.from_quat(rows[i]['state'][3:7]).inv().apply(future-current)
            valid[i] = True
    return target, valid


def metrics(truth, prediction):
    error = np.linalg.norm(truth-prediction, axis=1)*1000
    return dict(rows=len(error), median_mm=float(np.median(error)),
                p95_mm=float(np.quantile(error, .95)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--sc-labels', type=Path, required=True)
    p.add_argument('--hub', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-train-episode', type=int, default=10)
    p.add_argument('--per-validation-episode', type=int, default=10)
    a = p.parse_args()
    if a.output.exists(): p.error('Output exists; preserve previous result')
    torch.set_num_threads(8)
    rows = [json.loads(line) for line in (a.cache/'rows.jsonl').open()]
    target, valid = route_targets(rows)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row['family']=='sc' and valid[i]:
            groups[(row['split'],row['episode'])].append(i)
    indices=[]
    for (split, _), ids in sorted(groups.items()):
        n = a.per_train_episode if split=='train' else a.per_validation_episode
        indices.extend(np.asarray(ids)[np.linspace(0,len(ids)-1,min(n,len(ids))).astype(int)].tolist())
    wanted={(rows[i]['episode'],rows[i]['frame']) for i in indices}
    paths={}
    for line in a.sc_labels.open():
        row=json.loads(line)
        key=(int(row['episode_index']),int(row['frame']))
        if key in wanted: paths[key]=row['images']
    if set(paths)!=wanted: raise ValueError(f'Missing {len(wanted)-len(paths)} image triplets')
    encoder = torch.hub.load(str(a.hub),'dinov2_vits14',source='local',pretrained=False)
    encoder.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=False))
    encoder=encoder.cuda().eval()
    features = np.load(a.cache/'features.npy',mmap_mode='r')
    dino=[];dino_grid=[];duration=[]
    mean=torch.tensor((.485,.456,.406),device='cuda')[None,:,None,None]
    std=torch.tensor((.229,.224,.225),device='cuda')[None,:,None,None]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for number,i in enumerate(indices):
            item=rows[i];images=[]
            for camera in ('center','left','right'):
                with Image.open(paths[(item['episode'],item['frame'])][camera]) as f:
                    images.append(transport_rgb_tensor(np.asarray(f.convert('RGB'))))
            x=(torch.stack(images).cuda().float()/255.-mean)/std
            started=time.perf_counter()
            output=encoder.forward_features(x)
            torch.cuda.synchronize()
            duration.append((time.perf_counter()-started)*1000)
            # Keep the three view identities; compare only global features here.
            patch=output['x_norm_patchtokens'].mean(1)
            dino.append(patch.float().cpu().numpy().reshape(-1))
            spatial=output['x_norm_patchtokens'].transpose(1,2).reshape(3,384,16,16)
            dino_grid.append(F.adaptive_avg_pool2d(spatial.float(),(4,4)).cpu().numpy().reshape(-1))
            if number%100==0:print(json.dumps(dict(done=number,total=len(indices))),flush=True)
    dino=np.asarray(dino,np.float32)
    dino_grid=np.asarray(dino_grid,np.float32)
    base=np.asarray(features[indices],np.float32)
    state=np.asarray([rows[i]['state'] for i in indices],np.float32)
    task=np.asarray([rows[i]['task'] for i in indices],np.float32)
    y=target[indices]
    train=np.asarray([rows[i]['split']=='train' for i in indices]);validation=~train
    if {rows[indices[i]]['scene'] for i in np.where(train)[0]} & \
       {rows[indices[i]]['scene'] for i in np.where(validation)[0]}:
        raise ValueError('Scene leakage')
    results={}
    train_episodes=sorted({rows[i]['episode'] for i in indices if rows[i]['split']=='train'})
    calibration_episodes=set(train_episodes[::5])
    fit=np.asarray([rows[i]['split']=='train' and rows[i]['episode'] not in calibration_episodes for i in indices])
    calibration=train & ~fit
    for name,visual in [('state_task',None),('resnet18_global',base),
                        ('dinov2_small_global',dino),('dinov2_small_spatial4x4',dino_grid)]:
        x=np.concatenate((state,task,visual),axis=1) if visual is not None else np.concatenate((state,task),axis=1)
        center=x[train].mean(0)
        scale=x[train].std(0).clip(1e-6)
        x=np.clip((x-center)/scale,-10,10)
        xt=torch.as_tensor(x[train],device='cuda')
        xv=torch.as_tensor(x[validation],device='cuda')
        yt=torch.as_tensor(y[train],device='cuda')
        kernel=xt@xt.T
        identity=torch.eye(len(xt),device='cuda')
        arms={}
        calibration_scores={}
        xfit=torch.as_tensor(x[fit],device='cuda')
        xcal=torch.as_tensor(x[calibration],device='cuda')
        yfit=torch.as_tensor(y[fit],device='cuda')
        fit_kernel=xfit@xfit.T
        fit_identity=torch.eye(len(xfit),device='cuda')
        for alpha in (10.,100.,1000.):
            fit_coefficients=torch.linalg.solve(fit_kernel+alpha*fit_identity,yfit)
            calibration_prediction=(xcal@xfit.T@fit_coefficients).cpu().numpy()
            calibration_scores[str(alpha)]=metrics(y[calibration],calibration_prediction)
            coefficients=torch.linalg.solve(kernel+alpha*identity,yt)
            prediction=(xv@xt.T@coefficients).cpu().numpy()
            arms[str(alpha)]=metrics(y[validation],prediction)
        selected=min(calibration_scores,key=lambda alpha:calibration_scores[alpha]['p95_mm'])
        results[name]=dict(selected_alpha=float(selected),heldout=arms[selected],
                           train_calibration=calibration_scores,alpha_sweep=arms)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(schema='aic_sc_transport_frozen_vision_route_probe/v1',
                rows=len(indices),train_rows=int(train.sum()),validation_rows=int(validation.sum()),
                train_episodes=len({rows[i]['episode'] for i in indices if rows[i]['split']=='train'}),
                validation_episodes=len({rows[i]['episode'] for i in indices if rows[i]['split']=='validation'}),
                dino_encoder_only_ms=dict(p50=float(np.quantile(duration,.5)),
                                          p95=float(np.quantile(duration,.95))),
                results=results,final_scenes_opened=False,
                caveat='Teacher-frame route prediction only; neither closed-loop policy nor full inference latency')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
