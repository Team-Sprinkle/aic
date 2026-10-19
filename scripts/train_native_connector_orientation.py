#!/usr/bin/env python3
"""RGB-only full connector-in-TCP pose fit on immutable native grasp corpora.

Uses a trained landmark encoder, but directly supervises the complete rigid
pose. Labels, train-only normalization and fixed crop origins are not runtime
geometry inputs. No automatic deployment approval follows this experiment.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import cv2  # Must precede torch/torchvision in this environment.
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.nn import functional as F

from shared_port_tcp_geometry import matrix_from_pose9
from train_sc_native_keypoints import PretrainedKeypointRefiner
from train_sfp_native_grasp_keypoints import metrics, poses_from_pixels


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def rotation6(x):
    first = F.normalize(x[..., :3], dim=-1)
    second = F.normalize(x[..., 3:] - (first*x[..., 3:]).sum(-1, keepdim=True)*first, dim=-1)
    return torch.stack((first, second, torch.linalg.cross(first, second)), -1)


class ConnectorPose(nn.Module):
    def __init__(self, encoder_state, mean, scale):
        super().__init__()
        self.landmark = PretrainedKeypointRefiner(None, False, point_count=5, condition_camera=True)
        self.landmark.backbone.layer4 = nn.Identity()
        self.landmark.backbone.avgpool = nn.Identity()
        self.landmark.backbone.fc = nn.Identity()
        self.landmark.load_state_dict(encoder_state, strict=True)
        # Decoder remains checkpoint-compatible but is not on the direct path.
        for parameter in self.landmark.decoder.parameters():
            parameter.requires_grad_(False)
        self.project = nn.Conv2d(256, 32, 1)
        self.head = nn.Sequential(nn.LayerNorm(3*32*6*6), nn.Linear(3*32*6*6, 256),
                                  nn.GELU(), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 9))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('pose_mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('pose_scale', torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, x):
        # Camera order is fixed center/left/right, preserving camera identity.
        batch = len(x)
        x = x.reshape(-1, 3, 192, 192).float()/255.
        x = (x-self.landmark.mean)/self.landmark.std
        b = self.landmark.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x = b.layer3(b.layer2(b.layer1(x)))
        x = F.adaptive_avg_pool2d(self.project(x), (6,6)).reshape(batch, -1)
        pose = self.head(x)*self.pose_scale + self.pose_mean
        return pose[:, :3], rotation6(pose[:, 3:])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'pixels', 'checkpoint', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--family', choices=['sc', 'sfp'], required=True)
    p.add_argument('--updates', type=int, default=20000)
    p.add_argument('--minutes', type=float, default=90)
    p.add_argument('--min-updates', type=int, default=3000)
    p.add_argument('--patience', type=int, default=3000)
    p.add_argument('--seed', type=int, default=20261017)
    p.add_argument('--rotation-only', action='store_true')
    p.add_argument('--freeze-encoder', action='store_true')
    p.add_argument('--crop-jitter', type=int, default=0)
    a = p.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not physical.isdigit():
        raise ValueError('Select exactly one physical GPU with CUDA_VISIBLE_DEVICES')
    lock = open('/var/tmp/chmin_aic_gpu'+physical+'.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    a.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    rows = json.loads((a.source/'manifest.json').read_text())
    fit = np.array([r['split']=='train' for r in rows])
    val = np.flatnonzero(~fit)
    if not len(val) or not fit.any() or any(r['split'] not in ('train', 'validation') for r in rows):
        raise ValueError('Explicit train/validation splits required')
    for key in ('scene', 'episode'):
        if {r[key] for r in rows if r['split']=='train'} & {r[key] for r in rows if r['split']=='validation'}:
            raise ValueError(key+' leakage')
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    saved['model'] = {k: v for k, v in saved['model'].items()
                      if not k.startswith(('backbone.layer4.', 'backbone.fc.'))}
    metadata = saved['plan']
    if a.family == 'sc':
        truth = np.stack([matrix_from_pose9(y) for y in np.load(a.source/'labels.npy')])
        centres = metadata['crop_centres_train_only']
        origins = np.stack([np.rint(np.asarray(centres[c])-96) for c in ('center','left','right')])
        optics = json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())['tcp_to_optical']
        reference = metadata['CAD_reference_points']
    else:
        truth = np.load(a.source/'body_tip_matrices.npy')
        centres = metadata['crop_centres_train_only']
        origins = np.stack([np.rint(np.asarray(centres[c])-104)+8 for c in ('center','left','right')])
        optics = metadata['tcp_to_optical']
        reference = metadata['reference_points']
    pixels = np.load(a.pixels, mmap_mode='r')
    if len(pixels)!=len(rows) or pixels.shape[1]!=3 or pixels.dtype!=np.uint8:
        raise ValueError('Pixel/manifest mismatch')
    if pixels.shape[2:4] not in ((192,192),(208,208)):
        raise ValueError('Unexpected original crop size')
    targets = np.concatenate((truth[:, :3, 3], truth[:, :3, :2].transpose(0,2,1).reshape(-1,6)), axis=1)
    mean, scale = targets[fit].mean(0), targets[fit].std(0).clip(min=1e-4)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if fit[i]: groups[r['episode']].append(i)
    groups = list(groups.values())
    split_counts = {s: dict(samples=sum(r['split']==s for r in rows),
                          episodes=len({r['episode'] for r in rows if r['split']==s}),
                          resets=len({r['scene'] for r in rows if r['split']==s}))
                    for s in ('train','validation')}
    # Reproduce the current pixel-only decoder before comparing direct pose.
    baseline = PretrainedKeypointRefiner(None, False, point_count=5, condition_camera=True)
    baseline.backbone.layer4=nn.Identity(); baseline.backbone.avgpool=nn.Identity(); baseline.backbone.fc=nn.Identity()
    baseline.load_state_dict(saved['model'], strict=True)
    baseline.cuda().eval()
    def images(ids, jitter=False):
        array = np.array(pixels[ids], copy=True)
        if array.shape[2]==208:
            shifts = rng.integers(0,17,size=(len(ids),3,2)) if jitter else np.full((len(ids),3,2),8)
            array = np.stack([np.stack([array[i,c,sy:sy+192,sx:sx+192]
                for c,(sx,sy) in enumerate(shifts[i])]) for i in range(len(ids))])
        if jitter and a.crop_jitter:
            size=a.crop_jitter
            padded=np.pad(array,((0,0),(0,0),(size,size),(size,size),(0,0)),mode='edge')
            shifts=rng.integers(0,2*size+1,size=(len(ids),3,2))
            array=np.stack([np.stack([padded[i,c,sy:sy+192,sx:sx+192]
                for c,(sx,sy) in enumerate(shifts[i])]) for i in range(len(ids))])
            gain=rng.uniform(.85,1.15,size=(len(ids),3,1,1,3))
            array=np.clip(array*gain,0,255).astype(np.uint8)
        return torch.from_numpy(array).permute(0,1,4,2,3).cuda()
    baseline_predictions=[]
    with torch.inference_mode():
        for j in range(0,len(val),24):
            image=images(val[j:j+24])
            coords=baseline(image.reshape(-1,3,192,192).float()/255,
                camera_id=torch.eye(3,device='cuda').repeat(len(image),1)).cpu().numpy().reshape(-1,3,5,2)
            baseline_predictions.append(poses_from_pixels(coords*191+origins[None,:,None,:],optics,reference))
    baseline_predictions=np.concatenate(baseline_predictions)
    baseline_report=metrics(baseline_predictions,truth[val])
    del baseline
    constant=np.broadcast_to(np.eye(4),(len(val),4,4)).copy()
    constant[:,:3,3]=truth[fit,:3,3].mean(0)
    constant[:,:3,:3]=Rotation.from_matrix(truth[fit,:3,:3]).mean().as_matrix()
    plan=dict(schema='native_connector_full_pose/v1',family=a.family,
        hypothesis='The first direct SC pose fit overfits held-out grasps. Rotation-only supervision plus crop/colour augmentation tests orientation generalization; approved translation stays a separate component.',
        source=str(a.source),pixels=str(a.pixels),checkpoint=str(a.checkpoint),split_counts=split_counts,
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((a.source/'manifest.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs='three ordered original192 RGB crops only; fixed train-derived origins, no true geometry or robot state',
        crop_origins_native=origins.tolist(),camera_order=['center','left','right'],
        geometry_labels_only=True,final_scenes_opened=False,source_modified=False,
        maximum_updates=a.updates,minutes=a.minutes,min_updates=a.min_updates,patience=a.patience,
        stop='Joint translation/orientation validation score plateau, or explicit time/update cap; best and last kept.',
        selection='max(translation_p95/0.4mm,orientation_p95/1degree)',
        promotion='Offline candidate only, no runtime deployment or insertion claim.',
        rotation_only=a.rotation_only,freeze_encoder=a.freeze_encoder,crop_jitter=a.crop_jitter,
        physical_gpu=physical,seed=a.seed,baseline=baseline_report,constant_baseline=metrics(constant,truth[val]))
    save(a.output/'plan.json',plan)
    save(a.output/'split_manifest.json',[dict(index=i,episode=r['episode'],scene=r['scene'],split=r['split']) for i,r in enumerate(rows)])
    np.save(a.output/'baseline_heldout_poses.npy',baseline_predictions)
    model=ConnectorPose(saved['model'],mean,scale).cuda()
    if a.freeze_encoder:
        for v in model.landmark.parameters():v.requires_grad_(False)
    params=[x for x in model.parameters() if x.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-3)
    tt=torch.as_tensor(truth[:,:3,3],device='cuda',dtype=torch.float32)
    rr=torch.as_tensor(truth[:,:3,:3],device='cuda',dtype=torch.float32)
    def evaluate(ids):
        model.eval(); predicted=[]
        with torch.inference_mode():
            for j in range(0,len(ids),32):
                t,r=model(images(ids[j:j+32]))
                pose=np.broadcast_to(np.eye(4),(len(t),4,4)).copy()
                pose[:,:3,3]=t.cpu().numpy();pose[:,:3,:3]=r.cpu().numpy()
                predicted.append(pose)
        result=np.concatenate(predicted)
        return metrics(result,truth[ids]),result
    history=[];best=float('inf');best_step=0;reason='update_cap';started=time.monotonic()
    for step in range(1,a.updates+1):
        if time.monotonic()-started>a.minutes*60 and best_step:
            reason='time_cap';break
        selected=np.array([groups[g][rng.integers(len(groups[g]))] for g in rng.integers(len(groups),size=16)])
        model.train();
        if a.freeze_encoder:model.landmark.eval()
        opt.zero_grad(set_to_none=True)
        t,r=model(images(selected,jitter=True))
        position=F.smooth_l1_loss((t-tt[selected])/.0004,torch.zeros_like(t))
        angle=(r-rr[selected]).square().sum((-1,-2)).mean()/(2*(np.pi/180)**2)
        loss=angle if a.rotation_only else position+angle
        if not torch.isfinite(loss):raise ValueError('Non-finite full-pose loss')
        loss.backward();nn.utils.clip_grad_norm_(params,1);opt.step()
        if step%200==0:
            metric,_=evaluate(val)
            score=metric['orientation_p50_p95_deg'][1] if a.rotation_only else max(metric['translation_p50_p95_mm'][1]/.4,metric['orientation_p50_p95_deg'][1])
            history.append(dict(step=step,metrics=metric,score=score,wall_seconds=time.monotonic()-started))
            checkpoint=dict(model=model.state_dict(),plan=plan,pose_mean=mean,pose_scale=scale,step=step)
            if score<best*.995:
                best=score;best_step=step;torch.save(checkpoint,a.output/'best.pt')
            save(a.output/'history.json',history)
            print(json.dumps(history[-1]),flush=True)
            if step>=a.min_updates and step-best_step>=a.patience:
                reason='validation_plateau';break
    torch.save(dict(model=model.state_dict(),plan=plan,pose_mean=mean,pose_scale=scale,step=step),a.output/'last.pt')
    model.load_state_dict(torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)['model'])
    metric,pred=evaluate(val);np.save(a.output/'heldout_pose_predictions.npy',pred)
    np.save(a.output/'heldout_indices.npy',val)
    train_ids=np.flatnonzero(fit)[::max(1,int(fit.sum()/512))]
    train_metric,_=evaluate(train_ids)
    summary=dict(plan=plan,metrics=metric,train_subsample_metrics=train_metric,
        baseline=baseline_report,constant_baseline=plan['constant_baseline'],best_step=best_step,
        updates_completed=step,stop_reason=reason,plateau_reached=reason=='validation_plateau',
        gate_passed=metric['translation_p50_p95_mm'][1]<=.4 and metric['orientation_p50_p95_deg'][1]<=1,
        orientation_approved=False,automatic_runtime_change=False,
        parameters=sum(x.numel() for x in model.parameters()),trainable_parameters=sum(x.numel() for x in params),
        wall_seconds=time.monotonic()-started)
    save(a.output/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
