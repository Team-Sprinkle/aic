#!/usr/bin/env python3
"""Matched requested-SFP crop diagnostic with current spatial maps and distance history.

This is crop-only: projected-in-frame is not true visibility, so no deployment
stop/accept head is trained here. Simulator geometry is label/scoring only.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_requested_history_crop import (CAMERAS, HEIGHT, LOOKUP, WIDTH,
                                           relative_pose, rows_from)


def distance_histories(rows, step_mm=30., slots=12):
    grouped = defaultdict(list)
    for i, r in enumerate(rows):
        grouped[r['episode']].append(i)
    result = {}
    for indices in grouped.values():
        indices.sort(key=lambda i: rows[i]['time_s'])
        xyz = np.asarray([rows[i]['state'][:3] for i in indices], np.float64)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]*1000
        anchors = [0]
        for n, i in enumerate(indices):
            if arc[n]-arc[anchors[-1]] >= step_mm:
                anchors.append(n)
            prior = [a for a in anchors if a < n][-slots+1:]
            chosen = (prior+[n])[-slots:]
            result[i] = [(indices[a], float(arc[n]-arc[a])) for a in chosen]
    return result


def coarse_cache(full, path, device):
    if path.exists():
        a = np.load(path, mmap_mode='r')
        if a.shape != (len(full), 3, 256, 4, 5):
            raise ValueError('Coarse feature shape mismatch')
        return a
    a = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16,
                                  shape=(len(full), 3, 256, 4, 5))
    with torch.no_grad():
        for i in range(0, len(full), 16):
            x = torch.from_numpy(np.asarray(full[i:i+16], dtype=np.float32)).to(device)
            y = F.adaptive_avg_pool2d(x.flatten(0, 1), (4, 5))
            a[i:i+len(x)] = y.reshape(len(x), 3, 256, 4, 5).cpu().numpy().astype(np.float16)
    a.flush()
    return np.load(path, mmap_mode='r')


class CoarseCrop(nn.Module):
    def __init__(self, history=True, state_conditioned=False):
        super().__init__()
        d = 128
        self.history = history
        self.state_conditioned = state_conditioned
        self.target = nn.Embedding(12, d)
        self.camera = nn.Embedding(3, d)
        self.position = nn.Embedding(20, d)
        self.feature = nn.Linear(256, d)
        self.motion = nn.Sequential(nn.Linear(10, d), nn.GELU(), nn.Linear(d, d))
        self.state = nn.Sequential(nn.Linear(7, d), nn.GELU(), nn.Linear(d, d))
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, 4, batch_first=True)
                                   for _ in range(2)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.current = nn.Conv2d(256, d, 1)
        self.heat = nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(d, 1, 1))
        self.inframe = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, current, history, motion, mask, target, state=None,
                return_context=False):
        b, h = history.shape[:2]
        if not self.history:
            mask = mask.clone(); mask[:, :-1] = True
        spatial = history.float().permute(0, 1, 2, 4, 5, 3).reshape(b, h, 3, 20, 256)
        cams = self.camera(torch.arange(3, device=current.device))[None, None, :, None]
        pos = self.position(torch.arange(20, device=current.device))[None, None, None]
        keys = (self.feature(spatial) + cams + pos +
                self.motion(motion)[:, :, None, None]).reshape(b, h*3*20, 128)
        q = self.target(target)[:, None] + self.camera(
            torch.arange(3, device=current.device))[None]
        if self.state_conditioned:
            if state is None: raise ValueError('Missing measured TCP state')
            q = q + self.state(state)[:, None]
        key_mask = mask[:, :, None, None].expand(-1, -1, 3, 20).reshape(b, -1)
        for attn, norm in zip(self.attn, self.norm):
            updated, _ = attn(q, keys, keys, key_padding_mask=key_mask, need_weights=False)
            q = norm(q+updated)
        fused = self.current(current.float().reshape(b*3, 256, 32, 36))
        fused = fused + q.reshape(b*3, 128, 1, 1)
        logits = self.heat(fused).reshape(b, 3, 32, 36)
        score = F.softmax(logits.flatten(2)*4., dim=-1)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, 32, device=current.device),
                                torch.linspace(0, 1, 36, device=current.device), indexing='ij')
        xy = torch.stack(((score*xx.flatten()).sum(-1),
                          (score*yy.flatten()).sum(-1)), -1)
        output = (xy, self.inframe(q).squeeze(-1), logits)
        if return_context:
            return output + (q, fused.reshape(b, 3, 128, 32, 36))
        return output


def batch(rows, full, small, histories, chosen, device, random_target=False,
          mask_housing=False):
    n = len(chosen); slots = 12
    current = torch.from_numpy(np.asarray(full[chosen], np.float32)).to(device)
    ix = np.zeros((n, slots), np.int64)
    mask = np.ones((n, slots), bool)
    motion = np.zeros((n, slots, 10), np.float32)
    target_ids = np.zeros(n, np.int64)
    points = np.full((n, 3, 2), .5, np.float32)
    valid = np.zeros((n, 3), np.float32)
    states = np.zeros((n, 7), np.float32)
    for k, i in enumerate(chosen):
        states[k] = np.asarray(rows[i]['state'][:7], np.float32)
        seq = histories[i]
        for j, (past, path_mm) in enumerate(seq, start=slots-len(seq)):
            ix[k, j] = past; mask[k, j] = False
            motion[k, j, :9] = relative_pose(rows[i]['state'], rows[past]['state'])
            motion[k, j, 9] = path_mm/400.
        name = (f'sfp_card_{random.randrange(5)}_port_{random.randrange(2)}'
                if random_target and random.random() < .4 else rows[i]['selected_identity'])
        target_ids[k] = LOOKUP[name]
        t = next((v for v in rows[i]['targets'] if v['identity'] == name), None)
        if t is not None:
            for c, cam in enumerate(CAMERAS):
                uv = t['cameras'][cam]
                housing = t.get('housing_occluded_rgb_heuristic',{}).get(cam,False)
                if uv[3] and not (mask_housing and housing):
                    valid[k, c] = 1.
                    points[k, c] = (uv[0]/(WIDTH-1), uv[1]/(HEIGHT-1))
    hist = torch.from_numpy(np.asarray(small[ix], np.float32)).to(device)
    return (current, hist, torch.from_numpy(motion).to(device),
            torch.from_numpy(mask).to(device), torch.from_numpy(target_ids).to(device),
            torch.from_numpy(points).to(device), torch.from_numpy(valid).to(device),
            torch.from_numpy(states).to(device))


@torch.no_grad()
def evaluate(model, rows, full, small, histories, indices, device, output=None,
             mask_housing=False):
    model.eval(); records=[]
    for offset in range(0, len(indices), 8):
        chosen=indices[offset:offset+8]
        inp=batch(rows,full,small,histories,chosen,device,
                  mask_housing=mask_housing)
        xy, flag, _ = model(*inp[:5], state=inp[7])
        pred=xy.float().cpu().numpy()*[WIDTH-1,HEIGHT-1]
        probs=flag.float().sigmoid().cpu().numpy()
        for k,i in enumerate(chosen):
            r=rows[i]; t=next(v for v in r['targets'] if v['identity']==r['selected_identity'])
            error=[]; wrong=[]
            for c,cam in enumerate(CAMERAS):
                uv=t['cameras'][cam]; e=float(np.linalg.norm(pred[k,c]-uv[:2]))
                alternatives=[v['cameras'][cam][:2] for v in r['targets']
                              if v['identity']!=r['selected_identity'] and v['cameras'][cam][3]]
                other=min((float(np.linalg.norm(pred[k,c]-p)) for p in alternatives),default=1e9)
                error.append(e); wrong.append(e>other)
            records.append({'episode':r['episode'],'time_s':r['time_s'],
                            'requested':r['selected_identity'],'prediction_px':pred[k].tolist(),
                            'projected_in_frame_score':probs[k].tolist(),
                            'error_px_label_only':error,'wrong_port_label_only':wrong,
                            'projected_in_frame_label_only':[bool(t['cameras'][c][3]) for c in CAMERAS]})
    accepted=[r for r in records if min(r['projected_in_frame_score'])>=.5]
    by_ep=defaultdict(list)
    for r in records: by_ep[r['episode']].append(r)
    early=sum(any(min(r['projected_in_frame_score'])>=.5 and
                  r['time_s']-min(x['time_s'] for x in seq)<=3 for r in seq)
              for seq in by_ep.values())
    result={'episodes':len(by_ep),'rows':len(records),'accepted':len(accepted),
            'wrong_port_any_view':sum(any(r['wrong_port_label_only']) for r in accepted),
            'crop_coverage_60px_all_view':sum(max(r['error_px_label_only'])<=60 for r in accepted),
            'error_px_p50':float(np.median([max(r['error_px_label_only']) for r in records])),
            'early_episodes_3s':early}
    if output:
        with open(output,'w') as f:
            for r in records:f.write(json.dumps(r)+'\n')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--labels',type=Path,required=True)
    ap.add_argument('--stress-labels',type=Path,required=True)
    ap.add_argument('--feature-cache',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--updates',type=int,default=1000)
    ap.add_argument('--minutes',type=float,default=90)
    ap.add_argument('--history-off',action='store_true')
    ap.add_argument('--state-conditioned',action='store_true')
    ap.add_argument('--exclude-housing-occluded',action='store_true')
    ap.add_argument('--train-min-target-distance-mm',type=float,default=0.)
    ap.add_argument('--seed',type=int,default=20261004)
    a=ap.parse_args()
    if a.output.exists():ap.error('Output exists: preserve previous run')
    a.output.mkdir(parents=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(8)
    rows=rows_from(a.labels)+rows_from(a.stress_labels)
    train=[i for i,r in enumerate(rows) if r['split']=='train']
    if a.train_min_target_distance_mm:
        train=[i for i in train if np.linalg.norm(
            np.asarray(next(t for t in rows[i]['targets'] if
                            t['identity']==rows[i]['selected_identity'])['position_base_m'])-
            np.asarray(rows[i]['state'][:3]))*1000 >= a.train_min_target_distance_mm]
    dev=[i for i,r in enumerate(rows[:len(rows_from(a.labels))]) if r['split']=='validation']
    stress=list(range(len(rows)-len(rows_from(a.stress_labels)),len(rows)))
    groups={name:{rows[i]['scene_group'] for i in ix}
            for name,ix in [('train',train),('dev',dev),('stress',stress)]}
    if any(groups[x]&groups[y] for x,y in [('train','dev'),('train','stress'),('dev','stress')]):
        raise ValueError('Scene group leakage')
    full=np.load(a.feature_cache/'features.npy',mmap_mode='r')
    manifest=json.loads((a.feature_cache/'features_manifest.json').read_text())
    expected=[{'episode':r['episode'],'time_s':r['time_s'],
               'scene_group':r['scene_group'],'images':r['images']} for r in rows]
    if manifest!=expected:raise ValueError('Feature cache/row mismatch')
    small=coarse_cache(full,a.feature_cache/'distance_4x5_features.npy','cuda')
    histories=distance_histories(rows)
    (a.output/'manifest.json').write_text(json.dumps({
        'train_groups':sorted(groups['train']),'dev_groups':sorted(groups['dev']),
        'inspected_stress_groups':sorted(groups['stress']),'history_off':a.history_off,
        'history_step_mm':30,'max_slots':12,'state_conditioned':a.state_conditioned,
        'exclude_housing_occluded':a.exclude_housing_occluded,
        'train_min_target_distance_mm':a.train_min_target_distance_mm,
        'train_rows_after_phase_filter':len(train),
        'source_labels':str(a.labels),
        'stress_labels':str(a.stress_labels),'feature_cache':str(a.feature_cache),
        'true_visibility_labels':False,'stop_accept_trained':False,'final_scenes_opened':False},indent=2)+'\n')
    model=CoarseCrop(not a.history_off,a.state_conditioned).cuda()
    optim=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    best=float('inf');best_step=0;logs=[];start=time.monotonic()
    for step in range(1,a.updates+1):
        if time.monotonic()-start>a.minutes*60:break
        model.train();chosen=random.choices(train,k=8)
        inp=batch(rows,full,small,histories,chosen,'cuda',True,a.exclude_housing_occluded)
        optim.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            xy,flag,heat=model(*inp[:5],state=inp[7])
        label=inp[5];visible=inp[6]
        pixel=F.smooth_l1_loss(xy.float(),label,beta=.02,reduction='none').sum(-1)
        cells=(label[...,1]*31).round().long()*36+(label[...,0]*35).round().long()
        ce=F.cross_entropy((heat.float()*4).reshape(-1,32*36),cells.flatten(),
                           reduction='none').reshape_as(visible)
        loss=((pixel+.01*ce)*visible).sum()/visible.sum().clamp_min(1)
        loss+=.2*F.binary_cross_entropy_with_logits(flag.float(),visible)
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
        if step%100==0:
            dev_score=evaluate(model,rows,full,small,histories,dev,'cuda',
                               mask_housing=a.exclude_housing_occluded)
            score=dev_score['wrong_port_any_view']*10+(10-dev_score['early_episodes_3s'])*5
            score+=10*(1-dev_score['crop_coverage_60px_all_view']/max(1,dev_score['accepted']))
            log={'step':step,'loss':float(loss.detach()),'selection_score':score,'dev':dev_score}
            logs.append(log);print(json.dumps(log),flush=True)
            if score<best:
                best=score;best_step=step
                torch.save({'model':model.state_dict(),'step':step,
                            'history_off':a.history_off,'state_conditioned':a.state_conditioned},
                           a.output/'best.pt')
    if not best_step:raise RuntimeError('No selected checkpoint')
    model.load_state_dict(torch.load(a.output/'best.pt',weights_only=False)['model'])
    result={'schema':'distance_coarse_crop_v1','best_step':best_step,'updates':step,
            'train_episodes':len(groups['train']),'dev':evaluate(model,rows,full,small,histories,dev,'cuda',a.output/'dev_predictions.jsonl',a.exclude_housing_occluded),
            'inspected_stress':evaluate(model,rows,full,small,histories,stress,'cuda',a.output/'stress_predictions.jsonl',a.exclude_housing_occluded),
            'stop_accept_trained':False,'promotion':False}
    (a.output/'history.json').write_text(json.dumps(logs,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
