#!/usr/bin/env python3
"""Audit supervised SC transport direction on episode-grouped validation frames."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_shared_port_tcp_bc import read_rows, Samples, SharedPortTCPActor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--sc-labels', type=Path, nargs='+', required=True)
    p.add_argument('--sfp-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:3')
    a = p.parse_args()
    rows = read_rows([('sc', x) for x in a.sc_labels])
    rows = [r for r in rows if r['split'] == 'validation' and 3 <= r['elapsed_s'] <= 15 and r['frame'] % 5 == 0]
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    stats = {k: checkpoint['model'][k].numpy() for k in ('state_mean', 'state_std', 'pose_mean',
             'pose_std', 'target_mean', 'target_std')}
    model = SharedPortTCPActor(stats, pretrained=False, use_time=checkpoint.get('include_elapsed', False))
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(a.device).eval()
    loader = DataLoader(Samples(rows, a.sfp_cache), batch_size=16, shuffle=False, num_workers=2)
    all_rows = []
    with torch.inference_mode():
        offset = 0
        for images, state, task, pose, target, elapsed, families, tip in loader:
            pred, guessed = model(images.to(a.device), state.to(a.device), task.to(a.device), elapsed.to(a.device))
            truth = (target[:, 0, :3] - pose[:, :3]).numpy()
            proposed = (pred[:, 0, :3] - guessed[:, :3]).cpu().numpy()
            pose_error = np.linalg.norm((guessed.cpu().numpy()[:, :3] - pose.numpy()[:, :3]), axis=1)
            for i, (t, q) in enumerate(zip(truth, proposed)):
                tn = float(np.linalg.norm(t)); qn = float(np.linalg.norm(q))
                cosine = float(t @ q / (tn * qn)) if tn > .001 and qn > .001 else None
                r = rows[offset + i]
                all_rows.append({'episode':r['episode'],'scene':r['scene'],'frame':r['frame'],
                                 'elapsed_s':r['elapsed_s'],'teacher_delta_mm':(t*1000).tolist(),
                                 'actor_delta_mm':(q*1000).tolist(),'teacher_norm_mm':tn*1000,
                                 'actor_norm_mm':qn*1000,'direction_cosine':cosine,
                                 'pose_error_mm':float(pose_error[i]*1000)})
            offset += len(truth)
    valid = [r for r in all_rows if r['direction_cosine'] is not None]
    ep = defaultdict(list)
    for r in valid: ep[r['episode']].append(r['direction_cosine'])
    summary = {'checkpoint':str(a.checkpoint),'rows':len(all_rows),'episodes':len(set(r['episode'] for r in all_rows)),
               'moving_rows':len(valid),'opposite_direction_fraction':sum(r['direction_cosine'] < 0 for r in valid)/len(valid) if valid else None,
               'direction_cosine_median':float(np.median([r['direction_cosine'] for r in valid])) if valid else None,
               'pose_error_mm_median':float(np.median([r['pose_error_mm'] for r in all_rows])) if all_rows else None,
               'by_episode':{str(k):{'moving_rows':len(v),'direction_cosine_median':float(np.median(v)),
                                    'opposite_fraction':sum(x < 0 for x in v)/len(v)} for k,v in ep.items()},
               'rows_detail':all_rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows_detail'}))

if __name__=='__main__': main()
