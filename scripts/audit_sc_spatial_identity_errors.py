#!/usr/bin/env python3
"""Save exact held-out image/request errors for a frozen SC identity probe."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_sc_absent_presence_head import load_rows
from train_sc_spatial_identity_probe import CAMERAS, SelectedHead, frozen_maps
from train_two_family_port_locator import Locator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--variant',choices=('pooled','spatial'),default='spatial')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--full-development',action='store_true',
                   help='Score every saved development frame with frozen weights')
    a=p.parse_args()
    d=json.loads(a.summary.read_text())
    args=argparse.Namespace(two_port_labels=Path(d['two_port_labels']),
        one_port_train=[] if a.full_development else [Path(x) for x in d['one_port_train']],
        one_port_development=[Path(x) for x in d['one_port_development']],
        two_port_frames_per_group=6,
        one_port_frames_per_group=1000000 if a.full_development else 24)
    rows=load_rows(args)
    record=next(x for x in d['records'] if x['variant']==a.variant)
    saved=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
    locator=Locator(None).to(a.device).eval()
    locator.load_state_dict(torch.load(saved['frozen_locator'],map_location='cpu',weights_only=False)['model'])
    visual=frozen_maps(locator,rows,a.device).to(a.device).float()
    state=torch.tensor([r['state'] for r in rows],dtype=torch.float32,device=a.device)
    state=(state-saved['state_mean'].to(a.device))/saved['state_std'].to(a.device)
    camera=torch.tensor([[float(r['camera']==c) for c in CAMERAS] for r in rows],
                        dtype=torch.float32,device=a.device)
    request=torch.eye(2,device=a.device).repeat(len(rows),1)
    condition=torch.cat((camera.repeat_interleave(2,0),state.repeat_interleave(2,0),request),-1)
    model=SelectedHead(a.variant=='spatial',visual.shape[1],condition.shape[1]).to(a.device).eval()
    model.load_state_dict(saved['model'])
    with torch.inference_mode():
        logits=torch.cat([model(visual[torch.arange(i,min(i+256,len(condition)),device=a.device)//2],
                                condition[i:i+256]) for i in range(0,len(condition),256)])
        probabilities=logits.sigmoid().reshape(-1,2).cpu().numpy()
    errors=[];by_group={};development_predictions=[]
    for i,row in enumerate(rows):
        if row['split']!='development':continue
        for port in (0,1):
            truth=bool(row['labels'][port]);prob=float(probabilities[i,port]);pred=prob>=.5
            development_predictions.append({'group':row['group'],'image':row['image'],
                'camera':row['camera'],'requested_port':port,
                'projected_inframe_label':truth,'probability':prob})
            if pred==truth:continue
            item={'group':row['group'],'image':row['image'],'camera':row['camera'],
                'requested_port':port,'projected_inframe_label':truth,'probability':prob,
                'error':'false_accept' if pred else 'miss'}
            errors.append(item)
            group=by_group.setdefault(row['group'],{'false_accept':0,'miss':0})
            group[item['error']]+=1
    a.output.write_text(json.dumps({'schema':'sc_spatial_identity_errors/v1',
        'source_summary':str(a.summary),'variant':a.variant,'threshold':.5,
        'label_kind':'projected in-frame; not occlusion visibility',
        'by_group':by_group,'errors':errors,
        'development_predictions':development_predictions,
        'final_scenes_opened':False},indent=2)+'\n')
    print(json.dumps({'errors':len(errors),'by_group':by_group},indent=2))


if __name__=='__main__':main()
