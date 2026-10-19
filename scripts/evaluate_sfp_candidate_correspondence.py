#!/usr/bin/env python3
"""Replay frozen SFP point-to-request verifier on saved RGB cascade proposals."""

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from train_sfp_candidate_correspondence_probe import CAMERAS, Correspondence, frozen_maps
from train_two_family_port_locator import Locator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--cascade',type=Path,required=True)
    p.add_argument('--correspondence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--diagnostic-ablations',action='store_true',
                   help='Score labeled point and zero normalized TCP state for audit only')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous result')
    labels=[json.loads(x) for x in a.labels.open()]
    label_by_key={(r['episode'],float(r['time_s'])):r for r in labels}
    cascade=json.loads(a.cascade.read_text())
    saved=torch.load(a.correspondence,map_location='cpu',weights_only=False)
    samples=[];images=[]
    for item in cascade['episode_rows']:
        key=(item['episode'],float(item['time_s']))
        row=label_by_key.get(key)
        if row is None or 'fine_pixels_px' not in item['diagnostic']:continue
        if len(item['diagnostic']['fine_pixels_px'])!=3:continue
        for cam,point in zip(CAMERAS,item['diagnostic']['fine_pixels_px']):
            im=cv2.imread(row['images'][cam]);
            if im is None:raise FileNotFoundError(row['images'][cam])
            im=cv2.cvtColor(cv2.resize(im,(288,256),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
            images.append(im)
            samples.append({'episode':item['episode'],'time_s':item['time_s'],
                'selected':row['selected_identity'],'camera':cam,'state':row['state'][:7],
                'point':(np.asarray(point)/[1151,1023]).tolist(),
                'point_native_px':point,'targets':row['targets'],
                'cascade_accepted':bool(item['accepted']),
                'reprojection_px':item['diagnostic'].get('reprojection_px'),
                'image':row['images'][cam]})
    locator=Locator(None).to(a.device).eval()
    locator.load_state_dict(torch.load(saved['locator'],map_location='cpu',weights_only=False)['model'])
    visual=frozen_maps(locator,np.stack(images),a.device).to(a.device).float()
    state=torch.tensor([r['state'] for r in samples],device=a.device,dtype=torch.float32)
    state=(state-saved['state_mean'].to(a.device))/saved['state_std'].to(a.device)
    camera=torch.tensor([[float(r['camera']==c) for c in CAMERAS] for r in samples],
                        device=a.device)
    request=torch.tensor([[float(r['selected']==name) for name in saved['names']]
                          for r in samples],device=a.device)
    point=torch.tensor([r['point'] for r in samples],device=a.device,dtype=torch.float32)
    condition=torch.cat((request,camera,state,point),1)
    model=Correspondence().to(a.device).eval();model.load_state_dict(saved['model'])
    with torch.inference_mode():
        scores=[]
        for i in range(0,len(samples),128):
            scores.extend(model(visual[i:i+128],condition[i:i+128],point[i:i+128]).sigmoid().cpu().tolist())
        true_scores=[];zero_state_scores=[]
        if a.diagnostic_ablations:
            true_points=[]
            for r in samples:
                selected=next((t for t in r['targets'] if t['identity']==r['selected']),None)
                projection=selected['cameras'][r['camera']] if selected else None
                true_points.append((np.asarray(projection[:2])/[1151,1023]).tolist()
                                   if projection is not None and projection[3] else [0,0])
            true_point=torch.tensor(true_points,device=a.device,dtype=torch.float32)
            for i in range(0,len(samples),128):
                zero=condition[i:i+128].clone();zero[:,13:20]=0
                zero_state_scores.extend(model(visual[i:i+128],zero,point[i:i+128]).sigmoid().cpu().tolist())
                true_cond=condition[i:i+128].clone();true_cond[:,-2:]=true_point[i:i+128]
                true_scores.extend(model(visual[i:i+128],true_cond,true_point[i:i+128]).sigmoid().cpu().tolist())
    results=[];counts=Counter()
    for i in range(0,len(samples),3):
        views=samples[i:i+3];prob=scores[i:i+3]
        if len(views)!=3:continue
        requested=views[0]['selected']
        distances={}
        for target in views[0]['targets']:
            if target['identity'] not in saved['names']:continue
            errors=[]
            for v in views:
                projection=target['cameras'][v['camera']]
                if not projection[3]:break
                errors.append(float(np.linalg.norm(np.asarray(v['point_native_px'])-projection[:2])))
            if len(errors)==3:distances[target['identity']]=errors
        nearest=min(distances,key=lambda name:np.mean(distances[name])) if distances else None
        association=all(p>=.5 for p in prob)
        accepted=views[0]['cascade_accepted']
        result={'episode':views[0]['episode'],'time_s':views[0]['time_s'],
            'requested_identity':requested,'nearest_identity_audit_only':nearest,
            'selected_error_px_by_camera':distances.get(requested),
            'nearest_error_px_by_camera':distances.get(nearest),
            'cascade_accepted':accepted,'association_scores':dict(zip(CAMERAS,prob)),
            'zero_state_scores_audit_only':dict(zip(CAMERAS,zero_state_scores[i:i+3])) if a.diagnostic_ablations else None,
            'true_point_scores_audit_only':dict(zip(CAMERAS,true_scores[i:i+3])) if a.diagnostic_ablations else None,
            'association_all_three_accept':association,
            'combined_accept':accepted and association,
            'reprojection_px':views[0]['reprojection_px'],
            'images':[v['image'] for v in views]}
        results.append(result)
        if accepted:
            kind='correct' if nearest==requested else 'wrong'
            counts[f'cascade_{kind}']+=1
            counts[f'combined_{kind}']+=association
    out={'schema':'sfp_candidate_correspondence_live_proposal_replay/v1',
        'frozen_cascade':str(a.cascade),'frozen_correspondence':str(a.correspondence),
        'runtime_true_geometry':False,'truth_used_only_after_prediction':True,
        'threshold':.5,'all_three_required':True,'rows':results,
        'counts':dict(counts),'final_scenes_opened':False}
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'counts':out['counts'],
        'known_wrong':[r for r in results if r['episode'].startswith('card1_port1')
                       and r['time_s'] in (3.15,5.05)]},indent=2))


if __name__=='__main__':main()
