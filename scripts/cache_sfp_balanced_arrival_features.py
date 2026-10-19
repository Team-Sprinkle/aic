#!/usr/bin/env python3
"""Cache frozen three-view features from selected, scored SFP agent replays.

Recorded actions are ignored. Trial YAML supplies selected-port geometry for
offline labels only. Inference inputs are RGB, measured state, and public task.
"""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import torch
from torchvision.models import resnet18
import yaml

from build_sfp_agent_perception_labels import video_frames,reset_hash
from build_two_family_port_projection_labels import sfp_ports
from lerobot_robot_aic.task_encoding import encode_task_vector


CAMERAS=('center','left','right')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def source_episodes(root,calibration):
    selected=[]
    for row in csv.DictReader((root/'selection_report.csv').open()):
        if row['selected']!='True' or row['status']!='OK' or float(row['total_score'])<90:
            continue
        attempt=row['trial_id'];n=re.search(r'attempt_(\d+)',attempt).group(1)
        trial_id='trial_'+n;path=root/'trials'/f'{trial_id}.yaml'
        config=yaml.safe_load(path.read_text());task=next(iter(next(iter(config['trials'].values()))['tasks'].values()))
        card=int(task['target_module_name'].rsplit('_',1)[-1]);port=int(task['port_name'].rsplit('_',1)[-1])
        m=next(m for c,p,m in sfp_ports({'trial_yaml':str(path),'trial_id':trial_id},calibration)
               if (c,p)==(card,port))
        dataset=root/'replay_attempts'/attempt/'dataset'
        assert (dataset/'meta/info.json').exists() and len(list((dataset/'videos').rglob('*.mp4')))==3
        selected.append(dict(attempt=attempt,trial=str(path),dataset=str(dataset),
                             card=card,port=port,score=float(row['total_score']),
                             scene=reset_hash(config),port_pose=m))
    if len({e['scene'] for e in selected})!=len(selected):
        raise ValueError('Duplicate complete reset in selected replay')
    groups=defaultdict(list)
    for e in selected:groups[(e['card'],e['port'])].append(e)
    for group in groups.values():
        group.sort(key=lambda e:e['scene'])
        n=len(group);a=round(n*.6);b=round(n*.8)
        for i,e in enumerate(group):e['split']='train' if i<a else 'development' if i<b else 'audit'
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--resnet',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stride',type=int,default=4)
    p.add_argument('--max-episodes',type=int,default=0)
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing cache')
    if a.stride<1:p.error('Positive stride required')
    a.output.mkdir(parents=True)
    torch.set_num_threads(4)
    calibration=json.loads(a.calibration.read_text())
    selected=source_episodes(a.source,calibration)
    if a.max_episodes:selected=selected[:a.max_episodes]
    model=resnet18(weights=None);model.fc=torch.nn.Identity()
    weights=torch.load(a.resnet,map_location='cpu',weights_only=True)
    model.load_state_dict({k:v for k,v in weights.items() if not k.startswith('fc.')},strict=True)
    model=model.to(a.device).eval()
    means=torch.tensor([.485,.456,.406],device=a.device)[None,:,None,None]
    stds=torch.tensor([.229,.224,.225],device=a.device)[None,:,None,None]
    rows=[];features=[];failures=[];counts=Counter()
    for episode,e in enumerate(selected):
        try:
            dataset=Path(e['dataset']);info=json.loads((dataset/'meta/info.json').read_text())
            frame=pd.read_parquet(next((dataset/'data').rglob('*.parquet'))).sort_values('frame_index')
            meta=pd.read_parquet(next((dataset/'meta/episodes').rglob('*.parquet'))).iloc[0]
            indices=list(range(0,len(frame),a.stride))
            videos={c:video_frames(dataset,info,meta,indices,c) for c in CAMERAS}
            rgb=[];local_rows=[]
            m=e['port_pose'];base_port=Rotation.from_matrix(m[:3,:3])
            task=encode_task_vector(task_family='sfp_to_nic',target_card_index=e['card'],
                                    target_port_index=e['port']).tolist()
            for i in indices:
                state=np.asarray(frame.iloc[i]['observation.state'],np.float32)
                rel=base_port.inv().apply(state[:3]-m[:3,3]);lateral=float(np.linalg.norm(rel[:2]))
                label=int(-.10<=rel[2]<=-.07 and lateral<=.01)
                local_rows.append(dict(episode=episode,attempt=e['attempt'],scene=e['scene'],
                    split=e['split'],card=e['card'],port=e['port'],frame=int(i),
                    time_s=float(frame.iloc[i]['timestamp']),state=state.tolist(),task=task,
                    label=label,supervision_lateral_m=lateral,
                    supervision_axial_m=float(rel[2])))
                for c in CAMERAS:
                    rgb.append(cv2.resize(videos[c][i],(224,224),interpolation=cv2.INTER_AREA))
            rgb=np.stack(rgb)
            encoded=[]
            with torch.inference_mode():
                for first in range(0,len(rgb),96):
                    x=torch.from_numpy(rgb[first:first+96]).permute(0,3,1,2).to(a.device).float()/255
                    with torch.autocast('cuda',dtype=torch.bfloat16):y=model((x-means)/stds)
                    encoded.append(y.float().cpu().numpy().astype(np.float16))
            array=np.concatenate(encoded).reshape(len(indices),3*512)
            rows.extend(local_rows);features.append(array)
            counts[(e['split'],e['card'],e['port'])]+=1
            if (episode+1)%20==0:print(json.dumps(dict(episodes=episode+1,rows=len(rows))),flush=True)
        except Exception as exc:
            failures.append(dict(attempt=e['attempt'],error=repr(exc)))
            print(json.dumps(failures[-1]),flush=True)
    if not features:raise RuntimeError('No valid episode decoded')
    array=np.concatenate(features)
    assert len(array)==len(rows)
    np.save(a.output/'features.npy',array)
    with (a.output/'rows.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
    manifest=dict(schema='aic_sfp_balanced_arrival_frozen_rgb/v1',source=str(a.source),
        rows=len(rows),episodes=len({x['episode'] for x in rows}),feature_shape=list(array.shape),
        group_counts={str(k):v for k,v in sorted(counts.items())},failures=failures,
        source_sha256=digest(a.source/'selection_report.csv'),resnet_sha256=digest(a.resnet),
        calibration_sha256=digest(a.calibration),stride=a.stride,
        image_input='three synchronized recorded 288x256 RGB videos, resized 224x224 for frozen ImageNet ResNet18',
        inference_inputs='frozen RGB features, measured state, public task code',
        training_only_label='selected port-frame TCP axial [-100,-70] mm and lateral <=10 mm',
        action_labels_used=False,selected_score_floor=90,final_scenes_opened=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows),episodes=manifest['episodes'],failures=len(failures))),flush=True)


if __name__=='__main__':main()
