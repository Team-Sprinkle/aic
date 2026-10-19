#!/usr/bin/env python3
"""Score a frozen native-crop port locator on predicted-rail proposals."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_sfp_observed_rail_crop import Crops, evaluate
from train_target_conditioned_port_locator import ConditionalLocator
from train_two_family_port_locator import CAMERAS


def read(path):
    return [json.loads(s) for s in (path/'rows.jsonl').open()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--proposals',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous evaluation')
    labels={(r['episode'],r['time_s']):r for r in read(a.labels)}
    rows=[]
    for proposal in read(a.proposals):
        label=labels[(proposal['episode'],proposal['time_s'])]
        assert proposal['scene_group']==label['scene_group']
        rows.append({'episode':proposal['episode'],'time_s':proposal['time_s'],
                     'scene_group':proposal['scene_group'],'images':label['images'],
                     'state':label['state'],'targets':label['targets'],
                     'requested':proposal['requested'],'proposal':proposal['crop']})
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if ckpt.get('history_context') or ckpt.get('source_offset_meta'):
        p.error('This evaluation supports the no-history, no-offset crop model only')
    torch.set_num_threads(8)
    model=ConditionalLocator(None).cuda()
    model.load_state_dict(ckpt['model'])
    loader=DataLoader(Crops(rows,[(i,c) for i in range(len(rows)) for c in range(3)],False),
                      batch_size=24,num_workers=4,pin_memory=True,
                      multiprocessing_context='spawn')
    cal=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics={c:np.asarray(cal['tcp_to_optical'][c],float) for c in CAMERAS}
    a.output.mkdir(parents=True)
    summary=evaluate(model,rows,loader,optics,a.output/'predictions.jsonl')
    summary.update({'checkpoint':str(a.checkpoint),'labels':str(a.labels),
                    'proposals':str(a.proposals),'episode_groups':len({r['scene_group'] for r in rows}),
                    'crop_center_source':sorted({r['proposal']['source'] for r in rows}),
                    'final_scenes_opened':False,'runtime_true_geometry_input':False})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
