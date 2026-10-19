#!/usr/bin/env python3
"""Evaluate selected-port identity on independent native RGB scene groups."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_two_family_port_locator import Locator, Views, WIDTH, HEIGHT


def stats(values):
    return {'n':len(values),'median':float(np.median(values)),
            'p95':float(np.quantile(values,.95))} if values else {'n':0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--checkpoint',action='append',nargs=2,metavar=('NAME','PATH'),required=True)
    p.add_argument('--after-s',type=float,default=20.)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous result')
    rows=[r for r in map(json.loads,(a.labels/'rows.jsonl').open()) if r['time_s']>=a.after_s]
    groups=defaultdict(list)
    indices=[(i,c) for i in range(len(rows)) for c in range(3)]
    loader=DataLoader(Views(rows,None,indices,False),batch_size=48,shuffle=False,num_workers=2)
    torch.set_num_threads(4)
    results={}
    for name,path in a.checkpoint:
        model=Locator(None).cuda().eval()
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
        episode_values=defaultdict(lambda:defaultdict(list))
        with torch.inference_mode():
            for image,meta,points,present,inframe,selected,_,row_indices,cameras in loader:
                pred,flag=model(image.cuda().float()/255.,meta.cuda())
                pred=pred.float().cpu().numpy();flag=flag.float().sigmoid().cpu().numpy()
                points=points.numpy();inframe=inframe.numpy()
                for j,idx in enumerate(row_indices.tolist()):
                    candidate=selected[j].item();episode=rows[idx]['episode'];bucket=episode_values[episode]
                    expected=points[j,candidate];observed=pred[j,candidate]
                    error=float(np.linalg.norm((observed-expected)*[WIDTH-1,HEIGHT-1]))
                    bucket['selected_pixel_error'].append(error)
                    bucket['selected_flag'].append(float(flag[j,candidate]))
                    bucket['flag_pass'].append(float(flag[j,candidate]>=.5))
                    others=[float(np.linalg.norm((observed-points[j,k])*[WIDTH-1,HEIGHT-1]))
                            for k in np.flatnonzero(inframe[j]) if k!=candidate]
                    bucket['closer_to_other'].append(float(bool(others) and min(others)<error))
        results[name]={'checkpoint':str(path),'episodes':{
            episode:{key:(stats(value) if key in ('selected_pixel_error','selected_flag')
                          else float(np.mean(value))) for key,value in fields.items()}
            for episode,fields in episode_values.items()}}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'schema':'aic_sfp_locator_identity_heldout/v1',
        'after_s':a.after_s,'results':results,'geometry_scoring_only':True,
        'reserved_final_opened':False},indent=2)+'\n')
    print(json.dumps({name:{ep:{'p95_px':v['selected_pixel_error']['p95'],
                                  'wrong_fraction':v['closer_to_other'],'flag_pass':v['flag_pass']}
                            for ep,v in result['episodes'].items()}
                      for name,result in results.items()}))


if __name__=='__main__':main()
