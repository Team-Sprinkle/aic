#!/usr/bin/env python3
"""Matched one-port reset evaluation of two frozen spatial SC identity heads."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from train_sc_spatial_identity_probe import CAMERAS, SelectedHead, frozen_maps
from train_two_family_port_locator import Locator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--projection',nargs='+',type=Path,required=True)
    p.add_argument('--locator',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:3')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior result')
    torch.set_num_threads(4)
    rows=[json.loads(s) for path in a.projection for s in path.open()]
    locator=Locator(None).to(a.device).eval()
    locator.load_state_dict(torch.load(a.locator,map_location='cpu',weights_only=False)['model'])
    visual=frozen_maps(locator,rows,a.device).float().to(a.device)
    state=torch.tensor([r['measured_tcp_base_pose'] for r in rows],
                       device=a.device,dtype=torch.float32)
    camera=torch.tensor([[float(r['camera']==c) for c in CAMERAS] for r in rows],
                        device=a.device,dtype=torch.float32)
    request=torch.eye(2,device=a.device).repeat(len(rows),1)
    predictions={}
    for name,path in (('baseline',a.baseline),('candidate',a.candidate)):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=SelectedHead(True,256,12).to(a.device).eval()
        model.load_state_dict(saved['model'])
        normalized=(state-saved['state_mean'].to(a.device))/saved['state_std'].to(a.device)
        if saved.get('mask_tcp_orientation',False):normalized[:,3:7]=0
        condition=torch.cat((camera.repeat_interleave(2,0),
                             normalized.repeat_interleave(2,0),request),-1)
        with torch.inference_mode():
            prob=torch.cat([model(visual[torch.arange(i,min(i+256,2*len(rows)),device=a.device)//2],
                                  condition[i:i+256]).sigmoid()
                            for i in range(0,2*len(rows),256)]).reshape(-1,2).cpu().numpy()
        predictions[name]=prob
    grouped=defaultdict(dict)
    for i,r in enumerate(rows):grouped[(r['scene_group_id'],r['sim_ns'])][r['camera']]=i
    results=[]
    for (group,stamp),views in sorted(grouped.items()):
        if set(views)!=set(CAMERAS):continue
        ids=[views[c] for c in CAMERAS];r=rows[ids[0]];port=r['target_port']
        row={'group':group,'sim_ns':stamp,'target_port':port,
             'projected_all_views':all(rows[i]['projected_inframe'] for i in ids),
             'opening_tcp_distance_mm':float(np.linalg.norm(r['relative_opening_tcp_m'])*1000)}
        for name,prob in predictions.items():
            row[name]={'real':[float(prob[i,port]) for i in ids],
                       'absent':[float(prob[i,1-port]) for i in ids],
                       'real_all_three':bool(all(prob[ids,port]>=.5)),
                       'absent_all_three':bool(all(prob[ids,1-port]>=.5))}
        results.append(row)
    counts={}
    for name in predictions:
        counts[name]={}
        for scope,sample in (('all',results),
                             ('near_220mm',[r for r in results if r['opening_tcp_distance_mm']<=220])):
            pos=[r for r in sample if r['projected_all_views']]
            counts[name][scope]={'triplets':len(sample),'projected_all_triplets':len(pos),
                                 'projected_all_recalled':sum(r[name]['real_all_three'] for r in pos),
                                 'absent_all_three_accept':sum(r[name]['absent_all_three'] for r in sample)}
    result={'schema':'sc_spatial_identity_pair/v1','counts':counts,'rows':results,
            'projection_is_not_certified_visibility':True,
            'baseline':str(a.baseline),'candidate':str(a.candidate),
            'sources':[str(x) for x in a.projection],'final_scenes_opened':False}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
