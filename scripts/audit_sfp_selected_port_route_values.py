#!/usr/bin/env python3
"""Score a fixed episode-spread sample of frozen RGB route inputs posthoc."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from build_two_family_port_projection_labels import sfp_ports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--values',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous sample audit')
    rows=[json.loads(s) for s in (a.cache/'rows.jsonl').open()]
    values=np.load(a.values/'values.npy',mmap_mode='r')
    accepted=np.load(a.values/'accepted.npy',mmap_mode='r')
    if len(rows)!=len(values) or len(rows)!=len(accepted):raise ValueError('Cache row mismatch')
    sources={(int(r['episode_index']),int(r['frame_index'])):r
             for r in map(json.loads,a.labels.open()) if 'images' in r}
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        if r['family']=='sfp' and accepted[i] and (int(r['episode']),int(r['frame'])) in sources:
            groups[(r['split'],int(r['episode']))].append(i)
    rng=np.random.default_rng(20261007);chosen=[]
    for split in ('train','validation'):
        episodes=[key for key in groups if key[0]==split]
        rng.shuffle(episodes)
        for key in episodes:
            if sum(rows[i]['split']==split for i in chosen)>=50:break
            chosen.append(int(rng.choice(groups[key])))
        extra=0
        while sum(rows[i]['split']==split for i in chosen)<50:
            key=episodes[extra%len(episodes)];extra+=1
            candidates=[i for i in groups[key] if i not in chosen]
            if candidates:chosen.append(int(rng.choice(candidates)))
            if extra>1000:raise ValueError('Too few distinct accepted rows')
    cal=json.loads(Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())
    port_cache={};details=[]
    for i in chosen:
        row=rows[i];label=sources[(int(row['episode']),int(row['frame']))]
        source=Path(label['source_scored_stream']).parent
        if source not in port_cache:
            cfg=yaml.safe_load((source/'eval_config.yaml').read_text())
            tid=next(iter(cfg['trials']))
            port_cache[source]={(card,port):m for card,port,m in sfp_ports(
                {'trial_yaml':str(source/'eval_config.yaml'),'trial_id':tid},cal)}
        ports=port_cache[source]
        state=np.asarray(label['state'],float)
        pred=state[:3]+Rotation.from_quat(state[3:7]).apply(np.asarray(values[i],float)*.2)
        card=int(label['selected_card']);port=int(label['selected_port'])
        truth=ports[card,port][:3,3]
        nearest=min(ports,key=lambda key:np.linalg.norm(pred-ports[key][:3,3]))
        details.append(dict(cache_index=i,split=row['split'],episode=int(row['episode']),
                            frame=int(row['frame']),requested=[card,port],nearest=list(nearest),
                            error_mm=float(np.linalg.norm(pred-truth)*1000)))
    errors=[x['error_mm'] for x in details]
    wrong=sum(x['nearest']!=x['requested'] for x in details)
    result=dict(schema='aic_sfp_selected_port_route_value_sample_audit/v1',sample_size=len(details),
                sampled_episodes=len({(x['split'],x['episode']) for x in details}),
                wrong_accepted=wrong,error_mm_p50=float(np.median(errors)),
                error_mm_p95=float(np.percentile(errors,95)),
                gate_passed=bool(len(details)==100 and wrong==0 and np.percentile(errors,95)<=20),
                geometry_used_after_frozen_prediction_only=True,
                final_scenes_opened=False,details=details)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='details'}))


if __name__=='__main__':main()
