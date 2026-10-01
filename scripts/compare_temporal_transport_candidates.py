#!/usr/bin/env python3
"""Full held-out episode evaluation of frozen-vision temporal transport arms."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_temporal_transport_probe import TemporalTransport, build_data, evaluate
from audit_temporal_transport_labels import STAGES


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--base-checkpoint',type=Path,required=True)
    p.add_argument('--sc-labels',type=Path,required=True)
    p.add_argument('--sc-spec',type=Path,required=True)
    p.add_argument('--history',type=Path,required=True)
    p.add_argument('--no-history',type=Path,required=True)
    p.add_argument('--state-skip',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(8)
    data=build_data(a.cache,a.base_checkpoint,'cuda')
    spec=json.loads(a.sc_spec.read_text())
    card_by_case={e['case']:e['card_count'] for e in spec['episodes']}
    card_by_episode={}
    with a.sc_labels.open() as f:
        for line in f:
            r=json.loads(line)
            card_by_episode[int(r['episode_index'])]=card_by_case[r['episode_id']]
    for row in data['rows']:
        if row['family']=='sc':row['card_count']=card_by_episode[row['episode']]
    val=np.array([i for i,r in enumerate(data['rows']) if r['split']=='validation'],dtype=np.int64)
    reports={};paths=dict(history=a.history,no_history=a.no_history,state_skip=a.state_skip)
    for name,path in paths.items():
        cp=torch.load(path,map_location='cpu',weights_only=False)
        raw=cp['source_manifest'].get('raw_state_skip',False)
        model=TemporalTransport(data['mean'],data['std'],raw_state_skip=raw).cuda().eval()
        model.load_state_dict(cp['model'],strict=True)
        reports[name]=dict(checkpoint=str(path),sha256=sha(path),best_update=cp['update'],
                           history_enabled=cp['history_enabled'],raw_state_skip=raw,
                           validation=evaluate(model,data,val,not cp['history_enabled']))
        del model;torch.cuda.empty_cache()
    summary=dict(schema='aic_temporal_transport_full_heldout/v1',
                 validation_episode_counts=dict(sc=21,sfp=32),
                 validation_rows=dict(sc=sum(data['rows'][i]['family']=='sc' for i in val),
                                      sfp=sum(data['rows'][i]['family']=='sfp' for i in val)),
                 reports=reports,
                 scope='all cached 200ms decision rows in held-out complete episodes; teacher observations, not autonomous rollout',
                 offline_gate='SC stage accuracy >=.90; next-subgoal translation p95 <=20mm; no stage skips',
                 final_scenes_opened=False,rl_started=False)
    a.output.write_text(json.dumps(summary,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(10,4))
    names=list(STAGES);x=np.arange(6);width=.25
    for j,(name,report) in enumerate(reports.items()):
        stage=report['validation']['by_sc_stage']
        y=[stage[str(i)]['subgoal_translation_mm_p95'] for i in range(6)]
        ax.bar(x+(j-1)*width,y,width,label=name)
    ax.axhline(20,color='red',linestyle='--',linewidth=1,label='20 mm gate')
    ax.set_xticks(x,names,rotation=25,ha='right');ax.set_ylabel('Held-out subgoal p95 (mm)')
    ax.legend();fig.tight_layout()
    fig.savefig(a.output.with_suffix('.png'),dpi=160);plt.close(fig)
    print(json.dumps({k:v['validation']['overall'] for k,v in reports.items()}))


if __name__=='__main__':main()
