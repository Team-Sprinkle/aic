#!/usr/bin/env python3
"""Replay frozen state/task arrival proposal on existing autonomous SFP traces."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from build_two_family_port_projection_labels import sfp_ports
from train_sfp_arrival_probe import Probe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--run',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous replay')
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if saved['mode']!='state_only':raise ValueError('This replay requires state_only probe')
    model=Probe(1536,len(saved['state_mean']),'state_only').eval()
    model.load_state_dict(saved['model'])
    cal=json.loads(a.calibration.read_text());results=[]
    for run in a.run:
        trace=run/'runtime_trace.jsonl';scene=run/'eval_config.yaml'
        if not trace.exists() or not scene.exists():raise FileNotFoundError(run)
        cfg=yaml.safe_load(scene.read_text());trial_id=next(iter(cfg['trials']))
        ports={(c,p):m for c,p,m in sfp_ports({'trial_yaml':str(scene),'trial_id':trial_id},cal)}
        rows=[]
        for line in trace.open():
            x=json.loads(line)
            if x.get('event')!='proposal' or len(x.get('state',[]))!=32:continue
            task=x.get('task_vector')
            if task is None or task[0]!=1:continue
            rows.append(x)
        rows.sort(key=lambda x:x['sim_time'])
        run_length=0;trigger=None;observations=[]
        for x in rows:
            st=np.asarray(x['state']+x['task_vector'],np.float32)
            z=(st-saved['state_mean'])/saved['state_std']
            with torch.inference_mode():
                prob=float(model(torch.zeros(1,1536),torch.from_numpy(z[None])).sigmoid().item())
            run_length=run_length+1 if prob>=saved['threshold'] else 0
            card=int(np.argmax(x['task_vector'][4:9]));port=int(np.argmax(x['task_vector'][2:4]))
            m=ports[card,port];rel=m[:3,:3].T@(np.asarray(x['state'][:3])-m[:3,3])
            record=dict(time_s=float(x['sim_time']),prob=prob,card=card,port=port,
                posthoc_lateral_mm=float(np.linalg.norm(rel[:2])*1000),
                posthoc_axial_mm=float(rel[2]*1000))
            observations.append(record)
            if trigger is None and run_length>=saved['streak']:trigger=record
        results.append(dict(run=str(run),rows=len(rows),trigger=trigger,
                            closest_lateral_mm=min((x['posthoc_lateral_mm'] for x in observations),default=None),
                            max_probability=max((x['prob'] for x in observations),default=None),
                            trace=observations))
    a.output.write_text(json.dumps({'schema':'aic_sfp_arrival_onpolicy_replay/v1',
        'checkpoint':str(a.checkpoint),'runs':results,
        'port_geometry_used_only_after_proposal_for_scoring':True,
        'reserved_final_opened':False},indent=2)+'\n')
    print(json.dumps([{k:v[k] for k in ('run','rows','trigger','closest_lateral_mm','max_probability')}
                      for v in results],indent=2))


if __name__=='__main__':main()
