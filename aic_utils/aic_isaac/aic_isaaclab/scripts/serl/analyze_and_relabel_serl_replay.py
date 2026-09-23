#!/usr/bin/env python3
"""Summarize causal SERL replay and optionally add explicit episode outcomes."""
from __future__ import annotations
import argparse, collections, json
from pathlib import Path
import torch


def episode_id(t):
    m=t.get('metadata') or {}
    for value in (m.get('causal_episode'),m.get('episode_before_step'),m):
        if isinstance(value,dict) and value.get('episode_id'): return str(value['episode_id'])
    return 'unknown'

def final_geometry(t):
    terminal=t.get('terminal_observation') or {}
    return terminal.get('insertion_geometry') or (t.get('metadata') or {}).get('post_step_insertion_geometry') or {}

def scalar_by_env(g,key,env=0):
    v=g.get(key+'_by_env')
    if isinstance(v,list) and len(v)>env:return v[env]
    return g.get(key+'_env0',g.get(key+'_mean'))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('inputs',nargs='+'); ap.add_argument('--output'); ap.add_argument('--summary',required=True); ap.add_argument('--success-bonus',type=float,default=5.0); ap.add_argument('--failure-penalty',type=float,default=-5.0); a=ap.parse_args()
    groups=collections.OrderedDict(); source=[]
    for name in a.inputs:
        x=torch.load(name,map_location='cpu',weights_only=False); source.append({'path':name,'transitions':len(x['transitions'])})
        for t in x['transitions']:groups.setdefault(episode_id(t),[]).append(t)
    rows=[]; kept=[]
    for eid,ts in groups.items():
        last=ts[-1]; m=last.get('metadata') or {}; complete=bool(m.get('terminated') or m.get('truncated'))
        g=final_geometry(last); strict=scalar_by_env(g,'strict_success')
        success=bool(strict) if complete else False
        components=collections.Counter(); modes=collections.Counter(); forces=[]; actor_owned=[]
        for t in ts:
            md=t.get('metadata') or {}; ps=md.get('policy_sample') or {}; c=ps.get('component')
            if isinstance(c,list):c=c[0] if c else None
            if c is not None:components[int(c)]+=1
            rec=md.get('measured_path_recovery') or {}; modes[str(rec.get('mode_name','none'))]+=1
            state=t.get('obs',{}).get('state')
            if torch.is_tensor(state) and state.numel()>=29:forces.append(float(torch.linalg.norm(state[26:29])))
            actor_owned.append(bool(t.get('actor_owned',torch.tensor([True])).reshape(-1)[0]))
        row={'episode_id':eid,'transitions':len(ts),'complete':complete,'success':success,'terminated':bool(m.get('terminated')),'truncated':bool(m.get('truncated')),'signed_depth_mm':None if scalar_by_env(g,'signed_depth_m') is None else 1000*float(scalar_by_env(g,'signed_depth_m')),'lateral_mm':None if scalar_by_env(g,'lateral_error_m') is None else 1000*float(scalar_by_env(g,'lateral_error_m')),'orientation_deg':None if scalar_by_env(g,'orientation_error_rad') is None else 180/3.141592653589793*float(scalar_by_env(g,'orientation_error_rad')),'max_force_n':max(forces,default=0.0),'component_counts':dict(components),'recovery_mode_counts':dict(modes),'actor_owned_fraction':sum(actor_owned)/max(1,len(actor_owned))}
        rows.append(row)
        if complete:
            copied=[]
            for t in ts:
                q=dict(t); q['metadata']=dict(t.get('metadata') or {}); q['metadata']['relabel_schema']='episode_outcome_v1'; q['metadata']['episode_success']=success
                copied.append(q)
            copied[-1]['done']=torch.ones_like(copied[-1]['done'])
            copied[-1]['reward']=copied[-1]['reward'] + (a.success_bonus if success else a.failure_penalty)
            copied[-1]['metadata']['episode_outcome_reward_added']=a.success_bonus if success else a.failure_penalty
            kept.extend(copied)
    complete=[r for r in rows if r['complete']]; summary={'schema_version':1,'sources':source,'episodes_total':len(rows),'episodes_complete':len(complete),'successes':sum(r['success'] for r in complete),'failures':sum(not r['success'] for r in complete),'transitions_relabelled':len(kept),'episodes':rows}
    Path(a.summary).parent.mkdir(parents=True,exist_ok=True); Path(a.summary).write_text(json.dumps(summary,indent=2)+'\n')
    if a.output:
        Path(a.output).parent.mkdir(parents=True,exist_ok=True); torch.save({'capacity':max(len(kept),1),'size':len(kept),'saved_size':len(kept),'relabel_schema':'episode_outcome_v1','sources':source,'transitions':kept},a.output)
    print(json.dumps({k:v for k,v in summary.items() if k!='episodes'},indent=2))
if __name__=='__main__':main()
