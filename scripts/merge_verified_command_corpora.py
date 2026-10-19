#!/usr/bin/env python3
"""Freeze a non-destructive union of independently qualified command copies.

Rows and public observation prefixes remain immutable symlinked sources. Reject
duplicate episodes and physical reset leakage before producing the manifest.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    episodes=[];seen=set();splits={};counts=Counter();links=[];sources=[]
    for ordinal,root in enumerate(a.inputs):
        root=root.resolve();manifest=root/'summary.json'
        summary=json.loads(manifest.read_text())
        sources.append(dict(path=str(root),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))
        for old in summary['episodes']:
            e=dict(old)
            if e['episode'] in seen:raise ValueError('Duplicate source episode: '+e['episode'])
            seen.add(e['episode'])
            source=root/e['rows_file'];rows=0;scene=None
            with source.open() as f:
                for line in f:
                    r=json.loads(line);rows+=1
                    if not r.get('full_recorded_controller_contract_verified'):
                        raise ValueError('Unverified controller contract')
                    if r['episode']!=e['episode'] or r['split']!=e['split'] or r['family']!=e['family']:
                        raise ValueError('Row/manifest identity mismatch')
                    if scene is not None and scene!=r['scene']:raise ValueError('Mixed resets in one episode')
                    scene=r['scene']
            if rows!=e['rows'] or not rows:raise ValueError('Bad source row count')
            if scene in splits and splits[scene]!=e['split']:raise ValueError('Physical reset leakage: '+scene)
            splits[scene]=e['split'];e['scene']=scene
            for key in ('rows_file','observation_prefix_file'):
                if not e.get(key):continue
                src=(root/e[key]).resolve()
                if not src.is_file():raise FileNotFoundError(src)
                name=f'{ordinal:02d}_{Path(e[key]).name}'
                links.append((name,src));e[key]=name
            counts[f'{e["family"]}/{e["split"]}']+=rows;episodes.append(e)
    a.output.mkdir(parents=True,exist_ok=False)
    for name,source in links:(a.output/name).symlink_to(source)
    report=dict(schema='verified_command_union/v1',episodes=episodes,rows=dict(counts),
        source_episodes=len(episodes),physical_resets=len(splits),inputs=sources,
        original_data_modified=False,full_controller_contract_verified=True,
        causal_port_grasp_locks_still_required=True,actor_training_ready=False,
        final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'}))


if __name__=='__main__':main()
