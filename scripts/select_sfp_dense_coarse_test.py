#!/usr/bin/env python3
"""Freeze 18 new 3–5-card groups for a separate coarse crop stress test."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rail-labels',type=Path,required=True)
    p.add_argument('--port-labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing test selection')
    rail=[json.loads(line) for line in (a.rail_labels/'rows.jsonl').open()]
    port=[json.loads(line) for line in (a.port_labels/'rows.jsonl').open()]
    by_count=defaultdict(set)
    for r in rail:
        if r['split']=='train' and r['card_count'] in (3,4,5):
            by_count[r['card_count']].add(r['scene_group'])
    chosen=set()
    for count in (3,4,5):
        scenes=sorted(by_count[count],key=lambda s:hashlib.sha256(s.encode()).hexdigest())
        if len(scenes)<6:raise ValueError(f'Only {len(scenes)} scenes for count {count}')
        chosen.update(scenes[:6])
    selected_rail=[dict(r,split='validation') for r in rail if r['scene_group'] in chosen]
    selected_port=[dict(r,split='validation',fit_split='validation') for r in port
                   if r['scene_group'] in chosen]
    def key(r):return (r['scene_group'],r['episode'],round(float(r['time_s']),6))
    if {key(r) for r in selected_rail}!={key(r) for r in selected_port}:
        raise ValueError('Rail/port provenance mismatch')
    a.output.mkdir(parents=True)
    for name,items in (('rail',selected_rail),('port',selected_port)):
        folder=a.output/name;folder.mkdir()
        with (folder/'rows.jsonl').open('w') as f:
            for r in items:f.write(json.dumps(r,separators=(',',':'))+'\n')
    manifest=dict(schema='aic_sfp_dense_coarse_test/v1',scenes=sorted(chosen),
                  counts={str(c):6 for c in (3,4,5)},rows=len(selected_rail),
                  original_split='train',test_split_for_frozen_evaluators='validation',
                  source_rail=str(a.rail_labels),source_port=str(a.port_labels),
                  test_not_used_for_model_selection=True,
                  gate=dict(zero_wrong_accepts=True,early_anchor_min=16,
                            accepted_3d_p95_mm_max=20),
                  final_scenes_opened=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='scenes'}),flush=True)


if __name__=='__main__':main()
