#!/usr/bin/env python3
"""Bounded structural check for frozen SFP requested-port predictions.

Only selected-port and native-crop heads can produce an anchor. If the crop
source pair is on the requested card, the refined 3D point must agree with
that pair's requested-port endpoint to within 10 mm (less than half the
roughly 23 mm separation of the two ports). All geometry here is predicted
from images; true simulator positions are read only after the decision.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(path):
    return {(r['episode'],r['time_s']):r for line in path.open() if (r:=json.loads(line))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--selected',type=Path,required=True)
    ap.add_argument('--crop',type=Path,required=True)
    ap.add_argument('--proposals',type=Path,required=True)
    ap.add_argument('--candidates',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('Preserve previous audit')
    selected=read(args.selected);crop=read(args.crop);proposals=read(args.proposals)
    candidates=read(args.candidates)
    rows=[];groups=defaultdict(list)
    for key in sorted(selected):
        s=selected[key];c=crop.get(key);p=proposals[key];a=candidates[key]
        card=int(s['requested'].split('_')[2])
        same=bool(c and p['crop'] and p['crop']['source_card']==card)
        distance=None;crop_allowed=False
        if c and c['accepted']:
            if same:
                source=a['candidates'][s['requested']]['predicted_base_m']
                if source is not None and c['predicted_base_m'] is not None:
                    distance=float(np.linalg.norm(np.asarray(source)-np.asarray(c['predicted_base_m']))*1000)
                crop_allowed=distance is not None and distance<=10.0
            else:
                crop_allowed=True
        chosen=s if s['accepted'] else c if crop_allowed else None
        row=dict(episode=key[0],time_s=key[1],requested=s['requested'],
                 accepted=chosen is not None,
                 source='selected' if s['accepted'] else 'crop' if crop_allowed else None,
                 same_card_pair=same,crop_to_source_endpoint_mm=distance,
                 error_mm=chosen['error_mm'] if chosen else None,
                 nearest_identity_label_only=chosen['nearest_identity_label_only'] if chosen else None)
        rows.append(row);groups[key[0]].append(row)
    episodes=[];errors=[];wrong=0
    for name,items in sorted(groups.items()):
        items.sort(key=lambda x:x['time_s'])
        first=next((x for x in items if x['accepted']),None)
        accepted=[x for x in items if x['accepted']]
        wrong_items=[x for x in accepted if x['nearest_identity_label_only']!=x['requested']]
        errors.extend(x['error_mm'] for x in accepted)
        wrong+=len(wrong_items)
        episodes.append(dict(episode=name,requested=items[0]['requested'],
                             accepted=len(accepted),wrong=len(wrong_items),
                             first_accept_from_first_sample_s=(first['time_s']-items[0]['time_s'] if first else None),
                             early_within_3s=bool(first and first['time_s']-items[0]['time_s']<=3)))
    summary=dict(schema='sfp_fresh_observation_structural_gate/v1',
                 threshold_mm=10.0,threshold_basis='less than half 23 mm same-card port separation',
                 episodes=episodes,episode_count=len(episodes),accepted=sum(x['accepted'] for x in episodes),
                 accepted_wrong=wrong,early_within_3s=sum(x['early_within_3s'] for x in episodes),
                 any_accept=sum(x['accepted']>0 for x in episodes),
                 accepted_3d_p95_mm=float(np.quantile(errors,.95)) if errors else None,
                 deployment_promoted=False,development_diagnostic_only=True,
                 truth_used_in_decision=False,final_scenes_opened=False)
    args.output.mkdir(parents=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (args.output/'rows.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
