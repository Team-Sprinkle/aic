#!/usr/bin/env python3
"""Freeze a second, disjoint, unmarked 30-panel SFP identity review set."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_sfp_requested_card_review import choose, panel, prior_by_motion, read


SOURCE = Path('/home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/labels_all_ten/rows.jsonl')
PRIOR = Path('/home/nas/chmin/aic_archives/2026-10-06_sfp_requested_card_blind_review_v1/blind_manifest.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Preserve frozen review; choose a new output')
    earlier_hashes = {hashlib.sha256(Path(r['panel']).read_bytes()).hexdigest()
                      for r in json.loads(PRIOR.read_text())}
    groups = defaultdict(list)
    for row in read(SOURCE): groups[row['episode']].append(row)
    for rows in groups.values(): rows.sort(key=lambda r:r['time_s'])
    chosen=[]
    for card in range(5):
        for port in range(2):
            name=f'card{card}_port{port}'
            for phase,distance in [('route',220),('approach',90),('near',30)]:
                chosen.append((name,phase,choose(groups[name],distance)))
    assert len(chosen)==30
    args.output.mkdir(parents=True);(args.output/'panels').mkdir()
    order=np.random.default_rng(20261007).permutation(len(chosen))
    blind=[];sealed=[]
    for k,j in enumerate(order):
        episode,phase,row=chosen[int(j)]
        alias=f'N{k:02d}'
        history=prior_by_motion(groups[episode],row)
        dest=args.output/'panels'/f'{alias}.jpg'
        panel(row,history,alias,args.output/'panels')
        if hashlib.sha256(dest.read_bytes()).hexdigest() in earlier_hashes:
            raise RuntimeError(f'Panel duplicates earlier frozen review: {alias}')
        blind.append(dict(alias=alias,panel=str(dest.resolve()),requested=row['selected_identity'],
                          history_available=history is not None,current_card='unreviewed',
                          history_card='unreviewed',selected_port='unreviewed',visibility='unreviewed'))
        sealed.append(dict(alias=alias,source_episode=episode,phase=phase,time_s=row['time_s'],
                           scene_group=row['scene_group'],targets_label_only=row['targets'],
                           images=row['images'],history_time_s=history['time_s'] if history else None))
    (args.output/'blind_manifest.json').write_text(json.dumps(blind,indent=2)+'\n')
    (args.output/'sealed_mapping.json').write_text(json.dumps(sealed,indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(dict(schema='sfp_blind_review_followup/v1',
        panels=30,ten_requests=True,phases=['route','approach','near'],
        panel_duplicate_count=0,projections_hidden=True,final_scenes_opened=False),indent=2)+'\n')
    print(args.output)


if __name__=='__main__': main()
