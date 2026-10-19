#!/usr/bin/env python3
"""Read-only requested-SFP-port candidate check against RGB-predicted rails."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

CAMS = ('center', 'left', 'right')


def load_rows(path):
    return [json.loads(line) for line in path.open()]


def key(row):
    return (row['scene_group'], row['episode'], round(float(row['time_s']), 6))


def summarize(items, accepted_field):
    grouped = defaultdict(list)
    for row in items:
        grouped[row['scene_group']].append(row)
    early_good = early_wrong = ever_good = ever_wrong = 0
    accepts = wrong = 0
    for scene, rr in grouped.items():
        rr.sort(key=lambda r: r['time_s'])
        first = rr[0]['time_s']
        selected = [r for r in rr if r[accepted_field]]
        accepts += len(selected)
        wrong += sum(r['wrong_identity_label_only'] for r in selected)
        early = [r for r in selected if r['time_s']-first <= 3]
        early_good += any(not r['wrong_identity_label_only'] for r in early)
        early_wrong += any(r['wrong_identity_label_only'] for r in early)
        ever_good += any(not r['wrong_identity_label_only'] for r in selected)
        ever_wrong += any(r['wrong_identity_label_only'] for r in selected)
    return {'groups': len(grouped), 'accepted_rows': accepts, 'wrong_accepted_rows': wrong,
            'early_correct_groups': early_good, 'early_wrong_groups': early_wrong,
            'ever_correct_groups': ever_good, 'ever_wrong_groups': ever_wrong}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rail-predictions', type=Path, required=True)
    p.add_argument('--candidates', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Preserve existing output')
    rail = {key(r):r for r in load_rows(args.rail_predictions)}
    label = {key(r):r for r in load_rows(args.labels)}
    candidates = load_rows(args.candidates)
    if not candidates or len(rail) != len(candidates):
        raise ValueError('Expected one rail prediction per candidate observation')
    rail_by_scene=defaultdict(list)
    for item in rail.values():
        rail_by_scene[item['scene_group']].append(item)
    records=[]
    for old in candidates:
        k=key(old)
        if k not in rail or k not in label:
            raise ValueError(f'Missing causal same-frame join {k}')
        r=rail[k]; gt=label[k]
        if old['requested'] != gt['selected_identity']:
            raise ValueError('Different requested target')
        card=int(old['requested'].split('_')[2])
        history=[q['occupied_prob'] for q in rail_by_scene[r['scene_group']]
                 if 0 <= r['time_s']-q['time_s'] <= 3]
        prob=np.median(history,axis=0)[card]
        # The vote sees only this frame and observations in its past 3 seconds.
        candidate=old['candidates'][old['requested']]
        indices=[]
        for ci,cam in enumerate(CAMS):
            point=candidate['views'].get(cam)
            if point is None:
                continue
            endpoint=np.asarray(r['endpoint_xy_normalized'][ci])*[1151,1023]
            start,end=endpoint
            delta=end-start
            if np.linalg.norm(delta)<40:
                continue
            slot=4*float(np.dot(np.asarray(point)-start,delta)/np.dot(delta,delta))
            indices.append({'camera':cam,'predicted_slot_index':slot,
                            'matches_requested_rail':abs(slot-card)<=.5})
        geometry_ok=sum(x['matches_requested_rail'] for x in indices)>=2
        baseline=bool(candidate['three_view_consistent'])
        proposed=baseline and prob>=.5 and geometry_ok
        target=next(t for t in gt['targets'] if t['identity']==old['requested'])
        pos=np.asarray(candidate['predicted_base_m'],float)
        nearest=min(gt['targets'],key=lambda t:np.linalg.norm(pos-t['position_base_m']))
        wrong=nearest['identity']!=old['requested']
        records.append({'scene_group':old['scene_group'],'episode':old['episode'],
                        'time_s':old['time_s'],'requested':old['requested'],
                        'requested_occupancy_prob':float(prob),
                        'slot_indices':indices,'baseline_accept':baseline,
                        'rail_gate_accept':proposed,'wrong_identity_label_only':wrong,
                        'nearest_identity_label_only':nearest['identity'],
                        'requested_error_mm_label_only':float(np.linalg.norm(
                            pos-target['position_base_m'])*1000)})
    args.output.mkdir(parents=True)
    with (args.output/'rows.jsonl').open('w') as file:
        for r in records:
            file.write(json.dumps(r,separators=(',',':'))+'\n')
    result={'schema':'sfp_rail_candidate_gate_readonly/v1',
            'fixed_rule':'three-view-consistent and occupied >=0.5 and '
                         'two camera slot indices within 0.5 of requested card',
            'source_rows':len(records),'baseline':summarize(records,'baseline_accept'),
            'rail_gate':summarize(records,'rail_gate_accept'),
            'geometry_runtime_input':False,'final_scenes_opened':False,
            'offline_teacher_route_only':True}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
