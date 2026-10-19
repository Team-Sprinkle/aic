#!/usr/bin/env python3
"""Make selected-slot native crop proposals from frozen RGB rail predictions."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

CAMS=('center','left','right')


def key(row):
    return row['scene_group'],row['episode'],round(float(row['time_s']),6)


def read(path):
    return [json.loads(line) for line in path.open()]


def save(path,rows):
    path.mkdir(parents=True,exist_ok=True)
    with (path/'rows.jsonl').open('w') as out:
        for row in rows:out.write(json.dumps(row,separators=(',',':'))+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rail-manifest',type=Path,required=True)
    p.add_argument('--predictions',type=Path,action='append',required=True)
    p.add_argument('--source-labels',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expand-train-requests',action='store_true')
    p.add_argument('--skip-actor-diagnostic',action='store_true')
    args=p.parse_args()
    if args.output.exists():p.error('Preserve existing output')
    rail={key(row):row for row in read(args.rail_manifest/'rows.jsonl')}
    sources={}
    for path in args.source_labels:
        for row in read(path/'rows.jsonl'):
            k=key(row)
            if k in sources and sources[k]['images']!=row['images']:
                raise ValueError(f'Duplicate mismatched source row {k}')
            sources[k]=row
    predictions={}
    for path in args.predictions:
        for row in read(path):
            k=key(row)
            if k in predictions:raise ValueError(f'Duplicate prediction {k}')
            predictions[k]=row
    labels=[];proposals=defaultdict(list);missing=[];counts=Counter()
    for k,pred in predictions.items():
        if k not in rail or k not in sources:
            missing.append(k);continue
        base=rail[k];source=sources[k]
        if args.skip_actor_diagnostic and base['autonomous_actor_diagnostic']:
            continue
        if source['images']!=base['images']:
            raise ValueError(f'Image provenance mismatch {k}')
        if source['scene_group']!=base['scene_group']:
            raise ValueError(f'Scene identity mismatch {k}')
        split=base['split']
        label=dict(source);label['split']=split
        labels.append(label)
        requests=([t['identity'] for t in source['targets']]
                  if split=='train' and args.expand_train_requests
                  else [source['selected_identity']])
        for requested in requests:
            card=int(requested.split('_')[2])
            if not base['occupied_slots_training_label_only'][card]:
                raise ValueError('Requested present target has empty scene rail')
            centers={}
            for ci,cam in enumerate(CAMS):
                ends=np.asarray(pred['endpoint_xy_normalized'][ci],float)*[1151,1023]
                centers[cam]=(ends[0]+card/4*(ends[1]-ends[0])).tolist()
            proposals[split].append({'episode':source['episode'],
                'scene_group':source['scene_group'],'time_s':source['time_s'],
                'requested':requested,'crop':{'center_px':centers,
                'source':'frozen_RGB_predicted_selected_rail',
                'source_card':card},
                'selected_slot_occupied_prob':pred['occupied_prob'][card],
                'geometry_runtime_input':False})
            counts[(split,'requests')]+=1
        counts[(split,'frames')]+=1
    if missing:raise ValueError(f'Missing manifest/source joins: {len(missing)}')
    if not labels:raise ValueError('No joined labels')
    if {r['scene_group'] for r in labels if r['split']=='train'} & {
            r['scene_group'] for r in labels if r['split']=='validation'}:
        raise ValueError('Train/development reset leak')
    args.output.mkdir(parents=True)
    save(args.output/'labels',labels)
    for split,rows in proposals.items():
        save(args.output/f'{split}_proposals',rows)
    summary={'schema':'sfp_slot_crop_proposals/v1',
             'counts':{f'{a}_{b}':v for (a,b),v in counts.items()},
             'groups':{split:len({r['scene_group'] for r in rows})
                       for split,rows in proposals.items()},
             'expand_train_requests':args.expand_train_requests,
             'proposal_from_frozen_rgb':True,
             'geometry_runtime_input':False,'final_scenes_opened':False,
             'source_predictions':[str(x) for x in args.predictions],
             'source_labels':[str(x) for x in args.source_labels]}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
