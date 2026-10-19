#!/usr/bin/env python3
"""Check RGB-predicted selected-rail crop coverage using labels only to score."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

CAMS=('center','left','right')


def read(path):
    return [json.loads(line) for line in path.open()]


def key(row):
    return row['scene_group'],row['episode'],round(float(row['time_s']),6)


def score_group(rows,field):
    groups=defaultdict(list)
    for row in rows:groups[row['scene_group']].append(row)
    result={'reset_groups':len(groups),'early_groups_two_views':0,
            'early_groups_three_views':0,'ever_groups_two_views':0,
            'early_wrong_card_port_in_crop_groups':0}
    for rr in groups.values():
        rr.sort(key=lambda r:r['time_s']);first=rr[0]['time_s']
        early=[r for r in rr if r['time_s']-first<=3]
        result['early_groups_two_views']+=any(r[field]['target_views']>=2 for r in early)
        result['early_groups_three_views']+=any(r[field]['target_views']==3 for r in early)
        result['ever_groups_two_views']+=any(r[field]['target_views']>=2 for r in rr)
        result['early_wrong_card_port_in_crop_groups']+=any(
            r[field]['other_card_port_views']>=2 and r[field]['target_views']>=2
            for r in early)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rail-predictions',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Preserve existing output')
    rails=read(args.rail_predictions)
    labels={key(r):r for r in read(args.labels)}
    records=[]
    for pred in rails:
        if key(pred) not in labels:
            raise ValueError(f'Missing same-frame label {key(pred)}')
        label=labels[key(pred)]
        requested=label['selected_identity']
        card=int(requested.split('_')[2])
        target=next(t for t in label['targets'] if t['identity']==requested)
        occupied=float(pred['occupied_prob'][card])
        result={'scene_group':pred['scene_group'],'episode':pred['episode'],
                'time_s':pred['time_s'],'requested':requested,
                'selected_slot_occupied_prob':occupied,
                'camera_details':{}}
        for ci,cam in enumerate(CAMS):
            ends=np.asarray(pred['endpoint_xy_normalized'][ci])*[1151,1023]
            center=ends[0]+card/4*(ends[1]-ends[0])
            truth=target['cameras'][cam]
            projected=bool(truth[3])
            delta=(np.asarray(truth[:2],float)-center) if projected else None
            details={'predicted_crop_center_px':center.tolist(),
                     'requested_port_projected_in_frame_label_only':projected,
                     'requested_port_offset_px_label_only':delta.tolist() if projected else None,
                     'other_card_ports_inside_480_label_only':0}
            if projected:
                for other in label['targets']:
                    if other['identity']==requested or other['identity'].split('_')[2]==str(card):
                        continue
                    point=other['cameras'][cam]
                    if point[3] and max(abs(np.asarray(point[:2])-center))<=240:
                        details['other_card_ports_inside_480_label_only']+=1
            result['camera_details'][cam]=details
        for side in (160,320,480):
            usable=[v for v in result['camera_details'].values()
                    if v['requested_port_projected_in_frame_label_only']
                    and max(abs(np.asarray(v['requested_port_offset_px_label_only'])))<=side/2]
            result[str(side)]={'target_views':len(usable),
                               'other_card_port_views':sum(v['other_card_ports_inside_480_label_only']>0
                                                           for v in usable),
                               'occupied_vote':occupied>=.5}
        records.append(result)
    report={'schema':'sfp_slot_conditioned_crop_feasibility/v1',
            'rows':len(records),'label_rows_available':len(labels),
            'reset_groups':len({r['scene_group'] for r in records}),
            'crop_sizes_px':[160,320,480],
            'geometry_runtime_input':False,'projection_is_not_visibility':True,
            'final_scenes_opened':False,
            'by_size':{str(side):score_group(records,str(side)) for side in (160,320,480)}}
    args.output.mkdir(parents=True)
    with (args.output/'rows.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r,separators=(',',':'))+'\n')
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
