#!/usr/bin/env python3
"""Read-only target/neighbor coverage of RGB-selected SFP card crops."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


CAMS=('center','left','right')


def read(path):
    return [json.loads(line) for line in path.open()]


def key(row):
    return row['scene_group'],row['episode'],round(float(row['time_s']),6)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--proposals',type=Path,required=True)
    parser.add_argument('--size',type=int,default=320)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Preserve prior audit')
    labels={key(row):row for row in read(args.labels/'rows.jsonl')}
    groups=defaultdict(list)
    for proposal in read(args.proposals/'rows.jsonl'):
        k=key(proposal)
        if k not in labels:raise ValueError(f'Missing label for {k}')
        label=labels[k];requested=proposal['requested'];selected=next(
            t for t in label['targets'] if t['identity']==requested)
        port=requested.rsplit('_',1)[-1]
        rivals=[t for t in label['targets'] if t['identity']!=requested and
                t['identity'].rsplit('_',1)[-1]==port]
        checks={}
        for camera in CAMS:
            cx,cy=proposal['crop']['center_px'][camera]
            def inside(t):
                u,v,_,in_frame=t['cameras'][camera]
                return bool(in_frame and abs(u-cx)<=args.size/2 and abs(v-cy)<=args.size/2)
            chosen=inside(selected)
            wrong=[t['identity'] for t in rivals if inside(t)]
            checks[camera]={'selected_inside':chosen,'same_port_rivals_inside':wrong,
                            'selected_only':chosen and not wrong}
        group=(proposal['scene_group'],proposal['episode'])
        groups[group].append({'time_s':float(proposal['time_s']),'requested':requested,
                              'checks':checks,'selected_views':sum(x['selected_inside'] for x in checks.values()),
                              'clean_views':sum(x['selected_only'] for x in checks.values())})
    episodes=[]
    for (scene,ep),rows in sorted(groups.items()):
        rows.sort(key=lambda r:r['time_s']);start=rows[0]['time_s']
        early=[r for r in rows if r['time_s']-start<=3.0]
        episodes.append({'episode':ep,'scene_group':scene,'requested':rows[0]['requested'],
                         'rows':len(rows),'early_rows':len(early),
                         'early_selected_two_views':any(r['selected_views']>=2 for r in early),
                         'early_clean_two_views':any(r['clean_views']>=2 for r in early),
                         'all_selected_two_view_rows':sum(r['selected_views']>=2 for r in rows),
                         'all_clean_two_view_rows':sum(r['clean_views']>=2 for r in rows),
                         'first_early_clean_s':next((r['time_s']-start for r in early if r['clean_views']>=2),None)})
    summary={'schema':'sfp_tight_crop_feasibility/v1','crop_size_px':args.size,
             'proposal_source':'RGB_predicted_rail_only','geometry':'scoring_only',
             'episodes':len(episodes),'early_selected_two_view_episodes':sum(e['early_selected_two_views'] for e in episodes),
             'early_clean_two_view_episodes':sum(e['early_clean_two_views'] for e in episodes),
             'total_selected_two_view_rows':sum(e['all_selected_two_view_rows'] for e in episodes),
             'total_clean_two_view_rows':sum(e['all_clean_two_view_rows'] for e in episodes),
             'per_episode':episodes,'final_scenes_opened':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='per_episode'}))


if __name__=='__main__':main()
