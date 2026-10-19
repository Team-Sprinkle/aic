#!/usr/bin/env python3
"""Add grouped full-success teacher tails for head-only negative supervision."""
import argparse
from collections import Counter, defaultdict
import io
import json
from pathlib import Path

from materialize_sfp_scored_transport import materialize


def request(row):
    task=row['task_vector']
    return (int(max(range(5),key=lambda i:task[4+i])),
            int(max(range(2),key=lambda i:task[2+i])))


def choose(rows, split, limit):
    if limit%10:
        raise ValueError('Select equal complete episode counts for all ten requests')
    by_request=defaultdict(list)
    for r in rows.values():
        if (r['split']==split and r.get('source_scored_stream') and
                Path(r['source_scored_stream']).is_file()):
            by_request[request(r)].append(r)
    selected=[]
    for card in range(5):
        for port in range(2):
            items=by_request[(card,port)]
            # Fresh diverse episodes first, but include an old episode if it
            # is the only source of a requested identity.
            items.sort(key=lambda r:(int(r['episode_index'])<2400000,
                                     int(r['episode_index'])))
            if len(items)<limit//10:
                raise ValueError(f'Insufficient grouped tail sources: {(card,port)}')
            selected.extend(items[:limit//10])
    return selected


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-episodes',type=int,default=30)
    p.add_argument('--validation-episodes',type=int,default=10)
    p.add_argument('--tail-seconds',type=float,default=8.)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing output')
    a.output.mkdir(parents=True)
    first={}
    with a.labels.open() as source:
        for line in source:
            r=json.loads(line)
            first.setdefault(int(r['episode_index']),r)
    selected=choose(first,'train',a.train_episodes)+choose(first,'validation',a.validation_episodes)
    if len(selected)!=a.train_episodes+a.validation_episodes:
        raise ValueError('Insufficient grouped episodes')
    calibration=json.loads(a.calibration.read_text())
    reports=[];accepted=[];failures=[];new=[]
    for n,row in enumerate(selected):
        audit=Path(row['source_scored_stream']).parent
        buf=io.StringIO()
        episode=3100000+n
        try:
            report=materialize(audit,calibration,episode,buf,row['split'],
                               pause_tail_seconds=a.tail_seconds)
            rows=[json.loads(line) for line in buf.getvalue().splitlines()]
            if not rows or {r['scene_sha256'] for r in rows}!={row['scene_sha256']}:
                raise ValueError('Scene identity mismatch')
            after=[r for r in rows if r.get('past_transport_cutoff')]
            post_neg=[r for r in after if not (-.130<=r['observed_tcp_pose_port_frame'][2]<=-.060 and
                                                sum(x*x for x in r['observed_tcp_pose_port_frame'][:2])<=.03**2)]
            if len(post_neg)<4:
                raise ValueError(f'Only {len(post_neg)} post-window negatives')
            report.update(source_episode=row['episode_index'],new_episode=episode,
                          split=row['split'],post_cutoff_rows=len(after),
                          post_cutoff_negative_rows=len(post_neg))
            reports.append(report);accepted.extend(rows)
        except (ValueError,KeyError,FileNotFoundError) as exc:
            failures.append(dict(source_episode=row['episode_index'],audit=str(audit),
                                 error=type(exc).__name__+': '+str(exc)))
    if len([x for x in reports if x['split']=='train'])<24 or \
       len([x for x in reports if x['split']=='validation'])<8:
        raise ValueError(f'Insufficient accepted grouped tails: {len(reports)}; {failures}')
    with (a.output/'sfp_labels.jsonl').open('w') as out:
        with a.labels.open() as source:
            for line in source:out.write(line)
        for r in accepted:out.write(json.dumps(r,separators=(',',':'))+'\n')
    summary=dict(schema='aic_sfp_pause_tail_labels/v1',source_labels=str(a.labels),
                 new_rows=len(accepted),source_episodes=len(reports),
                 counts=dict(Counter(r['split'] for r in reports)),
                 post_cutoff_negative_rows=sum(r['post_cutoff_negative_rows'] for r in reports),
                 reports=reports,failures=failures,head_only=True,
                 actor_action_supervision_from_tails=False,
                 final_scenes_opened=False)
    (a.output/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('reports','failures')}),flush=True)


if __name__=='__main__':main()
