#!/usr/bin/env python3
"""Freeze anonymous current/history RGB panels for SFP card identity review."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
cv2.setNumThreads(1)
import numpy as np


CAMS=('center','left','right')
SOURCES={
 'fresh':Path('/home/nas/chmin/aic_archives/2026-10-06_sfp_joint_candidate_fresh10_labels_v1/rows.jsonl'),
 'older':Path('/home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/labels_all_ten/rows.jsonl'),
 'actor':Path('outputs/experiments/2026-10-05_spatial_transport_prerequisite_v1/actor_labels/rows.jsonl')}


def read(path):return [json.loads(s) for s in path.open()]


def distance_mm(row):
    target=next(t for t in row['targets'] if t['identity']==row['selected_identity'])
    return float(np.linalg.norm(np.asarray(row['state'][:3])-np.asarray(target['position_base_m']))*1000)


def choose(rows,distance):
    return min(rows,key=lambda r:(abs(distance_mm(r)-distance),r['time_s']))


def prior_by_motion(rows,current,minimum_mm=30):
    earlier=[r for r in rows if r['time_s']<current['time_s'] and
             np.linalg.norm(np.asarray(r['state'][:3])-np.asarray(current['state'][:3]))*1000>=minimum_mm]
    return max(earlier,key=lambda r:r['time_s']) if earlier else None


def panel(row,history,alias,out):
    strips=[]
    for item,title in [(row,'current'),(history,'earlier >=30 mm')]:
        ims=[]
        for cam in CAMS:
            if item is None:
                im=np.full((342,384,3),25,np.uint8)
            else:
                im=cv2.imread(item['images'][cam])
                if im is None:raise FileNotFoundError(item['images'][cam])
                im=cv2.resize(im,(384,342),interpolation=cv2.INTER_AREA)
            cv2.putText(im,f'{title} {cam}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),2)
            ims.append(im)
        strips.append(np.concatenate(ims,axis=1))
    cv2.imwrite(str(out/f'{alias}.jpg'),np.concatenate(strips,axis=0),[cv2.IMWRITE_JPEG_QUALITY,94])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve frozen review')
    groups={}
    for source,path in SOURCES.items():
        by=defaultdict(list)
        for row in read(path):by[row['episode']].append(row)
        for ep,rows in by.items():groups[source,ep]=sorted(rows,key=lambda r:r['time_s'])
    chosen=[]
    for card in range(5):
        for port in range(2):
            ep=f'card{card}_port{port}';rows=groups['fresh',ep]
            for phase,dist in [('route',250),('approach',120),('near',50)]:
                chosen.append(('fresh',ep,phase,choose(rows,dist)))
    for ep in sorted(e for source,e in groups if source=='actor'):
        rows=groups['actor',ep]
        for phase,row in [('first',rows[0]),('last',rows[-1])]:chosen.append(('actor',ep,phase,row))
    for ep,stamp in [('card1_port0',3.2),('card3_port1',10.75),('card1_port0',3.85),('card3_port0',3.85)]:
        source='fresh' if stamp in (3.2,10.75) else 'older';rows=groups[source,ep]
        row=min(rows,key=lambda r:abs(r['time_s']-stamp))
        chosen.append((source,ep,'known_neighbor_failure',row))
    if len(chosen)!=40:raise ValueError(f'Expected 40 review items, got {len(chosen)}')
    a.output.mkdir(parents=True);(a.output/'panels').mkdir()
    rng=np.random.default_rng(20261006);order=rng.permutation(len(chosen))
    blind=[];sealed=[]
    for i,j in enumerate(order):
        source,ep,phase,row=chosen[int(j)];alias=f'R{i:02d}'
        history=prior_by_motion(groups[source,ep],row)
        panel(row,history,alias,a.output/'panels')
        blind.append({'alias':alias,'panel':str((a.output/'panels'/f'{alias}.jpg').resolve()),
                      'requested':row['selected_identity'],'history_available':history is not None,
                      'single_view_card':'unreviewed','history_card':'unreviewed',
                      'port_pair':'unreviewed','actual_visibility':'unreviewed'})
        sealed.append({'alias':alias,'source':source,'episode':ep,'phase':phase,
                       'time_s':row['time_s'],'scene_group':row['scene_group'],
                       'distance_mm_label_only':distance_mm(row),
                       'images':row['images'],'history_time_s':history['time_s'] if history else None,
                       'targets_label_only':row['targets']})
    (a.output/'blind_manifest.json').write_text(json.dumps(blind,indent=2)+'\n')
    (a.output/'sealed_mapping.json').write_text(json.dumps(sealed,indent=2)+'\n')
    summary={'schema':'sfp_requested_card_review/v1','items':len(chosen),
             'fresh_route_approach_near':30,'saved_autonomous':6,'known_neighbor_failures':4,
             'source_geometry_for_sampling_only':True,'panels_have_no_target_projection':True,
             'history_minimum_tcp_displacement_mm':30,'final_scenes_opened':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
