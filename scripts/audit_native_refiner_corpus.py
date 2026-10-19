#!/usr/bin/env python3
"""Merge existing native-image labels conservatively; never relabel projections as visibility."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import pose_matrix
from build_sc_native_pose_labels import project


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repair-image-time-projections',action='store_true')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing audit')
    a.output.mkdir(parents=True)
    rows={};rejected=Counter();sources=[];shapes={};split_by_image=defaultdict(set);shifts=[]
    optics={c:np.asarray(v) for c,v in json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())['tcp_to_optical'].items()}
    @lru_cache(maxsize=4)
    def controller_stream(path):
        data=json.loads(Path(path).read_text());values=[]
        for stamp,message in data['streams']['/aic_controller/controller_state']:
            q=message['tcp_pose'];values.append((stamp,[q['position'][k] for k in 'xyz']+[q['orientation'][k] for k in 'xyzw']))
        return [r[0] for r in values],values
    for source in a.source:
        sources.append({'path':str(source.resolve()),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
        for source_index,line in enumerate(source.open()):
            if source_index%2000==0:print(json.dumps({'source':str(source),'rows_read':source_index,'accepted_unique_so_far':len(rows)}),flush=True)
            r=json.loads(line)
            if r.get('split') not in ('train','validation'):
                rejected['unsupported_split']+=1;continue
            paths=r.get('images',{})
            if any(not isinstance(paths.get(c),str) for c in ('center','left','right')):
                rejected['cache_indices_not_original_images']+=1;continue
            actual={c:str(Path(paths[c]).absolute()) for c in ('center','left','right')}
            if any('frozen_pipeline_coverage' in x or '/final' in x.lower() for x in actual.values()):
                rejected['excluded_evaluation_source']+=1;continue
            if any(not Path(x).is_file() for x in actual.values()):
                rejected['missing_original_image']+=1;continue
            valid=True
            for x in actual.values():
                if x not in shapes:
                    with Image.open(x) as image:shapes[x]=image.size
                if shapes[x]!=(1152,1024):valid=False
            if not valid:rejected['not_original_resolution']+=1;continue
            if a.repair_image_time_projections and 'label_pose_join' not in r:
                stream=r.get('source_stream') or r.get('source_scored_stream')
                if not stream or not Path(stream).is_file():
                    rejected['missing_image_time_controller_stream']+=1;continue
                times,values=controller_stream(stream);camera_poses={};ages={};image_state=None
                for c,x in actual.items():
                    stamp=int(Path(x).stem.split('_')[-2]);j=bisect_right(times,stamp)-1
                    if j<0 or stamp-times[j]>10_000_000:break
                    camera_poses[c]=pose_matrix(values[j][1]);ages[c]=(stamp-times[j])/1e6
                    if c=='center':image_state=values[j][1]
                if len(camera_poses)!=3:
                    rejected['stale_image_time_controller_pose']+=1;continue
                for target in r['targets']:
                    m=pose_matrix([*target['position_base_m'],*target['orientation_base_xyzw']])
                    for c in actual:
                        new=project(m,camera_poses[c]@optics[c]);old=target['cameras'][c]
                        if old[3] and new[3]:shifts.append(float(np.linalg.norm(np.asarray(new[:2])-old[:2])))
                        target['cameras'][c]=new
                r['state']=image_state;r['image_pose_age_ms']=ages;r['label_pose_join']='latest_past_controller_at_each_image_stamp'
            if not any(t['identity']==r['selected_identity'] for t in r['targets']):
                raise ValueError('Selected identity absent from labels')
            for x in actual.values():split_by_image[x].add(r['split'])
            key=tuple(actual[c] for c in ('center','left','right'))+(r['selected_identity'],)
            if key in rows:
                old=rows[key]
                if old['split']!=r['split']:rejected['duplicate_split_disagreement']+=1
                for c in actual:
                    t=next(t for t in r['targets'] if t['identity']==r['selected_identity'])['cameras'][c]
                    ot=next(t for t in old['targets'] if t['identity']==old['selected_identity'])['cameras'][c]
                    if np.linalg.norm(np.asarray(t[:2])-ot[:2])>.1:
                        raise ValueError(f'Duplicate images have conflicting selected labels: {key}')
                rejected['duplicate_triplet_request']+=1;continue
            r['images']=actual;r['label_source_manifest']=str(source.resolve())
            r['physical_visibility_certified']=False;r['geometry_training_label_only']=True
            rows[key]=r
    bad_images={k for k,v in split_by_image.items() if len(v)>1}
    bad_episodes={(r['family'],r['episode']) for r in rows.values() if any(x in bad_images for x in r['images'].values())}
    scene_splits=defaultdict(set)
    for r in rows.values():scene_splits[r['scene_group']].add(r['split'])
    bad_scenes={k for k,v in scene_splits.items() if len(v)>1}
    accepted=[r for r in rows.values() if (r['family'],r['episode']) not in bad_episodes and r['scene_group'] not in bad_scenes]
    rejected['rows_quarantined_for_group_or_image_split_overlap']=len(rows)-len(accepted)
    accepted.sort(key=lambda r:(r['family'],r['episode'],r['time_s'],r['selected_identity']))
    with (a.output/'rows.jsonl').open('w') as f:
        for r in accepted:f.write(json.dumps(r,separators=(',',':'))+'\n')
    episodes=defaultdict(set);coverage=defaultdict(set);counts=Counter()
    for r in accepted:
        k=f"{r['family']}/{r['split']}";episodes[k].add(r['episode']);counts[k]+=1
        coverage[f"{k}/cards{r['card_count']}/{r['selected_identity']}"].add(r['scene_group'])
    report={'schema':'aic_native_refiner_corpus_audit/v1','sources':sources,'rows':len(accepted),
            'rows_by_family_split':dict(counts),'episodes_by_family_split':{k:len(v) for k,v in episodes.items()},
            'coverage_distinct_scenes':{k:len(v) for k,v in sorted(coverage.items())},'rejected':dict(rejected),
            'quarantined_episodes':sorted(bad_episodes),'quarantined_scene_groups':sorted(bad_scenes),
            'projection_repairs':{'enabled':a.repair_image_time_projections,'n':len(shifts),'shift_px_p50_p95_max':np.percentile(shifts,[50,95,100]).tolist() if shifts else None},
            'native_resolution':[1152,1024],'projection_is_not_visibility':True,'final_scenes_opened':False}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    samples=[]
    for family in ('sc','sfp'):
        for count in range(1,6):
            candidates=[r for r in accepted if r['family']==family and r['card_count']==count and r['split']=='train']
            if candidates:
                # QA only: choose a frame with visible projected labels; no policy uses this selection.
                r=min(candidates,key=lambda r:abs(r['time_s']-25))
                target=next(t for t in r['targets'] if t['identity']==r['selected_identity'])
                for c in ('center','left','right'):
                    image=cv2.imread(r['images'][c]);assert image is not None
                    for t in r['targets']:
                        x,y,_,inside=t['cameras'][c]
                        if inside:cv2.circle(image,(round(x),round(y)),8,(0,255,0) if t['identity']==r['selected_identity'] else (255,180,0),2)
                    image=cv2.resize(image,(384,342))
                    cv2.putText(image,f"{family} n{count} {c} {r['selected_identity']}",(5,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
                    samples.append(image)
    if samples:
        while len(samples)%3:samples.append(np.zeros_like(samples[0]))
        for i in range(0,len(samples),15):
            panels=samples[i:i+15]
            cv2.imwrite(str(a.output/f'label_review_{i//15}.jpg'),np.vstack([np.hstack(panels[j:j+3]) for j in range(0,len(panels),3)]))
    print(json.dumps({k:report[k] for k in ['rows','episodes_by_family_split','rejected']}),flush=True)


if __name__=='__main__':main()
