#!/usr/bin/env python3
"""Audit causal alternate-request SFP teacher pairs before any branch fit."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def request(row):
    task=np.asarray(row['task'])
    return int(task[4:9].argmax()),int(task[2:4].argmax())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(line) for line in (a.cache/'rows.jsonl').open()]
    feat=np.load(a.cache/'features.npy',mmap_mode='r')
    scenes=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(rows):
        if r['family']=='sfp' and r['split']=='train':
            scenes[r['scene']][request(r)].append(i)
    records=[]
    for scene,groups in scenes.items():
        if len(groups)<2:continue
        keys=sorted(groups)
        for ka in range(len(keys)):
            for kb in range(ka+1,len(keys)):
                left,right=groups[keys[ka]],groups[keys[kb]]
                for i in left:
                    r=rows[i]
                    if not .0<=r['time_s']<=1.5:continue
                    j=min(right,key=lambda idx:abs(rows[idx]['time_s']-r['time_s']))
                    other=rows[j]
                    dt=abs(other['time_s']-r['time_s'])
                    if dt>.11:continue
                    xyz=np.linalg.norm(np.asarray(r['state'][:3])-other['state'][:3])*1000
                    angle=(Rotation.from_quat(r['state'][3:7]).inv()*
                           Rotation.from_quat(other['state'][3:7])).magnitude()*180/np.pi
                    action=np.linalg.norm(np.asarray(r['target'][0][:3])-other['target'][0][:3])*1000
                    x=np.asarray(feat[i],float);y=np.asarray(feat[j],float)
                    cosine=float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))
                    causal=bool(xyz<=2 and angle<=2 and cosine>=.995 and action>=3)
                    records.append(dict(scene=scene,first_request=list(keys[ka]),second_request=list(keys[kb]),
                                        first_index=i,second_index=j,time_s=r['time_s'],
                                        relative_time_gap_s=dt,measured_tcp_difference_mm=float(xyz),
                                        measured_orientation_difference_deg=float(angle),
                                        frozen_image_feature_cosine=cosine,
                                        teacher_first_action_difference_mm=float(action),
                                        safe_counterfactual_label_candidate=causal))
    valid=[r for r in records if r['safe_counterfactual_label_candidate']]
    result=dict(schema='aic_sfp_paired_branch_alignment/v1',paired_scenes=sum(len(g)>1 for g in scenes.values()),
                aligned_rows=len(records),safe_candidate_rows=len(valid),
                safe_candidate_scenes=len(set(r['scene'] for r in valid)),
                safe_teacher_action_delta_median_mm=(float(np.median([r['teacher_first_action_difference_mm'] for r in valid])) if valid else None),
                same_scene_not_equal_observation_after_motion=True,
                thresholds=dict(time_s=.11,tcp_mm=2,orientation_deg=2,image_cosine=.995,action_difference_mm=3),
                records=records,final_scenes_opened=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
