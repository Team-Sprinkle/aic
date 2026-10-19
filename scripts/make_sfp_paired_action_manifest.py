#!/usr/bin/env python3
"""Select causal alternate-request actor targets at matched visual states."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def request(row):
    task=row['task']
    return (int(np.argmax(task[4:9])),int(np.argmax(task[2:4])))


def pairs_for(split, rows, features, threshold):
    by_scene=defaultdict(lambda:defaultdict(list))
    for i,row in enumerate(rows):
        if row['family']=='sfp' and row['split']==split:
            by_scene[row['scene']][request(row)].append(i)
    candidates=[];paired_scenes=0
    for scene,by_request in by_scene.items():
        if len(by_request)<2:continue
        paired_scenes+=1
        keys=sorted(by_request)
        for a in range(len(keys)):
            for b in range(a+1,len(keys)):
                left=by_request[keys[a]];right=by_request[keys[b]]
                right_times=np.asarray([rows[j]['time_s'] for j in right])
                for i in left:
                    r=rows[i]
                    k=int(np.argmin(abs(right_times-r['time_s'])))
                    j=right[k];other=rows[j]
                    dt=abs(other['time_s']-r['time_s'])
                    if dt>.11:continue
                    xyz=float(np.linalg.norm(np.asarray(r['state'][:3])-other['state'][:3])*1000)
                    angle=float((Rotation.from_quat(r['state'][3:7]).inv()*
                                 Rotation.from_quat(other['state'][3:7])).magnitude()*180/np.pi)
                    feat_a=np.asarray(features[i],np.float32)
                    feat_b=np.asarray(features[j],np.float32)
                    cosine=float(np.dot(feat_a,feat_b)/
                                 (np.linalg.norm(feat_a)*np.linalg.norm(feat_b)))
                    delta=float(np.linalg.norm(np.asarray(r['target'][0][:3])-
                                               other['target'][0][:3])*1000)
                    if xyz>2 or angle>2 or cosine<.995 or delta<threshold:continue
                    candidates.append(dict(scene=scene,split=split,left_index=i,right_index=j,
                                           first_request=list(keys[a]),second_request=list(keys[b]),
                                           first_time_s=r['time_s'],second_time_s=other['time_s'],
                                           dt_s=dt,tcp_difference_mm=xyz,
                                           orientation_difference_deg=angle,
                                           image_feature_cosine=cosine,
                                           teacher_first_action_difference_mm=delta))
    by_scene=defaultdict(list)
    for row in candidates:by_scene[row['scene']].append(row)
    kept=[]
    for scene,items in by_scene.items():
        kept.extend(sorted(items,key=lambda r:r['first_time_s'])[:4])
    return kept,paired_scenes,len(candidates)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-train-pairs',type=int,default=30)
    a=p.parse_args()
    rows=[json.loads(line) for line in (a.cache/'rows.jsonl').open()]
    features=np.load(a.cache/'features.npy',mmap_mode='r')
    if len(rows)!=len(features):raise ValueError('Cache feature/row mismatch')
    train,train_scenes,train_candidates=pairs_for('train',rows,features,5.)
    val,val_scenes,val_candidates=pairs_for('validation',rows,features,3.)
    result=dict(schema='aic_sfp_paired_action_manifest/v1',cache=str(a.cache),
                thresholds=dict(max_time_s=.11,max_tcp_mm=2,max_orientation_deg=2,
                                min_feature_cosine=.995,min_train_action_difference_mm=5,
                                min_validation_action_difference_mm=3,max_per_scene=4),
                train_paired_scenes=train_scenes,validation_paired_scenes=val_scenes,
                train_candidates=train_candidates,validation_candidates=val_candidates,
                train_pairs=train,validation_pairs=val,
                sufficient_for_fit=len(train)>=a.min_train_pairs,
                final_scenes_opened=False,geometry_runtime_input=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('train_pairs','validation_pairs')}),flush=True)


if __name__=='__main__':main()
