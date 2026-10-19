#!/usr/bin/env python3
"""Replay recorded RGB/state through SFP pair orientation; truth is scoring only."""
import argparse
import bisect
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from build_two_family_port_projection_labels import sfp_ports
from sfp_port_cascade import SfpPortCascade


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--coarse',type=Path,required=True)
    p.add_argument('--broad',type=Path,required=True)
    p.add_argument('--fine',type=Path,required=True)
    p.add_argument('--orientation',type=Path,required=True)
    p.add_argument('--orientation-secondary',type=Path,required=True)
    p.add_argument('--camera-calibration',type=Path,required=True)
    p.add_argument('--scoring-calibration',type=Path,required=True)
    p.add_argument('--limit',type=int,default=20)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous result')
    cv2.setNumThreads(1);torch.set_num_threads(4)
    model=SfpPortCascade(a.coarse,a.broad,a.fine,a.orientation,a.camera_calibration,
                         orientation_path_secondary=a.orientation_secondary,
                         pair_orientation=True)
    config=yaml.safe_load((a.audit/'eval_config.yaml').read_text())
    trial_id=next(iter(config['trials']))
    task=next(iter(config['trials'][trial_id]['tasks'].values()))
    card=int(task['target_module_name'].rsplit('_',1)[-1])
    port=int(task['port_name'].rsplit('_',1)[-1])
    scoring=json.loads(a.scoring_calibration.read_text())
    truth=next(m for c,p,m in sfp_ports({'trial_yaml':str(a.audit/'eval_config.yaml'),
        'trial_id':trial_id},scoring) if c==card and p==port)
    frames={cam:[] for cam in ('center','left','right')}
    for row in csv.DictReader((a.audit/'frame_manifest.csv').open()):
        if row['camera'] in frames:
            frames[row['camera']].append((int(row['sim_ns']),a.audit/'frames'/row['file']))
    for entries in frames.values():entries.sort()
    traces=[json.loads(line) for line in (a.audit/'runtime_trace.jsonl').open()]
    proposals=[r for r in traces if r['event']=='proposal'][:a.limit]
    rows=[]
    for proposal in proposals:
        native={}
        for cam,entries in frames.items():
            stamps=[entry[0] for entry in entries]
            target=int(proposal['sim_time']*1e9)
            j=bisect.bisect_left(stamps,target)
            j=min(range(max(0,j-1),min(len(entries),j+1)),
                  key=lambda k:abs(stamps[k]-target))
            image=cv2.imread(str(entries[j][1]))
            if image is None:raise FileNotFoundError(entries[j][1])
            native[cam]=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        small={}
        for cam,image in native.items():
            resized=cv2.resize(image,(288,256),interpolation=cv2.INTER_AREA)
            _,encoded=cv2.imencode('.jpg',resized[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,95])
            small[cam]=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        task_vector=[1.,0.,float(port==0),float(port==1),
                     *[float(j==card) for j in range(5)],1.]
        start=time.perf_counter()
        predicted,diagnostic=model.predict(native,small,proposal['state'],task_vector)
        latency=(time.perf_counter()-start)*1000
        item={'sim_time_s':proposal['sim_time'],'accepted':predicted is not None,
              'latency_ms':latency,'diagnostic':diagnostic}
        if predicted is not None:
            item['orientation_error_deg']=float(np.rad2deg((
                Rotation.from_matrix(truth[:3,:3]).inv()*
                Rotation.from_matrix(predicted[:3,:3])).magnitude()))
            local=(np.linalg.inv(truth)@np.r_[predicted[:3,3],1.])[:3]
            item['lateral_error_mm']=float(np.linalg.norm(local[:2])*1000)
        rows.append(item)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'schema':'aic_sfp_pair_live_frame_replay/v1',
        'audit':str(a.audit),'rows':rows,'geometry_posthoc_scoring_only':True,
        'runtime_true_geometry':False,'reserved_final_opened':False},indent=2)+'\n')
    print(json.dumps({'frames':len(rows),'accepted':sum(x['accepted'] for x in rows),
        'orientation_error_deg':[x.get('orientation_error_deg') for x in rows]}))


if __name__=='__main__':main()
