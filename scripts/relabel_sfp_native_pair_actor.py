#!/usr/bin/env python3
"""Reexpress recorded SFP teacher targets in a paired RGB port frame.

The old dataset already stores targets in its own RGB-estimated port frame.
Recovering base-frame targets from that representation uses measured TCP and
no true scene geometry. New port frames come from native RGB only.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from audit_sc_port_targets import pose_matrix
from build_predicted_port_local_dataset import pose9
from shared_port_tcp_geometry import matrix_from_pose9
from sfp_port_cascade import SfpPortCascade


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--coarse',type=Path,required=True)
    parser.add_argument('--broad',type=Path,required=True)
    parser.add_argument('--fine',type=Path,required=True)
    parser.add_argument('--orientation',type=Path,required=True)
    parser.add_argument('--orientation-secondary',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--max-rows-per-episode',type=int,default=0)
    args=parser.parse_args()
    if args.output.exists():parser.error('Preserve existing output')
    import torch
    torch.set_num_threads(4);cv2.setNumThreads(1)
    cascade=SfpPortCascade(args.coarse,args.broad,args.fine,args.orientation,
        args.calibration,orientation_path_secondary=args.orientation_secondary,
        pair_orientation=True)
    args.output.mkdir(parents=True)
    counts=Counter();eps=defaultdict(set)
    with (args.source.open() as stream,
          (args.output/'rows.jsonl').open('w') as out,
          (args.output/'matched_old_rows.jsonl').open('w') as matched):
        for line in stream:
            row=json.loads(line)
            if row['split']=='test':
                continue
            if row.get('image_kind')!='native_1152x1024_rgb_jpeg':
                raise ValueError('This ablation accepts native SFP rows only')
            episode=row['episode'];split=row['split']
            if args.max_rows_per_episode and counts[('seen',episode)]>=args.max_rows_per_episode:
                continue
            counts[('seen',episode)]+=1
            native={cam:cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
                    for cam,path in row['images'].items()}
            if any(v is None for v in native.values()):raise FileNotFoundError(row['images'])
            small={}
            for cam,image in native.items():
                reduced=cv2.resize(image,(288,256),interpolation=cv2.INTER_AREA)
                _,encoded=cv2.imencode('.jpg',reduced[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,95])
                small[cam]=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
            estimate,diagnostic=cascade.predict(native,small,row['state'],row['task'])
            counts[(split,'seen')]+=1
            if estimate is None:
                counts[(split,diagnostic['reason'])]+=1
                continue
            base_tcp=pose_matrix(row['state'][:7])
            old_port_tcp=matrix_from_pose9(row['predicted_port_tcp_pose9'])
            old_base_port=base_tcp@np.linalg.inv(old_port_tcp)
            base_targets=[old_base_port@matrix_from_pose9(p)
                          for p in row['teacher_tcp_targets_predicted_port_pose9']]
            rel=np.linalg.inv(estimate)@base_tcp
            new=dict(row)
            new['predicted_port_tcp_pose9']=pose9(rel)
            new['teacher_tcp_targets_predicted_port_pose9']=[
                pose9(np.linalg.inv(estimate)@p) for p in base_targets]
            new['orientation_source']='observed_rgb_port_pair_and_fixed_mount_roll'
            new['relabel_from_recorded_targets_without_true_geometry']=True
            new['pair_reprojection_px']=diagnostic['sibling_reprojection_px']
            new['pair_views']=diagnostic['sibling_views_used']
            out.write(json.dumps(new,separators=(',',':'))+'\n')
            matched.write(line)
            counts[(split,'accepted')]+=1;eps[split].add(episode)
    summary={'schema':'sfp_native_pair_matched_actor_relabel/v1',
             'counts':{str(k):v for k,v in counts.items() if k[0]!='seen' or len(k)==2 and k[1]=='seen'},
             'accepted_episodes':{s:len(v) for s,v in eps.items()},
             'source':str(args.source),'pair_rgb_only':True,
             'true_geometry_runtime_or_relabel_input':False,'reserved_final_opened':False}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
