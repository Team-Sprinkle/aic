#!/usr/bin/env python3
"""Fit an SC mating-orientation prior from successful teacher targets only."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def stats(values):
    x=np.asarray(values,float)
    return {'n':len(x),'median_deg':float(np.median(x)),
            'p95_deg':float(np.quantile(x,.95)),'max_deg':float(x.max())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();quats={'train':[],'validation':[]};episodes={'train':set(),'validation':set()}
    for row in map(json.loads,a.source.open()):
        if row['official_tier3']!=75:continue
        if not -.10<row['observed_tcp_pose_opening_frame'][2]<-.015:continue
        split=row['split']
        if split not in quats:continue
        quats[split].append(row['teacher_tcp_target_poses_opening_frame'][0][3:7])
        episodes[split].add(row['episode_id'])
    if not quats['train'] or not quats['validation']:raise ValueError('Grouped train and validation required')
    prior=Rotation.from_quat(quats['train']).mean()
    result={'schema':'aic_sc_local_orientation_prior/v1',
            'source':str(a.source),'nominal_tcp_in_port_orientation_xyzw':prior.as_quat().tolist(),
            'near_definition_tcp_axial_m':[-.10,-.015],
            'episodes':{k:len(v) for k,v in episodes.items()},
            'relative_to_nominal':{k:stats((prior.inv()*Rotation.from_quat(v)).magnitude()*180/np.pi)
                                   for k,v in quats.items()},
            'fit_uses_train_only':True,'runtime_true_geometry':False,'reserved_final_opened':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
