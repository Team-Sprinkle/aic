#!/usr/bin/env python3
"""Posthoc RGB-crop versus oracle-center comparison for SC orientation.

Oracle centers are used solely to identify the source of an existing error.
They are never runtime or actor inputs.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from run_sc_native_crop_orientation_ablation import (
    CropEncoder, Residual, features_from_crops, oneport_examples,
)
from train_sc_oneport_orientation_adaptation import Model, oneport_data, stats
from train_two_family_port_locator import Locator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oneport-dev-glob',required=True)
    p.add_argument('--locator',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--refiner',type=Path,required=True)
    p.add_argument('--residual',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing diagnostic')
    torch.set_num_threads(4)
    paths=[Path(x) for x in sorted(glob.glob(a.oneport_dev_glob))]
    if len(paths)!=4:raise ValueError('Expected four development groups')
    locator=Locator(None).to(a.device).eval()
    locator.load_state_dict(torch.load(a.locator,map_location='cpu',weights_only=False)['model'])
    ex=oneport_examples(paths,locator,a.device)
    x,y,meta=oneport_data(paths,locator,json.loads(a.calibration.read_text()),a.device)
    if len(ex)!=len(meta):raise ValueError('Row-order mismatch')
    checkpoint=torch.load(a.baseline,map_location='cpu',weights_only=False)
    base=Model(checkpoint['model']['mean'],checkpoint['model']['std']).to(a.device).eval()
    base.load_state_dict(checkpoint['model'])
    with torch.inference_mode():
        rotations=np.concatenate([base(torch.from_numpy(x[i:i+256]).to(a.device)).cpu().numpy()
                                  for i in range(0,len(x),256)])
    encoder=CropEncoder(a.refiner).to(a.device).eval()
    saved=torch.load(a.residual,map_location='cpu',weights_only=False)
    model=Residual(3*768+6+1).to(a.device).eval()
    model.load_state_dict(saved['model'])
    images=[im for r in ex for im in r['images']]
    report={'schema':'sc_crop_orientation_oracle_center_diagnostic/v1','rows':len(ex),
            'oracle_geometry_in_deployable_model':False,'final_scenes_opened':False}
    before=np.rad2deg((Rotation.from_matrix(rotations).inv()*Rotation.from_matrix(y)).magnitude())
    report['baseline']=stats(before)
    report['by_episode']={}
    for kind in ('rgb','oracle'):
        centers=[pt for r in ex for pt in (r['centers'] if kind=='rgb' else r['true_xy'])]
        crops=features_from_crops(images,centers,encoder,a.device,160)
        coords=np.asarray(centers,np.float32).reshape(len(ex),6)
        port=np.asarray([r['port'] for r in ex],np.float32)[:,None]
        features=np.concatenate((crops,coords/np.tile([1151,1023],3),port),axis=1).astype(np.float32)
        with torch.inference_mode():
            correction=model(torch.from_numpy(features).to(a.device)).cpu().numpy()
        pred=Rotation.from_matrix(rotations)*Rotation.from_rotvec(correction)
        error=np.rad2deg((pred.inv()*Rotation.from_matrix(y)).magnitude())
        report[kind]=stats(error)
        for ep in sorted({r['episode'] for r in ex}):
            idx=[i for i,r in enumerate(ex) if r['episode']==ep]
            report['by_episode'].setdefault(ep,{})[kind]=stats(error[idx])
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
