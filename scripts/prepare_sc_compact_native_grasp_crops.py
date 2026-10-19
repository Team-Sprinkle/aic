#!/usr/bin/env python3
"""One train-derived native-resolution ROI ablation; no GT runtime crop choice."""
import cv2
cv2.setNumThreads(2)
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from shared_port_tcp_geometry import matrix_from_pose9
from train_sc_grasp_keypoint_probe import CAD
from evaluate_sc_native_triangulation import FOCAL


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=json.loads((a.source/'manifest.json').read_text())
    pose=np.stack([matrix_from_pose9(x) for x in np.load(a.source/'labels.npy')])
    train=np.array([x['split']=='train' for x in rows])
    plan=json.loads(a.plan.read_text());cloud=CAD@pose[:,:3,:3].transpose(0,2,1)+pose[:,None,:3,3]
    regions={};heldout_coverage={}
    for camera,optic in plan['tcp_to_optical'].items():
        inv=np.linalg.inv(optic);q=cloud@inv[:3,:3].T+inv[:3,3]
        projected=q[:,:,:2]/q[:,:,2:]*FOCAL+[576,512]
        lo=projected[train].min((0,1));hi=projected[train].max((0,1))
        # Twelve-pixel native padding, then round UP to a sixteen-pixel size.
        size=int(np.ceil(((hi-lo).max()+24)/16)*16)
        origin=np.rint((lo+hi)/2-(size-1)/2).astype(int)
        local=projected-origin
        regions[camera]=dict(origin_native_xy=origin.tolist(),native_size=size,
            train_bounds=[lo.tolist(),hi.tolist()],model_pixels=192,
            sampling='Endpoint-aligned affine native->192, before any global resize')
        heldout_coverage[camera]=dict(projected_references_outside=int(((local[~train]<0)|(local[~train]>size-1)).any(-1).sum()),
            used_to_select_or_enlarge_ROI=False)
    metadata=dict(schema='sc_compact_native_crop_ablation/v1',source=str(a.source),
        manifest_sha256=hashlib.sha256((a.source/'manifest.json').read_bytes()).hexdigest(),
        regions=regions,heldout_coverage=heldout_coverage,
        geometry_used_for_train_only_fixed_ROI=True,GT_runtime_crop_selection=False,
        split_changed=False,source_modified=False)
    (a.output/'crop_geometry.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if a.cache:
        cached=np.lib.format.open_memmap(a.output/'native_crops.npy',mode='w+',dtype=np.uint8,shape=(len(rows),3,192,192,3))
        for i,row in enumerate(rows):
            for ci,camera in enumerate(('center','left','right')):
                im=cv2.imread(row['images'][camera]);assert im is not None
                region=regions[camera];x,y=region['origin_native_xy'];scale=191/(region['native_size']-1)
                transform=np.array([[scale,0,-scale*x],[0,scale,-scale*y]],np.float32)
                cached[i,ci]=cv2.cvtColor(cv2.warpAffine(im,transform,(192,192)),cv2.COLOR_BGR2RGB)
        cached.flush()
        indices=np.linspace(0,len(rows)-1,6,dtype=int)
        cv2.imwrite(str(a.output/'native_crop_QA.jpg'),np.vstack([np.hstack(cached[i])[:,:,::-1] for i in indices]))
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()
