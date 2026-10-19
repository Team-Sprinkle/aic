#!/usr/bin/env python3
"""Immutable original SFP corpus plus 32 QA-approved wider-grasp resets."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
cv2.setNumThreads(2)
import numpy as np
from scipy.spatial.transform import Rotation
from shared_port_tcp_geometry import matrix_from_pose9

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
a=p.parse_args()
source=Path('/home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1/SFP_native_grasp_crops_v2')
out=a.root/'SFP_wide_orientation_dataset'
out.mkdir(exist_ok=False)
rows=json.loads((source/'manifest.json').read_text())
old_pixels=np.load(source/'native_crops.npy',mmap_mode='r');poses=list(np.load(source/'body_tip_matrices.npy'))
assert len(rows)==len(old_pixels)==len(poses)
old_count=len(rows)
meta=json.loads((source/'summary.json').read_text())
centres=meta['crop_centres_train_only']
origins={c:np.rint(np.asarray(centres[c])-104).astype(int) for c in ('center','left','right')}
new=[]
for worker in range(2):
    folder=a.root/('SFP_orientation_capture_worker'+str(worker))
    assert json.loads((folder/'first_six_visual_QA.json').read_text())['passed']
    complete=json.loads((folder/'summary.json').read_text())
    assert complete['completed']==16
    for episode in complete['results']:
        assert episode['valid']
        values=json.loads((Path(episode['audit'])/'grasp_labels.json').read_text())
        assert len(values)>=6
        for row in values:
            assert row['family']=='sfp' and row['geometry_label_only'] and not row['GT_crop_selection']
            assert row['episode']==episode['case']['case'] and row['split']==episode['case']['split']
            new.append(row)
for row in new:
    rows.append(row);poses.append(matrix_from_pose9(row['tip_in_tcp_pose9']))
for key in ('episode','scene'):
    assert not ({r[key] for r in rows if r['split']=='train'}&{r[key] for r in rows if r['split']=='validation'}),key+' leakage'
pixels=np.lib.format.open_memmap(out/'native_crops.npy',mode='w+',dtype=np.uint8,shape=(len(rows),3,208,208,3))
for i in range(0,old_count,32):pixels[i:min(old_count,i+32)]=old_pixels[i:min(old_count,i+32)]
qa=[]
for i,row in enumerate(new,old_count):
    for ci,c in enumerate(('center','left','right')):
        image=cv2.imread(row['images'][c]);assert image is not None and image.shape[:2]==(1024,1152)
        x,y=origins[c]
        crop=cv2.warpAffine(image,np.array([[1,0,-x],[0,1,-y]],np.float32),(208,208))
        pixels[i,ci]=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
    if i==old_count or row['episode']!=rows[i-1]['episode']:
        if len(qa)<4:
            image=np.hstack(pixels[i])[:,:,::-1].copy();path=out/('crop_QA_'+str(len(qa))+'.jpg');cv2.imwrite(str(path),image);qa.append(str(path))
pixels.flush();np.save(out/'body_tip_matrices.npy',np.stack(poses))
(out/'manifest.json').write_text(json.dumps(rows)+'\n')
angles=Rotation.from_matrix(np.stack(poses)[:,:3,:3]).as_euler('xyz',degrees=True)
summary=dict(meta,source=str(source),source_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),samples=len(rows),original_samples=old_count,new_samples=len(new),splits={s:dict(episodes=len({r['episode'] for r in rows if r['split']==s}),samples=sum(r['split']==s for r in rows)) for s in ('train','validation')},original_crop_origins_preserved=True,original_data_modified=False,geometry_training_only=True,near_evaluation_images_used_for_training=False,crop_QA=qa,angles_min_deg=angles.min(0).tolist(),angles_max_deg=angles.max(0).tolist())
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
