#!/usr/bin/env python3
"""Fixed native grasp crops: only training labels select the shared centres."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

CAMS = ('center','left','right')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((a.labels/'summary.json').read_text())
    rows=[r for e in source['episodes'] for r in map(json.loads,(a.labels/e['rows_file']).open())]
    train=[r for r in rows if r['split']=='train'];assert train
    groups={s:{r['scene'] for r in rows if r['split']==s} for s in ('train','validation')}
    if groups['train']&groups['validation']:raise ValueError('Physical reset split leakage')
    centres={c:np.median([r['projected_virtual_reference_pixels'][c][0] for r in train],axis=0).tolist() for c in CAMS}
    size=208;patches=np.lib.format.open_memmap(a.output/'native_crops.npy',mode='w+',dtype=np.uint8,shape=(len(rows),3,size,size,3))
    targets=[];poses=[];qa=[]
    for i,row in enumerate(rows):
        labels=[];panels=[]
        for ci,c in enumerate(CAMS):
            im=cv2.imread(row['images'][c]);assert im is not None and im.shape[:2]==(1024,1152)
            offset=np.rint(np.asarray(centres[c])-size/2).astype(int)
            patch=cv2.warpAffine(im,np.array([[1,0,-offset[0]],[0,1,-offset[1]]],np.float32),(size,size))
            patches[i,ci]=cv2.cvtColor(patch,cv2.COLOR_BGR2RGB)
            points=np.asarray(row['projected_virtual_reference_pixels'][c])-offset
            if points.min()<8 or points.max()>199:raise ValueError('Reference outside jitter-safe fixed crop; audit before training')
            labels.append(points)
            if len(qa)<4 and (i==0 or row['episode']!=rows[i-1]['episode']) and (row['split']=='validation' or not qa):
                for k,pt in enumerate(points):
                    xy=tuple(np.rint(pt).astype(int));cv2.circle(patch,xy,3,(0,255,0),1)
                    cv2.putText(patch,str(k),(xy[0]+3,xy[1]-3),0,.35,(0,255,0),1)
                cv2.putText(patch,c+' GT QA only',(3,16),0,.4,(0,255,0),1);panels.append(patch)
        if panels:
            path=a.output/f'QA_{len(qa)}_{row["split"]}.jpg';cv2.imwrite(str(path),np.concatenate(panels,axis=1));qa.append(str(path))
        targets.append(labels);poses.append(row['body_tip_matrix'])
    patches.flush();np.save(a.output/'keypoint_pixels.npy',np.asarray(targets,np.float32));np.save(a.output/'body_tip_matrices.npy',np.asarray(poses))
    (a.output/'manifest.json').write_text(json.dumps(rows)+'\n')
    summary=dict(samples=len(rows),source_episodes={s:len({r['episode'] for r in rows if r['split']==s}) for s in groups},
                 reset_groups={s:len(v) for s,v in groups.items()},crop_centres_train_only=centres,
                 crop_size_original_pixels=size,training_roi=192,training_jitter_pixels=8,
                 tcp_to_optical=source['tcp_to_optical'],reference_points=source['virtual_reference_points_m'],
                 source_labels=str(a.labels),geometry_input=False,GT_evaluation_crop_selection=False,
                 QA_images=qa,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
