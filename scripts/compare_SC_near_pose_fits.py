#!/usr/bin/env python3
"""Matched near-domain predictions for far-only/far+near; offline labels only."""
import cv2
cv2.setNumThreads(2)
from pathlib import Path
import json
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parent))
from sc_spatial_grasp_observation import SCSpatialGraspObservation
from shared_port_tcp_geometry import matrix_from_pose9
from train_sfp_native_grasp_keypoints import metrics,poses_from_pixels
from scipy.spatial.transform import Rotation
root=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1')
output=root/'matched_near_pose_comparison';output.mkdir(exist_ok=False)
source=root/'far_and_near_pose_dataset';rows=json.loads((source/'manifest.json').read_text())
pixels=np.load(source/'native_crops.npy',mmap_mode='r')
truth=np.stack([matrix_from_pose9(x) for x in np.load(source/'labels.npy')])
ids=np.array([i for i,r in enumerate(rows) if r.get('source_role','').startswith('Near25mm')])
torch.set_num_threads(4)
reports={}
for name in ['acquisition_spatial_joint','far_near_spatial_joint']:
    model=SCSpatialGraspObservation(root/name/'best.pt','cpu');assert model.native_sizes is None
    origins=np.stack([np.rint(np.asarray(model.centres[c])-96) for c in model.cameras]);pred=[]
    with torch.inference_mode():
        for start in range(0,len(ids),24):
            chosen=ids[start:start+24];x=torch.from_numpy(np.array(pixels[chosen])).permute(0,1,4,2,3).reshape(-1,3,192,192).float()/255
            pred.append(model.model(x,camera_id=torch.eye(3).repeat(len(chosen),1)).numpy().reshape(-1,3,5,2)*191+origins[None,:,None,:])
    poses=poses_from_pixels(np.concatenate(pred),model.optics,model.cad)
    np.save(output/(name+'_near_predictions.npy'),poses)
    report={}
    for split in ['train','validation']:
        chosen=np.array([j for j,i in enumerate(ids) if rows[int(i)]['split']==split]);part=poses[chosen];target=truth[ids[chosen]]
        report[split]=dict(metrics(part,target),samples=len(chosen),episodes=len({rows[int(ids[j])]['episode'] for j in chosen}),per_episode=[])
        for episode in sorted({rows[int(ids[j])]['episode'] for j in chosen}):
            mask=np.array([rows[int(ids[j])]['episode']==episode for j in chosen]);report[split]['per_episode'].append(dict(episode=episode,**metrics(part[mask],target[mask])))
    reports[name]=report
train=np.array([i for i in ids if rows[int(i)]['split']=='train']);val=np.array([i for i in ids if rows[int(i)]['split']=='validation']);constant=np.broadcast_to(np.eye(4),(len(val),4,4)).copy();constant[:,:3,3]=truth[train,:3,3].mean(0);constant[:,:3,:3]=Rotation.from_matrix(truth[train,:3,:3]).mean().as_matrix()
result=dict(models=reports,near_train_mean_baseline=metrics(constant,truth[val]),matched_complete_episode_split=True,geometry_labels_only=True,final_scenes_opened=False,selection_changed=False)
(output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(output/'indices.json').write_text(json.dumps(ids.tolist())+'\n')
print(json.dumps({n:{s:d[s]['orientation_p50_p95_deg'] for s in d} for n,d in reports.items()}),flush=True)
