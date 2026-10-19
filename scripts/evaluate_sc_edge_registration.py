#!/usr/bin/env python3
"""One held-out RGB triplet/episode; labels used only after pose refinement."""
import cv2
cv2.setNumThreads(2)
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from sc_grasp_edge_registration import SCGraspEdgeRegistration
from shared_port_tcp_geometry import matrix_from_pose9

p=argparse.ArgumentParser();p.add_argument('--rotation-only',action='store_true');p.add_argument('--fit',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
plan=json.loads((a.fit/'plan.json').read_text());refiner=SCGraspEdgeRegistration(plan['tcp_to_optical'],plan['crop_origins_native'],rotation_only=a.rotation_only);rows=json.loads((a.source/'manifest.json').read_text());patches=np.load(a.source/'native_crops.npy',mmap_mode='r');ids=np.load(a.fit/'heldout_indices.npy');pred=np.load(a.fit/'heldout_pose_predictions.npy');seen=set();results=[];updated=[]
for index,initial in zip(ids,pred):
 row=rows[int(index)]
 if row['episode'] in seen:continue
 seen.add(row['episode']);images=np.array(patches[index]);fixed,diagnostic=refiner.refine(images,initial)
 # Simulator label enters this evaluation branch only after refinement.
 truth=matrix_from_pose9(row['tip_in_tcp_pose9']);errors={}
 for label,pose in [('before',initial),('after',fixed)]:
  errors[label]=dict(translation_mm=float(np.linalg.norm(pose[:3,3]-truth[:3,3])*1000),orientation_deg=float(np.degrees((Rotation.from_matrix(pose[:3,:3]).inv()*Rotation.from_matrix(truth[:3,:3])).magnitude())))
 role=row.get('source_role','');domain='nominal_home' if role.startswith('Nominal-centred') else 'nominal_acquisition' if role else 'original'
 results.append(dict(episode=row['episode'],index=int(index),domain=domain,errors=errors,diagnostic=diagnostic));updated.append(fixed)
 if len(results)<=6:
  tiles=[]
  for ci,c in enumerate(refiner.cameras):
   im=cv2.cvtColor(images[ci],cv2.COLOR_RGB2BGR)
   for pose,color in [(initial,(0,0,255)),(fixed,(0,255,0))]:
    xy=refiner.project(refiner.vertices,pose,c,ci);hull=cv2.convexHull(xy.astype(np.float32));cv2.polylines(im,[np.rint(hull).astype(np.int32)],True,color,1)
   tiles.append(cv2.resize(im,(384,384)))
  cv2.imwrite(str(a.output/(row['episode']+'_predictions_only.jpg')),np.hstack(tiles))
 print(json.dumps(dict(episode=row['episode'],errors=errors)),flush=True)
summary={}
for domain in ['all','original','nominal_home','nominal_acquisition']:
 subset=[x for x in results if domain=='all' or x['domain']==domain]
 if subset:summary[domain]=dict(episodes=len(subset),**{stage:{key:np.percentile([x['errors'][stage][key] for x in subset],[50,95]).tolist() for key in ['translation_mm','orientation_deg']} for stage in ['before','after']})
(a.output/'summary.json').write_text(json.dumps(dict(results_by_domain=summary,results=results,pose_predictions_only=True,labels_only_after_refinement=True,automatic_promotion=False),indent=2)+'\n');np.save(a.output/'refined_predictions.npy',updated)
print(json.dumps(summary),flush=True)
