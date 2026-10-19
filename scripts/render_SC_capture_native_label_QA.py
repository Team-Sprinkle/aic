#!/usr/bin/env python3
"""Native training-label QA montage; deliberately not prediction evidence."""
import cv2
cv2.setNumThreads(2)
import argparse,json
from pathlib import Path
import numpy as np
from shared_port_tcp_geometry import matrix_from_pose9
from evaluate_sc_native_triangulation import FOCAL
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--captures',type=Path,required=True);p.add_argument('--count',type=int,required=True);a=p.parse_args()
root=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1');plan=json.loads((root/'far_near_spatial_joint/plan.json').read_text());cad=np.asarray(plan['CAD_reference_points']);progress=json.loads((a.captures/'progress.json').read_text());selected=progress['results'][:a.count];assert len(selected)==a.count and all(x['valid'] for x in selected);montage=[];report=[]
for n,result in enumerate(selected):
 labels=json.loads((Path(result['audit'])/'grasp_labels.json').read_text());row=labels[len(labels)//2];pose=matrix_from_pose9(row['tip_in_tcp_pose9']);cloud=cad@pose[:3,:3].T+pose[:3,3];panels=[];outside=0
 for ci,camera in enumerate(['center','left','right']):
  image=cv2.imread(row['images'][camera]);assert image is not None;left,top=plan['crop_origins_native'][ci];patch=cv2.warpAffine(image,np.array([[1,0,-left],[0,1,-top]],np.float32),(192,192));inv=np.linalg.inv(plan['tcp_to_optical'][camera]);q=cloud@inv[:3,:3].T+inv[:3,3];points=q[:,:2]/q[:,2:]*FOCAL+[576-left,512-top];outside+=int(((points<0)|(points>191)).any(-1).sum())
  for point in points:cv2.circle(patch,tuple(np.rint(point).astype(int)),2,(0,255,0),1)
  patch=cv2.copyMakeBorder(patch,22,0,0,0,cv2.BORDER_CONSTANT);cv2.putText(patch,f'{n} {camera} GT LABEL QA',(2,15),cv2.FONT_HERSHEY_SIMPLEX,.32,(0,255,0),1);panels.append(patch)
 montage.append(np.hstack(panels));report.append(dict(case=result['case']['case'],samples=len(labels),max_image_label_age_ms=max(x['image_to_label_age_ms'] for x in labels),projected_references_outside_fixed_crop=outside,geometry_supervision_only=True))
cv2.imwrite(str(a.captures/'native_label_QA.jpg'),np.vstack(montage));(a.captures/'label_QA_metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
