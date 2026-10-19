#!/usr/bin/env python3
"""Append image-time perception-only captures; preserve episode/reset splits."""
import cv2
cv2.setNumThreads(2)
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from shared_port_tcp_geometry import matrix_from_pose9


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--captures',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 capture=json.loads((a.captures/'summary.json').read_text());capture_plan=json.loads((a.captures/'plan.json').read_text());assert capture['completed']==capture_plan['planned'] and len(capture['results'])==capture_plan['planned']
 assert json.loads((a.captures/'first_six_visual_QA.json').read_text())['passed']
 rows=json.loads((a.source/'manifest.json').read_text());poses=list(np.load(a.source/'labels.npy'));pixels=list(np.load(a.source/'native_crops.npy',mmap_mode='r'))
 checkpoint=Path('/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/sc_grasp_joint_geometry_v6_continuation/joint_geometry/best.pt')
 centres=torch.load(checkpoint,map_location='cpu',weights_only=False)['plan']['crop_centres_train_only'];origins={c:np.rint(np.asarray(centres[c])-96) for c in ['center','left','right']};before=len(rows);added=[]
 for result in capture['results']:
  assert result['valid'];new=json.loads((Path(result['audit'])/'grasp_labels.json').read_text());assert len(new)>=result['case'].get('minimum_pose_samples',6)
  assert all(r['episode']==result['case']['case'] and r['split']==result['case']['split'] for r in new)
  for r in new:
   one=[]
   for c in ['center','left','right']:
    im=cv2.imread(r['images'][c]);assert im is not None
    rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);x,y=origins[c];one.append(cv2.warpAffine(rgb,np.array([[1,0,-x],[0,1,-y]],np.float32),(192,192)))
   rows.append(r);poses.append(r['tip_in_tcp_pose9']);pixels.append(np.stack(one))
  added.append(dict(case=result['case'],samples=len(new)))
 for key in ['episode','scene']:
  assert not ({r[key] for r in rows if r['split']=='train'}&{r[key] for r in rows if r['split']=='validation'}),key+' leakage'
 a.output.mkdir(parents=True);(a.output/'manifest.json').write_text(json.dumps(rows)+'\n');np.save(a.output/'labels.npy',np.asarray(poses));np.save(a.output/'native_crops.npy',np.stack(pixels))
 # Fixed crop montage: no privileged projection selects these pixels.
 montage=np.vstack([np.hstack(pixels[i])[:,:,::-1] for i in np.linspace(before,len(rows)-1,6,dtype=int)]);cv2.imwrite(str(a.output/'native_crop_QA.jpg'),montage)
 rotation=Rotation.from_matrix(np.stack([matrix_from_pose9(r['tip_in_tcp_pose9'])[:3,:3] for r in rows if r['split']=='train']));angles=(rotation.mean().inv()*rotation).magnitude()*180/np.pi
 summary=dict(source=str(a.source),captures=str(a.captures),source_manifest_sha256=hashlib.sha256((a.source/'manifest.json').read_bytes()).hexdigest(),samples=len(rows),episodes=len({r['episode'] for r in rows}),splits={s:dict(samples=sum(r['split']==s for r in rows),episodes=len({r['episode'] for r in rows if r['split']==s})) for s in ['train','validation']},added=added,train_rotation_distance_to_mean_p50_p95_max_deg=np.percentile(angles,[50,95,100]).tolist(),sources_modified=False,perception_only=True,BC_expert_admission=False,final_scenes_opened=False)
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
