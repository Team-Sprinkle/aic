#!/usr/bin/env python3
"""One RGB triplet per episode; separate original, acquisition and nominal domains."""
import cv2
cv2.setNumThreads(2)
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from native_connector_pose_observation import NativeConnectorPoseObservation
from shared_port_tcp_geometry import matrix_from_pose9


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 reports={}
 for version in ['best','last']:
  model=NativeConnectorPoseObservation(a.run/(version+'.pt'),'cpu');source=Path(model.plan['source']);rows=json.loads((source/'manifest.json').read_text());pixels=np.load(model.plan['pixels'],mmap_mode='r');seen=set();ids=[]
  for i,r in enumerate(rows):
   if r['episode'] not in seen:ids.append(i);seen.add(r['episode'])
  predicted=[]
  with torch.inference_mode():
   for offset in range(0,len(ids),16):
    batch=torch.from_numpy(np.array(pixels[ids[offset:offset+16]],copy=True)).permute(0,1,4,2,3)
    t,q=model.model(batch);predicted.extend(zip(t.numpy(),q.numpy()))
  groups={}
  for i,(position,rotation) in zip(ids,predicted):
   row=rows[i];truth=matrix_from_pose9(row['tip_in_tcp_pose9']);role=row.get('source_role','');domain='nominal_home' if role.startswith('Nominal-centred') else 'nominal_near_acquisition' if role else 'original_expert'
   key=row['split']+'/'+domain;e=np.linalg.norm(position-truth[:3,3])*1000;angle=np.degrees((Rotation.from_matrix(rotation).inv()*Rotation.from_matrix(truth[:3,:3])).magnitude());groups.setdefault(key,[]).append((float(e),float(angle)))
  reports[version]={k:dict(episodes=len(v),translation_mm_p50_p95=np.percentile(np.array(v)[:,0],[50,95]).tolist(),orientation_deg_p50_p95=np.percentile(np.array(v)[:,1],[50,95]).tolist()) for k,v in groups.items()}
 a.output.write_text(json.dumps(dict(run=str(a.run),reports=reports,scope='One original-crop triplet per complete episode, CPU inference; labels only used after prediction',ground_truth_model_input=False),indent=2)+'\n');print(json.dumps(reports),flush=True)


if __name__=='__main__':main()
