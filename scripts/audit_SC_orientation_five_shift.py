#!/usr/bin/env python3
"""One fixed five-shift inference comparison; freeze any active-fit checkpoint."""
import cv2
cv2.setNumThreads(2)
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from native_connector_orientation_tta import NativeConnectorOrientationTTA,SHIFTS
from shared_port_tcp_geometry import matrix_from_pose9
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);shutil.copyfile(a.fit/'best.pt',a.output/'checkpoint.pt');model=NativeConnectorOrientationTTA(a.output/'checkpoint.pt','cpu');source=Path(model.plan['source']);rows=json.loads((source/'manifest.json').read_text());ids=np.array([i for i,r in enumerate(rows) if r['split']=='validation' and r.get('source_role','').startswith('Near25mm')]);pixels=np.load(source/'native_crops.npy',mmap_mode='r');truth=np.stack([matrix_from_pose9(x) for x in np.load(source/'labels.npy')]);torch.set_num_threads(4);means=[];canonical=[];spans=[]
for start in range(0,len(ids),8):
 mean,raw,span=model.cropped_rotations(np.array(pixels[ids[start:start+8]]));means.extend(mean);canonical.extend(raw);spans.extend(span)
reports={}
for name,pred in [('canonical',np.stack(canonical)),('five_shift',np.stack(means))]:
 relative=truth[ids,:3,:3].transpose(0,2,1)@pred;angle=np.degrees(Rotation.from_matrix(relative).magnitude());tilt=np.degrees(np.arccos(np.clip(relative[:,2,2],-1,1)));reports[name]=dict(orientation_p50_p95_deg=np.percentile(angle,[50,95]).tolist(),axial_tilt_p50_p95_deg=np.percentile(tilt,[50,95]).tolist())
report=dict(checkpoint_sha256=hashlib.sha256((a.output/'checkpoint.pt').read_bytes()).hexdigest(),fit=str(a.fit),training_complete_at_audit=(a.fit/'summary.json').exists(),samples=len(ids),episodes=len({rows[int(i)]['episode'] for i in ids}),shifts=SHIFTS,reports=reports,augmentation_span_p50_p95_deg=np.percentile(spans,[50,95]).tolist(),geometry_for_scoring_only=True,automatic_deployment=False)
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');np.save(a.output/'five_shift_rotations.npy',np.stack(means));np.save(a.output/'indices.npy',ids);print(json.dumps(report),flush=True)
