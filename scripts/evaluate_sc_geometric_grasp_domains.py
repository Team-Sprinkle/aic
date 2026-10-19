#!/usr/bin/env python3
"""Audit the existing RGB landmark/triangulation grasp alternative on reset-heldout data."""
import cv2
cv2.setNumThreads(2)
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sc_native_grasp_observation import SCNativeGraspObservation
from shared_port_tcp_geometry import matrix_from_pose9


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 model=SCNativeGraspObservation(a.checkpoint,'cpu');rows=json.loads((a.source/'manifest.json').read_text());selected={}
 for row in rows:
  if row['split']=='validation':selected.setdefault(row['episode'],row)
 results=[]
 for row in selected.values():
  images={c:cv2.cvtColor(cv2.imread(row['images'][c]),cv2.COLOR_BGR2RGB) for c in ['center','left','right']};pred=model.predict(images,already_training_jpeg=True);truth=matrix_from_pose9(row['tip_in_tcp_pose9']);role=row.get('source_role','');domain='nominal_home' if role.startswith('Nominal-centred') else 'nominal_near_acquisition' if role else 'original_expert';rot=np.asarray(pred['orientation_for_diagnostic_only']);pos=np.asarray(pred['predicted_tip_position_tcp_m']);results.append(dict(episode=row['episode'],scene=row['scene'],domain=domain,translation_mm=float(np.linalg.norm(pos-truth[:3,3])*1000),orientation_deg=float(np.degrees((Rotation.from_matrix(rot).inv()*Rotation.from_matrix(truth[:3,:3])).magnitude())),predicted_position=pos.tolist(),predicted_rotation=rot.tolist(),input='Only original RGB + fixed training crop/camera calibration'))
 groups={d:{'episodes':sum(x['domain']==d for x in results),**{k:np.percentile([x[k] for x in results if x['domain']==d],[50,95]).tolist() for k in ['translation_mm','orientation_deg']}} for d in sorted({x['domain'] for x in results})};a.output.write_text(json.dumps(dict(checkpoint=str(a.checkpoint),source=str(a.source),metrics=groups,results=results,geometry_scoring_only=True,promotion=False),indent=2)+'\n');print(json.dumps(groups),flush=True)


if __name__=='__main__':main()
