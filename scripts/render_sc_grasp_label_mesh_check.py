#!/usr/bin/env python3
"""Project the official visible SC mesh using oracle labels, for label QA only."""
import cv2
cv2.setNumThreads(2)
import argparse,json,struct,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from shared_port_tcp_geometry import matrix_from_pose9
from evaluate_sc_native_triangulation import FOCAL


def clouds(path):
 blob=path.read_bytes();assert blob[:4]==b'glTF';offset=12;chunks={}
 while offset<len(blob):
  n,kind=struct.unpack_from('<II',blob,offset);chunks[kind]=blob[offset+8:offset+8+n];offset+=8+n
 data=json.loads(chunks[0x4e4f534a]);binary=chunks[0x004e4942];out=[]
 for node in data['nodes']:
  if 'mesh' not in node:continue
  if any(k in node for k in ['matrix','translation','rotation','scale','children']):raise ValueError('Handle explicit node transform before projecting a changed asset')
  for primitive in data['meshes'][node['mesh']]['primitives']:
   accessor=data['accessors'][primitive['attributes']['POSITION']];assert accessor['componentType']==5126 and accessor['type']=='VEC3' and 'sparse' not in accessor
   view=data['bufferViews'][accessor['bufferView']];vertices=np.ndarray((accessor['count'],3),dtype='<f4',buffer=binary,offset=view.get('byteOffset',0)+accessor.get('byteOffset',0),strides=(view.get('byteStride',12),4)).copy();out.append(vertices)
 return out


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--episode',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();row=next(r for r in json.loads((a.source/'manifest.json').read_text()) if r['episode']==a.episode);body=matrix_from_pose9(row['tip_in_tcp_pose9']);v=list(map(float,ET.parse('aic_assets/models/SC Plug/model.sdf').getroot().find("model/link[@name='sc_tip_link']/pose").text.split()));plug_tip=np.eye(4);plug_tip[:3,3]=v[:3];plug_tip[:3,:3]=Rotation.from_euler('xyz',v[3:]).as_matrix();tcp_mesh=body@np.linalg.inv(plug_tip);parts=clouds(Path('aic_assets/models/SC Plug/sc_plug_visual.glb'));optic=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())['tcp_to_optical'];plan=torch.load('/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/sc_grasp_joint_geometry_v6_continuation/joint_geometry/best.pt',map_location='cpu',weights_only=False)['plan'];tiles=[]
 for c in ['center','left','right']:
  raw=cv2.imread(row['images'][c]);origin=np.rint(np.asarray(plan['crop_centres_train_only'][c])-96);image=cv2.warpAffine(raw,np.array([[1,0,-origin[0]],[0,1,-origin[1]]],np.float32),(192,192));overlay=image.copy();camera_mesh=np.linalg.inv(np.asarray(optic[c]))@tcp_mesh
  for vertices in parts:
   points=vertices@camera_mesh[:3,:3].T+camera_mesh[:3,3];assert np.min(points[:,2])>0
   pixels=points[:,:2]/points[:,2:]*FOCAL+[576,512]-origin;hull=cv2.convexHull(np.rint(pixels).astype(np.int32));cv2.polylines(overlay,[hull],True,(0,255,0),1)
  left=cv2.resize(image,(384,384));right=cv2.resize(overlay,(384,384));cv2.putText(left,c+' RGB',(8,24),0,.6,(0,255,255),1);cv2.putText(right,'GT mesh QA ONLY',(8,24),0,.6,(0,255,255),1);tiles.append(np.hstack([left,right]))
 cv2.imwrite(str(a.output),np.vstack(tiles));print(json.dumps(dict(episode=row['episode'],mesh_bounds=np.stack([np.min(np.vstack(parts),0),np.max(np.vstack(parts),0)]).tolist(),GT_label_QA_only=True,GT_crop_selection=False,warning='Per-part projected convex hulls include occluded/back edges. This is not a trained-pose prediction visualization.')))


if __name__=='__main__':main()
