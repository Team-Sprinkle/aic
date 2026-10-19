#!/usr/bin/env python3
"""Frozen RGB port-orientation check on saved SC transport stop observations."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate
from train_two_family_port_locator import CAMERAS, NAMES, Locator
from train_two_family_port_orientation import Model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decision',type=Path,required=True)
    p.add_argument('--audit-root',type=Path,required=True)
    p.add_argument('--locator',type=Path,required=True)
    p.add_argument('--orientation',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--tcp-in-port-prior',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior replay')
    decision=json.loads(a.decision.read_text())
    calibration=json.loads(a.calibration.read_text())
    optics={c:np.asarray(calibration['tcp_to_optical'][c],float) for c in CAMERAS}
    nominal=Rotation.from_quat(json.loads(a.tcp_in_port_prior.read_text())
                                ['nominal_tcp_in_port_orientation_xyzw'])
    locator=Locator(None).to(a.device).eval()
    locator.load_state_dict(torch.load(a.locator,map_location='cpu',weights_only=False)['model'])
    saved=torch.load(a.orientation,map_location='cpu',weights_only=False)
    orient=Model(saved['model']['mean'],saved['model']['std']).to(a.device).eval()
    orient.load_state_dict(saved['model'])
    rows=[]
    for item in decision['sc_frozen_traces']:
        stop=item['visual']['stop_s']
        if stop is None:continue
        audit=a.audit_root/item['trace']
        trace=[json.loads(x) for x in (audit/'runtime_trace.jsonl').open()]
        proposals=[x for x in trace if x['event']=='proposal']
        proposal=min(proposals,key=lambda x:abs(x['sim_time']-stop))
        index={(r['camera'],int(r['sim_ns'])):audit/'frames'/r['file']
               for r in csv.DictReader((audit/'frame_manifest.csv').open())}
        images=[];paths=[]
        for c in CAMERAS:
            path=index[(c,int(proposal['image_sim_ns'][c]))]
            rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
            images.append(cv2.resize(rgb,(288,256),interpolation=cv2.INTER_AREA))
            paths.append(str(path))
        pixels=torch.from_numpy(np.stack(images)).permute(0,3,1,2).to(a.device).float()/255
        meta=np.zeros((3,5),np.float32);meta[:,0]=1
        for j in range(3):meta[j,2+j]=1
        with torch.inference_mode():
            points,flags,features=locator(pixels,torch.from_numpy(meta).to(a.device),
                                          return_features=True)
        points=points.float().cpu().numpy()*[1151,1023]
        flags=flags.float().sigmoid().cpu().numpy()
        state=np.asarray(proposal['state'][:7],float);base_tcp=pose_matrix(state)
        cameras={c:base_tcp@optics[c] for c in CAMERAS}
        candidates=[];mask=[]
        for k in range(len(NAMES)):
            visible={c:points[j,k] for j,c in enumerate(CAMERAS) if flags[j,k]>=.5}
            if len(visible)>=2:
                try:position=triangulate(visible,cameras);valid=np.isfinite(position).all()
                except np.linalg.LinAlgError:valid=False
            else:valid=False
            candidates.extend(position.tolist() if valid else [0.,0.,0.]);mask.append(float(valid))
        port=int(np.argmax(proposal['task_vector'][2:4]));code=np.zeros(len(NAMES))
        code[NAMES.index(f'sc_port_{port}')]=1
        feat=np.r_[candidates,mask,code,state,features.float().cpu().numpy().reshape(-1)].astype(np.float32)
        with torch.inference_mode():
            pred=orient(torch.from_numpy(feat).to(a.device)[None]).float().cpu().numpy()[0]
        predicted_port=Rotation.from_matrix(pred)
        predicted_tcp=predicted_port*nominal
        measured=Rotation.from_quat(state[3:7])
        diagnostic=json.loads((audit/'teacher_path_diagnostic.json').read_text())
        plan=json.loads(Path(diagnostic['plan']).read_text())
        goal=Rotation.from_quat(plan['waypoints'][-1]['tcp_pose']['orientation_xyzw'])
        angle=lambda lhs,rhs:float(np.rad2deg((lhs.inv()*rhs).magnitude()))
        rows.append({'trace':item['trace'],'sim_time':proposal['sim_time'],
            'posthoc_false_stop':item['visual']['false_handoff'],
            'requested_port':port,'selected_rgb_flags':flags[:,port].tolist(),
            'selected_valid_triangulation':bool(mask[port]),
            'predicted_tcp_to_measured_deg':angle(predicted_tcp,measured),
            'predicted_tcp_to_teacher_goal_deg_posthoc':angle(predicted_tcp,goal),
            'measured_to_teacher_goal_deg_posthoc':angle(measured,goal),
            'predicted_port_quat_xyzw':predicted_port.as_quat().tolist(),
            'images':paths})
    output={'schema':'aic_sc_frozen_visual_orientation_stop_replay/v1',
        'checkpoint':str(a.orientation),'locator':str(a.locator),
        'nominal_tcp_in_port_prior':str(a.tcp_in_port_prior),
        'rule_candidate':'predicted TCP to measured TCP <=3 degrees',
        'threshold_selected_after_development_failures':True,
        'scored_teacher_goal_is_posthoc_only':True,
        'runtime_true_geometry_input':False,'rows':rows,'final_scenes_opened':False}
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ('trace','posthoc_false_stop','selected_valid_triangulation',
          'predicted_tcp_to_measured_deg','predicted_tcp_to_teacher_goal_deg_posthoc',
          'measured_to_teacher_goal_deg_posthoc')} for r in rows],indent=2))


if __name__=='__main__':main()
