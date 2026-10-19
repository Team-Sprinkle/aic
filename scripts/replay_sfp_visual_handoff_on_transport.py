#!/usr/bin/env python3
"""Frozen native-RGB SFP visual pause check on saved autonomous traces.

Samples are fixed by simulation clock, before any ground-truth geometry is read.
The predicted port-relative TCP pose determines the candidate pause. Truth is
loaded only after inference to score accepted/rejected candidates.
"""
import argparse
from bisect import bisect_left
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import yaml

from audit_sc_port_targets import pose_matrix
from build_two_family_port_projection_labels import sfp_ports
from sfp_port_cascade import SfpPortCascade
from train_sfp_candidate_correspondence_probe import Correspondence,frozen_maps
from train_two_family_port_locator import Locator


CAMERAS=('center','left','right')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root',type=Path,required=True)
    p.add_argument('--camera-calibration',type=Path,required=True)
    p.add_argument('--port-calibration',type=Path,required=True)
    p.add_argument('--run',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--first-sec',type=int,default=10)
    p.add_argument('--last-sec',type=int,default=40)
    p.add_argument('--period-sec',type=int,default=1)
    p.add_argument('--correspondence',type=Path,
                   help='Frozen selected-request point verifier; require all three views')
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve earlier replay')
    r=a.model_root
    model=SfpPortCascade(r/'joint_locator_native_sfp_v5_heatmap/best.pt',
        r/'sfp_broad_crop480_v2/best.pt',r/'sfp_fine_crop160_v2_inside_selection/best.pt',
        r/'sfp_native_orientation_v2_highnative/best.pt',a.camera_calibration,
        device=a.device,orientation_path_secondary=r/'sfp_native_orientation_v3_runtime_matched/best.pt')
    calibration=json.loads(a.port_calibration.read_text())
    correspondence=None
    if a.correspondence:
        cp=torch.load(a.correspondence,map_location='cpu',weights_only=False)
        locator=Locator(None).to(a.device).eval()
        locator.load_state_dict(torch.load(cp['locator'],map_location='cpu',weights_only=False)['model'])
        head=Correspondence().to(a.device).eval();head.load_state_dict(cp['model'])
        correspondence=(cp,locator,head)
    output=[]
    for run in a.run:
        trace=[json.loads(s) for s in (run/'runtime_trace.jsonl').open()]
        proposals=sorted((x for x in trace if x.get('event')=='proposal' and len(x.get('state',[]))==32),
                         key=lambda x:x['sim_time'])
        cfg=yaml.safe_load((run/'eval_config.yaml').read_text());trial_id=next(iter(cfg['trials']))
        truth_ports={(c,p):m for c,p,m in sfp_ports({'trial_yaml':str(run/'eval_config.yaml'),
                                                     'trial_id':trial_id},calibration)}
        paths={c:sorted((run/'frames').glob('*_'+c+'.jpg'),
                        key=lambda f:int(f.stem.split('_')[1])) for c in CAMERAS}
        stamps={c:np.asarray([int(f.stem.split('_')[1])/1e9 for f in fs]) for c,fs in paths.items()}
        records=[]
        for sec in range(a.first_sec,a.last_sec+1,a.period_sec):
            x=min(proposals,key=lambda q:abs(q['sim_time']-sec))
            if abs(x['sim_time']-sec)>.15:continue
            native={};age={}
            for c in CAMERAS:
                j=int(np.argmin(abs(stamps[c]-sec)));f=paths[c][j]
                age[c]=float(abs(stamps[c][j]-sec))
                if age[c]>.075:break
                native[c]=cv2.cvtColor(cv2.imread(str(f)),cv2.COLOR_BGR2RGB)
            if len(native)<3:continue
            small={c:cv2.resize(native[c],(288,256),interpolation=cv2.INTER_AREA)
                   for c in CAMERAS}
            started=time.perf_counter()
            with torch.inference_mode():pred,diag=model.predict(native,small,
                np.asarray(x['state']),np.asarray(x['task_vector']))
            scores=None
            if pred is not None and correspondence is not None:
                cp,locator,head=correspondence
                ims=np.stack([small[c] for c in CAMERAS])
                visual=frozen_maps(locator,ims,a.device).to(a.device).float()
                condition=[];points=[]
                for j,c in enumerate(CAMERAS):
                    request=np.zeros(len(cp['names']),np.float32)
                    request[cp['names'].index(diag['identity'])]=1
                    camera=np.zeros(3,np.float32);camera[j]=1
                    state=(np.asarray(x['state'][:7],np.float32)-cp['state_mean'].numpy())/cp['state_std'].numpy()
                    point=np.asarray(diag['fine_pixels_px'][j],np.float32)/[1151,1023]
                    condition.append(np.r_[request,camera,state,point]);points.append(point)
                cond=torch.tensor(np.stack(condition),device=a.device,dtype=torch.float32)
                points=torch.tensor(np.stack(points),device=a.device,dtype=torch.float32)
                with torch.inference_mode():
                    scores=head(visual,cond,points).sigmoid().cpu().numpy()
            correspondence_accept=bool(pred is not None and
                                       (scores is None or np.all(scores>=.5)))
            elapsed=(time.perf_counter()-started)*1000
            card=int(np.argmax(x['task_vector'][4:9]));port=int(np.argmax(x['task_vector'][2:4]))
            measured=pose_matrix(x['state'][:7]);actual=truth_ports[card,port]
            actual_tcp=np.linalg.inv(actual)@measured
            pred_tcp=np.linalg.inv(pred)@measured if pred is not None else None
            est_lateral=np.linalg.norm(pred_tcp[:2,3])*1000 if pred_tcp is not None else None
            est_axial=pred_tcp[2,3]*1000 if pred_tcp is not None else None
            candidate=(correspondence_accept and est_lateral<=10 and -100<=est_axial<=-70)
            records.append(dict(time_s=sec,measured_time_s=float(x['sim_time']),
                image_age_s=age,card=card,port=port,visual_accept=pred is not None,
                visual_reason=diag.get('reason'),reprojection_px=diag.get('reprojection_px'),
                correspondence_scores=scores.tolist() if scores is not None else None,
                correspondence_accept=correspondence_accept,
                estimated_lateral_mm=float(est_lateral) if est_lateral is not None else None,
                estimated_axial_mm=float(est_axial) if est_axial is not None else None,
                candidate_pause=bool(candidate),
                posthoc_actual_lateral_mm=float(np.linalg.norm(actual_tcp[:2,3])*1000),
                posthoc_actual_axial_mm=float(actual_tcp[2,3]*1000),
                posthoc_port_position_error_mm=float(np.linalg.norm(pred[:3,3]-actual[:3,3])*1000)
                                            if pred is not None else None,
                model_and_preprocess_ms=elapsed))
        output.append(dict(run=str(run),inspections=len(records),candidate_pauses=sum(x['candidate_pause'] for x in records),
            records=records))
    a.output.write_text(json.dumps({'schema':'aic_sfp_visual_pause_replay/v1',
        'schedule':dict(first=a.first_sec,last=a.last_sec,period=a.period_sec),
        'candidate_rule':'RGB cascade and optional all-three selected-port correspondence accepted; predicted port-frame TCP lateral <=10 mm and axial [-100,-70] mm',
        'correspondence_checkpoint':str(a.correspondence) if a.correspondence else None,
        'runtime_truth_input':False,'final_scenes_opened':False,'runs':output},indent=2)+'\n')
    for run in output:
        print(json.dumps(dict(run=run['run'],inspections=run['inspections'],
            candidate_pauses=run['candidate_pauses'],
            candidates=[x for x in run['records'] if x['candidate_pause']])),flush=True)


if __name__=='__main__':main()
