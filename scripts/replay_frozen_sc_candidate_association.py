#!/usr/bin/env python3
"""Read-only check of proposed regions at frozen historical SC acquisitions.

No model, threshold, scene or historical result is changed. Geometry-derived
old wrong-accept annotations are consumed ONLY after image/state inference for
stratified scoring. This is a proposal rejection replay, not a new rollout.
"""
import argparse
from bisect import bisect_right
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from train_two_family_port_locator import Locator
from train_native_candidate_history_association import CandidateHistoryAssociation,distance_history
from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate,FOCAL
from build_sc_native_pose_labels import project
from scipy.spatial.transform import Rotation

CAMS=('center','left','right')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--locator',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reproject-past-snapshot',action='store_true',help='Account for older saved images using predicted candidate and public TCP only, under fixed motion bounds')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(6);cv2.setNumThreads(1)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=CandidateHistoryAssociation(saved['variant']=='distance_history',saved.get('always_current_global',False)).cuda().eval();model.load_state_dict(saved['model'])
    locator=Locator(None).cuda().eval();locator.load_state_dict(torch.load(a.locator,map_location='cpu',weights_only=False)['model'])
    mean=saved['state_mean'].cuda();std=saved['state_std'].cuda()
    optics={c:np.asarray(m) for c,m in json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())['tcp_to_optical'].items()}
    cases=json.loads((a.frozen/'progress.json').read_text())['results'];results=[];latency=[]
    with torch.inference_mode():
        for case in cases:
            if case['family']!='sc' or not case['rule_visual_acquisition']:continue
            audit=Path(case['audit']);trace=[json.loads(l) for l in (audit/'runtime_trace.jsonl').open()]
            terminals=[e for e in trace if e.get('event')=='rule_terminal' and e.get('acquired')]
            if not terminals:continue
            stop=terminals[-1]['sim_time']
            accepted=[e for e in trace if e.get('event')=='rule_visual_inspection' and e['sim_time']<=stop and e['report'].get('valid')]
            if not accepted:raise ValueError('Missing frozen accepted report')
            proposal=accepted[-1];points=np.asarray(proposal['report']['coarse_pixels'],np.float32)
            stream=json.loads((audit/'scored_stream.json').read_text());controller=stream['streams']['/aic_controller/controller_state']
            ct=[r[0] for r in controller];by_stamp={}
            with (audit/'frame_manifest.csv').open() as f:
                for r in csv.DictReader(f):
                    if r['camera'] in CAMS and int(r['sim_ns'])<=round(proposal['sim_time']*1e9):
                        by_stamp.setdefault(int(r['sim_ns']),{})[r['camera']]=str(audit/'frames'/Path(r['file']).name)
            rows=[]
            for stamp,images in sorted(by_stamp.items()):
                if len(images)!=3:continue
                j=bisect_right(ct,stamp)-1
                if j<0 or stamp-ct[j]>100_000_000:continue
                q=controller[j][1]['tcp_pose'];state=[q['position'][k] for k in 'xyz']+[q['orientation'][k] for k in 'xyzw']
                rows.append(dict(episode=case['case'],time_s=stamp/1e9,state=state,images=images))
            if not rows:raise ValueError('No causal native image triplet')
            age=proposal['sim_time']-rows[-1]['time_s']
            proposal_stamp=round(proposal['sim_time']*1e9);j=bisect_right(ct,proposal_stamp)-1
            if j<0 or proposal_stamp-ct[j]>100_000_000:raise ValueError('Missing past proposal TCP')
            q=controller[j][1]['tcp_pose'];proposal_state=[q['position'][k] for k in 'xyz']+[q['orientation'][k] for k in 'xyzw']
            motion_mm=float(np.linalg.norm(np.asarray(proposal_state[:3])-rows[-1]['state'][:3])*1000)
            motion_deg=float((Rotation.from_quat(proposal_state[3:]).inv()*Rotation.from_quat(rows[-1]['state'][3:])).magnitude()*180/np.pi)
            if a.reproject_past_snapshot:
                if age>.25 or motion_mm>1 or motion_deg>.2:
                    raise ValueError(f'Past snapshot outside fixed replay bounds age={age}, motion_mm={motion_mm}, motion_deg={motion_deg}')
                at_proposal={c:pose_matrix(proposal_state)@optics[c] for c in CAMS}
                candidate_base=triangulate({c:points[i] for i,c in enumerate(CAMS)},at_proposal)
                points=np.asarray([project(np.block([[np.eye(3),candidate_base[:,None]],[np.zeros((1,3)),np.ones((1,1))]]),pose_matrix(rows[-1]['state'])@optics[c])[:2] for c in CAMS],np.float32)
            elif age>.1:raise ValueError(f'Image too stale for frozen candidate replay {age}')
            history,mask,_=distance_history(rows,saved['spacing_mm']/1000)
            selected=history[-1][mask[-1]].tolist();indices=sorted(set(selected+[len(rows)-1]));features={};images_current=None
            for ri in indices:
                images=[cv2.imread(rows[ri]['images'][c]) for c in CAMS]
                if any(im is None or im.shape!=(1024,1152,3) for im in images):raise ValueError('Non-native or missing frozen RGB')
                small=np.stack([cv2.cvtColor(cv2.resize(im,(288,256),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB) for im in images])
                x=torch.from_numpy(small).permute(0,3,1,2).cuda().float()/255
                meta=torch.zeros(3,5,device='cuda');meta[:,0]=1;meta[:,2:]=torch.eye(3,device='cuda')
                _,_,_,spatial=locator(x,meta,return_features=True,return_spatial=True)
                features[ri]=F.adaptive_avg_pool2d(spatial,(8,9))
                if ri==len(rows)-1:images_current=images
            states=torch.tensor([r['state'] for r in rows],dtype=torch.float32,device='cuda')
            states[:,3:7]*=torch.where(states[:,6:7]<0,-1.,1.);states=(states-mean)/std
            hist=torch.zeros(10,3,6,256,device='cuda');hstate=torch.zeros(10,7,device='cuda');hmask=torch.zeros(10,dtype=torch.bool,device='cuda')
            for i,ri in enumerate(selected):
                hist[i]=F.adaptive_avg_pool2d(features[ri],(2,3)).flatten(2).permute(0,2,1);hstate[i]=states[ri];hmask[i]=True
            current=features[len(rows)-1]
            torch.cuda.synchronize();tick=time.perf_counter()
            pred=model(current[None].expand(3,-1,-1,-1,-1),hist[None].expand(3,-1,-1,-1,-1),hstate[None].expand(3,-1,-1),hmask[None].expand(3,-1),states[-1:,:].expand(3,-1),torch.arange(3,device='cuda'),torch.from_numpy(points).cuda()/torch.tensor([1151,1023],device='cuda'),torch.full((3,),case['port'],dtype=torch.long,device='cuda')).sigmoid().cpu().numpy()
            latency.append((time.perf_counter()-tick)*1000)
            # The posthoc wrong-port tag is first accessed below; it never selects history or image crops.
            out=dict(case=case['case'],arm=case['arm'],old_wrong_accept=bool(case['wrong_accepted_port']),
                probabilities=pred.tolist(),new_association_pass=bool(np.all(pred>=.9)),
                proposal_sim_time=proposal['sim_time'],image_time=rows[-1]['time_s'],image_age_ms=age*1000,
                measured_motion_since_image_mm=motion_mm,measured_motion_since_image_deg=motion_deg,
                candidate_reprojected_to_past_image=a.reproject_past_snapshot,
                history_keyframes=len(selected),geometry_prediction_input=False)
            results.append(out)
            if out['old_wrong_accept']:
                panels=[]
                for ci,im in enumerate(images_current):
                    cv2.drawMarker(im,tuple(np.rint(points[ci]).astype(int)),(0,0,255),cv2.MARKER_CROSS,24,2)
                    cv2.putText(im,f"request SC{case['port']} | candidate match {pred[ci]:.3f}",(12,35),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,255,255),2)
                    panels.append(cv2.resize(im,(576,512)))
                cv2.imwrite(str(a.output/(case['case']+'__'+case['arm']+'.jpg')),np.concatenate(panels,axis=1))
            with (a.output/'progress.json').open('w') as f:json.dump(dict(completed=len(results),last=out),f,indent=2)
            print(json.dumps(out),flush=True)
    summary=dict(schema='frozen_SC_candidate_association_replay/v1',cases=len(results),
        old_wrong_accepts=sum(r['old_wrong_accept'] for r in results),
        wrong_accepts_rejected=sum(r['old_wrong_accept'] and not r['new_association_pass'] for r in results),
        correct_acquisitions=sum(not r['old_wrong_accept'] for r in results),
        correct_retained=sum(not r['old_wrong_accept'] and r['new_association_pass'] for r in results),
        head_only_latency_ms=dict(p50=float(np.median(latency)),p95=float(np.percentile(latency,95))) if latency else {},
        threshold=.9,results=results,read_only_historical_replay=True,not_counterfactual_live_success=True,
        historical_results_modified=False,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)


if __name__=='__main__':main()
