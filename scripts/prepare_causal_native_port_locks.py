#!/usr/bin/env python3
"""Freeze each teacher episode's earliest stable RGB-only requested-port pose.

Only original images, causal public robot state, the public task code and
training-only camera/fixture calibration enter inference. No target geometry
is read. Two successive accepted observations must agree; their second time
is the lock time. Earlier actor rows must be excluded, never backfilled.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from requested_native_port_pose import RequestedNativePortPose


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('commands','output','backbone','sc-coarse','sfp-coarse','native-refiner','mount-priors'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--family',choices=('sc','sfp','both'),default='both')
    p.add_argument('--sc-margin-px',type=float,default=50.)
    p.add_argument('--sfp-margin-px',type=float,default=64.)
    p.add_argument('--sc-reprojection-px',type=float,default=2.)
    p.add_argument('--sfp-reprojection-px',type=float,default=2.)
    p.add_argument('--sfp-native-refiner',type=Path,
                   help='Optional SFP-only targeted repair; keep the validated SC version frozen')
    p.add_argument('--inspection-period-s',type=float,default=.2)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=a.resume)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    source=json.loads((a.commands/'summary.json').read_text())
    native_paths={f:a.native_refiner for f in ('sc','sfp')}
    projection_gates={f:.9 for f in native_paths}
    if a.native_refiner.suffix=='.json':
        bundle=json.loads(a.native_refiner.read_text())
        if bundle.get('schema')!='native_refiner_family_bundle/v1':raise ValueError('Unknown native family bundle')
        for f in native_paths:
            item=bundle['checkpoints'][f];path=Path(item['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Native family checkpoint hash changed')
            native_paths[f]=path
            projection_gates[f]=item.get('minimum_projection_probability',.9)
    if a.sfp_native_refiner:native_paths['sfp']=a.sfp_native_refiner
    models={f:RequestedNativePortPose(a.backbone,a.sc_coarse if f=='sc' else a.sfp_coarse,
        native_paths[f],
        a.mount_priors,family=f,maximum_shift_px=a.sc_margin_px if f=='sc' else a.sfp_margin_px,
        maximum_reprojection_px=a.sc_reprojection_px if f=='sc' else a.sfp_reprojection_px,
        minimum_projection_probability=projection_gates[f])
        for f in ('sc','sfp') if a.family in ('both',f)}
    plan=dict(command_source=str(a.commands),models={str(v):hashlib.sha256(v.read_bytes()).hexdigest()
        for v in (a.backbone,a.sc_coarse,a.sfp_coarse,a.native_refiner,a.mount_priors,*native_paths.values())},
        native_refiner_by_family={f:str(path) for f,path in native_paths.items()},
        minimum_projection_probability_by_family=projection_gates,
        lock_rule='two successive accepted observations within1mm/1degree; static lock at second observation; reset context each episode',
        maximum_refiner_shift_px=dict(sc=a.sc_margin_px,sfp=a.sfp_margin_px),
        maximum_SC_reprojection_px=a.sc_reprojection_px,maximum_SFP_reprojection_px=a.sfp_reprojection_px,
        inspection_period_s=a.inspection_period_s,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        geometry_input=False,backfill_prelock_rows=False,final_scenes_opened=False)
    if (a.output/'plan.json').exists() and json.loads((a.output/'plan.json').read_text())!=plan:raise ValueError('Frozen lock plan changed')
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    progress=a.output/'progress.jsonl';reports=list(map(json.loads,progress.open())) if progress.exists() and a.resume else []
    done={r['episode'] for r in reports}
    for episode in source['episodes']:
        family=episode['family']
        if family not in models or episode['episode'] in done:continue
        model=models[family];model.reset();previous=None;locked=None;reasons={};seen=0;last_inspection=-float('inf')
        observations=list(map(json.loads,(a.commands/(episode.get('observation_prefix_file') or episode['rows_file'])).open()))
        for row in observations:
            if row['time_s']-last_inspection<a.inspection_period_s-1e-6:continue
            last_inspection=row['time_s']
            images={c:cv2.cvtColor(cv2.imread(row['images'][c]),cv2.COLOR_BGR2RGB) for c in ('center','left','right')}
            prediction=model.predict(images,np.asarray(row['state']),row['task']);seen+=1
            reasons[prediction['reason']]=reasons.get(prediction['reason'],0)+1
            if not prediction['accepted']:previous=None;continue
            pose=np.asarray(prediction['base_port_matrix'])
            if previous is not None:
                translation=float(np.linalg.norm(pose[:3,3]-previous[:3,3]));angle=float((Rotation.from_matrix(previous[:3,:3]).inv()*Rotation.from_matrix(pose[:3,:3])).magnitude())
                if translation<=.001 and angle<=np.deg2rad(1):
                    # Average only measurements already available now.
                    locked=pose.copy();locked[:3,3]=(pose[:3,3]+previous[:3,3])/2
                    locked[:3,:3]=Rotation.from_matrix(np.stack([pose[:3,:3],previous[:3,:3]])).mean().as_matrix()
                    locked=dict(episode=row['episode'],scene=row['scene'],split=row['split'],family=family,
                        task=row['task'],time_s=row['time_s'],base_port_matrix=locked.tolist(),
                        geometry_input=False,source_images=row['images'],static_port_lock=True,
                        lock_measurements=2,translation_spread_mm=translation*1000,rotation_spread_deg=float(np.rad2deg(angle)))
                    key=hashlib.sha256(row['episode'].encode()).hexdigest()
                    (a.output/(key+'.lock.json')).write_text(json.dumps(locked)+'\n')
                    break
            previous=pose
        report=dict(episode=episode['episode'],family=family,split=episode['split'],rows_inspected=seen,
                    locked=locked is not None,lock_time_s=locked['time_s'] if locked else None,reasons=reasons)
        reports.append(report)
        with progress.open('a') as f:f.write(json.dumps(report)+'\n')
        print(json.dumps(report),flush=True)
    locks=[]
    for r in reports:
        if r['locked']:
            key=hashlib.sha256(r['episode'].encode()).hexdigest()
            locks.append(json.loads((a.output/(key+'.lock.json')).read_text()))
    (a.output/'locks.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in locks))
    summary=dict(episodes=len(reports),locked_episodes=sum(r['locked'] for r in reports),
                 by_family={f:dict(episodes=sum(r['family']==f for r in reports),locked=sum(r['family']==f and r['locked'] for r in reports)) for f in models},
                 geometry_input=False,prelock_rows_must_be_excluded=True,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':
    from project_gpu_guard import selected_gpu_reservation
    with selected_gpu_reservation():main()
