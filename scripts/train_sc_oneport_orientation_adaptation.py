#!/usr/bin/env python3
"""Bounded frozen-encoder orientation-head adaptation for one-port SC scenes.

Original grouped SFP/SC examples are replayed to measure and limit forgetting.
No development or final reset enters fitting, and simulator pose is a label only.
"""
import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate
from train_two_family_port_locator import CAMERAS,NAMES,Locator
from train_two_family_port_orientation import Model


def stats(a):
    a=np.asarray(a,float)
    return {'n':len(a),'median_deg':float(np.median(a)) if len(a) else None,
            'p95_deg':float(np.quantile(a,.95)) if len(a) else None}


def error(model,x,y,device):
    out=[]
    with torch.inference_mode():
        for i in range(0,len(x),512):
            out.append(model(torch.from_numpy(x[i:i+512]).to(device)).float().cpu().numpy())
    if not out:return np.asarray([])
    return np.rad2deg((Rotation.from_matrix(np.concatenate(out)).inv()*
                       Rotation.from_matrix(y)).magnitude())


def original_data(labels,sc_pred,sfp_pred,visual_path,calibration):
    rows=[json.loads(s) for s in labels.open()]
    pred=defaultdict(dict)
    for path in (sc_pred,sfp_pred):
        for item in map(json.loads,path.open()):pred[item['row']][item['camera']]=item
    visual=np.load(visual_path,mmap_mode='r')
    if visual.shape!=(len(rows),3,768):raise ValueError('Visual cache mismatch')
    optics={c:np.asarray(v,float) for c,v in calibration['tcp_to_optical'].items()}
    x=[];y=[];meta=[]
    for i,r in enumerate(rows):
        tcp=pose_matrix(r['state'][:7]);cameras={c:tcp@optics[c] for c in CAMERAS}
        xyz=[];mask=[]
        for name in NAMES:
            pixels={c:pred[i][c]['predicted'][name]['xy'] for c in CAMERAS
                    if c in pred[i] and name in pred[i][c]['predicted'] and
                    pred[i][c]['predicted'][name]['projected_inframe_probability']>=.5}
            if len(pixels)>=2:
                try:point=triangulate(pixels,cameras);valid=np.isfinite(point).all()
                except np.linalg.LinAlgError:valid=False
            else:valid=False
            xyz.extend(point.tolist() if valid else [0.,0.,0.]);mask.append(float(valid))
        code=np.zeros(len(NAMES));code[NAMES.index(r['selected_identity'])]=1
        x.append(np.r_[xyz,mask,code,r['state'][:7],np.asarray(visual[i]).reshape(-1)])
        target=next(t for t in r['targets'] if t['identity']==r['selected_identity'])
        y.append(Rotation.from_quat(target['orientation_base_xyzw']).as_matrix())
        meta.append((r['split'],r['family'],r['episode'],bool(mask[NAMES.index(r['selected_identity'])])))
    return np.asarray(x,np.float32),np.asarray(y,np.float32),meta


def grouped_triplets(paths):
    groups=[]
    for path in paths:
        by_camera=defaultdict(list)
        for row in map(json.loads,path.open()):by_camera[row['camera']].append(row)
        for camera in CAMERAS:by_camera[camera].sort(key=lambda r:r['sim_ns'])
        if not by_camera['center']:continue
        ids=np.unique(np.linspace(0,len(by_camera['center'])-1,
                                  min(len(by_camera['center']),35),dtype=int))
        for j in ids:
            center=by_camera['center'][j];views={'center':center}
            for camera in ('left','right'):
                if not by_camera[camera]:break
                other=min(by_camera[camera],key=lambda r:abs(r['sim_ns']-center['sim_ns']))
                if abs(other['sim_ns']-center['sim_ns'])>150_000_000:break
                views[camera]=other
            if len(views)==3 and all(views[c]['projected_inframe'] for c in CAMERAS):
                groups.append((path,views))
    return groups


def oneport_data(paths,locator,calibration,device):
    samples=grouped_triplets(paths)
    optics={c:np.asarray(v,float) for c,v in calibration['tcp_to_optical'].items()}
    x=[];y=[];meta=[]
    for path,views in samples:
        images=[]
        for c in CAMERAS:
            bgr=cv2.imread(views[c]['image'])
            if bgr is None:raise FileNotFoundError(views[c]['image'])
            images.append(cv2.cvtColor(cv2.resize(bgr,(288,256),interpolation=cv2.INTER_AREA),
                                       cv2.COLOR_BGR2RGB))
        rgb=torch.from_numpy(np.stack(images)).permute(0,3,1,2).to(device).float()/255
        hint=np.zeros((3,5),np.float32);hint[:,0]=1
        for j in range(3):hint[j,2+j]=1
        with torch.inference_mode():
            points,flags,features=locator(rgb,torch.from_numpy(hint).to(device),return_features=True)
        points=points.float().cpu().numpy()*[1151,1023]
        flags=flags.float().sigmoid().cpu().numpy()
        state=np.asarray(views['center']['measured_tcp_base_pose'],float)
        # Camera frames can be up to 150 ms apart. Each projection needs the
        # measured TCP at *its own* timestamp, while the head's state is the
        # center decision state.
        cameras={c:pose_matrix(views[c]['measured_tcp_base_pose'])@optics[c]
                 for c in CAMERAS}
        xyz=[];mask=[]
        for k in range(len(NAMES)):
            pixels={c:points[j,k] for j,c in enumerate(CAMERAS) if flags[j,k]>=.5}
            if len(pixels)>=2:
                try:point=triangulate(pixels,cameras);valid=np.isfinite(point).all()
                except np.linalg.LinAlgError:valid=False
            else:valid=False
            xyz.extend(point.tolist() if valid else [0.,0.,0.]);mask.append(float(valid))
        port=int(views['center']['target_port']);code=np.zeros(len(NAMES));code[port]=1
        x.append(np.r_[xyz,mask,code,state,features.float().cpu().numpy().reshape(-1)])
        y.append((Rotation.from_quat(state[3:7])*Rotation.from_matrix(
            views['center']['relative_opening_rotation_tcp'])).as_matrix())
        distance=float(np.linalg.norm(views['center']['relative_opening_tcp_m'])*1000)
        meta.append({'episode':views['center']['episode_id'],
                     'group':views['center']['scene_group_id'],'port':port,
                     'distance_mm':distance,'valid':bool(mask[port]),
                     'max_camera_gap_ms':max(abs(views[c]['sim_ns']-views['center']['sim_ns'])
                                             for c in CAMERAS)/1e6,
                     'center_sim_ns':views['center']['sim_ns'],
                     'images':{c:views[c]['image'] for c in CAMERAS}})
    if not x:raise ValueError('No matched one-port triplets')
    return np.asarray(x,np.float32),np.asarray(y,np.float32),meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--sc-predictions',type=Path,required=True)
    p.add_argument('--sfp-predictions',type=Path,required=True)
    p.add_argument('--visual-features',type=Path,required=True)
    p.add_argument('--oneport-train-glob',required=True)
    p.add_argument('--oneport-development-glob',required=True)
    p.add_argument('--locator',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=600)
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous experiment')
    a.output.mkdir(parents=True)
    torch.manual_seed(20261003);np.random.seed(20261003);torch.set_num_threads(4)
    calibration=json.loads(a.calibration.read_text())
    train_paths=[Path(x) for x in sorted(glob.glob(a.oneport_train_glob))]
    dev_paths=[Path(x) for x in sorted(glob.glob(a.oneport_development_glob))]
    if len(train_paths)!=8 or len(dev_paths)!=4:raise ValueError('Expected eight train and four dev groups')
    if {x.parent.parent.name for x in train_paths}&{x.parent.parent.name for x in dev_paths}:
        raise ValueError('One-port scene overlap')
    ordinary_x,ordinary_y,ordinary_meta=original_data(a.labels,a.sc_predictions,
        a.sfp_predictions,a.visual_features,calibration)
    locator=Locator(None).to(a.device).eval()
    locator.load_state_dict(torch.load(a.locator,map_location='cpu',weights_only=False)['model'])
    train_x,train_y,train_meta=oneport_data(train_paths,locator,calibration,a.device)
    dev_x,dev_y,dev_meta=oneport_data(dev_paths,locator,calibration,a.device)
    train_groups={r['group'] for r in train_meta}
    dev_groups={r['group'] for r in dev_meta}
    if train_groups&dev_groups:raise ValueError('One-port reset-group leakage')
    saved=torch.load(a.initial,map_location='cpu',weights_only=False)
    model=Model(saved['model']['mean'],saved['model']['std']).to(a.device)
    model.load_state_dict(saved['model'])
    original_val={f:np.asarray([i for i,m in enumerate(ordinary_meta)
                               if m[0]=='validation' and m[1]==f and m[3]],dtype=int)
                  for f in ('sc','sfp')}
    def evaluate():
        model.eval();report={}
        for f,ids in original_val.items():
            report[f+'_original_validation']=stats(error(model,ordinary_x[ids],ordinary_y[ids],a.device))
        e=error(model,dev_x,dev_y,a.device)
        report['oneport_development_all']=stats(e)
        near=[i for i,m in enumerate(dev_meta) if m['distance_mm']<100 and m['valid']]
        report['oneport_development_near_100mm']=stats(e[near])
        report['oneport_by_group']={name:stats([e[i] for i,m in enumerate(dev_meta)
             if m['episode']==name and m['valid']]) for name in sorted({m['episode'] for m in dev_meta})}
        return report
    baseline=evaluate()
    train_ids=np.asarray([i for i,m in enumerate(ordinary_meta) if m[0]=='train'])
    xx=torch.from_numpy(ordinary_x).to(a.device);yy=torch.from_numpy(ordinary_y).to(a.device)
    one_x=torch.from_numpy(train_x).to(a.device);one_y=torch.from_numpy(train_y).to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    rng=np.random.default_rng(20261003);history=[]
    for step in range(1,a.updates+1):
        base=torch.as_tensor(rng.choice(train_ids,96),device=a.device)
        one=torch.as_tensor(rng.choice(len(train_x),32),device=a.device)
        model.train();opt.zero_grad(set_to_none=True)
        pred=model(torch.cat((xx[base],one_x[one])))
        loss=(pred-torch.cat((yy[base],one_y[one]))).square().mean()
        loss.backward();opt.step()
        if step%200==0:history.append({'step':step,'train_loss':float(loss.detach())})
    updated=evaluate()
    torch.save({'model':model.state_dict(),'source_initial':str(a.initial),
        'updates':a.updates,'fit_oneport_groups':[x.parent.parent.name for x in train_paths],
        'final_scenes_opened':False},a.output/'adapted.pt')
    report={'schema':'aic_sc_oneport_orientation_adaptation/v1',
        'hypothesis':'Explicit one-port RGB fitting repairs frozen orientation head without losing original tasks',
        'updates':a.updates,'ordinary_training_rows':int(len(train_ids)),
        'ordinary_validation_rows':{k:int(len(v)) for k,v in original_val.items()},
        'oneport_train_views':len(train_x),'oneport_development_views':len(dev_x),
        'oneport_train_groups':len(train_groups),'oneport_train_requested_groups':len(train_paths),
        'oneport_development_groups':len(dev_groups),
        'baseline':baseline,'adapted':updated,'history':history,
        'checkpoint':str(a.output/'adapted.pt'),
        'runtime_true_geometry_input':False,'truth_for_training_and_scoring_only':True,
        'final_scenes_opened':False}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'baseline':baseline,'adapted':updated},indent=2))


if __name__=='__main__':main()
