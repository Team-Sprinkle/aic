#!/usr/bin/env python3
"""Diagnose port orientation from RGB-predicted multiport geometry.

Input candidate points come from the frozen RGB locator and measured TCP.
The true port quaternions are training/scoring labels only. Complete scene
groups are held out. This model is a diagnostic, not a visibility model.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import random

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.nn import functional as F

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate
from train_two_family_port_locator import NAMES,CAMERAS


def six_to_rotation(x):
    a=F.normalize(x[...,:3],dim=-1)
    b=x[...,3:]-torch.sum(x[...,3:]*a,dim=-1,keepdim=True)*a
    b=F.normalize(b,dim=-1)
    return torch.stack((a,b,torch.cross(a,b,dim=-1)),dim=-1)


class Model(nn.Module):
    def __init__(self,mean,std):
        super().__init__();self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
        self.head=nn.Sequential(nn.Linear(len(mean),256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),
                                nn.Linear(256,128),nn.SiLU(),nn.Linear(128,6))
    def forward(self,x):return six_to_rotation(self.head((x-self.mean)/self.std))


def err_deg(output,target):
    cosine=((output.transpose(-1,-2)@target).diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
    return torch.rad2deg(torch.acos(cosine.clamp(-1+1e-6,1-1e-6)))


def stats(values):
    a=np.asarray(values)
    return {'n':len(a),'median_deg':float(np.median(a)) if len(a) else None,
            'p95_deg':float(np.quantile(a,.95)) if len(a) else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--sc-predictions',type=Path)
    p.add_argument('--sfp-predictions',type=Path)
    p.add_argument('--camera-calibration',type=Path,required=True)
    p.add_argument('--visual-features',type=Path,
                   help='Frozen RGB-only layer3+4 features, three cameras per label row')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=3000)
    p.add_argument('--selection-manifest',type=Path,
                   help='Reserve native validation episodes not used for checkpoint selection')
    p.add_argument('--native-sample-fraction',type=float,default=None,
                   help='Episode-balanced training mass for native RGB source')
    a=p.parse_args()
    if not (a.sc_predictions or a.sfp_predictions):p.error('At least one prediction file required')
    if a.output.exists():p.error('Preserve existing run')
    a.output.mkdir(parents=True)
    torch.set_num_threads(8);torch.manual_seed(20261002);np.random.seed(20261002);random.seed(20261002)
    rows=[json.loads(l) for l in (a.labels/'rows.jsonl').open()]
    pred=defaultdict(dict)
    for path in (a.sc_predictions,a.sfp_predictions):
        if path is None:continue
        for item in map(json.loads,path.open()):pred[item['row']][item['camera']]=item
    calibration=json.loads(a.camera_calibration.read_text())
    optics={c:np.asarray(calibration['tcp_to_optical'][c],float) for c in CAMERAS}
    x=[];y=[];splits=[];families=[];episodes=[];identities=[];selected_mask=[]
    for i,r in enumerate(rows):
        base_tcp=pose_matrix(r['state'][:7]);camera_pose={c:base_tcp@optics[c] for c in CAMERAS}
        candidates=[];present=[]
        for name in NAMES:
            pixels={c:pred[i][c]['predicted'][name]['xy'] for c in CAMERAS
                    if c in pred[i] and name in pred[i][c]['predicted'] and
                    pred[i][c]['predicted'][name]['projected_inframe_probability']>=.5}
            if len(pixels)>=2:
                try:point=triangulate(pixels,camera_pose)
                except np.linalg.LinAlgError:point=np.zeros(3)
                valid=float(np.isfinite(point).all())
            else:point=np.zeros(3);valid=0.
            candidates.extend(point.tolist() if valid else [0.,0.,0.]);present.append(valid)
        code=np.zeros(len(NAMES));code[NAMES.index(r['selected_identity'])]=1.
        feature=np.r_[candidates,present,code,r['state'][:7]].astype(np.float32)
        target=next(t for t in r['targets'] if t['identity']==r['selected_identity'])
        x.append(feature);y.append(Rotation.from_quat(target['orientation_base_xyzw']).as_matrix())
        splits.append(r['split']);families.append(r['family']);episodes.append(r['episode'])
        identities.append(r['selected_identity']);selected_mask.append(present[NAMES.index(r['selected_identity'])])
    x=np.asarray(x,np.float32);y=np.asarray(y,np.float32)
    if a.visual_features:
        visual=np.load(a.visual_features,mmap_mode='r')
        if visual.shape!=(len(rows),3,768):raise ValueError(visual.shape)
        x=np.concatenate((x,np.asarray(visual,dtype=np.float32).reshape(len(rows),-1)),axis=1)
    train=np.flatnonzero(np.asarray(splits)=='train');val=np.flatnonzero(np.asarray(splits)=='validation')
    test=np.asarray([],dtype=int)
    if a.selection_manifest:
        selection=set(json.loads(a.selection_manifest.read_text())['native_selection_episodes'])
        test=np.asarray([i for i in val if episodes[i] not in selection],dtype=int)
        val=np.asarray([i for i in val if episodes[i] in selection],dtype=int)
        if not len(val) or not len(test):raise ValueError('Selection/test episodes missing')
    train_eps=Counter(episodes[i] for i in train)
    if a.native_sample_fraction is None:
        weight=np.asarray([1/train_eps[episodes[i]] for i in train])
    else:
        if not 0<a.native_sample_fraction<1:raise ValueError('Invalid native fraction')
        native={episodes[i] for i in train if rows[i].get('image_kind')=='native_1152x1024_rgb_jpeg'}
        agent={episodes[i] for i in train if episodes[i] not in native}
        if not native or not agent:raise ValueError('Both native and agent training episodes required')
        weight=np.asarray([((a.native_sample_fraction/len(native)) if episodes[i] in native
                            else ((1-a.native_sample_fraction)/len(agent))) / train_eps[episodes[i]]
                           for i in train])
    weight/=weight.sum()
    mean=x[train].mean(0);std=x[train].std(0);std=np.maximum(std,.01)
    model=Model(mean,std).cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    xx=torch.from_numpy(x).cuda();yy=torch.from_numpy(y).cuda()
    mean_rotation={key:Rotation.from_matrix(y[[i for i in train if (families[i],identities[i])==key]]).mean()
                   for key in set((families[i],identities[i]) for i in train)}
    history=[];best=float('inf');best_step=0;stale=0
    for step in range(1,a.updates+1):
        batch=np.random.choice(train,size=256,p=weight)
        model.train();opt.zero_grad(set_to_none=True)
        output=model(xx[batch]);loss=(output-yy[batch]).square().mean()
        loss.backward();opt.step()
        if step%100==0:
            model.eval()
            with torch.inference_mode():
                predicted=torch.cat([model(part).cpu() for part in xx[val].split(1024)]).numpy()
            error=np.rad2deg((Rotation.from_matrix(predicted).inv()*Rotation.from_matrix(y[val])).magnitude())
            report={fam:stats(error[[j for j,i in enumerate(val) if families[i]==fam and selected_mask[i]]])
                    for fam in ('sc','sfp')}
            score=max(value['p95_deg'] for value in report.values() if value['p95_deg'] is not None)
            history.append({'step':step,'loss':float(loss.detach()),'validation':report})
            print(json.dumps({'step':step,'sc_p95':report['sc']['p95_deg'],
                              'sfp_p95':report['sfp']['p95_deg']}),flush=True)
            if score<best-.05:
                best=score;best_step=step;stale=0
                torch.save({'model':model.state_dict(),'architecture':'predicted_multiport_geometry_mlp',
                            'runtime_geometry':False,'visibility_certified':False},a.output/'best.pt')
            else:stale+=1
            if step>=500 and stale>=8:break
    model.load_state_dict(torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)['model'])
    model.eval()
    with torch.inference_mode():predicted=torch.cat([model(part).cpu() for part in xx[val].split(1024)]).numpy()
    error=np.rad2deg((Rotation.from_matrix(predicted).inv()*Rotation.from_matrix(y[val])).magnitude())
    baseline=[]
    for i in val:
        key=(families[i],identities[i]);base=mean_rotation.get(key)
        if base is None:base=Rotation.from_matrix(y[[j for j in train if families[j]==families[i]]]).mean()
        baseline.append(np.rad2deg((base.inv()*Rotation.from_matrix(y[i])).magnitude()))
    report={'schema':'two_family_port_orientation_geometry_mlp/v1','best_step':best_step,
            'updates_completed':step,'train_episodes':len({episodes[i] for i in train}),
            'validation_episodes':len({episodes[i] for i in val}),
            'frozen_rgb_features':str(a.visual_features) if a.visual_features else None,
            'results':{fam:{'model':stats([error[j] for j,i in enumerate(val) if families[i]==fam and selected_mask[i]]),
                            'constant_mean':stats([baseline[j] for j,i in enumerate(val) if families[i]==fam and selected_mask[i]])}
                       for fam in ('sc','sfp')},'runtime_geometry':False,'visibility_certified':False,
            'reserved_final_opened':False}
    if len(test):
        with torch.inference_mode():test_predicted=torch.cat([model(part).cpu() for part in xx[test].split(1024)]).numpy()
        test_error=np.rad2deg((Rotation.from_matrix(test_predicted).inv()*Rotation.from_matrix(y[test])).magnitude())
        report['selection_episodes']=sorted({episodes[i] for i in val})
        report['untouched_test_episodes']=sorted({episodes[i] for i in test})
        report['untouched_test']={fam:stats([test_error[j] for j,i in enumerate(test) if families[i]==fam and selected_mask[i]])
                                  for fam in ('sc','sfp')}
    report['native_sample_fraction']=a.native_sample_fraction
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
