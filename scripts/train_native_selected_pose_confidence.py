#!/usr/bin/env python3
"""Calibrate selected-port pose reliability from RGB features on grouped scenes.

The label is held-out 0.5 mm lateral accuracy, not literal visual visibility.
True geometry is read for training and scoring only.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate,reprojection_residual_px
from train_two_family_port_locator import CAMERAS,NAMES


def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    if not len(y):return {'n':0}
    bins=np.linspace(0,1,6);ece=0.
    for lo,hi in zip(bins[:-1],bins[1:]):
        keep=(p>=lo)&(p<=hi if hi==1 else p<hi)
        if keep.any():ece+=float(keep.mean()*abs(y[keep].mean()-p[keep].mean()))
    accepted=p>=.9
    return {'n':len(y),'positive_rate':float(y.mean()),
            'brier':float(np.mean((p-y)**2)),'ece_5bins':ece,
            'confidence_ge_0p9_coverage':float(accepted.mean()),
            'confidence_ge_0p9_precision':float(y[accepted].mean()) if accepted.any() else None}


class Head(nn.Module):
    def __init__(self,d):
        super().__init__();self.net=nn.Sequential(nn.Linear(d,96),nn.GELU(),nn.Dropout(.1),
            nn.Linear(96,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x):return self.net(x).flatten()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--camera-calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--family',choices=('sc','sfp'),default='sfp')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous confidence fit')
    a.output.mkdir(parents=True)
    torch.manual_seed(20261002);np.random.seed(20261002);torch.set_num_threads(4)
    rows=[json.loads(s) for s in (a.labels/'rows.jsonl').open()]
    features=np.load(a.features,mmap_mode='r')
    if features.shape!=(len(rows),3,768):raise ValueError(features.shape)
    pred=defaultdict(dict)
    for item in map(json.loads,a.predictions.open()):pred[item['row']][item['camera']]=item
    cal=json.loads(a.camera_calibration.read_text())
    optics={c:np.asarray(cal['tcp_to_optical'][c],float) for c in CAMERAS}
    examples=[];unestimable=[]
    for i,row in enumerate(rows):
        if row['family']!=a.family:continue
        name=row['selected_identity'];target=next(t for t in row['targets'] if t['identity']==name)
        sel={c:pred[i][c]['predicted'][name] for c in CAMERAS}
        pixels={c:v['xy'] for c,v in sel.items() if v['projected_inframe_probability']>=.5}
        if len(pixels)<2:
            unestimable.append({'episode':row['episode'],'split':row['split'],'identity':name})
            continue
        poses={c:pose_matrix(row['state'][:7])@optics[c] for c in CAMERAS}
        try:
            point=triangulate(pixels,poses)
            reproj=reprojection_residual_px(point,pixels,poses)
        except np.linalg.LinAlgError:continue
        true=np.asarray(target['position_base_m'])
        from scipy.spatial.transform import Rotation
        rotation=Rotation.from_quat(target['orientation_base_xyzw']).as_matrix()
        error=rotation.T@(point-true)
        lateral=float(np.linalg.norm(error[:2])*1000)
        axial=float(abs(error[2])*1000)
        tcpm=float(np.linalg.norm(point-row['state'][:3]))
        camera_metadata=np.concatenate(([sel[c]['projected_inframe_probability'] for c in CAMERAS],
            [v for c in CAMERAS for v in np.asarray(sel[c]['xy'])/[1151,1023]]))
        code=np.zeros(len(NAMES),np.float32);code[NAMES.index(name)]=1.
        x=np.r_[np.asarray(features[i],dtype=np.float32).reshape(-1),code,
                camera_metadata,[min(10.,reproj)/10,tcpm]].astype(np.float32)
        examples.append({'x':x,'y':float(lateral<=.5),'lateral_mm':lateral,
                         'axial_mm':axial,'near':tcpm<.1,'episode':row['episode'],
                         'split':row['split'],'identity':name})
    train=[r for r in examples if r['split']=='train']
    valid_names=sorted({r['episode'] for r in examples if r['split']=='validation'})
    selection_names=set(valid_names[::2]);test_names=set(valid_names[1::2])
    selection=[r for r in examples if r['episode'] in selection_names]
    test=[r for r in examples if r['episode'] in test_names]
    if not train or not selection or not test:raise ValueError('Grouped train/selection/test needed')
    x=np.stack([r['x'] for r in train]);mean=x.mean(0);std=x.std(0).clip(min=.01)
    tx=torch.tensor((x-mean)/std);ty=torch.tensor([r['y'] for r in train],dtype=torch.float32)
    vx=torch.tensor((np.stack([r['x'] for r in selection])-mean)/std)
    vy=np.asarray([r['y'] for r in selection]);test_x=torch.tensor((np.stack([r['x'] for r in test])-mean)/std)
    model=Head(x.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    best=float('inf');best_epoch=0;stale=0;history=[]
    for epoch in range(1,301):
        order=torch.randperm(len(tx));model.train()
        for ids in order.split(64):
            opt.zero_grad(set_to_none=True)
            loss=nn.functional.binary_cross_entropy_with_logits(model(tx[ids]),ty[ids])
            loss.backward();opt.step()
        if epoch%5:continue
        model.eval()
        with torch.inference_mode():prob=model(vx).sigmoid().numpy()
        score=metrics(vy,prob)['brier'];history.append({'epoch':epoch,'selection_brier':score})
        if score<best-1e-4:
            best=score;best_epoch=epoch;stale=0
            torch.save({'model':model.state_dict(),'mean':mean,'std':std,
                        'feature_dim':x.shape[1],'trained_reliability_not_visibility':True,
                        'runtime_true_geometry':False},a.output/'best.pt')
        else:stale+=1
        if epoch>=30 and stale>=10:break
    ckpt=torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['model']);model.eval()
    with torch.inference_mode():
        vp=model(vx).sigmoid().numpy();tp=model(test_x).sigmoid().numpy()
    report={'schema':'aic_selected_pose_confidence/v1','family':a.family,
            'target':'P(3D lateral pose error <= 0.5 mm)',
            'actual_visibility_label':False,'best_epoch':best_epoch,'stopped_epoch':epoch,
            'episodes':{'train':len({r['episode'] for r in train}),
                        'selection':len(selection_names),'test':len(test_names)},
            'rows':{'train':len(train),'selection':len(selection),'test':len(test),
                    'unestimable':len(unestimable)},
            'selection':metrics(vy,vp),
            'test':metrics([r['y'] for r in test],tp),
            'test_constant_mean':metrics([r['y'] for r in test],
                                         np.full(len(test),ty.mean().item())),
            'test_near':metrics([r['y'] for r in test if r['near']],
                                [p for r,p in zip(test,tp) if r['near']]),
            'test_by_identity':{name:metrics([r['y'] for r in test if r['identity']==name],
                                             [p for r,p in zip(test,tp) if r['identity']==name])
                                for name in sorted({r['identity'] for r in test})},
            'final_scenes_opened':False}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
