#!/usr/bin/env python3
"""Matched full-image versus original-resolution grasp crop pose probes.

RGB is the only learned input. Simulator TF is label-only. Crop centres are
fixed from TRAIN episodes, never chosen using evaluation geometry.
"""
import argparse,json,hashlib,time
from pathlib import Path
from collections import defaultdict
from bisect import bisect_left
import cv2,numpy as np,torch
from torch import nn
from torchvision import models
from scipy.spatial.transform import Rotation,Slerp
from PIL import Image
from analyze_local_actor_scored_tip import record_at,transform
from shared_port_tcp_geometry import matrix_from_xyz_xyzw,matrix_from_pose9
from build_sc_native_pose_labels import project

ROOT=Path(__file__).resolve().parents[1]
CAMERAS=('center','left','right')

def interpolate(rows,t):
    stamps=[r[0] for r in rows];j=bisect_left(stamps,t)
    if j==0 or j==len(rows):raise ValueError('No bracketing label TF')
    t0,d0=rows[j-1];t1,d1=rows[j]
    if t1-t0>65e6:raise ValueError('Label TF gap exceeds validated interval')
    a,b=transform(d0),transform(d1);f=(t-t0)/(t1-t0);m=np.eye(4)
    m[:3,3]=a[:3,3]+f*(b[:3,3]-a[:3,3])
    m[:3,:3]=Slerp([0.,1.],Rotation.from_matrix(np.stack([a[:3,:3],b[:3,:3]])))([f]).as_matrix()[0]
    return m

def pose9(m):return np.r_[m[:3,3],m[:3,:2].T.reshape(-1)]

class Head(nn.Module):
    def __init__(self):
        super().__init__();self.project=nn.Linear(128,32)
        self.head=nn.Sequential(nn.LayerNorm(1536),nn.Linear(1536,256),nn.GELU(),
                                nn.Linear(256,64),nn.GELU(),nn.Linear(64,9))
    def forward(self,x):
        return self.head(self.project(x.permute(0,1,3,4,2)).flatten(1))

def errors(pred,true):
    position=np.linalg.norm(pred[:,:3]-true[:,:3],axis=1)*1000
    angles=[]
    for p,t in zip(pred,true):
        angles.append(float((Rotation.from_matrix(matrix_from_pose9(p)[:3,:3]).inv()*
                             Rotation.from_matrix(matrix_from_pose9(t)[:3,:3])).magnitude()*180/np.pi))
    return {'translation_p50_p95_mm':np.percentile(position,[50,95]).tolist(),
            'orientation_p50_p95_deg':np.percentile(angles,[50,95]).tolist()}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--validation-audit',type=Path,required=True)
    p.add_argument('--imagenet',type=Path,required=True);p.add_argument('--updates',type=int,default=5000)
    p.add_argument('--budget-minutes-per-arm',type=float,default=120);a=p.parse_args()
    if a.output.exists():p.error('Preserve prior experiment')
    if not json.loads(a.validation_audit.read_text())['passed']:p.error('High-rate label audit must pass first')
    a.output.mkdir(parents=True);torch.set_num_threads(4);cv2.setNumThreads(2)
    config=json.loads((ROOT/'configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics={c:np.asarray(config['tcp_to_optical'][c]) for c in CAMERAS}
    groups=defaultdict(list)
    for r in map(json.loads,a.source.open()):
        if r['observed_sc_tip_pose_opening_frame'][2]<-.03:groups[r['episode_id']].append(r)
    selected=[]
    for name,rows in sorted(groups.items()):
        rows.sort(key=lambda r:r['sim_time'])
        positions=np.asarray([r['state'][:3] for r in rows]);travel=np.r_[0,np.cumsum(np.linalg.norm(np.diff(positions,axis=0),axis=1))]
        indices=sorted(set(int(np.argmin(abs(travel-x))) for x in np.linspace(0,travel[-1],12)))
        stream=json.loads((ROOT/'outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized/streams'/f'{name}.json').read_text());e=stream['edges']
        wb=record_at(e['world|tabletop'],0)@record_at(e['tabletop|base_link'],0)
        for j in indices:
            r=rows[j];t=r['sim_time']*1e9
            worldtip=interpolate(e['aic_world|cable_1'],t)@interpolate(e['cable_1|cable_1/sc_tip_link'],t)
            tip=np.linalg.inv(wb@matrix_from_xyz_xyzw(r['state'][:7]))@worldtip
            selected.append({'episode':name,'scene':r['scene_sha256'],'split':r['split'],
                             'frame':r['frame'],'time_s':r['sim_time'],'images':r['images'],
                             'tip_in_tcp_pose9':pose9(tip).tolist()})
    fit=np.array([r['split']=='train' for r in selected]);validation=~fit
    assert not {r['scene'] for r in selected if r['split']=='train'}&{r['scene'] for r in selected if r['split']=='validation'}
    y=np.asarray([r['tip_in_tcp_pose9'] for r in selected],np.float32)
    centers={}
    for c in CAMERAS:
        points=[]
        for r in selected:
            if r['split']=='train':points.append(project(matrix_from_pose9(r['tip_in_tcp_pose9']),optics[c])[:2])
        centers[c]=np.median(points,axis=0).tolist()
    plan={'schema':'aic_sc_grasp_native_probe/v1','source':str(a.source),
          'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'rows':len(selected),
          'train_episodes':len({r['episode'] for r in selected if r['split']=='train'}),
          'validation_episodes':len({r['episode'] for r in selected if r['split']=='validation'}),
          'crop_centres_native_pixels_train_only':centers,'crop_size':160,
          'labels':'Interpolated TF validated against original high-rate bags; not actor inputs',
          'encoder':'Same frozen ImageNet ResNet18 through layer2, same 4x4 spatial features and shared head in both arms',
          'comparison':'Full native image reduced 288x256 then 224x224 vs crop160 before any global resizing',
          'updates':a.updates,'budget_minutes_per_arm':a.budget_minutes_per_arm,
          'gate':{'heldout_translation_p95_mm_max':.4,'heldout_orientation_p95_deg_max':1.,
                  'relative_translation_improvement_minimum':.25},'final_scenes_opened':False}
    (a.output/'frozen_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (a.output/'manifest.json').write_text(json.dumps(selected,indent=2)+'\n')
    backbone=models.resnet18(weights=None);backbone.load_state_dict(torch.load(a.imagenet,map_location='cpu',weights_only=True))
    encoder=nn.Sequential(*list(backbone.children())[:6],nn.AdaptiveAvgPool2d((4,4))).cuda().eval()
    features={name:[] for name in ['full','crop']};review=[]
    for r in selected:
        batches={'full':[],'crop':[]}
        for c in CAMERAS:
            im=cv2.imread(r['images'][c]);assert im.shape[:2]==(1024,1152)
            rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);left,top=np.rint(np.asarray(centers[c])-80).astype(int)
            patch=cv2.warpAffine(rgb,np.array([[1,0,-left],[0,1,-top]],np.float32),(160,160),borderMode=cv2.BORDER_CONSTANT)
            small=cv2.resize(rgb,(288,256),interpolation=cv2.INTER_AREA)
            batches['full'].append(np.asarray(Image.fromarray(small).resize((224,224),Image.Resampling.BILINEAR)).copy())
            batches['crop'].append(patch)
        if len(review)<6:review.append(np.hstack([cv2.cvtColor(x,cv2.COLOR_RGB2BGR) for x in batches['crop']]))
        for arm in features:
            x=torch.from_numpy(np.stack(batches[arm])).permute(0,3,1,2).cuda().float()/255
            x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
            with torch.inference_mode():features[arm].append(encoder(x).cpu().numpy().astype(np.float16))
    cv2.imwrite(str(a.output/'crop_input_review.jpg'),np.vstack(review))
    np.save(a.output/'labels.npy',y)
    mean=y[fit].mean(0);std=y[fit].std(0).clip(min=1e-4);summary={}
    for arm in features:
        array=np.stack(features[arm]);np.save(a.output/f'{arm}_features.npy',array)
        torch.manual_seed(20261014);np.random.seed(20261014);head=Head().cuda()
        xf=torch.from_numpy(array[fit]).cuda().float();xv=torch.from_numpy(array[validation]).cuda().float()
        yy=torch.from_numpy((y[fit]-mean)/std).cuda();optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
        best=float('inf');last_best=0;history=[];started=time.monotonic();reason='update_cap'
        for step in range(1,a.updates+1):
            if time.monotonic()-started>a.budget_minutes_per_arm*60:reason='time_cap';break
            ids=torch.randint(len(xf),(64,),device='cuda');head.train();optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.smooth_l1_loss(head(xf[ids]),yy[ids],beta=.2);loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
            if step%100==0:
                head.eval()
                with torch.inference_mode():pred=head(xv).cpu().numpy()*std+mean
                metric=errors(pred,y[validation]);score=metric['translation_p50_p95_mm'][1]+.25*metric['orientation_p50_p95_deg'][1]
                history.append({'update':step,'score':score,'metrics':metric})
                if score<best*.995:
                    best=score;last_best=step;torch.save({'model':head.state_dict(),'mean':mean,'std':std,'arm':arm,'plan':plan},a.output/f'{arm}_best.pt')
                if step>=2000 and step-last_best>=1500:reason='validation_plateau';break
        checkpoint=torch.load(a.output/f'{arm}_best.pt',map_location='cpu',weights_only=False);head.load_state_dict(checkpoint['model']);head.eval()
        with torch.inference_mode():pred=head(xv).cpu().numpy()*std+mean
        np.save(a.output/f'{arm}_heldout_predictions.npy',pred)
        summary[arm]={'updates_completed':step,'best_update':last_best,'stop_reason':reason,'metrics':errors(pred,y[validation])}
        (a.output/f'{arm}_history.json').write_text(json.dumps(history,indent=2)+'\n')
    summary['constant_mean']=errors(np.broadcast_to(mean,y[validation].shape),y[validation])
    crop=summary['crop']['metrics'];base=summary['full']['metrics']
    summary['passed']=(crop['translation_p50_p95_mm'][1]<=.4 and crop['orientation_p50_p95_deg'][1]<=1 and crop['translation_p50_p95_mm'][1]<=base['translation_p50_p95_mm'][1]*.75)
    summary['train_episodes']=plan['train_episodes'];summary['heldout_episodes']=plan['validation_episodes'];summary['geometry_actor_input']=False;summary['final_scenes_opened']=False
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
