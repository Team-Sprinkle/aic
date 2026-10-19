#!/usr/bin/env python3
"""Bounded, task-conditioned SFP keypoint experiment on original wide RGB.

Geometry is read for labels and offline scoring only. Inference receives image,
camera name, family bit and requested card/port identity; triangulation receives
measured TCP and fixed camera calibration, never the true opening.
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate, reprojection_residual_px
from train_target_conditioned_port_locator import ConditionalLocator
from train_two_family_port_locator import CAMERAS, LOOKUP


class BalancedViews(Dataset):
    def __init__(self, rows, indices, training):
        self.rows=rows;self.indices=indices;self.training=training
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        row_idx, cam_idx=self.indices[i];r=self.rows[row_idx]
        if cam_idx is None:cam_idx=random.randrange(3)
        cam=CAMERAS[cam_idx]
        image=cv2.imread(r['images'][cam],cv2.IMREAD_COLOR)
        if image is None:raise FileNotFoundError(r['images'][cam])
        image=cv2.cvtColor(cv2.resize(image,(576,512),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        if self.training:
            target=random.choice(r['targets'])
        else:
            target=next(t for t in r['targets'] if t['identity']==r['selected_identity'])
        proj=target['cameras'][cam]
        point=[proj[0]/1151,proj[1]/1023] if proj[3] else [0.,0.]
        meta=np.zeros(5,np.float32);meta[1]=1;meta[2+cam_idx]=1
        return (torch.from_numpy(image.copy()).permute(2,0,1),torch.from_numpy(meta),
                LOOKUP[target['identity']],torch.tensor(point,dtype=torch.float32),
                float(proj[3]),row_idx,cam_idx)


@torch.inference_mode()
def evaluate(model,rows,loader,device,optics,save_predictions=None):
    model.eval();by_row=defaultdict(dict);pixel=[];inframe=[]
    for image,meta,target,point,visible,row_idx,cam_idx in loader:
        xy,logit,_=model(image.to(device).float()/255.,meta.to(device),target.to(device))
        xy=xy.float().cpu().numpy();prob=logit.float().sigmoid().cpu().numpy()
        for i in range(len(xy)):
            j=int(row_idx[i]);cam=CAMERAS[int(cam_idx[i])]
            pixels=(xy[i]*[1151,1023]).tolist()
            by_row[j][cam]={'xy':pixels,'prob':float(prob[i]),'label_xy':(point[i].numpy()*[1151,1023]).tolist(),
                            'label_inframe':bool(visible[i])}
            inframe.append((float(prob[i]),bool(visible[i])))
            if visible[i]:pixel.append(float(np.linalg.norm(np.asarray(pixels)-np.asarray(by_row[j][cam]['label_xy']))))
    errors=[];wrong=0;accepted=0;total=0;valid=0;predictions=[]
    for j,views in sorted(by_row.items()):
        r=rows[j];selected=next(t for t in r['targets'] if t['identity']==r['selected_identity'])
        cameras={c:pose_matrix(r['state'])@optics[c] for c in CAMERAS}
        requested={c:views[c]['xy'] for c in CAMERAS if c in views and views[c]['prob']>=.5}
        record={'episode':r['episode'],'scene_group':r['scene_group'],'time_s':r['time_s'],
                'requested':r['selected_identity'],'views':views,'accepted':False,
                'predicted_base_m':None,'error_mm':None,'nearest_identity_label_only':None}
        total+=1
        if len(requested)>=2:
            try:
                pred=triangulate(requested,cameras);reproj=reprojection_residual_px(pred,requested,cameras)
                if np.isfinite(pred).all():
                    err=float(np.linalg.norm(pred-np.asarray(selected['position_base_m']))*1000)
                    record.update(predicted_base_m=pred.tolist(),error_mm=err,reprojection_px=float(reproj))
                    valid+=1
                    if len(requested)==3 and reproj<=5:
                        record['accepted']=True;accepted+=1;errors.append(err)
                        alt=min(r['targets'],key=lambda t:np.linalg.norm(pred-np.asarray(t['position_base_m'])))
                        record['nearest_identity_label_only']=alt['identity']
                        if alt['identity']!=r['selected_identity']:wrong+=1
            except np.linalg.LinAlgError:pass
        predictions.append(record)
    report={'rows':total,'valid_3d':valid,'accepted':accepted,'accept_rate':accepted/max(total,1),
            'accepted_wrong_identity':wrong,'accepted_3d_mm_p50':float(np.median(errors)) if errors else None,
            'accepted_3d_mm_p95':float(np.quantile(errors,.95)) if errors else None,
            'pixel_px_p50':float(np.median(pixel)) if pixel else None,
            'pixel_px_p95':float(np.quantile(pixel,.95)) if pixel else None,
            'inframe_accuracy':float(np.mean([(p>=.5)==y for p,y in inframe])) if inframe else None}
    if save_predictions is not None:
        with save_predictions.open('w') as f:
            for record in predictions:f.write(json.dumps(record,separators=(',',':'))+'\n')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--init',type=Path,required=True)
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--minutes',type=float,default=120)
    p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args()
    if a.output.exists():p.error('Output exists')
    if a.updates>2000:p.error('Bounded gate allows at most 2000 updates')
    a.output.mkdir(parents=True)
    rows=[json.loads(s) for s in (a.labels/'rows.jsonl').open()]
    train_idx=[(i,None) for i,r in enumerate(rows) if r['split']=='train']
    validation_idx=[(i,c) for i,r in enumerate(rows) if r['split']=='validation' for c in range(3)]
    if {rows[i]['scene_group'] for i,_ in train_idx}&{rows[i]['scene_group'] for i,_ in validation_idx}:
        raise ValueError('Reset overlap')
    if len({rows[i]['episode_index'] for i,_ in train_idx})!=50 or len({rows[i]['episode_index'] for i,_ in validation_idx})!=10:
        raise ValueError('Unexpected episode split')
    seed=20261003;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(8)
    calibration=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics={c:np.asarray(calibration['tcp_to_optical'][c],float) for c in CAMERAS}
    train_data=BalancedViews(rows,train_idx,True);val_data=BalancedViews(rows,validation_idx,False)
    sampler=torch.utils.data.RandomSampler(train_data,replacement=True,num_samples=a.batch_size*a.updates)
    loader=DataLoader(train_data,batch_size=a.batch_size,sampler=sampler,num_workers=4,pin_memory=True,persistent_workers=True)
    val_loader=DataLoader(val_data,batch_size=a.batch_size,num_workers=4,pin_memory=True,persistent_workers=True)
    model=ConditionalLocator(None).cuda()
    initial=torch.load(a.init,map_location='cpu',weights_only=False)
    model.load_state_dict(initial['model'])
    optimizer=torch.optim.AdamW((v for v in model.parameters() if v.requires_grad),lr=8e-5,weight_decay=1e-4)
    history=[];best=float('inf');best_step=0;start=time.monotonic();reason='update_cap';completed=0
    for step,batch in enumerate(loader,1):
        if step>a.updates:break
        if time.monotonic()-start>a.minutes*60:reason='wall_budget';break
        image,meta,target,point,inframe,*_=batch
        image=image.cuda(non_blocking=True).float()/255.;meta=meta.cuda(non_blocking=True)
        target=target.cuda(non_blocking=True);point=point.cuda(non_blocking=True)
        inframe=inframe.cuda(non_blocking=True)
        model.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): pred,flag,heat=model(image,meta,target)
        h,w=heat.shape[-2:]
        cell=(point[:,1]*(h-1)).round().long().clamp(0,h-1)*w+(point[:,0]*(w-1)).round().long().clamp(0,w-1)
        pixel=F.smooth_l1_loss(pred.float(),point,beta=.01,reduction='none').sum(-1)
        spatial=F.cross_entropy(heat.float().flatten(1)*4,cell,reduction='none')
        loss=((pixel+.02*spatial)*inframe).sum()/inframe.sum().clamp_min(1)
        loss+=.1*F.binary_cross_entropy_with_logits(flag.float(),inframe)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();completed=step
        if step%200==0:
            report=evaluate(model,rows,val_loader,'cuda',optics)
            score=(report['accepted_3d_mm_p95'] if report['accepted_3d_mm_p95'] is not None else 1e6)
            score+=report['accepted_wrong_identity']*100+100*(1-report['accept_rate'])
            history.append({'step':step,'loss':float(loss.detach()),'score':score,'validation':report})
            print(json.dumps(history[-1]),flush=True)
            if score<best:
                best=score;best_step=step
                torch.save({'model':model.state_dict(),'step':step,'architecture':'conditional_resnet18_576x512',
                            'training_geometry_input':False,'runtime_geometry_input':False},a.output/'best.pt')
    if not best_step:raise RuntimeError('No checkpoint selected')
    model.load_state_dict(torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)['model'])
    final=evaluate(model,rows,val_loader,'cuda',optics,a.output/'development_predictions.jsonl')
    summary={'schema':'sfp_balanced_anchor_probe/v1','train_episodes':50,'development_episodes':10,
             'rows_train':len(train_idx),'rows_development':len(validation_idx)//3,
             'best_step':best_step,'updates_completed':completed,'stop_reason':reason,
             'initial_checkpoint':str(a.init),'image_source':'original_1152x1024_jpeg_resized_to_576x512_for_network',
             'alternate_present_ports_sampled_as_training_requests':True,
             'runtime_geometry_input':False,'final_scenes_opened':False,
             'acceptance_rule':'all_3_inframe_probability>=0.5_and_reprojection<=5px',
             'offline_gate_zero_wrong_and_p95_le_20mm':bool(final['accepted_wrong_identity']==0 and
               final['accepted_3d_mm_p95'] is not None and final['accepted_3d_mm_p95']<=20),
             'development':final}
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
