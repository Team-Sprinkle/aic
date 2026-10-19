#!/usr/bin/env python3
"""Fine-tune the selected-port standoff probe on recorded actor states.

Four complete development scenes are adaptation data; two different five-card
scenes are held out. No teacher geometry enters the autonomous runtime.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn

from train_sc_transport_goal_probe import GoalProbe, labels


SCALE=np.asarray([.2,.2,.2,.3,.3,.3],dtype=np.float32)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(cache,base):
    rows=[json.loads(s) for s in (cache/'rows.jsonl').open()]
    features=np.load(cache/'features.npy',mmap_mode='r')
    state=np.asarray([r['state'] for r in rows],dtype=np.float32)
    state=(state-base['model']['state_mean'].numpy())/base['model']['state_std'].numpy()
    state[:,base.get('masked_state_indices',[])]=0
    task=np.asarray([r['task'] for r in rows],dtype=np.float32)
    return rows,np.concatenate((np.asarray(features),state,task),axis=1).astype(np.float32)


def alternate(rows,selected,deltas,onpolicy=False):
    result=selected.copy()
    if not onpolicy:
        goals={}
        for r in rows:
            if r['family']=='sc' and r['stage']==5:
                tcp=Rotation.from_quat(r['state'][3:7])
                goals[r['episode']]=np.asarray(r['state'][:3])+tcp.apply(r['subgoal'][:3])
    for i,r in enumerate(rows):
        if not onpolicy and r['family']!='sc':continue
        spec=deltas[str(r['episode'])] if not onpolicy else deltas[r['case']]
        selected_port=(int(np.argmax(r['task'][2:4])) if not onpolicy else r['selected_port'])
        if selected_port!=spec['selected_port']:raise ValueError('Wrong selected-port metadata')
        delta=np.asarray(spec['port0_minus_port1_standoff_base_m'])
        goal=(goals[r['episode']] if not onpolicy else np.asarray(r['goal_base_position']))
        other=goal+(-delta if selected_port==0 else delta)
        tcp=Rotation.from_quat(r['state'][3:7])
        result[i,:3]=tcp.inv().apply(other-np.asarray(r['state'][:3]))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--teacher-cache',type=Path,required=True)
    p.add_argument('--onpolicy-cache',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--alternate-goals',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=3000)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing run')
    a.output.mkdir(parents=True)
    torch.set_num_threads(8);random.seed(20261001);np.random.seed(20261001);torch.manual_seed(20261001)
    base=torch.load(a.base,map_location='cpu',weights_only=False)
    tr,xtr=arrays(a.teacher_cache,base)
    op,xop=arrays(a.onpolicy_cache,base)
    if xtr.shape[1]!=1578 or xop.shape[1]!=1578:raise ValueError('Feature dimension mismatch')
    ytr=labels(tr)
    yop=np.asarray([r['goal_relative'] for r in op],dtype=np.float32)
    specs=json.loads(a.alternate_goals.read_text())['episodes']
    by_case={v['case']:v for v in specs.values()}
    atr=alternate(tr,ytr,specs);aop=alternate(op,yop,by_case,onpolicy=True)
    device='cuda'
    xtr=torch.as_tensor(xtr,device=device);xop=torch.as_tensor(xop,device=device)
    ytr=torch.as_tensor(ytr/SCALE,device=device);yop=torch.as_tensor(yop/SCALE,device=device)
    atr=torch.as_tensor(atr/SCALE,device=device);aop=torch.as_tensor(aop/SCALE,device=device)
    teacher_train={s:np.asarray([i for i,r in enumerate(tr) if r['family']=='sc' and
        r['split']=='train' and r['stage']==s]) for s in range(6)}
    teacher_val=np.asarray([i for i,r in enumerate(tr) if r['family']=='sc' and r['split']=='validation'])
    ontrain=np.asarray([i for i,r in enumerate(op) if r['split']=='adapt_train'])
    onval=np.asarray([i for i,r in enumerate(op) if r['split']=='adapt_validation'])
    onnear=ontrain[np.linalg.norm(np.asarray([op[int(i)]['goal_relative'][:3] for i in ontrain]),axis=1)<.08]
    if not len(onval) or len(onnear)<20:raise ValueError('Insufficient grouped on-policy data')
    model=GoalProbe().cuda()
    initial=torch.load(a.initial,map_location='cpu',weights_only=False)
    if not initial.get('counterfactual_port_augmentation'):
        raise ValueError('Expected selected-port contrastive initialization')
    model.load_state_dict(initial['model'])
    opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=1e-4)
    def eval_group(x,y,indices):
        model.eval();parts=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for batch in np.array_split(indices,max(1,(len(indices)+511)//512)):
                parts.append(model(x[torch.as_tensor(batch,device=device)]).float().cpu().numpy())
        pred=np.concatenate(parts)*SCALE;truth=y[torch.as_tensor(indices,device=device)].cpu().numpy()*SCALE
        err=np.linalg.norm(pred[:,:3]-truth[:,:3],axis=1)*1000
        truth_dist=np.linalg.norm(truth[:,:3],axis=1)*1000
        near=truth_dist<50
        return {'rows':len(indices),'near50_rows':int(near.sum()),
            'translation_p95_mm':float(np.quantile(err,.95)),
            'near50_translation_p95_mm':float(np.quantile(err[near],.95)) if near.any() else None,
            'near50_translation_median_mm':float(np.median(err[near])) if near.any() else None}
    def port_flip():
        xx=xop[torch.as_tensor(onval,device=device)].clone()
        yy=xx.clone();yy[:,-8:-6]=xx[:,-8:-6].flip(-1)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            p=model(xx).float();q=model(yy).float()
        mm=(p[:,:3]-q[:,:3]).norm(dim=-1).cpu().numpy()*200
        return float(np.median(mm))
    def report(update,loss):
        t=eval_group(xtr,ytr,teacher_val);o=eval_group(xop,yop,onval)
        return {'update':update,'train_loss':loss,'teacher_validation':t,
            'onpolicy_validation':o,'onpolicy_port_flip_median_mm':port_flip()}
    history=[report(0,None)];print(json.dumps(history[-1]),flush=True)
    best=float('inf');best_update=0;bad=0
    for update in range(1,a.updates+1):
        ti=np.r_[*[np.random.choice(teacher_train[s],4) for s in range(6)]]
        oi=np.r_[np.random.choice(ontrain,16),np.random.choice(onnear,24)]
        tx=xtr[torch.as_tensor(ti,device=device)].clone();ty=ytr[torch.as_tensor(ti,device=device)].clone()
        ox=xop[torch.as_tensor(oi,device=device)].clone();oy=yop[torch.as_tensor(oi,device=device)].clone()
        tf=torch.rand(len(ti),device=device)<.5;of=torch.rand(len(oi),device=device)<.5
        tx[tf,-8:-6]=tx[tf,-8:-6].flip(-1);ty[tf]=atr[torch.as_tensor(ti[tf.cpu().numpy()],device=device)]
        ox[of,-8:-6]=ox[of,-8:-6].flip(-1);oy[of]=aop[torch.as_tensor(oi[of.cpu().numpy()],device=device)]
        x=torch.cat((tx,ox));y=torch.cat((ty,oy))
        model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model(x);loss=nn.functional.smooth_l1_loss(pred,y,beta=.05)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if update%250==0 or update==a.updates:
            item=report(update,float(loss.detach()));history.append(item)
            print(json.dumps(item),flush=True)
            o=item['onpolicy_validation'];t=item['teacher_validation']
            score=o['near50_translation_p95_mm']+.3*t['near50_translation_p95_mm']
            eligible=(t['near50_translation_p95_mm']<=8 and
                      item['onpolicy_port_flip_median_mm']>=60)
            if eligible and score<best-.05:
                best=score;best_update=update;bad=0
                torch.save({'model':model.state_dict(),'update':update,
                    'counterfactual_port_augmentation':True,'onpolicy_adaptation':True,
                    'four_group_adaptation':True,'final_scenes_opened':False},a.output/'best.pt')
            else:bad+=1
            if update>=1000 and bad>=6:break
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    manifest={'schema':'aic_sc_onpolicy_goal_adaptation/v1',
        'teacher_cache_manifest_sha256':sha(a.teacher_cache/'manifest.json'),
        'onpolicy_cache_manifest_sha256':sha(a.onpolicy_cache/'manifest.json'),
        'initial_checkpoint_sha256':sha(a.initial),'alternate_labels_sha256':sha(a.alternate_goals),
        'adapt_train_cases':list({r['case'] for r in op if r['split']=='adapt_train'}),
        'adapt_validation_cases':list({r['case'] for r in op if r['split']=='adapt_validation'}),
        'update_cap':a.updates,'selected_update':best_update,'final_scenes_opened':False}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps({'best_update':best_update,
        'updates_completed':update,'stop_reason':'validation_plateau' if update<a.updates else 'update_cap',
        'baseline':history[0],
        'selected':next((v for v in history if v['update']==best_update),None),
        'final_scenes_opened':False},indent=2)+'\n')


if __name__=='__main__':main()
