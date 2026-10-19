#!/usr/bin/env python3
"""Bounded native orientation repair: five virtual rigid port-frame keypoints.

All160px crops come from the existing frozen RGB candidate cache. Pose labels
project five fixed port-frame reference points for supervision only. Runtime
uses predicted pixels, measured TCP, camera calibration and known reference
geometry; true port pose is never an inference argument.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler

from train_joint_native_opening_refiner import JointRefiner
from evaluate_sc_native_triangulation import triangulate,FOCAL,reprojection_residual_px
from audit_sc_port_targets import pose_matrix

CAMS=('center','left','right')


def reference_points(family):
    width=.008 if family=='sc' else .005
    return np.array([[0,0,0],[-width,0,0],[width,0,0],[0,-.003,0],[0,.003,0]],float)


class NativePoseKeypoints(JointRefiner):
    def __init__(self,imagenet=None):
        super().__init__(imagenet);self.decoder[-1]=nn.Conv2d(64,5,1)
    def forward(self,image,task):
        x=(image-self.mean)/self.std;b=image.shape[0]
        x=self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        early=self.backbone.layer1(x);deep=self.backbone.layer3(self.backbone.layer2(early))
        z=torch.cat((early,F.interpolate(deep,size=early.shape[-2:],mode='bilinear',align_corners=False),task[:,:,None,None].expand(-1,-1,*early.shape[-2:])),1)
        probability=(self.decoder(z).reshape(b,5,-1).float()*4).softmax(-1);side=early.shape[-1]
        y,x=torch.meshgrid(torch.linspace(0,1,side,device=image.device),torch.linspace(0,1,side,device=image.device),indexing='ij')
        return torch.stack(((probability*x.flatten()).sum(-1),(probability*y.flatten()).sum(-1)),-1)


def projected_labels(samples,optics):
    q=np.asarray([m['state'][3:7] for m in samples]);rt=Rotation.from_quat(q).as_matrix();pt=np.asarray([m['state'][:3] for m in samples])
    rp=Rotation.from_quat([m['target_orientation_base_xyzw'] for m in samples]).as_matrix();pp=np.asarray([m['target_position_base_m'] for m in samples])
    o=np.stack([optics[m['camera']] for m in samples]);ro=rt@o[:,:3,:3];po=pt+np.einsum('nij,nj->ni',rt,o[:,:3,3])
    cad=np.stack([reference_points(m['family']) for m in samples]);world=np.einsum('nij,nkj->nki',rp,cad)+pp[:,None]
    local=np.einsum('nji,nkj->nki',ro,world-po[:,None]);uv=local[...,:2]/local[...,2:]*FOCAL+[576,512]
    discrepancy=np.linalg.norm(uv[:,0]-np.asarray([m['truth_xy'] for m in samples]),axis=-1)
    if discrepancy.max()>.01:raise ValueError(f'Centre projection does not match native labels: max{discrepancy.max()}px')
    return uv.astype(np.float32)


class Crops(Dataset):
    def __init__(self,cache,samples,labels,training):
        self.array=np.load(cache/'native_crops_192.npy',mmap_mode='r');self.samples=samples;self.labels=labels;self.training=training
        self.extra_arrays={}
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        m=self.samples[i];start=16+(np.random.randint(-12,13,2) if self.training else np.zeros(2,int))
        array=self.array
        if m.get('crop_store'):
            if m['crop_store'] not in self.extra_arrays:
                self.extra_arrays[m['crop_store']]=np.load(m['crop_store'],mmap_mode='r')
            array=self.extra_arrays[m['crop_store']]
        image=np.array(array[m['index'],start[1]:start[1]+160,start[0]:start[0]+160]).copy()
        origin=np.asarray(m['truth_xy'])-m['truth_local_192']+start
        target=(self.labels[m['pose_label_index']]-origin)/159
        valid=np.all((target>=0)&(target<=1),axis=-1)
        return torch.from_numpy(image).permute(2,0,1),torch.tensor(m['task']),torch.tensor(target,dtype=torch.float32),torch.tensor(valid),i


def rigid_pose(cad,cloud):
    a=cad-cad.mean(0);b=cloud-cloud.mean(0);u,s,vh=np.linalg.svd(a.T@b)
    if s[1]<1e-10:raise ValueError('Degenerate rigid reference prediction')
    rotation=vh.T@np.diag([1.,1.,np.linalg.det(vh.T@u.T)])@u.T
    position=cloud.mean(0)-rotation@cad.mean(0)
    return position,rotation


def summary_stats(v):
    return dict(n=len(v),p50=float(np.median(v)) if v else None,p95=float(np.percentile(v,95)) if v else None)


@torch.inference_mode()
def evaluate(model,loader,optics,prediction_path=None):
    model.eval();errors=defaultdict(list);groups=defaultdict(dict);writer=prediction_path.open('w') if prediction_path else None
    for image,task,target,mask,indices in loader:
        pred=model(image.cuda().float()/255,task.cuda()).float().cpu().numpy()
        for j,i in enumerate(indices.tolist()):
            m=loader.dataset.samples[i];origin=np.asarray(m['truth_xy'])-m['truth_local_192']+16
            pixels=pred[j]*159+origin;distance=np.linalg.norm((pred[j]-target[j].numpy())*159,axis=-1)
            if m['inside_margin']:errors[m['family']+'/centre_px'].append(float(distance[0]))
            errors[m['family']+'/valid_reference_px'].extend(distance[mask[j].numpy()].tolist())
            if m.get('source_domain') and m['inside_margin']:
                prefix=m['family']+'/'+m['source_domain']
                errors[prefix+'/centre_px'].append(float(distance[0]))
                errors[prefix+'/reference_px'].extend(distance[mask[j].numpy()].tolist())
            if m['selected']:groups[(m['family'],m['episode'],m['time_s'])][m['camera']]=(m,pixels,bool(mask[j].all()))
            if writer:writer.write(json.dumps(dict(m,predicted_keypoints_xy=pixels.tolist(),geometry_access='labels for scoring only'))+'\n')
    if writer:writer.close()
    for key,views in groups.items():
        if len(views)!=3 or not all(v[2] and v[0]['inside_margin'] for v in views.values()):continue
        m=next(iter(views.values()))[0];base_tcp=pose_matrix(m['state']);cameras={c:base_tcp@optics[c] for c in CAMS}
        stamps=[int(Path(v[0]['source_image']).stem.split('_')[-2]) for v in views.values()]
        if max(stamps)-min(stamps)>1_000_000:continue
        try:
            cloud=np.stack([triangulate({c:v[1][point] for c,v in views.items()},cameras) for point in range(5)])
            position,rotation=rigid_pose(reference_points(m['family']),cloud)
        except (ValueError,np.linalg.LinAlgError):continue
        true=Rotation.from_quat(m['target_orientation_base_xyzw']);delta=true.inv().apply(position-m['target_position_base_m'])*1000
        angular=float(np.degrees((true.inv()*Rotation.from_matrix(rotation)).magnitude()))
        family=m['family'];errors[family+'/lateral_mm'].append(float(np.linalg.norm(delta[:2])));errors[family+'/xyz_mm'].append(float(np.linalg.norm(delta)));errors[family+'/orientation_deg'].append(angular)
    return {k:summary_stats(v) for k,v in errors.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--init',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=12000);p.add_argument('--minutes',type=float,default=120)
    p.add_argument('--balance-source-domains',action='store_true')
    p.add_argument('--selection-domain',help='Bounded repair selection; old full validation remains a separate regression gate')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(0);torch.set_num_threads(6);torch.manual_seed(20261015);np.random.seed(20261015)
    optics={c:np.asarray(m) for c,m in json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())['tcp_to_optical'].items()}
    source=[json.loads(l) for l in (a.cache/'samples.jsonl').open()]
    by_frame=defaultdict(list)
    for m in source:by_frame[(m['episode'],m['time_s'])].append(m)
    samples=[];excluded=0
    for group in by_frame.values():
        stamps=[int(Path(m['source_image']).stem.split('_')[-2]) for m in group]
        # A missing/offscreen centre-camera sample can hide an asynchronous
        # triplet if only left/right remain. Metadata state is at time_s.
        if max(stamps)-min(stamps)>1_000_000 or any(abs(s-round(group[0]['time_s']*1e9))>1_000_000 for s in stamps):
            excluded+=len(group);continue
        samples.extend(group)
    for i,m in enumerate(samples):m['pose_label_index']=i
    labels=projected_labels(samples,optics);np.save(a.output/'projected_reference_pixels.npy',labels)
    with (a.output/'manifest.jsonl').open('w') as f:
        for m in samples:f.write(json.dumps(m)+'\n')
    train=[m for m in samples if m['split']=='train'];val=[m for m in samples if m['split']=='validation']
    assert {m['scene_group'] for m in train}.isdisjoint({m['scene_group'] for m in val})
    triplets=defaultdict(list)
    for m in val:
        if m['selected'] and m['inside_margin']:triplets[(m['family'],m['episode'],m['time_s'])].append(m)
    episodes=defaultdict(list)
    for key,group in triplets.items():
        if len(group)==3:episodes[key[:2]].append(group)
    screen=[]
    for groups in episodes.values():
        groups.sort(key=lambda g:g[0]['time_s'])
        for i in np.linspace(0,len(groups)-1,min(6,len(groups)),dtype=int):screen.extend(groups[i])
    count=Counter((m['family'],m['episode'],m['identity'],m['camera']) for m in train);cells=Counter(k[0] for k in count)
    domains=Counter((m['family'],m.get('source_domain','original')) for m in train)
    episode_domain={(m['family'],m['episode']):m.get('source_domain','original') for m in train}
    domain_cells=Counter((f,episode_domain[f,e]) for f,e,_,_ in count) if a.balance_source_domains else None
    family_domains=Counter(f for f,_ in domains)
    weights=[1/count[(m['family'],m['episode'],m['identity'],m['camera'])]/(
        domain_cells[(m['family'],m.get('source_domain','original'))]*family_domains[m['family']]
        if a.balance_source_domains else cells[m['family']]) for m in train]
    opts=dict(num_workers=4,persistent_workers=True,pin_memory=True)
    loader=DataLoader(Crops(a.cache,train,labels,True),batch_size=96,sampler=WeightedRandomSampler(weights,a.updates*96,replacement=True),**opts)
    screen_loader=DataLoader(Crops(a.cache,screen,labels,False),batch_size=192,**opts)
    full_loader=DataLoader(Crops(a.cache,val,labels,False),batch_size=192,**opts)
    model=NativePoseKeypoints().cuda();saved=torch.load(a.init,map_location='cpu',weights_only=False);state=saved['model']
    initial_updates=saved.get('cumulative_step',saved.get('step',0))
    if state['decoder.4.weight'].shape[0]==1:
        state['decoder.4.weight']=state['decoder.4.weight'].repeat(5,1,1,1);state['decoder.4.bias']=state['decoder.4.bias'].repeat(5)
    elif state['decoder.4.weight'].shape[0]!=5:raise ValueError('Expected one or five native landmarks')
    model.load_state_dict(state);opt=torch.optim.AdamW([w for w in model.parameters() if w.requires_grad],lr=1e-4,weight_decay=1e-4)
    start=time.monotonic();best=float('inf');best_step=0;stale=0;reason='update_cap'
    for step,(image,task,target,valid,_) in enumerate(loader,1):
        if time.monotonic()-start>a.minutes*60:reason='time_cap_not_proven_plateau';break
        model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(image.cuda().float()/255,task.cuda())
        loss=F.smooth_l1_loss(pred,target.cuda(),reduction='none',beta=.01);mask=valid.cuda()[...,None]
        loss=(loss*mask).sum()/(mask.sum()*2).clamp_min(1);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%250==0:
            report=evaluate(model,screen_loader,optics)
            needed=[(f,k,scale) for f in ('sc','sfp') for k,scale in [('centre_px',1.5),('lateral_mm',.5),('orientation_deg',1.)]]
            if any(f+'/'+k not in report for f,k,_ in needed):raise ValueError('Missing family/pose validation coverage')
            score=max(report[f+'/'+k]['p95']/scale for f,k,scale in needed)
            if a.selection_domain:
                keys=[f'sfp/{a.selection_domain}/'+k for k in ('centre_px','reference_px')]
                if any(k not in report for k in keys):raise ValueError('Repair domain missing from grouped validation')
                score=max(report[k]['p95']/1.5 for k in keys)
            record=dict(step=step,loss=float(loss.detach()),screen=report,score=score,elapsed_s=time.monotonic()-start)
            with (a.output/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            checkpoint=dict(schema='native_requested_pose_keypoints/v1',model=model.state_dict(),step=step,cumulative_step=initial_updates+step,optimizer_continued=False,geometry_input=False,reference_points={f:reference_points(f).tolist() for f in ('sc','sfp')},condition_code='family2_port2_card5_valid1_camera3',cache=str(a.cache),init=str(a.init),balanced_source_domains=a.balance_source_domains,selection_domain=a.selection_domain)
            torch.save(checkpoint,a.output/'last.pt')
            if score<best-.01:best=score;best_step=step;stale=0;torch.save(checkpoint,a.output/'best.pt')
            else:stale+=1
            if step>=4000 and stale>=10:reason='validation_plateau';break
    model.load_state_dict(torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)['model'])
    metrics=evaluate(model,full_loader,optics,prediction_path=a.output/'validation_predictions.jsonl')
    summary=dict(schema='native_requested_pose_keypoints/v1',best_step=best_step,updates_completed=step,stop_reason=reason,full_validation=metrics,
        initial_updates=initial_updates,cumulative_best_step=initial_updates+best_step,
        cumulative_updates=initial_updates+step,optimizer_continued=False,
        training_views=len(train),validation_views=len(val),asynchronous_views_excluded=excluded,
        balanced_source_domains=a.balance_source_domains,selection_domain=a.selection_domain,
        conditional_geometry_accuracy_not_acquisition_rate=True,geometry_prediction_input=False,final_scenes_opened=False,promoted=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':
    from project_gpu_guard import selected_gpu_reservation
    with selected_gpu_reservation():main()
