#!/usr/bin/env python3
"""Native SC/SFP opening refinement with frozen RGB crop selection and grouped validation."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from train_sc_native_keypoints import PretrainedKeypointRefiner
from train_two_family_port_locator import Locator, NAMES

CAMS=('center','left','right')


def code(row,identity,camera):
    result=np.zeros(13,np.float32);result[0 if row['family']=='sfp' else 1]=1
    result[2+int(identity.rsplit('_',1)[1])]=1
    if row['family']=='sfp':result[4+int(identity.split('_')[2])]=1;result[9]=1
    result[10+CAMS.index(camera)]=1
    return result


class JointRefiner(PretrainedKeypointRefiner):
    def __init__(self,imagenet=None):
        super().__init__(imagenet,False,point_count=1)
        self.decoder[0]=nn.Conv2d(64+256+13,128,3,padding=1)

    def forward(self,image,task):
        x=(image-self.mean)/self.std
        x=self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        early=self.backbone.layer1(x);deep=self.backbone.layer3(self.backbone.layer2(early))
        z=torch.cat((early,F.interpolate(deep,size=early.shape[-2:],mode='bilinear',align_corners=False),
                     task[:,:,None,None].expand(-1,-1,*early.shape[-2:])),1)
        heat=self.decoder(z).flatten(1).float();prob=(heat*4).softmax(-1)
        side=early.shape[-1];y,x=torch.meshgrid(torch.linspace(0,1,side,device=image.device),torch.linspace(0,1,side,device=image.device),indexing='ij')
        return torch.stack(((prob*x.flatten()).sum(-1),(prob*y.flatten()).sum(-1)),-1)


def prepare(a):
    if (a.output/'cache_summary.json').exists():return
    rows=[json.loads(l) for l in a.labels.open()]
    model=Locator(None).cuda().eval();model.load_state_dict(torch.load(a.coarse,map_location='cpu',weights_only=False)['model'])
    cache=np.lib.format.open_memmap(a.output/'native_crops_192.npy',mode='w+',dtype=np.uint8,shape=(len(rows)*6,192,192,3))
    spatial_cache=None;spatial_value=[];context=None
    if a.cache_coarse_spatial:
        spatial_cache=np.lib.format.open_memmap(a.output/'coarse_spatial_8x9.npy',mode='w+',dtype=np.float16,shape=(len(rows),3,256,8,9))
        model.backbone.layer3.register_forward_hook(lambda module,inputs,output:spatial_value.__setitem__(slice(None),[output]))
        context=(a.output/'frame_context.jsonl').open('w')
    metadata=[];reject=Counter();coarse_error=defaultdict(list);started=time.monotonic()
    with torch.inference_mode():
        for ri,r in enumerate(rows):
            rgb=[];meta=np.zeros((3,5),np.float32);meta[:,0 if r['family']=='sc' else 1]=1
            for ci,c in enumerate(CAMS):
                im=cv2.imread(r['images'][c]);assert im is not None and im.shape[:2]==(1024,1152)
                rgb.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));meta[ci,2+ci]=1
            small=torch.from_numpy(np.stack([cv2.resize(im,(288,256),interpolation=cv2.INTER_AREA) for im in rgb])).permute(0,3,1,2).cuda().float()/255
            points,*_=model(small,torch.from_numpy(meta).cuda());points=points.float().cpu().numpy()*[1151,1023]
            if spatial_cache is not None:
                spatial_cache[ri]=F.adaptive_avg_pool2d(spatial_value[0],(8,9)).half().cpu().numpy()
                context.write(json.dumps(dict(r,feature_row=ri,coarse_pixels_all_slots=points.tolist(),
                                              coarse_inframe_probabilities=_[0].sigmoid().float().cpu().tolist()),separators=(',',':'))+'\n')
            selected=r['selected_identity'];sibling=selected.rsplit('_',1)[0]+'_'+str(1-int(selected.rsplit('_',1)[1]))
            wanted=[t for t in r['targets'] if t['identity'] in (selected,sibling)]
            for t in wanted:
                for ci,c in enumerate(CAMS):
                    gt=t['cameras'][c]
                    if not gt[3]:reject['projected_offscreen']+=1;continue
                    center=points[ci,NAMES.index(t['identity'])];truth=np.asarray(gt[:2]);inside=bool(np.max(np.abs(truth-center))<=64)
                    coarse_error[f"{r['family']}/{r['split']}"] .append(float(np.linalg.norm(truth-center)))
                    if r['split']=='train' and not inside:reject['train_target_outside_rgb_crop_margin']+=1;continue
                    origin=np.round(center-96).astype(int)
                    crop=cv2.warpAffine(rgb[ci],np.array([[1,0,-origin[0]],[0,1,-origin[1]]],np.float32),(192,192))
                    idx=len(metadata);cache[idx]=crop
                    metadata.append(dict(index=idx,family=r['family'],episode=r['episode'],scene_group=r['scene_group'],split=r['split'],
                        camera=c,identity=t['identity'],selected=t['identity']==selected,cards=r['card_count'],time_s=r['time_s'],
                        state=r['state'][:7],target_position_base_m=t['position_base_m'],target_orientation_base_xyzw=t['orientation_base_xyzw'],
                        task=code(r,t['identity'],c).tolist(),truth_local_192=(truth-origin).tolist(),inside_margin=inside,
                        source_image=r['images'][c],coarse_xy=center.tolist(),truth_xy=truth.tolist()))
            if ri%250==0:
                print(json.dumps({'stage':'materialize','rows':ri,'total_rows':len(rows),'crops':len(metadata),'elapsed_s':time.monotonic()-started}),flush=True)
    cache.flush();del cache
    if spatial_cache is not None:spatial_cache.flush();context.close()
    with (a.output/'samples.jsonl').open('w') as f:
        for m in metadata:f.write(json.dumps(m,separators=(',',':'))+'\n')
    by=Counter(f"{m['family']}/{m['split']}" for m in metadata)
    report=dict(crops=len(metadata),counts=dict(by),rejected=dict(reject),native_crop_selection='frozen_RGB_locator_only',projection_is_not_visibility=True,
                coarse_error={k:{'n':len(v),'p50':float(np.median(v)),'p95':float(np.percentile(v,95))} for k,v in coarse_error.items()})
    (a.output/'cache_summary.json').write_text(json.dumps(report,indent=2)+'\n')


class Crops(Dataset):
    def __init__(self,root,samples,training):
        self.array=np.load(root/'native_crops_192.npy',mmap_mode='r');self.samples=samples;self.training=training
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        m=self.samples[i];offset=np.random.randint(-12,13,2) if self.training else np.zeros(2,int);start=16+offset
        image=np.array(self.array[m['index'],start[1]:start[1]+160,start[0]:start[0]+160]).copy()
        target=(np.asarray(m['truth_local_192'])-start)/159
        return torch.from_numpy(image).permute(2,0,1),torch.tensor(m['task']),torch.tensor(target,dtype=torch.float32),i


@torch.inference_mode()
def evaluate(model,loader,baseline=False,prediction_path=None):
    model.eval();errors=defaultdict(list)
    writer=prediction_path.open('w') if prediction_path else None
    for image,code_,target,indices in loader:
        image=image.cuda(non_blocking=True).float()/255
        pred=model(image) if baseline else model(image,code_.cuda(non_blocking=True))
        error=torch.linalg.vector_norm((pred.float().cpu()-target)*159,dim=1).numpy()
        for i,e in zip(indices.tolist(),error):
            m=loader.dataset.samples[i];errors[f"{m['family']}/all"].append(float(e))
            if m['inside_margin']:errors[f"{m['family']}/inside"].append(float(e))
            if m['selected']:errors[f"{m['family']}/selected"].append(float(e))
        if writer:
            for j,i in enumerate(indices.tolist()):
                m=loader.dataset.samples[i]
                origin=np.asarray(m['truth_xy'])-m['truth_local_192']+16
                predicted=origin+pred[j].float().cpu().numpy()*159
                writer.write(json.dumps(dict(m,predicted_xy=predicted.tolist(),truth_for_scoring_only=True))+'\n')
    if writer:writer.close()
    return {k:{'n':len(v),'p50_px':float(np.median(v)),'p95_px':float(np.percentile(v,95))} for k,v in errors.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--coarse',type=Path,required=True)
    p.add_argument('--imagenet',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=12000);p.add_argument('--minutes',type=float,default=240)
    p.add_argument('--batch-size',type=int,default=96);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--old-sc-refiner',type=Path,required=True)
    p.add_argument('--cache-coarse-spatial',action='store_true',help='Reuse frozen RGB features for the subsequent coarse association audit')
    p.add_argument('--cache-root',type=Path,help='Reuse completed immutable native crops without rematerializing')
    p.add_argument('--init-checkpoint',type=Path,help='Continue from saved model weights; older exports do not retain optimizer state')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'history.jsonl').exists():p.error('Do not accidentally overwrite training; use a new run')
    torch.set_num_threads(6);cv2.setNumThreads(0);torch.manual_seed(20261015);np.random.seed(20261015);random.seed(20261015)
    (a.output/'command_config.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},indent=2)+'\n')
    cache_root=a.cache_root or a.output
    if a.cache_root:
        if not (cache_root/'cache_summary.json').exists():p.error('Shared cache must be complete')
    else:prepare(a)
    samples=[json.loads(l) for l in (cache_root/'samples.jsonl').open()]
    train=[m for m in samples if m['split']=='train'];val=[m for m in samples if m['split']=='validation']
    assert {m['scene_group'] for m in train}.isdisjoint({m['scene_group'] for m in val})
    assert {m['family'] for m in train}=={'sc','sfp'} and {m['family'] for m in val}=={'sc','sfp'}
    # Checkpoints use a fixed scene-balanced validation screen; report every held-out view at completion.
    per=defaultdict(list)
    for m in val:per[(m['family'],m['episode'],m['camera'])].append(m)
    screen=[v[i] for v in per.values() for i in np.linspace(0,len(v)-1,min(4,len(v)),dtype=int)]
    counts=Counter((m['family'],m['episode'],m['identity'],m['camera']) for m in train)
    cells=Counter(key[0] for key in counts)
    weights=[1/counts[(m['family'],m['episode'],m['identity'],m['camera'])]/cells[m['family']] for m in train]
    sampler=WeightedRandomSampler(weights,a.updates*a.batch_size,replacement=True)
    opts=dict(num_workers=a.workers,persistent_workers=a.workers>0,pin_memory=True)
    loader=DataLoader(Crops(cache_root,train,True),batch_size=a.batch_size,sampler=sampler,**opts)
    screen_loader=DataLoader(Crops(cache_root,screen,False),batch_size=192,**opts)
    full_loader=DataLoader(Crops(cache_root,val,False),batch_size=192,**opts)
    old=PretrainedKeypointRefiner(None,False,point_count=1).cuda();old.load_state_dict(torch.load(a.old_sc_refiner,map_location='cpu',weights_only=False)['model'])
    sc_loader=DataLoader(Crops(cache_root,[m for m in screen if m['family']=='sc'],False),batch_size=192,**opts)
    (a.output/'old_SC_refiner_matched_screen.json').write_text(json.dumps(evaluate(old,sc_loader,True),indent=2)+'\n');del old
    model=JointRefiner(a.imagenet).cuda();opt=torch.optim.AdamW([w for w in model.parameters() if w.requires_grad],lr=2e-4,weight_decay=1e-4)
    initial_step=0
    if a.init_checkpoint:
        previous=torch.load(a.init_checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(previous['model']);initial_step=previous['step']
    best=float('inf');stale=0;best_step=0;started=time.monotonic();reason='update_cap';minimum=2000 if a.init_checkpoint else max(2000,int(np.ceil(3*len(train)/a.batch_size)))
    for step,(image,task,target,_) in enumerate(loader,1):
        if time.monotonic()-started>a.minutes*60:reason='time_cap_not_proven_plateau';break
        model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(image.cuda(non_blocking=True).float()/255,task.cuda(non_blocking=True))
        loss=F.smooth_l1_loss(pred.float(),target.cuda(non_blocking=True),beta=.01)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%250==0:
            report=evaluate(model,screen_loader);score=max(report[f'{family}/inside']['p95_px'] for family in ('sc','sfp'))
            record={'step':step,'cumulative_step':initial_step+step,'loss':float(loss.detach()),'screen':report,'selection_max_family_inside_p95_px':score,'elapsed_s':time.monotonic()-started}
            with (a.output/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            saved={'model':model.state_dict(),'schema':'joint_native_opening_refiner/v1','condition_code':'family2_port2_card5_valid1_camera3','crop_size':160,'runtime_geometry_input':False,'step':initial_step+step}
            torch.save(saved,a.output/'last.pt')
            if score<best-.02:best=score;best_step=step;stale=0;torch.save(saved,a.output/'best.pt')
            else:stale+=1
            if step>=minimum and stale>=10:reason='validation_plateau';break
    model.load_state_dict(torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)['model'])
    full=evaluate(model,full_loader,prediction_path=a.output/'validation_predictions.jsonl')
    summary={'status':'trained_not_promoted','best_step':best_step,'updates_completed':step,'stop_reason':reason,'minimum_updates':minimum,
             'initial_step':initial_step,'cumulative_updates_completed':initial_step+step,
             'weight_continuation_with_fresh_optimizer':bool(a.init_checkpoint),'cache_root':str(cache_root),
             'train_views':len(train),'validation_views':len(val),'selection_views':len(screen),'full_validation':full,
             'episodes':{f'{fam}/{split}':len({m['episode'] for m in samples if m['family']==fam and m['split']==split}) for fam in ('sc','sfp') for split in ('train','validation')},
             'runtime_crop_from_frozen_RGB_only':True,'true_visibility_certified':False,'final_scenes_opened':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
