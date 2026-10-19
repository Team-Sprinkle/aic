#!/usr/bin/env python3
"""Bounded requested-port crop refiner from observation-selected rail proposals."""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
cv2.setNumThreads(1)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset

from audit_sc_port_targets import pose_matrix
from audit_sfp_observed_rail_crop import project as project_native
from evaluate_sc_native_triangulation import triangulate,reprojection_residual_px
from train_target_conditioned_port_locator import ConditionalLocator
from train_two_family_port_locator import CAMERAS,LOOKUP


def crop_native(image,center,size=480,out=320):
    x0=int(round(center[0]))-size//2;y0=int(round(center[1]))-size//2
    x1=x0+size;y1=y0+size
    sx0=max(0,x0);sy0=max(0,y0);sx1=min(image.shape[1],x1);sy1=min(image.shape[0],y1)
    patch=np.zeros((size,size,3),dtype=image.dtype)
    if sx1>sx0 and sy1>sy0:
        patch[sy0-y0:sy1-y0,sx0-x0:sx1-x0]=image[sy0:sy1,sx0:sx1]
    return cv2.cvtColor(cv2.resize(patch,(out,out),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)


class Crops(Dataset):
    def __init__(self,rows,indices,training,hard_negative=False,source_offset_meta=False,
                 history_context=False):
        self.rows=rows;self.indices=indices;self.training=training
        self.hard_negative=hard_negative
        self.source_offset_meta=source_offset_meta
        self.history_context=history_context
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        row=self.rows[self.indices[i][0]];ci=self.indices[i][1]
        if ci is None:ci=random.randrange(3)
        cam=CAMERAS[ci];proposal=row['proposal'];center=proposal['center_px'][cam]
        image=cv2.imread(row['images'][cam])
        if image is None:raise FileNotFoundError(row['images'][cam])
        crop=crop_native(image,center)
        target=next(t for t in row['targets'] if t['identity']==row['requested'])
        uv=target['cameras'][cam]
        xy=[.5+(uv[0]-center[0])/480,.5+(uv[1]-center[1])/480] if uv[3] else [.5,.5]
        inside=bool(uv[3] and 0<=xy[0]<=1 and 0<=xy[1]<=1)
        meta=np.zeros(5,np.float32);meta[1]=1;meta[2+ci]=1
        if self.source_offset_meta:
            requested_card=int(row['requested'].split('_')[2])
            meta[0]=(requested_card-int(proposal['source_card']))/4.
        basic=(torch.from_numpy(crop.copy()).permute(2,0,1),torch.from_numpy(meta),
                LOOKUP[row['requested']],torch.tensor(xy,dtype=torch.float32),float(inside),
                self.indices[i][0],ci)
        if self.history_context:
            basic=basic+(torch.from_numpy(row['history_context'][ci].astype(np.float32)),)
        if not self.hard_negative:return basic
        alternatives=[]
        for other in row['targets']:
            if other['identity']==row['requested']:continue
            point=other['cameras'][cam]
            if not point[3]:continue
            other_xy=np.asarray([.5+(point[0]-center[0])/480,
                                 .5+(point[1]-center[1])/480],np.float32)
            if np.all((other_xy>=0)&(other_xy<=1)):
                alternatives.append((float(np.linalg.norm(other_xy-np.asarray(xy))),other_xy))
        if alternatives:
            nearest=min(alternatives,key=lambda item:item[0])[1]
            return basic+(torch.from_numpy(nearest),1.)
        return basic+(torch.tensor([.5,.5],dtype=torch.float32),0.)


class HistoryConditionalLocator(nn.Module):
    """Frozen native crop detector plus learned distance-history heatmap residual."""
    uses_history=True
    def __init__(self,init):
        super().__init__()
        self.base=ConditionalLocator(None)
        if init is not None:
            self.base.load_state_dict(torch.load(init,map_location='cpu',weights_only=False)['model'])
        for parameter in self.base.parameters():parameter.requires_grad_(False)
        channels=64+256+5+16+128
        self.residual=nn.Sequential(nn.Conv2d(channels,96,3,padding=1),nn.GELU(),
                                    nn.Conv2d(96,1,1))
        self.flag_residual=nn.Sequential(nn.Linear(256+5+16+128,96),nn.GELU(),nn.Linear(96,1))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.flag_residual[-1].weight);nn.init.zeros_(self.flag_residual[-1].bias)
    def train(self,mode=True):
        super().train(mode);self.base.eval();return self
    def forward(self,rgb,meta,target,context):
        with torch.no_grad():
            x=(rgb-self.base.mean)/self.base.std
            x=self.base.backbone.maxpool(self.base.backbone.relu(self.base.backbone.bn1(self.base.backbone.conv1(x))))
            early=self.base.backbone.layer1(x)
            deep=self.base.backbone.layer3(self.base.backbone.layer2(early))
            condition=torch.cat((meta,self.base.identity(target)),-1)
            joined=torch.cat((early,F.interpolate(deep,size=early.shape[-2:],mode='bilinear',align_corners=False),
                              condition[:,:,None,None].expand(-1,-1,*early.shape[-2:])),dim=1)
            old_heat=self.base.decoder(joined).squeeze(1)
            old_flag=self.base.inframe(torch.cat((deep.mean((-1,-2)),condition),-1)).squeeze(-1)
        heat=old_heat+self.residual(torch.cat((joined,context[:,:,None,None].expand(-1,-1,*early.shape[-2:])),dim=1)).squeeze(1)
        flag=old_flag+self.flag_residual(torch.cat((deep.mean((-1,-2)),condition,context),-1)).squeeze(-1)
        b,h,w=heat.shape
        probs=F.softmax(heat.reshape(b,-1).float()*4,dim=-1)
        yy,xx=torch.meshgrid(torch.linspace(0,1,h,device=rgb.device),
                             torch.linspace(0,1,w,device=rgb.device),indexing='ij')
        xy=torch.stack(((probs*xx.flatten()).sum(-1),(probs*yy.flatten()).sum(-1)),-1)
        return xy,flag,heat


@torch.inference_mode()
def evaluate(model,rows,loader,optics,save=None,rail_band_mm=0.,
             rail_band_adjacent_only=False):
    model.eval();predictions=defaultdict(dict);pixel_errors=[]
    for data in loader:
        image,meta,target,truth,inside,ri,ci,*extra=data
        if getattr(model,'uses_history',False):
            xy,logit,heat=model(image.cuda().float()/255.,meta.cuda(),target.cuda(),extra[0].cuda())
        else:
            xy,logit,heat=model(image.cuda().float()/255.,meta.cuda(),target.cuda())
        if rail_band_mm:
            h,w=heat.shape[-2:]
            masks=np.zeros((len(image),h,w),np.uint8)
            for k in range(len(image)):
                row=rows[int(ri[k])];cam=CAMERAS[int(ci[k])]
                if rail_band_adjacent_only:
                    requested_card=int(row['requested'].split('_')[2])
                    source_card=int(row['proposal']['source_card'])
                    if (abs(requested_card-source_card)!=1 or
                        row['proposal'].get('source')=='native_full_row_direct'):
                        masks[k]=1
                        continue
                center=np.asarray(row['proposal']['estimated_center_base_m'],float)
                crop_center=np.asarray(row['proposal']['center_px'][cam],float)
                optical=pose_matrix(row['state'])@optics[cam]
                vertices=[]
                for dx in (-.065,.065):
                    for dy in (-rail_band_mm/1000,rail_band_mm/1000):
                        for dz in (-.008,.008):
                            uv=project_native(center+np.asarray([dx,dy,dz]),optical)
                            if uv is not None:
                                point=(uv-crop_center)/480+.5
                                vertices.append([round(point[0]*(w-1)),round(point[1]*(h-1))])
                if len(vertices)>=3:
                    hull=cv2.convexHull(np.asarray(vertices,np.int32))
                    cv2.fillConvexPoly(masks[k],hull,1)
                if not masks[k].any():masks[k]=True
            score=(heat.float()*4).masked_fill(~torch.from_numpy(masks).to(heat.device).bool(),-1e4)
            prob=F.softmax(score.flatten(1),dim=-1)
            yy,xx=torch.meshgrid(torch.linspace(0,1,h,device=heat.device),
                                 torch.linspace(0,1,w,device=heat.device),indexing='ij')
            xy=torch.stack(((prob*xx.flatten()).sum(-1),(prob*yy.flatten()).sum(-1)),-1)
        xy=xy.float().cpu().numpy();prob=logit.float().sigmoid().cpu().numpy()
        for i in range(len(xy)):
            j=int(ri[i]);cam=CAMERAS[int(ci[i])];center=np.asarray(rows[j]['proposal']['center_px'][cam])
            pixel=(center+(xy[i]-.5)*480).tolist()
            lab=(center+(truth[i].numpy()-.5)*480).tolist()
            predictions[j][cam]={'xy':pixel,'prob':float(prob[i]),'truth_xy_label_only':lab,
                                  'truth_inside_label_only':bool(inside[i])}
            if inside[i]:pixel_errors.append(float(np.linalg.norm(np.asarray(pixel)-lab)))
    result=[];wrong=0;accepted=0;errors=[];byepisode=defaultdict(list)
    for j,views in sorted(predictions.items()):
        r=rows[j];points={c:views[c]['xy'] for c in CAMERAS if views[c]['prob']>=.5}
        rec={'episode':r['episode'],'time_s':r['time_s'],'requested':r['requested'],
             'views':views,'accepted':False,'predicted_base_m':None,'error_mm':None,
             'nearest_identity_label_only':None,'reprojection_px':None}
        if len(points)==3:
            cameras={c:pose_matrix(r['state'])@optics[c] for c in CAMERAS}
            try:
                pred=triangulate(points,cameras);reproj=reprojection_residual_px(pred,points,cameras)
                rec['predicted_base_m']=pred.tolist();rec['reprojection_px']=float(reproj)
                requested=next(t for t in r['targets'] if t['identity']==r['requested'])
                rec['error_mm']=float(np.linalg.norm(pred-np.asarray(requested['position_base_m']))*1000)
                if reproj<=5:
                    rec['accepted']=True;accepted+=1;errors.append(rec['error_mm'])
                    nearest=min(r['targets'],key=lambda t:np.linalg.norm(pred-np.asarray(t['position_base_m'])))
                    rec['nearest_identity_label_only']=nearest['identity'];wrong+=nearest['identity']!=r['requested']
            except np.linalg.LinAlgError:pass
        result.append(rec);byepisode[r['episode']].append(rec)
    early=0;any_episode=0
    for items in byepisode.values():
        items.sort(key=lambda r:r['time_s']);first=next((r for r in items if r['accepted']),None)
        any_episode+=first is not None;early+=first is not None and first['time_s']-items[0]['time_s']<=3
    summary={'proposed_rows':len(result),'accepted':accepted,'wrong':wrong,
             'accepted_3d_p95_mm':float(np.quantile(errors,.95)) if errors else None,
             'pixel_p95_px':float(np.quantile(pixel_errors,.95)) if pixel_errors else None,
             'episodes_with_any_crop_anchor':any_episode,'episodes_with_crop_anchor_within_3s_from_first_proposal':early}
    if save:
        with save.open('w') as f:
            for r in result:f.write(json.dumps(r,separators=(',',':'))+'\n')
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--train-proposals',type=Path,required=True)
    p.add_argument('--development-proposals',type=Path,required=True)
    p.add_argument('--init',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=1200)
    p.add_argument('--minutes',type=float,default=90)
    p.add_argument('--batch-size',type=int,default=12)
    p.add_argument('--neighbor-margin-weight',type=float,default=0.)
    p.add_argument('--selection-mode',choices=('original','early_safety'),default='original')
    p.add_argument('--source-offset-meta',action='store_true',
                   help='Encode requested minus observed source card in unused metadata scalar')
    p.add_argument('--history-context',type=Path,
                   help='Frozen 30 mm distance-history context cache; fit residual over frozen native crop model')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous run')
    if a.updates>6000:p.error('Bounded crop fit cap is 6000')
    if a.history_context and a.neighbor_margin_weight:
        p.error('History and neighbor margin are separate ablations')
    a.output.mkdir(parents=True)
    labels=[json.loads(s) for s in (a.labels/'rows.jsonl').open()]
    labels_by_key={(r['episode'],r['time_s']):r for r in labels}
    rows=[]
    for split,path in [('train',a.train_proposals),('validation',a.development_proposals)]:
        for line in (path/'rows.jsonl').open():
            item=json.loads(line)
            if not item['crop']:continue
            label=labels_by_key[(item['episode'],item['time_s'])]
            if label['split']!=split:raise ValueError('Split mismatch')
            rows.append({'episode':item['episode'],'time_s':item['time_s'],'split':split,
                         'scene_group':label['scene_group'],'images':label['images'],
                         'state':label['state'],'targets':label['targets'],
                         'requested':item['requested'],'proposal':item['crop']})
    if a.history_context:
        manifest=json.loads((a.history_context/'manifest.json').read_text())
        context=np.load(a.history_context/'context.npy',mmap_mode='r')
        if context.shape!=(len(manifest),3,128):raise ValueError('History context shape mismatch')
        by_key={(m['episode'],m['time_s']):i for i,m in enumerate(manifest)}
        for row in rows:
            source=labels_by_key[(row['episode'],row['time_s'])]
            index=by_key[(row['episode'],row['time_s'])]
            if manifest[index]['images']!=source['images']:
                raise ValueError('History context image mismatch')
            row['history_context']=np.asarray(context[index])
    train=[(i,None) for i,r in enumerate(rows) if r['split']=='train']
    val=[(i,c) for i,r in enumerate(rows) if r['split']=='validation' for c in range(3)]
    if {rows[i]['scene_group'] for i,_ in train}&{rows[i]['scene_group'] for i,_ in val}:
        raise ValueError('Reset split leak')
    random.seed(20261004);np.random.seed(20261004);torch.manual_seed(20261004);torch.set_num_threads(4 if a.history_context else 8)
    workers=4
    # Workers are started after CUDA model setup when the iterator is created;
    # always spawn so they do not inherit CUDA's state through fork.
    worker_context='spawn'
    loader=DataLoader(Crops(rows,train,True,a.neighbor_margin_weight>0,a.source_offset_meta,bool(a.history_context)),batch_size=a.batch_size,
        sampler=torch.utils.data.RandomSampler(Crops(rows,train,True,a.neighbor_margin_weight>0,a.source_offset_meta,bool(a.history_context)),replacement=True,
                                               num_samples=a.updates*a.batch_size),
        num_workers=workers,pin_memory=True,persistent_workers=True,multiprocessing_context=worker_context)
    vloader=DataLoader(Crops(rows,val,False,False,a.source_offset_meta,bool(a.history_context)),batch_size=24,num_workers=workers,pin_memory=True,persistent_workers=True,multiprocessing_context=worker_context)
    cal=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics={c:np.asarray(cal['tcp_to_optical'][c],float) for c in CAMERAS}
    model=(HistoryConditionalLocator(a.init) if a.history_context else ConditionalLocator(None)).cuda()
    if not a.history_context:
        model.load_state_dict(torch.load(a.init,map_location='cpu',weights_only=False)['model'])
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=6e-5,weight_decay=1e-4)
    history=[];best=float('inf');best_step=0;start=time.monotonic();reason='update_cap';completed=0
    for step,batch in enumerate(loader,1):
        if step>a.updates:break
        if time.monotonic()-start>a.minutes*60:reason='wall_budget';break
        image,meta,target,truth,inside,*extra=batch
        image=image.cuda(non_blocking=True).float()/255.;meta=meta.cuda(non_blocking=True)
        target=target.cuda(non_blocking=True);truth=truth.cuda(non_blocking=True);inside=inside.cuda(non_blocking=True)
        model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            xy,logit,heat=(model(image,meta,target,extra[2].cuda(non_blocking=True))
                           if a.history_context else model(image,meta,target))
        h,w=heat.shape[-2:]
        cell=(truth[:,1]*(h-1)).round().long().clamp(0,h-1)*w+(truth[:,0]*(w-1)).round().long().clamp(0,w-1)
        coord=F.smooth_l1_loss(xy.float(),truth,beta=.01,reduction='none').sum(-1)
        spatial=F.cross_entropy(heat.float().reshape(len(image),-1)*4,cell,reduction='none')
        loss=((coord+.02*spatial)*inside).sum()/inside.sum().clamp_min(1)
        loss+=.1*F.binary_cross_entropy_with_logits(logit.float(),inside)
        if a.neighbor_margin_weight:
            other_xy=extra[-2].cuda(non_blocking=True)
            other_valid=extra[-1].cuda(non_blocking=True)
            scores=heat.float().unsqueeze(1)*4
            tscore=F.grid_sample(scores,(truth*2-1)[:,None,None],align_corners=True).flatten()
            nscore=F.grid_sample(scores,(other_xy*2-1)[:,None,None],align_corners=True).flatten()
            pair_valid=inside*other_valid
            margin=F.relu(.5+nscore-tscore)
            loss+=a.neighbor_margin_weight*(margin*pair_valid).sum()/pair_valid.sum().clamp_min(1)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();completed=step
        if step%200==0:
            report=evaluate(model,rows,vloader,optics,a.output/f'predictions_step{step}.jsonl')
            score=(report['accepted_3d_p95_mm'] if report['accepted_3d_p95_mm'] is not None else 1e6)
            if a.selection_mode=='early_safety':
                score+=2000*report['wrong']
                score+=200*(10-report['episodes_with_crop_anchor_within_3s_from_first_proposal'])
                score+=5*(len(val)//3-report['accepted'])
            else:
                score+=200*report['wrong']+20*(len(val)//3-report['accepted'])
            item={'step':step,'loss':float(loss.detach()),'score':score,'development':report}
            history.append(item);print(json.dumps(item),flush=True)
            if score<best:
                best=score;best_step=step
                torch.save({'model':model.state_dict(),'step':step,
                            'architecture':'resnet18_task_conditioned_observed_rail_crop480',
                            'source_offset_meta':a.source_offset_meta,
                            'history_context':bool(a.history_context),
                            'history_context_path':str(a.history_context) if a.history_context else None,
                            'runtime_true_geometry_input':False},a.output/'best.pt')
    if not best_step:raise RuntimeError('No checkpoint')
    selected=next(r for r in history if r['step']==best_step)
    summary={'schema':'sfp_observed_rail_crop_fit/v1','train_rows':len(train),'development_rows':len(val)//3,
             'train_episode_groups':len({rows[i]['scene_group'] for i,_ in train}),
             'development_episode_groups':len({rows[i]['scene_group'] for i,_ in val}),
             'updates_completed':completed,'best_step':best_step,'stop_reason':reason,
             'original_crop_size_px':480,'network_size_px':320,
             'neighbor_margin_weight':a.neighbor_margin_weight,
             'source_offset_meta':a.source_offset_meta,
             'history_context':str(a.history_context) if a.history_context else None,
             'selection_mode':a.selection_mode,
             'crop_center_source':sorted({r['proposal'].get('source','unspecified') for r in rows}),
             'runtime_true_geometry_input':False,'final_scenes_opened':False,'selected':selected}
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
