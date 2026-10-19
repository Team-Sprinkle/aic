#!/usr/bin/env python3
"""Locate ONE requested SC port using current RGB maps and motion-spaced context.

This predicts a crop proposal and projected-in-frame flag, not certified
visibility or safe handoff. Frozen image features, public request and measured
relative motion are inputs. True projected pixels supervise/scoring only.
The complete physical-reset split is inherited from the native corpus.
"""
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.nn import functional as F
from train_native_candidate_history_association import distance_history


class RequestedHistoryLocator(nn.Module):
    def __init__(self,use_history=True,num_requests=2):
        super().__init__();self.use_history=use_history
        self.visual=nn.Linear(256,96);self.motion=nn.Linear(9,96)
        self.site=nn.Parameter(torch.randn(3,72,96)*.01)
        self.history_site=nn.Parameter(torch.randn(3,6,96)*.01)
        self.age=nn.Parameter(torch.randn(10,96)*.01)
        self.request=nn.Embedding(num_requests,96);self.camera=nn.Parameter(torch.randn(3,96)*.01)
        layer=nn.TransformerEncoderLayer(96,4,192,dropout=.1,batch_first=True,norm_first=True)
        self.memory=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.attend=nn.MultiheadAttention(96,4,dropout=.1,batch_first=True)
        self.heat=nn.Sequential(nn.Linear(192,96),nn.GELU(),nn.Linear(96,1))
        self.flag=nn.Sequential(nn.LayerNorm(96),nn.Linear(96,1))
    def forward(self,current,history,motion,mask,request):
        b=len(current)
        now=self.visual(current.flatten(-2).transpose(-1,-2).float())+self.site[None]
        memory=now.flatten(1,2)
        padding=torch.zeros(b,216,dtype=torch.bool,device=current.device)
        if self.use_history:
            past=self.visual(history.float())+self.history_site[None,None]+self.age[None,:,None,None]+self.motion(motion)[:,:,None,None]
            memory=torch.cat((past.flatten(1,3),memory),1)
            padding=torch.cat(((~mask)[:,:,None].expand(-1,-1,18).flatten(1),padding),1)
        memory=self.memory(memory,src_key_padding_mask=padding)
        query=self.request(request)[:,None]+self.camera[None]
        answer,_=self.attend(query,memory,memory,key_padding_mask=padding,need_weights=False)
        query=query+answer
        logits=self.heat(torch.cat((now,query[:,:,None].expand(-1,-1,72,-1)),-1)).squeeze(-1)
        y,x=torch.meshgrid(torch.linspace(0,1,8,device=current.device),torch.linspace(0,1,9,device=current.device),indexing='ij')
        grid=torch.stack((x.flatten(),y.flatten()),-1)
        point=torch.einsum('bck,kd->bcd',logits.softmax(-1),grid)
        return point,self.flag(query).squeeze(-1),logits


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=10000);p.add_argument('--minutes',type=float,default=45)
    p.add_argument('--init-checkpoint',type=Path,
                   help='Weight continuation with a fresh optimizer; preserves the original fit')
    p.add_argument('--balance-training-sources',action='store_true',
                   help='Give original-layout and new rail-layout reset groups equal mass')
    p.add_argument('--family',choices=('sc','sfp'),default='sc')
    p.add_argument('--current-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(6);torch.manual_seed(20261015);rng=np.random.default_rng(2026101531)
    rows=[];dropped_async=0
    for line in (a.cache/'frame_context.jsonl').open():
        r=json.loads(line)
        if r['family']!=a.family:continue
        stamps=r.get('image_sim_ns',{})
        if stamps and max(stamps.values())-min(stamps.values())>1_000_000:
            dropped_async+=1;continue
        rows.append(r)
    if not rows:raise ValueError('Missing requested family cache')
    num_requests=2 if a.family=='sc' else 10
    identity=lambda q:f'sc_port_{q}' if a.family=='sc' else f'sfp_card_{q//2}_port_{q%2}'
    groups={s:{r['scene_group'] for r in rows if r['split']==s} for s in ('train','validation')}
    assert groups['train'].isdisjoint(groups['validation'])
    maps=np.load(a.cache/'coarse_spatial_8x9.npy',mmap_mode='r')
    current=torch.from_numpy(np.stack([maps[r['feature_row']] for r in rows])).cuda()
    pooled=F.adaptive_avg_pool2d(current.flatten(0,1).float(),(2,3)).reshape(len(rows),3,256,6).permute(0,1,3,2).half()
    hi,mask,audit=distance_history(rows,.04,10,10)
    motion=np.zeros((len(rows),10,9),np.float32)
    for i,r in enumerate(rows):
        xyz=np.asarray(r['state'][:3]);rot=Rotation.from_quat(r['state'][3:7])
        for j,k in enumerate(hi[i]):
            if not mask[i,j]:continue
            old=rows[k]['state'];rel=rot.inv()*Rotation.from_quat(old[3:7])
            motion[i,j]=np.r_[rot.inv().apply(np.asarray(old[:3])-xyz)/.2,rel.as_matrix()[:,:2].T.reshape(-1)]
    hi=torch.from_numpy(hi).cuda();mask=torch.from_numpy(mask).cuda();motion=torch.from_numpy(motion).cuda()
    targets=np.zeros((len(rows),num_requests,3,2),np.float32);flags=np.zeros((len(rows),num_requests,3),np.float32);present=np.zeros((len(rows),num_requests),bool)
    for i,r in enumerate(rows):
        for port in range(num_requests):
            t=next((t for t in r['targets'] if t['identity']==identity(port)),None)
            if t is None:continue
            present[i,port]=True
            for ci,c in enumerate(('center','left','right')):
                v=t['cameras'][c];flags[i,port,ci]=v[3]
                if v[0] is not None and v[1] is not None:
                    targets[i,port,ci]=np.asarray(v[:2])/[1151,1023]
    target=torch.from_numpy(targets).cuda();flag=torch.from_numpy(flags).cuda();present=torch.from_numpy(present).cuda()
    train=np.array([i for i,r in enumerate(rows) if r['split']=='train']);val=np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    by_episode=defaultdict(list)
    for i in val:by_episode[rows[i]['episode']].append(i)
    screen=np.array([v[j] for v in by_episode.values() for j in np.linspace(0,len(v)-1,min(8,len(v)),dtype=int)])
    counts=Counter(rows[i]['scene_group'] for i in train);weights=np.array([1/counts[rows[i]['scene_group']] for i in train]);weights/=weights.sum()
    if a.balance_training_sources:
        source_groups=defaultdict(set)
        for i in train:source_groups[rows[i].get('training_source','original')].add(rows[i]['scene_group'])
        weights=np.array([1/counts[rows[i]['scene_group']]/len(source_groups[rows[i].get('training_source','original')]) for i in train]);weights/=weights.sum()
    model=RequestedHistoryLocator(not a.current_only,num_requests).cuda();initial_step=0
    if a.init_checkpoint:
        saved=torch.load(a.init_checkpoint,map_location='cpu',weights_only=False)
        if saved['use_history']!=model.use_history:raise ValueError('History architecture mismatch')
        model.load_state_dict(saved['model']);initial_step=saved.get('cumulative_step',saved['step'])
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    y,x=torch.meshgrid(torch.linspace(0,1,8,device='cuda'),torch.linspace(0,1,9,device='cuda'),indexing='ij');grid=torch.stack((x.flatten(),y.flatten()),-1)
    def forward(ids,request):return model(current[ids],pooled[hi[ids]],motion[ids],mask[ids],request)
    @torch.inference_mode()
    def evaluate(ids,save=False):
        model.eval();errors=[];wrong=0;positive=0;inside=0;records=[];by_source=defaultdict(list);false_absent=0;absent_rows=0
        capture=defaultdict(Counter)
        for port in range(num_requests):
            for start in range(0,len(ids),32):
                ix=ids[start:start+32];req=torch.full((len(ix),),port,device='cuda',dtype=torch.long)
                point,logit,_=forward(ix,req);expected=target[ix,port]
                err=((point-expected)*point.new_tensor([1151,1023])).norm(dim=-1)
                all_alt=((point[:,:,None]-target[ix].permute(0,2,1,3))*point.new_tensor([1151,1023])).norm(dim=-1)
                other_valid=flag[ix].permute(0,2,1)>.5;other_valid[:,:,port]=False
                alt=all_alt.masked_fill(~other_valid,float('inf')).min(-1).values
                valid=flag[ix,port]>.5;errors.extend(err[valid].cpu().tolist());positive+=int(valid.sum())
                probabilities=logit.sigmoid();absent=~present[ix,port]
                false_absent+=int((absent&(probabilities.min(-1).values>=.9)).sum());absent_rows+=int(absent.sum())
                eligible=valid.all(-1);captured=eligible&(probabilities.min(-1).values>=.9)&(err.max(-1).values<=64)
                for j,i in enumerate(ix):
                    group=rows[i].get('training_source','original')
                    by_source[group].extend(err[j][valid[j]].cpu().tolist())
                    capture[group]['projected_all3_eligible']+=int(eligible[j])
                    capture[group]['all3_flag_and_crop_capture']+=int(captured[j])
                wrong+=int(((alt+6<err)&valid).sum());inside+=int(((err<=64)&valid).sum())
                if save:
                    for j,i in enumerate(ix):records.append(dict(episode=rows[i]['episode'],scene_group=rows[i]['scene_group'],time_s=rows[i]['time_s'],requested_port=port,predicted_pixels=(point[j]*point.new_tensor([1151,1023])).cpu().tolist(),projected_inframe_probability=logit[j].sigmoid().cpu().tolist(),pixel_errors=err[j].cpu().tolist(),geometry_input=False))
        return dict(p50_px=float(np.median(errors)),p95_px=float(np.percentile(errors,95)),views=positive,wrong_port_nearest_views=wrong,within64px=inside,
                    absent_request_rows=absent_rows,all_three_false_absent_accepts=false_absent,
                    projected_triple_capture_by_source={k:{**dict(v),'recall':v['all3_flag_and_crop_capture']/v['projected_all3_eligible'] if v['projected_all3_eligible'] else None} for k,v in capture.items()},
                    by_source={k:dict(n=len(v),p95_px=float(np.percentile(v,95))) for k,v in by_source.items() if v}),records
    best=float('inf');stale=0;best_step=0;start=time.monotonic();reason='update_cap'
    if a.init_checkpoint:
        initial_metrics,_=evaluate(screen);best=max(v['p95_px'] for v in initial_metrics['by_source'].values())
        ck=dict(saved,step=0,cumulative_step=initial_step)
        torch.save(ck,a.output/'best.pt')
    for step in range(1,a.updates+1):
        if time.monotonic()-start>a.minutes*60:reason='time_cap_not_proven_plateau';break
        ix=rng.choice(train,48,replace=True,p=weights);req=torch.from_numpy(rng.integers(0,num_requests,len(ix))).cuda();model.train();opt.zero_grad(set_to_none=True)
        point,logit,heat=forward(ix,req);expected=target[ix,req];valid=flag[ix,req]
        diff=(grid[None,None]-expected[:,:,None])*point.new_tensor([8,7]);soft=(-(diff.square().sum(-1))/.5).softmax(-1)
        spatial=-(soft*heat.log_softmax(-1)).sum(-1)
        loss=((20*(point-expected).square().sum(-1)+spatial)*valid).sum()/valid.sum().clamp_min(1)+.2*F.binary_cross_entropy_with_logits(logit,valid)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%250==0:
            metrics,_=evaluate(screen);score=max(v['p95_px'] for v in metrics['by_source'].values());record=dict(step=step,loss=float(loss.detach()),metrics=metrics,elapsed_s=time.monotonic()-start)
            with (a.output/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            ck=dict(schema=f'{a.family.upper()}_requested_history_locator/v1',family=a.family,num_requests=num_requests,model=model.state_dict(),use_history=model.use_history,step=step,cumulative_step=initial_step+step,geometry_input=False,absolute_state_input=False,feature_checkpoint='frozen joint_locator_v1',history_spacing_mm=40,history_rotation_deg=10)
            torch.save(ck,a.output/'last.pt')
            if score<best-.25:best=score;best_step=step;stale=0;torch.save(ck,a.output/'best.pt')
            else:stale+=1
            if step>=3000 and stale>=8:reason='validation_plateau';break
    model.load_state_dict(torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)['model']);metrics,records=evaluate(val,True)
    (a.output/'validation_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    from prepare_verified_insertion_source_index import source as canonical_source
    source_by_episode={}
    for r in rows:
        if r['episode'] not in source_by_episode:source_by_episode[r['episode']]=canonical_source(r)
    summary=dict(schema=f'{a.family.upper()}_requested_history_locator_fit/v1',family=a.family,num_requests=num_requests,best_step=best_step,last_step=step,stop_reason=reason,metrics=metrics,
        rows=len(rows),excluded_asynchronous_triplets=dropped_async,
        initial_step=initial_step,best_cumulative_step=initial_step+best_step,optimizer_continued=False,
        balanced_training_sources=a.balance_training_sources,
        train_episodes=len({rows[i]['episode'] for i in train}),validation_episodes=len(by_episode),
        train_source_episodes=len({source_by_episode[rows[i]['episode']] for i in train}),
        validation_source_episodes=len({source_by_episode[rows[i]['episode']] for i in val}),
        training_resets=len(groups['train']),validation_resets=len(groups['validation']),
        history=not a.current_only,parameters=sum(p.numel() for p in model.parameters()),
        label='projected-inframe, not physical visibility or safe handoff',promoted=False,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':
    from project_gpu_guard import selected_gpu_reservation
    with selected_gpu_reservation():main()
