#!/usr/bin/env python3
"""Matched conditional diffusion transport probe on frozen three-camera features.

This is an offline supervised comparison. Route labels and simulator geometry
never enter the network; autonomous rollout requires a separate live adapter.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from train_temporal_transport_probe import build_data, LONG_AGES, sha


class DiffusionTransport(nn.Module):
    def __init__(self, feature_dim, target_mean, target_std):
        super().__init__()
        self.register_buffer('target_mean', torch.as_tensor(target_mean,dtype=torch.float32))
        self.register_buffer('target_std', torch.as_tensor(target_std,dtype=torch.float32))
        self.register_buffer('alphas', torch.cumprod(1-torch.linspace(.0001,.02,100),0))
        self.history_input=nn.Sequential(nn.LayerNorm(feature_dim+48+42),
                                          nn.Linear(feature_dim+48+42,256),nn.GELU())
        history_layer=nn.TransformerEncoderLayer(256,8,512,dropout=.05,
                                                  activation='gelu',batch_first=True,norm_first=True)
        self.history=nn.TransformerEncoder(history_layer,2)
        self.noisy_input=nn.Linear(9,256)
        self.time_embedding=nn.Embedding(100,256)
        self.action_position=nn.Embedding(4,256)
        denoise_layer=nn.TransformerEncoderLayer(256,8,768,dropout=.05,
                                                  activation='gelu',batch_first=True,norm_first=True)
        self.denoiser=nn.TransformerEncoder(denoise_layer,3)
        self.output=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,9))

    def context(self,data,indices):
        h=data['history'][indices];mask=data['mask'][indices]
        clock=data['camera_clock'][h].float().clone();clock[...,:3]/=.05
        pieces=(data['features'][h].float(),data['prev'][h].float(),
                data['motion'][h].float(),data['age'][indices][...,None],
                data['gap'][indices][...,None],clock,
                data['state_task_full'][h].float())
        x=self.history_input(torch.cat(pieces,-1))
        return self.history(x,src_key_padding_mask=~mask)[:,-1]

    def forward(self,context,noisy,timestep):
        t=self.time_embedding(timestep)
        pos=self.action_position(torch.arange(4,device=noisy.device))[None]
        actions=self.noisy_input(noisy)+t[:,None]+pos
        x=torch.cat((context[:,None]+t[:,None],actions),1)
        return self.output(self.denoiser(x)[:,1:])

    @torch.no_grad()
    def sample(self,context,seed,steps=16):
        generator=torch.Generator(device=context.device).manual_seed(seed)
        x=torch.randn((len(context),4,9),generator=generator,device=context.device)
        indices=np.linspace(99,0,steps,dtype=int).tolist()
        for j,index in enumerate(indices):
            t=torch.full((len(context),),index,device=context.device,dtype=torch.long)
            a=self.alphas[index]
            eps=self(context,x,t)
            clean=(x-(1-a).sqrt()*eps)/a.sqrt()
            next_index=indices[j+1] if j+1<len(indices) else -1
            ap=self.alphas[next_index] if next_index>=0 else a.new_tensor(1.)
            x=ap.sqrt()*clean+(1-ap).sqrt()*eps
        return x


@torch.inference_mode()
def evaluate(model,data,selected,seed=1234,batch_size=96):
    model.eval();groups=defaultdict(list)
    for offset in range(0,len(selected),batch_size):
        idx=torch.as_tensor(selected[offset:offset+batch_size],device=data['target'].device)
        context=model.context(data,idx)
        action=model.sample(context,seed+offset)
        pred=action*model.target_std+model.target_mean
        err=(pred[:,:,:3]-data['target'][idx,:,:3]).norm(dim=-1)*1000
        for j,i in enumerate(selected[offset:offset+batch_size]):
            row=data['rows'][int(i)]
            groups[row['family']].append([float(err[j,0]),float(err[j,3])])
    return {family:dict(rows=len(v),first_mm_p50=float(np.quantile(np.array(v)[:,0],.5)),
                         first_mm_p95=float(np.quantile(np.array(v)[:,0],.95)),
                         fourth_mm_p95=float(np.quantile(np.array(v)[:,1],.95)))
            for family,v in groups.items()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=3000)
    p.add_argument('--budget-minutes',type=float,default=60)
    p.add_argument('--seed',type=int,default=20261001)
    a=p.parse_args()
    if a.output.exists():p.error('Output already exists')
    a.output.mkdir(parents=True)
    torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(8)
    data=build_data(a.cache,a.checkpoint,'cuda',LONG_AGES)
    rows=data['rows']
    train=np.array([i for i,r in enumerate(rows) if r['split']=='train'],dtype=np.int64)
    val=np.array([i for i,r in enumerate(rows) if r['split']=='validation'],dtype=np.int64)
    counts=Counter((r['family'],r['episode']) for r in rows if r['split']=='train')
    episode_counts=Counter(family for family,episode in counts)
    weights=np.array([1/(2*episode_counts[rows[i]['family']]*counts[(rows[i]['family'],rows[i]['episode'])])
                      for i in train],dtype=np.float64)
    weights/=weights.sum()
    by_episode=defaultdict(list)
    for i in val:by_episode[(rows[i]['family'],rows[i]['episode'])].append(i)
    selected=sorted({int(i) for items in by_episode.values() for i in
                     np.array(items)[np.linspace(0,len(items)-1,min(12,len(items))).astype(int)]})
    model=DiffusionTransport(data['features'].shape[1],data['mean'],data['std']).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    manifest=dict(schema='aic_diffusion_transport_probe/v1',architecture='2-layer 256D history transformer + 3-layer 256D action diffusion transformer',
                  training_steps=100,inference_steps=16,history_ages_s=list(LONG_AGES),
                  cache_sha256=sha(a.cache/'manifest.json'),checkpoint_sha256=sha(a.checkpoint),
                  train_episodes={f:episode_counts[f] for f in episode_counts},
                  validation_episodes=dict(Counter(f for f,e in by_episode)),
                  target='four complete TCP-body pose9 targets, 50ms spacing',
                  update_cap=a.updates,budget_minutes=a.budget_minutes,
                  privileged_inputs=False,final_scenes_opened=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    started=time.monotonic();best=float('inf');best_update=0;history=[];reason='update_cap'
    for update in range(1,a.updates+1):
        if time.monotonic()-started>a.budget_minutes*60:reason='wall_budget';break
        selected_train=np.random.choice(train,64,replace=True,p=weights)
        idx=torch.as_tensor(selected_train,device='cuda')
        model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            context=model.context(data,idx)
            target=(data['target'][idx]-model.target_mean)/model.target_std
            t=torch.randint(0,100,(len(idx),),device='cuda')
            noise=torch.randn_like(target)
            alpha=model.alphas[t,None,None]
            noisy=alpha.sqrt()*target+(1-alpha).sqrt()*noise
            pred=model(context,noisy,t)
            loss=nn.functional.mse_loss(pred,noise)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if update%250==0 or update==a.updates:
            metrics=evaluate(model,data,selected,seed=a.seed)
            score=sum(metrics[f]['first_mm_p95'] for f in ('sc','sfp'))
            entry=dict(update=update,elapsed_min=(time.monotonic()-started)/60,
                       train_loss=float(loss),score=score,validation=metrics)
            history.append(entry)
            print(json.dumps(entry),flush=True)
            if score<best:
                best=score;best_update=update
                torch.save(dict(model=model.state_dict(),update=update,source_manifest=manifest,
                                validation=metrics),a.output/'best.pt')
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    result=dict(best_update=best_update,updates_completed=update,stop_reason=reason,
                best_validation=next(e['validation'] for e in history if e['update']==best_update),
                autonomous_gate_passed=False)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
