#!/usr/bin/env python3
"""One bounded on-policy DPPO update from an Isaac stochastic rollout."""
from __future__ import annotations
import argparse,copy,importlib.util,json,sys
from pathlib import Path
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
def load(name,file):
 spec=importlib.util.spec_from_file_location(name,HERE/file);m=importlib.util.module_from_spec(spec)
 sys.modules[name]=m;spec.loader.exec_module(m);return m
training=load("rpdp_training_dppo_update","train_rpdp_diffusion.py")
dppo=load("rpdp_dppo_update","rpdp_dppo.py")

class Value(nn.Module):
 def __init__(self,n):
  super().__init__();self.net=nn.Sequential(nn.LayerNorm(n),nn.Linear(n,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,1))
 def forward(self,x):return self.net(x).squeeze(-1)

def arguments():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
 p.add_argument('--replay',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument('--epochs',type=int,default=8);p.add_argument('--lr',type=float,default=1e-5)
 p.add_argument('--value-updates',type=int,default=1000);p.add_argument('--clip-ratio',type=float,default=.01)
 p.add_argument('--anchor-weight',type=float,default=.02);p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=20260922)
 p.add_argument('--max-post-update-log-ratio',type=float,default=.02)
 return p.parse_args()

def main():
 a=arguments();a.output_dir.mkdir(parents=True,exist_ok=True);device=torch.device(a.device);torch.manual_seed(a.seed)
 bundle=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=bundle['model']
 model=training.TrajectoryDiffusion(cfg['condition_dim'],cfg['horizon'],cfg['width'],cfg['layers'],fusion=cfg.get('fusion',False),visual_dim=cfg.get('visual_dim',0)).to(device)
 model.load_state_dict(bundle['model_state_dict']);model.eval()
 dppo.disable_mha_fastpath()
 anchor=copy.deepcopy(model).eval().requires_grad_(False)
 _,_,alpha_bar=training.schedule(int(bundle['diffusion_steps']),device)
 source=torch.load(a.replay,map_location='cpu',weights_only=False)['transitions']
 rows=[];episodes=[];current=[];excluded_restore=0
 for item in source:
  chain=item.get('dppo_chain');restore=(item.get('metadata') or {}).get('episode_requested_tip_restore') or {}
  if not chain or not bool((restore.get('released_by_env') or [True])[0]): excluded_restore+=1
  else: current.append(item)
  m=item.get('metadata') or {}
  if m.get('terminated') or m.get('truncated'):
   if current:episodes.append(current)
   current=[]
 if current:episodes.append(current)
 gamma=.99;returns=[]
 for episode in episodes:
  running=0.;local=[]
  for item in reversed(episode):
   running=float(item['reward'].reshape(-1)[0])+float(item['discount'].reshape(-1)[0])*running
   local.append(running)
  returns.extend(reversed(local));rows.extend(episode)
 if len(rows)<8:raise RuntimeError(f'only {len(rows)} usable on-policy transitions')
 conditions=torch.stack([r['dppo_chain'][0]['condition'].reshape(-1) for r in rows]).to(device)
 returns=torch.tensor(returns,dtype=torch.float32,device=device)
 value=Value(conditions.shape[1]).to(device);vopt=torch.optim.AdamW(value.parameters(),lr=3e-4)
 for _ in range(a.value_updates):
  pred=value(conditions);loss=nn.functional.mse_loss(pred,returns)
  vopt.zero_grad(set_to_none=True);loss.backward();vopt.step()
 with torch.no_grad():advantages=returns-value(conditions)
 def collate(indices):
  output=[]
  for k in range(len(rows[0]['dppo_chain'])):
   values={key:torch.stack([rows[i]['dppo_chain'][k][key] for i in indices]).to(device)
           for key in ('noisy','denoised','timestep','previous_timestep','condition','old_log_prob')}
   output.append(dppo.DenoisingTransition(**values))
  return output
 optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);history=[];indices=list(range(len(rows)))
 eta=float(rows[0].get('dppo_eta',.1));prediction_type=str(bundle.get('prediction_type','sample'))
 minimum_variance=float(rows[0].get('dppo_minimum_variance',1e-5))
 initial_chain=collate(indices)
 with torch.no_grad():
  initial_log_error=max(float((dppo.transition_log_prob(model,item,alpha_bar,eta=eta,prediction_type=prediction_type,minimum_variance=minimum_variance)-item.old_log_prob).abs().max()) for item in initial_chain)
 # GPU-to-CPU serialization of summed 36D log likelihoods introduces a small
 # float32 roundoff; this bound still implies an initial ratio within 0.3%.
 if initial_log_error>3e-3:raise RuntimeError(f'rollout likelihood replay mismatch: {initial_log_error}')
 for epoch in range(1,a.epochs+1):
  chain=collate(indices);policy_loss,ratio=dppo.clipped_dppo_loss(model,chain,advantages,alpha_bar,clip_ratio=a.clip_ratio,eta=eta,prediction_type=prediction_type,minimum_variance=minimum_variance)
  anchor_losses=[]
  for transition in chain:
   with torch.no_grad():reference=anchor(transition.noisy,transition.timestep,transition.condition)
   anchor_losses.append(nn.functional.mse_loss(model(transition.noisy,transition.timestep,transition.condition),reference))
  anchor_loss=torch.stack(anchor_losses).mean();loss=policy_loss+a.anchor_weight*anchor_loss
  before={name:value.detach().clone() for name,value in model.state_dict().items()}
  optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
  def post_ratios():
   with torch.no_grad():
    return torch.cat([torch.exp((dppo.transition_log_prob(model,item,alpha_bar,eta=eta,prediction_type=prediction_type,minimum_variance=minimum_variance)-item.old_log_prob).clamp(-20,20)) for item in chain])
  post=post_ratios();backtracks=0
  while float(torch.log(post).abs().max())>a.max_post_update_log_ratio and backtracks<20:
   with torch.no_grad():
    for name,parameter_value in model.state_dict().items():
     parameter_value.copy_(before[name]+.5*(parameter_value-before[name]))
   backtracks+=1;post=post_ratios()
  if float(torch.log(post).abs().max())>a.max_post_update_log_ratio:
   model.load_state_dict(before);post=post_ratios()
  history.append({'epoch':epoch,'loss':float(loss),'policy_loss':float(policy_loss),'anchor_loss':float(anchor_loss),'pre_ratio_mean':float(ratio.mean()),'pre_ratio_min':float(ratio.min()),'pre_ratio_max':float(ratio.max()),'post_ratio_mean':float(post.mean()),'post_ratio_min':float(post.min()),'post_ratio_max':float(post.max()),'post_max_abs_log_ratio':float(torch.log(post).abs().max()),'backtracks':backtracks})
 output=dict(bundle);output['model_state_dict']=model.cpu().state_dict();output['dppo']={'source_replay':str(a.replay),'usable_transitions':len(rows),'episodes':len(episodes),'excluded_restore_transitions':excluded_restore,'eta':eta,'minimum_variance':minimum_variance,'epochs':a.epochs,'learning_rate':a.lr,'clip_ratio':a.clip_ratio,'anchor_weight':a.anchor_weight,'max_post_update_log_ratio':a.max_post_update_log_ratio,'initial_log_prob_max_abs_error':initial_log_error,'return_mean':float(returns.mean()),'return_std':float(returns.std()),'advantage_mean':float(advantages.mean()),'advantage_std':float(advantages.std()),'history':history,'actor_privileged_geometry':False,'reserved_final_opened':False}
 torch.save(output,a.output_dir/'checkpoint.pt');torch.save(value.cpu().state_dict(),a.output_dir/'value.pt')
 report={'schema_version':1,'status':'complete',**output['dppo'],'checkpoint':str(a.output_dir/'checkpoint.pt')}
 (a.output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
