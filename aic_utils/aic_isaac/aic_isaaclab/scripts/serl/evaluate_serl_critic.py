#!/usr/bin/env python3
"""Episode-held-out outcome ranking for the RPDP SERL critics."""
from __future__ import annotations
import argparse, json, math, zlib
from pathlib import Path
import numpy as np, torch
from torch import nn
CAMERAS=['observation.images.center_camera','observation.images.left_camera','observation.images.right_camera']
class Encoder(nn.Module):
 def __init__(self):
  super().__init__(); self.image_encoder=nn.Sequential(nn.Conv2d(3,16,5,4,2),nn.ReLU(),nn.Conv2d(16,32,3,4,1),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(32,64),nn.ReLU()); self.proj=nn.Sequential(nn.Linear(42+64*3,256),nn.ReLU())
 def forward(self,o):return self.proj(torch.cat([o['state'],*[self.image_encoder(o['images'][k]) for k in CAMERAS]],-1))
class Critic(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=Encoder();self.q=nn.Sequential(nn.Linear(280,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1))
 def forward(self,o,a):return self.q(torch.cat((self.encoder(o),a),-1))
def eid(t):
 m=t['metadata']; return (m.get('causal_episode') or m.get('episode_before_step') or m).get('episode_id')
def unpack(x):
 if torch.is_tensor(x):return x
 a=np.frombuffer(zlib.decompress(x['data']),dtype=np.dtype(x['dtype'])).copy().reshape(x['shape']);return torch.from_numpy(a)
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--replay',action='append',required=True);p.add_argument('--episodes',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();dev=torch.device(a.device)
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); critics=[]
 for key in ('critic1','critic2'):
  c=Critic().to(dev);c.load_state_dict(ck[key]);c.eval();critics.append(c)
 ts=[]
 for name in a.replay:
  x=torch.load(name,map_location='cpu',weights_only=False);ts += [t for t in x['transitions'] if eid(t) in set(a.episodes)]
 rows=[]
 with torch.no_grad():
  for start in range(0,len(ts),16):
   b=ts[start:start+16];obs={'state':torch.stack([t['obs']['state'] for t in b]).to(dev),'images':{k:torch.stack([unpack(t['obs']['images'][k]) for t in b]).to(dev) for k in CAMERAS}};act=torch.stack([t['action'] for t in b]).to(dev);q=torch.minimum(critics[0](obs,act),critics[1](obs,act)).flatten().cpu()
   rows += [{'episode_id':eid(t),'q':float(v),'reward':float(t['reward'].reshape(-1)[0]),'success':bool((t['metadata'] or {}).get('episode_success',eid(t).startswith('rpdp_fresh_test5_t05_05')))} for t,v in zip(b,q)]
 by={}
 for e in a.episodes:
  r=[x for x in rows if x['episode_id']==e];by[e]={'count':len(r),'q_mean':sum(x['q'] for x in r)/max(1,len(r)),'q_first':r[0]['q'] if r else None,'q_last':r[-1]['q'] if r else None,'success':r[0]['success'] if r else None}
 succ=[v['q_mean'] for v in by.values() if v['success']];fail=[v['q_mean'] for v in by.values() if v['success'] is False]
 out={'checkpoint':a.checkpoint,'heldout_episode_ids':a.episodes,'episodes':by,'success_q_mean':sum(succ)/max(1,len(succ)),'failure_q_mean':sum(fail)/max(1,len(fail)),'success_ranked_above_failure':bool(succ and fail and min(succ)>max(fail)),'note':'two-episode diagnostic only; not sufficient critic validation'}
 Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
