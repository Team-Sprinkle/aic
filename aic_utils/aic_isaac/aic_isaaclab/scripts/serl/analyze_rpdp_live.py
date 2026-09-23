#!/usr/bin/env python3
"""Summarize autonomous RPDP replay without joining post-reset geometry."""
import argparse,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch


def main():
 p=argparse.ArgumentParser();p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 transitions=torch.load(a.replay,map_location='cpu',weights_only=False)['transitions'];groups=defaultdict(list);lat=[];dispatch_lat=[];proposal_differences=[]
 for i,t in enumerate(transitions):
  m=t.get('metadata') or {};ep=(m.get('causal_episode') or {}).get('episode_id') or m.get('episode_id')
  g=m.get('causal_insertion_geometry') or {};terminal=t.get('terminal_observation') if (m.get('terminated') or m.get('truncated')) else None
  pg=(terminal or {}).get('insertion_geometry') if terminal else m.get('post_step_insertion_geometry')
  groups[ep].append({'i':i,'causal':g,'post':pg,'force':float(np.linalg.norm(m.get('causal_force_xyz_n') or [0,0,0])),
                     'action':t['action'].reshape(4,6),'terminated':bool(m.get('terminated')),'truncated':bool(m.get('truncated'))})
  if m.get('prefetch_inference_s') is not None:lat.append(float(m['prefetch_inference_s'])*1000)
  if m.get('model_inference_s') is not None:dispatch_lat.append(float(m['model_inference_s'])*1000)
  proposal=m.get('model_proposal_24d')
  if proposal is not None:
   proposal=torch.as_tensor(proposal).reshape(-1);executed=torch.as_tensor(t['action']).reshape(-1)
   if proposal.numel()==executed.numel():proposal_differences.append(float((proposal-executed).abs().max()))
 episodes=[]
 for ep,rows in groups.items():
  samples=[]
  for r in rows:
   for g in (r['causal'],r['post']):
    if g:samples.append(g)
  lateral=[float(g['lateral_error_m_env0'])*1000 for g in samples];depth=[float(g['signed_depth_m_env0'])*1000 for g in samples];ori=[float(g['orientation_error_rad_env0'])*180/math.pi for g in samples]
  success=[bool(g['success_geometry_by_env'][0]) for g in samples]
  action=torch.stack([r['action'] for r in rows]);force=[r['force'] for r in rows]
  episodes.append({'episode_id':ep,'macro_decisions':len(rows),'success':any(success),'best_lateral_mm':min(lateral),
   'max_signed_depth_mm':max(depth),'final_signed_depth_mm':depth[-1],'final_lateral_mm':lateral[-1],
   'final_orientation_deg':ori[-1],'max_force_n':max(force),
   'post_reset_force_p95_n':float(np.quantile(force[1:] if len(force)>1 else force,.95)),
   'post_reset_force_max_n':max(force[1:] if len(force)>1 else force),
   'translation_command_max_mm':float(action[...,:3].norm(dim=-1).max()*1000),
   'rotation_command_max_deg':float(action[...,3:].norm(dim=-1).max()*180/math.pi),
   'terminated':any(r['terminated'] for r in rows),'truncated':any(r['truncated'] for r in rows)})
 latency={'source':'prefetch_inference_s','count':len(lat),'p50_ms':float(np.quantile(lat,.5)),'p95_ms':float(np.quantile(lat,.95)),'p99_ms':float(np.quantile(lat,.99)),'max_ms':max(lat)} if lat else None
 dispatch_latency={'source':'model_inference_s','count':len(dispatch_lat),'p50_ms':float(np.quantile(dispatch_lat,.5)),'p95_ms':float(np.quantile(dispatch_lat,.95)),'p99_ms':float(np.quantile(dispatch_lat,.99)),'max_ms':max(dispatch_lat)} if dispatch_lat else None
 guide_blends=[float((t.get('metadata') or {}).get('target_action_guide_collect_blend_effective') or 0) for t in transitions]
 override_fraction=(sum(value>1e-7 for value in proposal_differences)/len(proposal_differences)
                    if proposal_differences else 0.0)
 report={'schema_version':1,'status':'complete','replay':str(a.replay),'transition_count':len(transitions),'episode_count':len(episodes),
         'success_count':sum(e['success'] for e in episodes),'episodes':episodes,'model_inference_latency':latency,'cached_dispatch_latency':dispatch_latency,
         'guide_enabled':max(guide_blends,default=0)>0,
         'guide_blend_max':max(guide_blends,default=0),
         'executed_vs_model_proposal_override_fraction':override_fraction,
         'executed_vs_model_proposal_max_abs':max(proposal_differences,default=0.0),
         'guard_enabled':False,'exploration_enabled':False,'reserved_final_opened':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
