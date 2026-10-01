#!/usr/bin/env python3
"""Audit pinned Bullet 8.3.0 contacts; apply upstream #890 reporting math only.
Original JSONL is immutable. Force is normal impulse/dt, not full friction force.
"""
import argparse,collections,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('audit',type=Path);p.add_argument('--dt',type=float,default=.002);a=p.parse_args()
pairs=collections.defaultdict(lambda:dict(samples=0,max_normal_force_n=0,max_depth_m=0,first_s=None,last_s=None)); timelines=[]
with (a.audit/'oracle_contacts.jsonl').open() as f:
 for line in f:
  row=json.loads(line); t=row['sim_ns']/1e9; events=[]
  for c in row['contacts']:
   names=[c['collision1']['name'],c['collision2']['name']];key=' | '.join(sorted(names));v=pairs[key];v['samples']+=1;v['first_s']=t if v['first_s'] is None else v['first_s'];v['last_s']=t
   forces=[]
   corrected_vectors=[]
   for point_index,w in enumerate(c['wrenches']):
    raw=w['body_1_wrench']['force']; xyz=[raw[k] for k in ['x','y','z']]
    if max(xyz)-min(xyz)>1e-10:raise ValueError('Not legacy scalar impulse: do not apply correction')
    forces.append(abs(xyz[0])/a.dt)
    normal=c['normals'][point_index]
    corrected_vectors.append({axis:normal[axis]*xyz[0]/a.dt for axis in ['x','y','z']})
   force=max(forces,default=0);depth=max([-d for d in c['depths']],default=0)
   v['max_normal_force_n']=max(v['max_normal_force_n'],force);v['max_depth_m']=max(v['max_depth_m'],depth)
   if force>.05: events.append(dict(pair=key,max_point_normal_force_n=force,max_depth_m=depth,positions=c['positions'],normal_force_vectors_world_n=corrected_vectors,depths_m=[-d for d in c['depths']]))
  timelines.append(dict(sim_ns=row['sim_ns'],contacts=events,raw_wrench=row.get('raw_wrench'),controller=row.get('controller')))
summary=dict(legacy_engine='gz-physics Bullet Featherstone 8.3.0',correction='upstream f34d6e388a (#890): normal*impulse/dt; depth=-distance',step_s=a.dt,warning='Normal force only; maxima per point, not net wrist force. 100Hz samples may miss brief 500Hz contacts.',pairs=dict(sorted(pairs.items(),key=lambda kv:-kv[1]['max_normal_force_n'])))
(a.audit/'contact_force_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (a.audit/'corrected_contact_timeline.jsonl').open('w') as f:
 for r in timelines:f.write(json.dumps(r)+'\n')
print(json.dumps(list(summary['pairs'].items())[:12],indent=2))
