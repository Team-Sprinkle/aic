#!/usr/bin/env python3
"""Bounded changed-card-layout development suite; reserved final files are never read."""
from pathlib import Path
import json,hashlib,shutil,yaml
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'outputs/experiments/2026-09-30_transport_continuation';base.mkdir(exist_ok=True)
sources={0:Path('/var/tmp/chmin_aic_20260918_act/sc_low_cross_forward_minus10_20260926'),1:Path('/var/tmp/chmin_aic_20260918_act/moveit_sc_port1_generalization_source_20260929')}
cases=[('port0_four_shiftplus',0,[0,1,2,4],.005,'train'),('port1_five_shiftminus',1,[0,1,2,3,4],-.005,'train'),('port0_two_mixed',0,[1,4],-.005,'validation'),('port1_three_mixed',1,[0,2,4],.005,'validation')]
rows=[]
for i,(name,port,rails,shift,split) in enumerate(cases):
 out=base/'scenes'/name
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True)
 src=sources[port];cfg=yaml.safe_load((src/'eval_config.yaml').read_text());trial=next(iter(cfg['trials'].values()));board=trial['scene']['task_board']
 for rail in range(5):
  item=board[f'nic_rail_{rail}'];item['entity_present']=rail in rails
  if rail in rails:item['entity_pose']['translation']=max(-.0215,min(.0234,item['entity_pose']['translation']+shift))
 cfg['trials']={f'trial_{1300100+i}':trial};cfg.pop('generated',None)
 (out/'eval_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 for f in ['world_audit.sdf','bridge_audit.yaml','route_plan.json']:shutil.copy2(src/f,out/f)
 # Same recorded robot/board/port/cable start; changed card geometry only.
 joint=src/'measured_start_joint_state.json'
 if not joint.exists():joint=ROOT/'outputs/experiments/2026-09-29_moveit_gazebo_pilot/measured_start_joint_state.json'
 shutil.copy2(joint,out/'measured_start_joint_state.json')
 rows.append(dict(name=name,target_port=port,nic_rails=rails,card_shift_m=shift,split=split,scene_sha256=hashlib.sha256((out/'eval_config.yaml').read_bytes()).hexdigest(),source_scene=str(src),recorded_start_provenance=str(joint),changes='card presence and rail translation; robot/board/port/cable start unchanged',online_insertion=split=='train',final_scene=False))
(base/'scene_manifest.json').write_text(json.dumps(dict(schema='aic_bounded_transport_suite/v1',cases=rows,stopping_rule='four matched MoveIt checks; diagnose any transport blockage before further data collection',gate='transport only; insertion independent'),indent=2)+'\n')
print(json.dumps(rows,indent=2))
