#!/usr/bin/env python3
"""Ten fresh sample-derived SFP starts; never read released/final scene settings."""
import argparse,copy,hashlib,json,re,shutil
from pathlib import Path
import numpy as np
import yaml


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);repo=Path.cwd();sample=repo/'configs/evaluation_protocol/qualification_public_predeadline/sample_config.yaml';sha=hashlib.sha256(sample.read_bytes()).hexdigest();assert sha=='21f6eb9ac0d71f7dddac3657c67c41b5fa212e5aaa1fd68528b08ec219585c8d';base=yaml.safe_load(sample.read_text());rng=np.random.default_rng(2026101813);cases=[]
 for card in range(5):
  for port in [0,1]:
   count=1+(card+port)%5;cfg=copy.deepcopy(base);trial=cfg['trials']['trial_1'];cfg['trials']={'trial_1':trial};task=trial['tasks']['task_1'];task['target_module_name']=f'nic_card_mount_{card}';task['port_name']=f'sfp_port_{port}';board=trial['scene']['task_board'];present={card,*rng.choice([j for j in range(5) if j!=card],count-1,replace=False).tolist()}
   for j in range(5):
    board[f'nic_rail_{j}']=dict(entity_present=j in present)
    if j in present:board[f'nic_rail_{j}'].update(entity_name=f'nic_card_{j}',entity_pose=dict(translation=float(rng.uniform(-.02,.02)),roll=0.,pitch=0.,yaw=float(rng.uniform(-.015,.015))))
   for j in range(2):
    if board[f'sc_rail_{j}'].get('entity_present'):board[f'sc_rail_{j}']['entity_pose']['translation']=float(rng.uniform(-.04,.04))
   for axis in 'xy':board['pose'][axis]+=float(rng.uniform(-.025,.025))
   board['pose']['yaw']+=float(rng.uniform(-.12,.12));trial['scene']['cables']['cable_0']['pose']['gripper_offset']['z']=.04245 if (card+port)%2==0 else .04545
   for key in cfg['robot']['home_joint_positions']:cfg['robot']['home_joint_positions'][key]+=float(rng.uniform(-.005,.005))
   robot=(repo/'aic_description/urdf/ur_gz.urdf.xacro').read_text()
   for key,value in cfg['robot']['home_joint_positions'].items():robot,n=re.subn(rf'({re.escape(key)}=)(-?[0-9]+\.[0-9]+)',rf'\g<1>{value:.7f}',robot,count=1);assert n==1
   name=f'fresh_servo_sfp_n{count}_c{card}_p{port}_seed1813';scene=a.output/'scenes'/name;scene.mkdir(parents=True);(scene/'eval_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));(scene/'ur_llb_start.xacro').write_text(robot)
   for filename in ['world_audit.sdf','bridge_audit.yaml']:shutil.copyfile(repo/'configs/hierarchical_recovery'/filename,scene/filename)
   cases.append(dict(case=name,scene=str(scene),family='sfp',card_count=count,port=port,requested_card=card,replicate=0,domain='fresh_sample_derived_development',split='development',final_scene=False,present_cards=sorted(present)))
 (a.output/'evaluation_cases.json').write_text(json.dumps(dict(cases=cases),indent=2)+'\n');(a.output/'coverage_plan.json').write_text(json.dumps(dict(seed=2026101813,source_sample_sha256=sha,episodes=10,coverage='Each NIC identity and port pair once; each card count1to5twice. Stratified subset, not every joint combination.',changes='Boardxy±25mm/yaw±0.12rad; rail translation±20mm/yaw±0.015rad; public sample grasp heights; robot homejoints±0.005rad.',gate='At least9/10strict insertions; zero wrong-port accepts/scored harm; each successful acquisition p95below300ms. Freeze all10before first rollout. Preserve failures; no final scenes.',new_training=False),indent=2)+'\n')


if __name__=='__main__':main()
