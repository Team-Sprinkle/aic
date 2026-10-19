#!/usr/bin/env python3
"""Prepare48 TRAIN-only shell/volume SO3 grasps and three isolated capture shards."""
import copy,json,subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
repo=Path.cwd();py=repo/'.pixi/envs/default/bin/python';root=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1');bank=root/'spacefilled48_captures';bank.mkdir(exist_ok=False)
command=[py,repo/'scripts/collect_sc_grasp_variations.py','--output',bank/'template','--viewpoint','near','--seed','2026101831','--repeats','6','--card-counts','0','1','3','5','--episode-prefix','shell48','--train-only','--prepare-only']
subprocess.run(list(map(str,command)),check=True)
plan=json.loads((bank/'template/plan.json').read_text());assert len(plan['cases'])==48
sample=yaml.safe_load((repo/'configs/evaluation_protocol/qualification_public_predeadline/sample_config.yaml').read_text());nominal=sample['trials']['trial_3']['scene']['cables']['cable_1']['pose'];base=Rotation.from_euler('xyz',[nominal[k] for k in ['roll','pitch','yaw']]);rng=np.random.default_rng(2026101831);permutation=rng.permutation(48);cases=[]
for i,case in enumerate(plan['cases']):
 j=int(permutation[i]);z=1-2*(j+.5)/48;phi=j*np.pi*(3-np.sqrt(5));axis=np.array([np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z]);radius=.06 if j%2==0 else .06*((j//2+.5)/24)**(1/3)
 rpy=(base*Rotation.from_rotvec(axis*radius)).as_euler('xyz');direction=rng.normal(size=3);direction/=np.linalg.norm(direction);shift=direction*.0015*rng.random()**(1/3)
 original=Path(case['scene']);config=yaml.safe_load((original/'eval_config.yaml').read_text());pose=config['trials']['trial_3']['scene']['cables']['cable_1']['pose']
 for k,v in zip(['roll','pitch','yaw'],rpy):pose[k]=float(v)
 for k,v in zip('xyz',shift):pose['gripper_offset'][k]=float(nominal['gripper_offset'][k]+v)
 (original/'eval_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False));reset=bank/'reset_scenes'/case['case']
 subprocess.run(list(map(str,[py,repo/'scripts/prepare_grasp_viewpoint_reset.py','--scene',original,'--output',reset,'--home-state',root/'public_home_state.json','--view-state',root/'public_near25_view_state.json'])),check=True)
 case.update(original_scene=str(original),scene=str(reset),pose_only_reset_hold=True,sampling_window_s=4.5,near_public_robot_viewpoint_reset=True,actual_scene_25mm_standoff_not_asserted=True,rotation_vector_perturbation_rad=(axis*radius).tolist(),rotation_radius_rad=float(radius),translation_perturbation_world_m=shift.tolist(),sampling='48Fibonacci sphere directions; half outerradius.06rad, halfvolume-uniformradii; independentvolume-uniform1.5mmoffsets',split='train')
 cases.append(case)
for worker,(gpu,domain) in enumerate([(3,125),(4,126),(5,127)]):
 directory=bank/f'worker{worker}';directory.mkdir();subset=cases[worker::3];assert len(subset)==16
 shard=copy.deepcopy(plan);shard.update(cases=subset,planned=16,training=16,validation=0,first_visual_QA_count=3,physical_gpu=gpu,domain=domain,seed=2026101831,pose_only=True,BC_expert_admission=False,validation_rows_added=0,hypothesis='32-grasp data improveheldoutR but radial-uniformsampling remainscentre-heavy; coverouter SO3directions without usingvalidationlabels tochooseposes.',gate='First3perworker needactualvisual/causalQA.16admittedepisodes perworker,48total; stop oninvalidcapture.')
 (directory/'plan.json').write_text(json.dumps(shard,indent=2)+'\n');(directory/'state.json').write_text(json.dumps(dict(stage='prepared',collected=0))+'\n')
(bank/'plan.json').write_text(json.dumps(dict(cases=cases,planned=48,training_only=True,validation_unchanged=True,physical_GPUs=[3,4,5],ROS_domains=[125,126,127],legacyGPU0_preserved=True,maximum_project_GPUs=4,sample_only_not_final=True,seed=2026101831),indent=2)+'\n')
print('Prepared48traininggrasps; no simulator or model has run',flush=True)
