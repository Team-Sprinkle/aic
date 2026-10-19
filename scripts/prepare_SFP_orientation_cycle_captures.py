#!/usr/bin/env python3
"""32 bounded SFP perception resets; public sample derived, no insertion experts."""
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import numpy as np
import yaml

root=Path('/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1')
root.mkdir(exist_ok=False)
source=Path('configs/evaluation_protocol/qualification_public_predeadline/sample_config.yaml')
base=yaml.safe_load(source.read_text())
rng=np.random.default_rng(20261018101)
robot=Path('aic_description/urdf/ur_gz.urdf.xacro').read_text()
for name,value in base['robot']['home_joint_positions'].items():
    robot,count=re.subn(rf'({re.escape(name)}=)(-?[0-9]+\.[0-9]+)',rf'\g<1>{value:.7f}',robot,count=1)
    assert count==1
for worker in range(2):
    out=root/('SFP_orientation_capture_worker'+str(worker))
    out.mkdir()
    cases=[]
    for i in range(16):
        index=i*2+worker
        cfg=copy.deepcopy(base)
        trial=copy.deepcopy(base['trials']['trial_1' if index%2==0 else 'trial_2'])
        cfg['trials']={'trial_1':trial}
        board=trial['scene']['task_board'];count=[1,2,3,5][i%4];port=(i//4)%2;card=(i//2+worker)%count
        for j in range(5):
            board[f'nic_rail_{j}']=dict(entity_present=j<count)
            if j<count:
                board[f'nic_rail_{j}'].update(entity_name=f'nic_card_mount_{j}',entity_pose=dict(translation=float(rng.uniform(-.021,.023)),roll=0.,pitch=0.,yaw=float(rng.uniform(-.04,.04))))
        board['pose']['x']+=float(rng.uniform(-.03,.03));board['pose']['y']+=float(rng.uniform(-.03,.03));board['pose']['yaw']+=float(rng.uniform(-.15,.15))
        task=trial['tasks']['task_1'];task['target_module_name']=f'nic_card_mount_{card}';task['port_name']=f'sfp_port_{port}'
        pose=trial['scene']['cables']['cable_0']['pose']
        perturb=rng.uniform(-.12,.12,3)
        for j,key in enumerate(('roll','pitch','yaw')):pose[key]+=float(perturb[j])
        for key in 'xyz':pose['gripper_offset'][key]+=float(rng.uniform(-.001,.001))
        name=f'SFP_wide_grasp_{index:02d}_n{count}_c{card}_p{port}'
        scene=out/'scenes'/name;scene.mkdir(parents=True)
        (scene/'eval_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));(scene/'ur_llb_start.xacro').write_text(robot)
        for filename in ['world_audit.sdf','bridge_audit.yaml']:shutil.copyfile(Path('configs/hierarchical_recovery')/filename,scene/filename)
        cases.append(dict(case=name,scene=str(scene),family='sfp',card_count=count,port=port,requested_card=card,replicate=index,split='validation' if i%4==3 else 'train',domain='public_sample_derived_wide_SFP_grasp',final_scene=False,BC_expert=False,viewpoint='home',source_role='Wide SFP grasp coverage; perception-only training label',pose_only_reset_hold=True))
    (out/'plan.json').write_text(json.dumps(dict(cases=cases,planned=16,training=12,validation=4,seed=20261018101,pose_only=True,BC_expert_admission=False,orientation_perturbation_rad_each_axis=.12,translation_perturbation_m_each_axis=.001,sample_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),first_visual_QA_count=2,reserved_final_scenes_opened=False),indent=2)+'\n')
(root/'state.json').write_text(json.dumps(dict(stage='cycle1_SFP_orientation_coverage',maximum_cycles=5,target_score_strictly_above=290,original_full_sample_unchanged=True,final_scenes_sealed=True),indent=2)+'\n')
print(root)
