#!/usr/bin/env python3
"""Freeze a grouped nonfinal evaluation matrix; no training or policy tuning."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import numpy as np
import yaml


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=2026101407)
    a=ap.parse_args()
    if a.output.exists():
        ap.error('Preserve existing frozen plan')
    root=Path.cwd()
    old=Path('/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1')
    sc=old/'tracked_crop_fresh_scenes_v1/llb_1400'
    sfp=root/'outputs/experiments/2026-10-02_two_family_perception/sfp_native_scene_selection_v1/card1_port0_cards5_train'
    official=yaml.safe_load((old/'official_public_eval_config_20261014.yaml').read_text())
    templates={f:yaml.safe_load((p/'eval_config.yaml').read_text()) for f,p in [('sc',sc),('sfp',sfp)]}
    public={f:copy.deepcopy(official['trials'][k]['scene']['task_board']['pose'])
            for f,k in [('sc','trial_3'),('sfp','trial_2')]}
    joints=official['robot']['home_joint_positions']
    source=(root/'aic_description/urdf/ur_gz.urdf.xacro').read_text()
    rng=np.random.default_rng(a.seed)
    cases=[]
    a.output.mkdir(parents=True)
    for domain in ['primary','alternate_public_board']:
        for repeat in range(5):
            for count in (range(1,6) if domain=='primary' else [3]):
                for port in range(2):
                    for family in ['sc','sfp']:
                        name=f'{family}_{domain}_n{count}_p{port}_r{repeat}'
                        scene=a.output/'scenes'/name
                        scene.mkdir(parents=True)
                        config=copy.deepcopy(templates[family])
                        trial=next(iter(config['trials'].values()))
                        config['trials']={f'trial_{1840000+len(cases)}':trial}
                        board=trial['scene']['task_board']
                        if domain!='primary':
                            board['pose']=copy.deepcopy(public[family])
                        board['pose']['x']+=float(rng.uniform(-.006,.006))
                        board['pose']['y']+=float(rng.uniform(-.006,.006))
                        board['pose']['yaw']+=float(rng.uniform(-.006,.006))
                        present={(repeat+i)%5 for i in range(count)}
                        for i in range(5):
                            item=board[f'nic_rail_{i}']
                            item['entity_present']=i in present
                            item['entity_name']=f'nic_card_{i}'
                            item.setdefault('entity_pose',{'translation':0.,'roll':0.,'pitch':0.,'yaw':0.})
                            item['entity_pose']['translation']=float(rng.uniform(-.0215,.0234))
                        for i in range(2):
                            item=board[f'sc_rail_{i}']
                            item['entity_present']=True
                            item['entity_name']=f'sc_mount_{i}'
                            item.setdefault('entity_pose',{'translation':0.,'roll':0.,'pitch':0.,'yaw':0.})
                            item['entity_pose']['translation']=float(rng.uniform(-.06,.055))
                        task=next(iter(trial['tasks'].values()))
                        if family=='sc':
                            task['target_module_name']=f'sc_port_{port}'
                            board[f'sc_rail_{port}']['entity_pose']['translation']=float(np.linspace(-.06,.055,5)[repeat])
                            card=None
                        else:
                            card=repeat
                            task['target_module_name']=f'nic_card_mount_{card}'
                            task['port_name']=f'sfp_port_{port}'
                            board[f'nic_rail_{card}']['entity_pose']['translation']=float(np.linspace(-.0215,.0234,5)[repeat])
                        task['time_limit']=180
                        cable=trial['scene']['cables'][task['cable_name']]['pose']
                        for axis in 'xyz':
                            cable['gripper_offset'][axis]+=float(rng.uniform(-.0005,.0005))
                        for axis in ['roll','pitch','yaw']:
                            cable[axis]+=float(rng.uniform(-.005,.005))
                        initial={k:float(v+rng.uniform(-.012,.012)) for k,v in joints.items()}
                        config['robot']={'home_joint_positions':initial}
                        (scene/'eval_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
                        xacro=source
                        for k,v in initial.items():
                            xacro,n=re.subn(rf'({re.escape(k)}=)(-?[0-9]+\.[0-9]+)',rf'\g<1>{v:.7f}',xacro,count=1)
                            if n!=1:
                                raise ValueError(f'Missing real initial joint {k}')
                        (scene/'ur_llb_start.xacro').write_text(xacro)
                        for f in ['world_audit.sdf','bridge_audit.yaml']:
                            shutil.copyfile((sc if family=='sc' else sfp)/f,scene/f)
                        arms=(['near_approach','far_actor_control'] if family=='sc' and domain=='primary' and count in [1,5]
                              else ['near_approach'] if family=='sc' else ['learned_transport_only'])
                        cases.append({'case':name,'scene':str(scene),'family':family,'domain':domain,
                            'card_count':count,'port':port,'requested_card':card,'present_cards':sorted(present),
                            'replicate':repeat,'arms':arms,'configured_start_joints':initial,
                            'board_pose':board['pose'],'scene_hash':digest(scene/'eval_config.yaml'),
                            'start_xacro_hash':digest(scene/'ur_llb_start.xacro'),
                            'partition':'synthetic_fresh_development','final_scene':False})
    models={'local_actor':old/'local_causal_actor_correction_v1/best.pt',
            'transport_actor':old/'matched_actor_repaired_fixed20k_v2/best.pt',
            'grasp':old/'sc_grasp_translation_observation_v1.pt',
            'template':old/'common_corridor_candidate_live_v3.json'}
    watched=[root/'scripts/run_clean_transport_eval.sh',root/'scripts/train_shared_port_tcp_bc.py']
    watched+=list((root/'aic_example_policies/aic_example_policies/ros').glob('Run*.py'))
    plan={'schema':'aic_frozen_pipeline_coverage/v1','seed':a.seed,'cases':cases,
        'models':{k:{'path':str(v),'sha256':digest(v)} for k,v in models.items()},
        'runtime_hashes':{str(p):digest(p) for p in watched},
        'rollouts':sum(len(c['arms']) for c in cases),'scene_count':len(cases),
        'coverage':'100 primary:1–5cards x2ports x2families x5freshreplicates;20matchedSCfar controls at1/5cards;20alternate-public-board stress at3cards x2ports x2families x5.',
        'SFP_limit':'Frozen shared learned transport only. Latest trained stop head and strict visual verifier support SC only. No integrated autonomous SFP handoff/insertion exists in this selected hybrid. SFP full insertion is not attempted or counted as a failure rate.',
        'scene_limits':'Released rail ranges; small added board/start/grasp perturbations. Primary board pose follows existing development template. Alternate board poses follow released public examples. Not all card subsets or robot starts are exhausted.',
        'gates':{'SC_physical_depth_mm':14,'SC_terminal_lateral_mm':.5,'SC_orientation':'Explicitly reported; no retrospective threshold',
            'contacts':'No official prohibited-contact/sustained-force penalties; harmless raw cable contact allowed',
            'SFP_posthoc_region':'Tip axial[-100,-15]mm/lateral<=10mm; broad diagnostic visit only, NOT approved handoff',
            'episodes_per_cell':5},
        'protocol':'No training/tuning; all functional failures retained; continue full matrix after ordinary failures. Infrastructure errors are invalid runs, retry only after cause fixed. Final scenes sealed.',
        'resources':'Sequential rootless Gazebo on one free GPU, domain86lock, max4projectGPUs including legacyGPU0. New bulk data NAS only.',
        'budget':'140 sequential rollouts; approximately7–12h including launch,scoring andrecording. Each runtime keeps existing180s task,110–115simulation second transport limits and70s local actor; no extending individual failed policy budgets.',
        'final_scenes_opened':False}
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'scenes':len(cases),'rollouts':plan['rollouts'],'output':str(a.output)}))


if __name__=='__main__':
    main()
