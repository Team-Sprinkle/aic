#!/usr/bin/env python3
"""Collect bounded public qualification-family CheatCode transport anchors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import re
import subprocess

import yaml


ROOT=Path(__file__).resolve().parents[1]
OFFICIAL=ROOT/'artifacts/prod_cheatcode_audit/official_qualification/eval_config.yaml'
WORLD=ROOT/'artifacts/prod_cheatcode_audit/ordinary_broad_followup/cable_route_probe/smooth_wide_repeat_02'
BASE=ROOT/'outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt'
IMAGE='ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923'
START_XACRO=ROOT/'aic_description/urdf/ur_gz.urdf.xacro'
START_JOINTS=('shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
              'wrist_1_joint','wrist_2_joint','wrist_3_joint')
BASE_MEASURED_START=dict(shoulder_pan_joint=-0.15803,shoulder_lift_joint=-1.35146,
                         elbow_joint=-1.69620,wrist_1_joint=-1.66472,
                         wrist_2_joint=1.57073,wrist_3_joint=1.41262)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root,seed,mode,vary_starts=False):
    source=yaml.safe_load(OFFICIAL.read_text());rng=random.Random(seed)
    specs=([('sfp_trial1',1,4,1),('sfp_trial2',2,12,3),('sc_trial3',3,16,4)]
           if mode=='general' else [('sc_trial3_exact',3,6,2)]
           if mode=='sc_exact' else [('sfp_trial1',1,4,1),('sfp_trial2',2,12,3)]
           if mode=='sfp_only' else [('sfp_grid_trial1',1,50,10),('sfp_grid_trial2',2,50,10)])
    scenes=[]
    for family,trial_id,ntrain,ndev in specs:
        for k in range(ntrain+ndev):
            cfg={'scoring':source['scoring'], 'task_board_limits':source['task_board_limits'],
                 'trials':{f'trial_{810000+len(scenes)}':source['trials'][f'trial_{trial_id}'].copy()},
                 'robot':source['robot']}
            # Deep-copy nested YAML values before changing the selected layout.
            cfg=json.loads(json.dumps(cfg))
            trial=next(iter(cfg['trials'].values()));board=trial['scene']['task_board']
            pose=board['pose']
            pose['x']=round(pose['x']+rng.uniform(-.015,.015),5)
            pose['y']=round(pose['y']+rng.uniform(-.015,.015),5)
            pose['yaw']=round(pose['yaw']+rng.uniform(-.05,.05),5)
            if family.startswith('sfp_grid'):
                # Each board side has five train card-counts for every one of
                # the ten requested card/port codes, plus one held-out reset.
                code=k//6;repeat=k%6
                card,port=divmod(code,2)
                count=1+(repeat if repeat<5 else code%5)
                present={card}|set(rng.sample([c for c in range(5) if c!=card],count-1))
                trial['tasks']['task_1']['target_module_name']=f'nic_card_mount_{card}'
                trial['tasks']['task_1']['port_name']=f'sfp_port_{port}'
                for rail in range(5):
                    item=board[f'nic_rail_{rail}'];item['entity_present']=rail in present
                    if rail in present:
                        item['entity_name']=f'nic_card_{rail}'
                        item['entity_pose']=dict(translation=round(rng.uniform(-.015,.015),5),
                                                 roll=0.,pitch=0.,yaw=0.)
                    else:item.pop('entity_pose',None)
            if family.startswith('sc_trial3'):
                count=(1,2,3,4,5)[k%5] if mode=='general' else 3
                present=set(rng.sample(range(5),count)) if mode=='general' else {0,1,2}
                for card in range(5):
                    item=board[f'nic_rail_{card}'];item['entity_present']=card in present
                    if card in present:
                        item['entity_name']=f'nic_card_{card}'
                        item['entity_pose']=dict(translation=round(rng.uniform(-.015,.015),5),
                                                 roll=0.,pitch=0.,yaw=0.)
                    else:item.pop('entity_pose',None)
                port=k%2 if mode=='general' else 1
                trial['tasks']['task_1']['target_module_name']=f'sc_port_{port}'
                for p in (0,1):
                    board[f'sc_rail_{p}']['entity_pose']['translation']=round(
                        max(-.06,min(.055,board[f'sc_rail_{p}']['entity_pose']['translation']+
                                     rng.uniform(-.006,.006))),5)
            else:
                count=sum(board[f'nic_rail_{card}']['entity_present'] for card in range(5))
                present={card for card in range(5) if board[f'nic_rail_{card}']['entity_present']}
                port=int(trial['tasks']['task_1']['port_name'].split('_')[-1])
                for card in present:
                    item=board[f'nic_rail_{card}']['entity_pose']
                    item['translation']=round(max(-.0215,min(.0234,item['translation']+
                                                  rng.uniform(-.004,.004))),5)
            split=('validation' if k%6==5 else 'train') if family.startswith('sfp_grid') else (
                'train' if k<ntrain else 'validation')
            name=f'{family}_{split}_{k:02d}'
            folder=root/'scenes'/name;folder.mkdir(parents=True)
            start_record=None
            if vary_starts:
                # Gazebo's first trial starts from xacro joints. Merely changing
                # robot.home_joint_positions in YAML changes later resets only.
                xacro=START_XACRO.read_text()
                base=source['robot']['home_joint_positions']
                requested={joint:round(base[joint]+rng.uniform(-.02,.02),5)
                           for joint in START_JOINTS}
                if max(abs(requested[j]-base[j]) for j in START_JOINTS)<.005:
                    requested['shoulder_pan_joint']=round(base['shoulder_pan_joint']+.01,5)
                cfg['robot']['home_joint_positions'].update(requested)
                for joint in START_JOINTS:
                    pattern=rf'({re.escape(joint)}=)(-?[0-9]+\.[0-9]+)'
                    xacro,n=re.subn(pattern,rf'\g<1>{requested[joint]:.5f}',xacro,count=1)
                    if n!=1:raise ValueError(f'Cannot set {joint} in xacro')
                (folder/'ur_llb_start.xacro').write_text(xacro)
                start_record=dict(requested_joints=requested,
                                  xacro_sha256=digest(folder/'ur_llb_start.xacro'),
                                  verification='scored_first_trial_joint_observation_required')
            (folder/'eval_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
            for file in ('world_audit.sdf','bridge_audit.yaml'):
                shutil.copy2(WORLD/file,folder/file)
            scenes.append(dict(name=name,family='sfp' if family.startswith('sfp') else 'sc',
                               official_anchor=f'trial_{trial_id}',split=split,card_count=count,
                               cards=sorted(present),target_port=port,
                               target_card=(int(trial['tasks']['task_1']['target_module_name'].split('_')[-1])
                                            if family.startswith('sfp') else None),
                               board_pose=pose,scene_dir=str(folder),
                               config_sha256=digest(folder/'eval_config.yaml'),
                               start_variant=start_record))
    plan=dict(schema='aic_official_anchor_route_coverage/v1',official_config=str(OFFICIAL),
              official_sha256=digest(OFFICIAL),seed=seed,mode=mode,
              vary_starts=vary_starts,scenes=scenes,
              split_policy='complete distinct generated reset; development withheld from fit',
              acceptance='official tier3 75, scored stream and causal camera/action join required',
              pilot_gate='at least 1/2 full insertions in each family before continuing',
              final_scenes_opened=False)
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    return plan


def score(audit):
    path=audit/'results/scoring.yaml'
    if not path.exists():return None
    data=yaml.safe_load(path.read_text())
    trial=next((v for k,v in data.items() if k.startswith('trial_')),None)
    return None if trial is None else float(trial.get('tier_3',{}).get('score',0))


def audit_varied_start(audit,scene,start_record,baseline):
    bags=sorted((audit/'results').glob('bag_*/metadata.yaml'))
    if len(bags)!=1:raise ValueError(f'Expected one scored bag, found {len(bags)}')
    bag='/audit/results/'+bags[0].parent.name
    cmd=['docker','run','--rm','--net=none','-v',f'{ROOT}:/repo:ro',
         '-v',f'{audit}:/audit','-w','/repo','--entrypoint','bash',IMAGE,'-lc',
         'source /opt/ros/kilted/setup.bash; source /ws_aic/install/setup.bash; '
         f'python3 scripts/audit_sc_llb_start_pose.py {bag} /audit/start_joint_observations.json']
    with (audit/'start_joint_observations.log').open('w') as log:
        subprocess.run(cmd,env=dict(os.environ,DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock'),
                       stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
    rows=json.loads((audit/'start_joint_observations.json').read_text())['samples']
    if not rows:raise ValueError('No scored first-trial joint observation')
    actual=rows[0]['positions'];requested=start_record['requested_joints']
    expected={j:BASE_MEASURED_START[j]+requested[j]-baseline[j] for j in START_JOINTS}
    error={j:abs(actual[j]-expected[j]) for j in START_JOINTS}
    deviation={j:abs(actual[j]-BASE_MEASURED_START[j]) for j in START_JOINTS}
    valid=max(error.values())<=.015 and max(deviation.values())>=.003
    report=dict(valid=valid,actual=actual,expected=expected,
                max_prediction_error_rad=max(error.values()),
                max_deviation_from_fixed_start_rad=max(deviation.values()),
                requested_joints=requested,source='first_scored_bag_joint_states')
    (audit/'start_variant_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    if not valid:raise ValueError('Varied first-trial start was not observed')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--seed',type=int,default=20261011)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--max-new',type=int,default=1000)
    p.add_argument('--plan-mode',choices=('general','sc_exact','sfp_only','sfp_grid'),default='general')
    p.add_argument('--vary-starts',action='store_true',
                   help='Vary actual first-trial robot joints via per-scene xacro')
    p.add_argument('--sc-transport-pilot-revision',action='store_true',
                   help='For exact SC only, use the separately recorded verified transport pilot revision')
    a=p.parse_args()
    if not 0<=a.gpu<=7:p.error('invalid GPU')
    a.output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((a.output/'plan.json').read_text()) if (a.output/'plan.json').exists() else prepare(a.output,a.seed,a.plan_mode,a.vary_starts)
    if (plan['schema']!='aic_official_anchor_route_coverage/v1' or plan['final_scenes_opened'] or
        plan.get('mode','general')!=a.plan_mode or plan.get('vary_starts',False)!=a.vary_starts):
        p.error('unexpected plan')
    families=sorted({s['family'] for s in plan['scenes']})
    print(json.dumps(dict(planned=len(plan['scenes']),by_family={f:sum(s['family']==f for s in plan['scenes'])
                                                 for f in families})),flush=True)
    if a.plan_only:return
    progress_path=a.output/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    new=0
    # Pilot includes two per family, then all remaining train, then withheld development.
    scenes=plan['scenes'];pilot=[]
    for family in families:
        if a.plan_mode=='sfp_grid' and family=='sfp':
            pilot.extend([next(s for s in scenes if s['family']==family and
                               s['split']=='train' and s['official_anchor']==anchor)
                          for anchor in ('trial_1','trial_2')])
        else:
            pilot.extend([s for s in scenes if s['family']==family and s['split']=='train'][:2])
    remaining_train=[s for s in scenes if s not in pilot and s['split']=='train']
    if a.plan_mode=='sfp_grid':
        # One pass over every requested card/port on both board sides before
        # spending more rollouts on a second card-count repetition.
        def grid_order(s):
            k=int(s['name'].rsplit('_',1)[-1])
            return (k%6,k//6,s['official_anchor'])
        remaining_train.sort(key=grid_order)
    ordered=pilot+remaining_train+[s for s in scenes if s['split']=='validation']
    for i,s in enumerate(ordered):
        if i==len(pilot):
            for family in families:
                subset=[progress[x['name']] for x in pilot if x['family']==family]
                required=2 if a.plan_mode=='sfp_grid' and family=='sfp' else 1
                if sum(x['official_tier3']==75 and x['returncode']==0 for x in subset)<required:
                    if a.plan_mode=='sfp_grid' and family=='sfp':
                        revision_path=a.output/'pilot_transport_revision.json'
                        if revision_path.exists():
                            revision=json.loads(revision_path.read_text())
                            expected={x['name'] for x in pilot if x['family']==family}
                            rows=revision.get('verified_safe_transport',[])
                            if ({x.get('name') for x in rows}==expected and
                                all(x.get('safe_buffer') and x.get('no_scored_contact') and
                                    x.get('start_variant_valid') and x.get('chunks',0)>=30
                                    for x in rows)):
                                continue
                    if (a.plan_mode=='sc_exact' and family=='sc' and
                        a.sc_transport_pilot_revision):
                        revision_path=a.output/'pilot_revision.json'
                        if not revision_path.exists():raise FileNotFoundError(revision_path)
                        revision=json.loads(revision_path.read_text())
                        expected={x['name'] for x in pilot if x['family']==family}
                        if (revision.get('pilot_names')!=sorted(expected) or
                            revision.get('verified_safe_transport_count',0)<2):
                            raise ValueError('Pilot revision does not verify both frozen SC starts')
                        continue
                    print(json.dumps(dict(stopped='pilot_gate',family=family)),flush=True)
                    return
        if s['name'] in progress:continue
        if new>=a.max_new:break
        folder=Path(s['scene_dir'])
        if digest(folder/'eval_config.yaml')!=s['config_sha256']:raise ValueError('changed scene')
        if s.get('start_variant') and digest(folder/'ur_llb_start.xacro')!=s['start_variant']['xacro_sha256']:
            raise ValueError('changed first-trial start xacro')
        audit=a.output/'rollouts'/s['name']
        if audit.exists():raise FileExistsError(f'inspect incomplete {audit}')
        env=os.environ.copy();env.update(DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
            AIC_TRANSPORT_EVAL_GPU=str(a.gpu),AIC_TRANSPORT_EVAL_POLICY_GPU=str(a.gpu),
            AIC_TRANSPORT_EVAL_SCENE_DIR=str(folder),AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',
            AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.CheatCode',
            AIC_TRANSPORT_EVAL_TEACHER_PREFIX='true',
            AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='100',
            AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='300')
        cmd=['bash','scripts/run_clean_transport_eval.sh',s['name'],str(BASE),str(audit)]
        with (a.output/(s['name']+'.launch.log')).open('w') as log:
            try:run=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=700)
            except subprocess.TimeoutExpired:run=None
        result=dict(returncode=(run.returncode if run else None),official_tier3=score(audit),
                    audit=str(audit),command=cmd)
        if s.get('start_variant') and run and run.returncode==0 and result['official_tier3'] is not None:
            try:
                result['start_variant_audit']=audit_varied_start(
                    audit,folder,s['start_variant'],
                    yaml.safe_load(OFFICIAL.read_text())['robot']['home_joint_positions'])
            except Exception as exc:
                result['start_variant_audit']=dict(valid=False,error=f'{type(exc).__name__}: {exc}')
                result['returncode']=1
        progress[s['name']]=result
        progress_path.write_text(json.dumps(progress,indent=2)+'\n')
        new+=1
        print(json.dumps(dict(name=s['name'],split=s['split'],score=result['official_tier3'],
                              returncode=result['returncode'])),flush=True)
        if run is None or run.returncode or result['official_tier3'] is None:
            print(json.dumps(dict(stopped='run_error',name=s['name'])),flush=True)
            break
    print(json.dumps(dict(completed=len(progress),full=sum(x['official_tier3']==75 for x in progress.values()))),flush=True)


if __name__=='__main__':main()
