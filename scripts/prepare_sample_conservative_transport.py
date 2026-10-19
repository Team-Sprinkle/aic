#!/usr/bin/env python3
"""Development-only transport specification derived from pinned public sample."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from audit_sc_port_targets import matrix
from build_sfp_port_tcp_waypoints import transform
from build_two_family_port_projection_labels import sfp_ports


SAMPLE_SHA = '21f6eb9ac0d71f7dddac3657c67c41b5fa212e5aaa1fd68528b08ec219585c8d'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['pilot','development','evaluation'],default='pilot')
    a=p.parse_args();repo=Path.cwd()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    sample=repo/'configs/evaluation_protocol/qualification_public_predeadline/sample_config.yaml'
    assert hashlib.sha256(sample.read_bytes()).hexdigest()==SAMPLE_SHA
    base=yaml.safe_load(sample.read_text())
    old=Path('/home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1')
    # Existing frozen vision and relative tool calibrations are reused. No
    # old scene-conditioned port means, corridors or potential boxes are read.
    bundle=json.loads((old/'shared_perception_bundle_approved_v1.json').read_text())
    bundle={k:bundle[k] for k in ['backbone','sc_coarse','sfp_coarse','sfp_native_refiner','mount_priors']}
    bundle['sc_native_refiner']=str(old/'native_pose_SC_rail_repair_v1/best.pt')
    sfp=json.loads((old/'SFP_rule_training_priors_global_region_v2.json').read_text())
    scq=json.loads((repo/'outputs/experiments/2026-10-02_two_family_perception/local_sc_orientation_prior_v1.json').read_text())
    calibration=json.loads((repo/'configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())
    worldbase=matrix(calibration['world_to_base']);baseworld=np.linalg.inv(worldbase)
    spec=dict(schema='sample_conservative_transport/v1',sample_sha256=SAMPLE_SHA,
        geometry_runtime_input=False,source='Pinned predeadline sample; static asset/tool calibrations. No post-release scene positions.',
        perception_lineage='Existing frozen estimators reused; strict pre-release-training provenance remains unresolved. No new fit or final-scene-specific calibration.',
        bundle=bundle,control=dict(speed_m_s=.055,angular_speed_rad_s=.25,stage_tolerance_m=.015,
            position_integral_gain=.12,maximum_tracking_compensation_m=.012,
            acquisition_standoff_m=.150,minimum_standoff_m=.130,maximum_standoff_m=.180,
            maximum_lateral_m=.025,maximum_angle_deg=10.,hold_s=1.,inspection_interval_s=.3,
            scan_step_m=.010,maximum_scan_m=.025,acquisition_budget_s=35.,stage_no_progress_s=15.),
        families={})
    for family,key in [('sc','trial_3'),('sfp','trial_1')]:
        t=base['trials'][key];bp=t['scene']['task_board']['pose']
        board=baseworld@transform([bp[k] for k in 'xyz'],[bp[k] for k in ['roll','pitch','yaw']])
        ports={}
        if family=='sc':
            for port in (0,1):
                r=t['scene']['task_board'][f'sc_rail_{port}'].get('entity_pose',{})
                shift=r.get('translation',0.)
                opening=(board@transform([-.075+shift,.0295+.041*port,.0165],[1.57,0.,1.57])@
                    transform([0.,-.002,0.],[1.5708,3.14159,0.])@transform([0.,0.,-.01564],[0.,0.,0.]))
                ports[str(port)]=opening.tolist()
            nominal_tcp_port=scq['nominal_tcp_in_port_orientation_xyzw']
            # Public static body-to-tip prior only determines a conservative
            # TCP acquisition height; visual pose is mandatory for approval.
            body_offset=None
        else:
            config=copy.deepcopy(base)
            b=config['trials'][key]['scene']['task_board']
            for card in range(5):
                b[f'nic_rail_{card}']=dict(entity_present=True,entity_name=f'nic_card_{card}',
                    entity_pose=dict(translation=0.,roll=0.,pitch=0.,yaw=0.))
            temp=a.output/'nominal_sfp_asset_reference.yaml';temp.write_text(yaml.safe_dump(config))
            for card,port,m in sfp_ports(dict(trial_yaml=str(temp),trial_id=key),calibration):ports[f'{card}:{port}']=m.tolist()
            nominal_tcp_port=sfp['nominal_TCP_in_port_xyzw'];body_offset=sfp['mean_body_tip_position_m']
        spec['families'][family]=dict(nominal_board_base=board.tolist(),nominal_ports_base=ports,
            nominal_tcp_in_port_xyzw=nominal_tcp_port,body_tip_offset_m=body_offset,
            sc_opening_to_tcp_base_m=[0.,0.,.1766] if family=='sc' else None,
            # Public row bounds + margins, selected before any new rollout.
            bypass=dict(outside_board_x_m=-.27,behind_board_y_m=-.25,beyond_board_y_m=.15,low_tcp_z_m=.16),
            potential_margin_m=.20)
    (a.output/'transport_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    cases=[];rng=np.random.default_rng(2026101627 if a.phase!='evaluation' else 2026101673)
    world=repo/'configs/hierarchical_recovery/world_audit.sdf'
    bridge=repo/'configs/hierarchical_recovery/bridge_audit.yaml'
    if not bridge.exists():bridge=repo/'aic_bringup/config/ros_gz_bridge_config.yaml'
    robot=(repo/'aic_description/urdf/ur_gz.urdf.xacro').read_text()
    for k,v in base['robot']['home_joint_positions'].items():
        robot,n=re.subn(rf'({re.escape(k)}=)(-?[0-9]+\.[0-9]+)',rf'\g<1>{v:.7f}',robot,count=1)
        assert n==1

    def add(family,count,port,rep,split,trial_key=None,target_card=None):
        key=trial_key or ('trial_3' if family=='sc' else 'trial_1')
        cfg=copy.deepcopy(base);trial=cfg['trials'][key];cfg['trials']={key:trial}
        board=trial['scene']['task_board'];task=next(iter(trial['tasks'].values()))
        card=None
        if trial_key is None:
            selected=rep%5 if target_card is None else target_card
            present={(selected+i)%5 for i in range(count)}
            if a.phase=='evaluation':
                if family=='sfp':
                    others=[i for i in range(5) if i!=selected]
                    present={selected,*map(int,rng.choice(others,size=count-1,replace=False))}
                else:present=set(map(int,rng.choice(5,size=count,replace=False)))
            for i in range(5):
                board[f'nic_rail_{i}']=dict(entity_present=i in present)
                if i in present:board[f'nic_rail_{i}'].update(entity_name=f'nic_card_{i}',entity_pose=dict(
                    translation=float(rng.uniform(-.0215,.0234)),roll=0.,pitch=0.,yaw=float(rng.uniform(-.05,.05))))
            for i in range(2):board[f'sc_rail_{i}']=dict(entity_present=True,entity_name=f'sc_mount_{i}',entity_pose=dict(
                translation=float(rng.uniform(-.06,.055)),roll=0.,pitch=0.,yaw=0.))
            if a.phase=='evaluation' and family=='sc' and rep==0:
                board[f'sc_rail_{1-port}']['entity_present']=False
            if family=='sc':task['target_module_name']=f'sc_port_{port}'
            else:
                card=selected;task['target_module_name']=f'nic_card_mount_{card}';task['port_name']=f'sfp_port_{port}'
            # Our independently declared stress envelope, not a distribution
            # inferred from the released leaderboard scene coordinates.
            for axis in 'xy':board['pose'][axis]+=float(rng.uniform(-.035,.035))
            board['pose']['yaw']+=float(rng.uniform(-.20,.20))
            cable=trial['scene']['cables'][task['cable_name']]['pose']
            displacement=rng.normal(size=3);displacement*=float(rng.uniform(0.,.002))/np.linalg.norm(displacement)
            angular=rng.normal(size=3);angular*=float(rng.uniform(0.,.04))/np.linalg.norm(angular)
            for j,axis in enumerate('xyz'):cable['gripper_offset'][axis]+=float(displacement[j])
            for j,axis in enumerate(['roll','pitch','yaw']):cable[axis]+=float(angular[j])
        elif family=='sfp':card=int(task['target_module_name'].rsplit('_',1)[1])
        name=f'{split}_{key}' if trial_key else f'{split}_{family}_n{count}_p{port}_r{rep}'
        if target_card is not None:name+=f'_c{target_card}'
        scene=a.output/'scenes'/name;scene.mkdir(parents=True)
        (scene/'eval_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        (scene/'ur_llb_start.xacro').write_text(robot)
        shutil.copyfile(world,scene/'world_audit.sdf');shutil.copyfile(bridge,scene/'bridge_audit.yaml')
        cases.append(dict(case=name,scene=str(scene),family=family,card_count=count,port=port,replicate=rep,
            requested_card=card,domain='sample_heldout' if split=='evaluation' else 'sample_development',split=split,official_trial=key,arm='sample_conservative',
            configured_start_joints=base['robot']['home_joint_positions'],final_scene=False))
    if a.phase!='evaluation':
        add('sfp',1,0,0,'sample',trial_key='trial_1');add('sfp',1,0,0,'sample',trial_key='trial_2');add('sc',0,1,0,'sample',trial_key='trial_3')
    if a.phase=='pilot':
        for family in ['sc','sfp']:
            for count in ([0,1,5] if family=='sc' else [1,5]):
                for port in range(2):add(family,count,port,0,'development')
    elif a.phase=='development':
        for count in [0,1,5]:
            for port in range(2):add('sc',count,port,0,'development')
        for count in [1,5]:
            for port in range(2):
                for card in range(5):add('sfp',count,port,0,'development',target_card=card)
    else:
        for rep in range(2):
            for count in range(6):
                for port in range(2):add('sc',count,port,rep,'evaluation')
            for count in range(1,6):
                for port in range(2):
                    for card in range(5):add('sfp',count,port,rep,'evaluation',target_card=card)
    plan=dict(cases=cases,planned=len(cases),gate='All3sample; >=90%development per family, zero wrong/harm; complete active latency p95<300ms. Diagnose only these development traces. Never use post-release evaluation results to repair rules.',
        sample_sha256=SAMPLE_SHA,rule_spec=str(a.output/'transport_spec.json'),bundle=bundle,
        final_scenes_opened=False,training_started=False,
        phase=a.phase,
        evaluation_after_gate='Freeze successful rules; 24SC (0..5cards x2ports x2repeats) +100SFP (1..5cards x5requestedcards x2ports x2repeats) =124heldout sample-derived runs. Post-release scenes, if rerun, evaluation-only afterfreeze; never tuning.')
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),development_cases=len(cases))))


if __name__=='__main__':main()
