#!/usr/bin/env python3
"""Select n400 five-card train resets covering high initial port-frame lateral-y."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

from build_two_family_port_projection_labels import sfp_ports


SOURCE = Path('/home/nas/chmin/aic_archives/2026-10-06_sfp_diverse_teacher_v1')
CALIBRATION = Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_relative(scene, scored, calibration):
    config = yaml.safe_load((scene/'eval_config.yaml').read_text())
    trial_id = next(iter(config['trials']))
    opening = {(card, port): frame for card, port, frame in sfp_ports(
        {'trial_yaml': str(scene/'eval_config.yaml'), 'trial_id': trial_id}, calibration)}[1,1]
    states = json.loads(scored.read_text())['streams']['/aic_controller/controller_state']
    first = states[0][1]['tcp_pose']['position']
    tcp = np.asarray([first[k] for k in 'xyz'])
    return (opening[:3,:3].T @ (tcp-opening[:3,3])*1000).tolist()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diffusion',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Preserve existing plan')
    plan=json.loads((SOURCE/'plan.json').read_text())
    if not plan['train_only'] or plan['final_scenes_opened']:
        parser.error('Expected training-only source')
    progress=json.loads((SOURCE/'progress.json').read_text())['results']
    calibration=json.loads(CALIBRATION.read_text())
    eligible=[]
    for source in plan['scenes']:
        if source['card_count']!=5 or 'n400__ec2_02_sfp_hard' not in source['source_trial']:
            continue
        if progress.get(source['name'],{}).get('official_score',{}).get('tier3',0)<75:
            continue
        scored=SOURCE/'rollouts'/source['name']/'scored_stream.json'
        if not scored.exists():continue
        rel=initial_relative(Path(source['scene_dir']),scored,calibration)
        eligible.append((rel[1],source,rel))
    eligible.sort(key=lambda item:(-item[0],item[1]['name']))
    if len(eligible)<10:parser.error(f'Only {len(eligible)} eligible n400 five-card resets')
    selected=eligible[:10]
    if len({x['scene_digest'] for _,x,_ in selected})!=10:
        parser.error('Duplicate reset groups')
    args.output.mkdir(parents=True)
    scenes=[]
    for _,source,rel in selected:
        config=yaml.safe_load(Path(source['source_trial']).read_text())
        trial=next(iter(config['trials'].values()))
        task=next(iter(trial['tasks'].values()))
        task['target_module_name']='nic_card_mount_1'
        task['port_name']='sfp_port_1'
        name='matched_n400_'+source['name']
        scene=args.output/'scenes'/name
        scene.mkdir(parents=True)
        (scene/'eval_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        for filename in ('world_audit.sdf','bridge_audit.yaml'):
            shutil.copy2(Path(source['scene_dir'])/filename,scene/filename)
        scenes.append(dict(name=name,source_split='train',source_scene_hash=source['scene_digest'],
                           source_trial=source['source_trial'],source_teacher=str(SOURCE/'rollouts'/source['name']),
                           initial_port_frame_xyz_mm=rel,card_count=5,alternate_request=[1,1],
                           scene_dir=str(scene),config_sha256=sha(scene/'eval_config.yaml')))
    result=dict(schema='aic_sfp_card1p1_early_recovery/v1',selection='top ten initial port-frame lateral-y among '
        'successful n400 hard five-card training resets; selected before new actor rollouts',
        scenes=scenes,source_plan=str(SOURCE/'plan.json'),source_plan_sha256=sha(SOURCE/'plan.json'),
        actor_checkpoint=str(args.diffusion.resolve()),actor_prefix_s=4,teacher_cap_s=60,
        first_three_gate='at least 2/3 accepted',total_gate='at least 8/10 accepted',
        acceptance='full insert, no scored contact, cable-board point force <10N, '
                   'at least 12 actor proposals, actor p95 <300ms',
        source_episode_grouped=True,final_scenes_opened=False,privileged_teacher_after_takeover=True)
    (args.output/'plan.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(planned=len(scenes),initial_lateral_y_mm=[round(x[0],1) for x in selected],
                          unique_reset_groups=len({x['source_scene_hash'] for x in scenes}))))


if __name__=='__main__':main()
