#!/usr/bin/env python3
"""Freeze ten untouched nonfinal source-development SFP camera scenes."""
import argparse
import json
from pathlib import Path
import shutil

import yaml

from cache_sfp_balanced_arrival_features import source_episodes
from run_sfp_cheatcode_target_pilot import TEMPLATE,card_count


SOURCE=Path('outputs/trajectory_datasets/clean_including_no_insert_trajs/sfp_to_nic/agent/nic_cards_mixed/n400__ec2_02_sfp_hard')
CAL=Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json')
OLD=Path('/home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/plan.json')
FIT=Path('/home/nas/chmin/aic_archives/2026-10-06_sfp_slot_crop_training_materialized_v1/labels/rows.jsonl')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve frozen scene plan')
    used={r['reset_sha256'] for r in json.loads(OLD.read_text())['scenes']}
    used.update(json.loads(s)['scene_group'] for s in FIT.open())
    episodes=source_episodes(SOURCE,json.loads(CAL.read_text()))
    chosen=[]
    for card in range(5):
        for port in range(2):
            pool=[e for e in episodes if e['split']=='development' and e['card']==card and
                  e['port']==port and e['scene'] not in used]
            if not pool:raise ValueError(f'No unused nonfinal development reset for {card},{port}')
            pool.sort(key=lambda e:(-card_count(e['trial']),str(e['trial'])))
            e=pool[0];name=f'card{card}_port{port}'
            scene=a.output/'scenes'/name;scene.mkdir(parents=True)
            shutil.copy2(e['trial'],scene/'eval_config.yaml')
            for item in ('world_audit.sdf','bridge_audit.yaml'):
                shutil.copy2(TEMPLATE/item,scene/item)
            chosen.append({'name':name,'card':card,'port':port,
                           'card_count':card_count(e['trial']),
                           'scene':str(scene.resolve()),'source_trial':str(Path(e['trial']).resolve()),
                           'source_split':'development','provenance':'unused nonfinal source development reset; source actions excluded',
                           'reset_sha256':e['scene']})
            used.add(e['scene'])
    assert len({r['reset_sha256'] for r in chosen})==10
    manifest={'schema':'sfp_joint_candidate_fresh_nonfinal_plan/v1','scenes':chosen,
              'model_weights_frozen':True,'candidate_rule_frozen':True,
              'final_scenes_opened':False,'source_action_arrays_used':False,
              'gate':{'wrong_accepted_max':0,'early_acquisition_min':9,
                      'early_seconds':3,'accepted_position_p95_mm_max':20,
                      'complete_live_p95_ms_max':300}}
    (a.output/'plan.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'scenes':len(chosen),'requests':[r['name'] for r in chosen],
                      'new_scene_hashes':[r['reset_sha256'] for r in chosen]}))


if __name__=='__main__':main()
