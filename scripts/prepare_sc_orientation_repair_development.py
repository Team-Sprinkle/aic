#!/usr/bin/env python3
"""Freeze one non-final SC port-0 reset for a matched orientation correction."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import yaml

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--prior-manifest',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output exists; preserve prior scene')
    old=set()
    for path in a.prior_manifest:
        old.update(x['complete_reset_sha256'] for x in json.loads(path.read_text())['scenes'])
    a.output.mkdir(parents=True)
    for file in ('world_audit.sdf','bridge_audit.yaml'):
        shutil.copy2(a.source/file,a.output/file)
    cfg=yaml.safe_load((a.source/'eval_config.yaml').read_text())
    trial=next(iter(cfg['trials'].values()))
    task=next(iter(trial['tasks'].values()))
    assert task['target_module_name']=='sc_port_0'
    board=trial['scene']['task_board']
    board['pose'].update(x=.161,y=-.204,yaw=3.209)
    board['sc_rail_1']={'entity_present':False}
    board['sc_rail_0']['entity_pose']['translation']=.022
    identity=json.dumps({'scene':trial['scene'],'robot':cfg.get('robot'),'task':task},sort_keys=True)
    digest=hashlib.sha256(identity.encode()).hexdigest()
    if digest in old:raise ValueError('Duplicate prior reset')
    (a.output/'eval_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    manifest={'schema':'sc_orientation_repair_development/v1','name':a.output.name,
              'split':'development','selected_port':0,'other_port_present':False,
              'cards':[0,1,2,3,4],'board_xy_yaw':[.161,-.204,3.209],
              'rail_translation_m':.022,'complete_reset_sha256':digest,
              'frozen_before_live_comparison':True,'final_scenes_opened':False,
              'prior_manifests':[str(x) for x in a.prior_manifest]}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))
if __name__=='__main__':main()
