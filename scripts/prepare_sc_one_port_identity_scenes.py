#!/usr/bin/env python3
"""Create bounded port-1-only SC development resets for an identity-head probe."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml


CASES = (
    ('train_a', .142, -.220, 3.010, -.030, (0, 3)),
    ('train_b', .172, -.190, 3.080,  .020, (0, 1, 3, 4)),
    ('development_a', .130, -.160, 3.160, .040, (2,)),
    ('development_b', .180, -.230, 2.960, -.015, (0, 1, 2, 3, 4)),
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve existing scenes')
    source = yaml.safe_load((a.source/'eval_config.yaml').read_text())
    a.output.mkdir(parents=True)
    seen = set()
    report = []
    for name, x, y, yaw, rail, cards in CASES:
        scene_dir = a.output/name
        scene_dir.mkdir()
        for file in ('world_audit.sdf','bridge_audit.yaml'):
            shutil.copy2(a.source/file,scene_dir/file)
        cfg = yaml.safe_load((a.source/'eval_config.yaml').read_text())
        trial = next(iter(cfg['trials'].values()))
        task = next(iter(trial['tasks'].values()))
        if task['target_module_name'] != 'sc_port_1':
            raise ValueError('Expected port-1 SC source')
        board = trial['scene']['task_board']
        board['pose'].update(x=x,y=y,yaw=yaw)
        board['sc_rail_0'] = {'entity_present':False}
        board['sc_rail_1']['entity_pose']['translation'] = rail
        for card in range(5):
            if card not in cards:
                board[f'nic_rail_{card}'] = {'entity_present':False}
        identity = json.dumps({'scene':trial['scene'],'robot':cfg.get('robot'),
                               'task':task},sort_keys=True)
        digest = hashlib.sha256(identity.encode()).hexdigest()
        if digest in seen:
            raise ValueError('Duplicate reset')
        seen.add(digest)
        (scene_dir/'eval_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        report.append({'name':name,'scene_dir':str(scene_dir),'split':name.split('_')[0],
                       'selected_port':1,'other_port_present':False,'cards':list(cards),
                       'board_xy_yaw':[x,y,yaw], 'rail1_translation_m':rail,
                       'complete_reset_sha256':digest})
    (a.output/'manifest.json').write_text(json.dumps({'schema':'sc_one_port_identity_scenes/v1',
        'source':str(a.source),'purpose':'presence-head repair only',
        'scenes':report,'final_scenes_opened':False},indent=2)+'\n')
    print(json.dumps({'scenes':len(report),'output':str(a.output)}))


if __name__ == '__main__':
    main()
