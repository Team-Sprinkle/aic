#!/usr/bin/env python3
"""Freeze ten non-final SFP reset configs for a read-only anchor check."""
import argparse
import json
from pathlib import Path
import shutil

import yaml

from build_sfp_agent_perception_labels import reset_hash
from cache_sfp_balanced_arrival_features import source_episodes
from run_sfp_cheatcode_target_pilot import TEMPLATE, card_count


SOURCE = Path('outputs/trajectory_datasets/clean_including_no_insert_trajs/sfp_to_nic/agent/nic_cards_mixed/n400__ec2_02_sfp_hard')
CAL = Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json')
PRIOR = Path('/home/nas/chmin/aic_archives/2026-10-03_sfp_balanced_arrival_v1')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Output already exists; preserve frozen plan')
    episodes = source_episodes(SOURCE, json.loads(CAL.read_text()))
    used = set()
    for split in ('train', 'development'):
        progress = json.loads((PRIOR/f'balanced_teacher_{split}'/'progress.json').read_text())
        used.update(Path(v['source_trial']).stem for v in progress['results'].values())
    chosen = []
    for card in range(5):
        for port in range(2):
            if (card, port) == (4, 1):
                # The single existing development reset for this code was used
                # in model selection. Make a distinct, explicitly non-final reset.
                original = SOURCE/'trials/trial_000526.yaml'
                cfg = yaml.safe_load(original.read_text())
                old_key = next(iter(cfg['trials']))
                trial = cfg['trials'].pop(old_key)
                board = trial['scene']['task_board']
                board['pose']['x'] = round(board['pose']['x'] + .006, 5)
                board['pose']['y'] = round(board['pose']['y'] - .004, 5)
                rail = board['nic_rail_4']['entity_pose']
                rail['translation'] = round(rail['translation'] + .003, 5)
                cfg['trials']['trial_990041'] = trial
                provenance = 'synthetic perturbation of used development trial_000526: board x +6 mm, y -4 mm, target rail +3 mm'
                source_split = 'synthetic_nonfinal'
            else:
                candidates = [e for e in episodes if e['card'] == card and e['port'] == port
                              and e['split'] == 'development' and Path(e['trial']).stem not in used]
                if not candidates:
                    raise RuntimeError(f'No unused development reset for {card},{port}')
                # Deterministic, favor more cards and avoid selecting by measured model performance.
                candidates.sort(key=lambda e: (-card_count(e['trial']), e['scene']))
                original = Path(candidates[0]['trial'])
                cfg = yaml.safe_load(original.read_text())
                provenance = 'unused source development scene config; source action arrays excluded'
                source_split = 'development'
            name = f'card{card}_port{port}'
            scene = args.output/'scenes'/name
            scene.mkdir(parents=True)
            (scene/'eval_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
            for item in ('world_audit.sdf', 'bridge_audit.yaml'):
                shutil.copy2(TEMPLATE/item, scene/item)
            chosen.append(dict(name=name, card=card, port=port,
                               card_count=card_count(scene/'eval_config.yaml'),
                               scene=str(scene.resolve()), source_trial=str(original),
                               source_split=source_split, provenance=provenance,
                               reset_sha256=reset_hash(cfg)))
    previous = {e['scene'] for e in episodes if Path(e['trial']).stem in used}
    if len({e['reset_sha256'] for e in chosen}) != 10:
        raise ValueError('Duplicate fresh reset')
    if previous & {e['reset_sha256'] for e in chosen}:
        raise ValueError('Overlap with prior fit/development reset')
    # The source audit partition is reserved even when its filenames are known.
    audit = {e['scene'] for e in episodes if e['split'] == 'audit'}
    if audit & {e['reset_sha256'] for e in chosen}:
        raise ValueError('Overlap with source audit partition')
    manifest = dict(schema='sfp_fresh_anchor_nonfinal_plan/v1', scenes=chosen,
                    model_weights_frozen=True, final_scenes_opened=False,
                    source_action_arrays_used=False,
                    gate=dict(wrong_accepted_max=0, early_acquisition_min=9,
                              early_seconds=3, accepted_position_p95_mm_max=20,
                              complete_live_p95_ms_max=300))
    (args.output/'plan.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(scenes=len(chosen), by_request=[e['name'] for e in chosen],
                          previous_reset_overlap=0, source_audit_overlap=0)))


if __name__ == '__main__':
    main()
