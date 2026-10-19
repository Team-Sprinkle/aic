#!/usr/bin/env python3
"""Score frozen local actors on complete, separate corrective expert episodes.

This is offline action agreement, not autonomous insertion.  Each checkpoint
sees identical RGB, measured state, task code, and frozen RGB-estimated pose.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_predicted_port_local_actor import LocalRows, evaluate
from train_shared_port_tcp_bc import SharedPortTCPActor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--sfp-cache', type=Path, required=True)
    parser.add_argument('--checkpoint', action='append', nargs=2, metavar=('NAME', 'PATH'), required=True)
    parser.add_argument('--training-rows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Preserve prior result')
    rows = [json.loads(line) for line in args.rows.open()]
    trained = [json.loads(line) for line in args.training_rows.open()]
    if not rows or any(row['family'] != 'sfp' for row in rows):
        raise ValueError('Expected nonempty SFP development rows')
    test_scenes = {row['scene'] for row in rows}
    if test_scenes & {row['scene'] for row in trained if row['split'] == 'train'}:
        raise ValueError('Development scene occurs in training data')
    torch.set_num_threads(4)
    phases = {
        'all': rows,
        'far_40_to_80mm': [r for r in rows if .04 <= abs(r['predicted_port_tcp_pose9'][2]) < .08],
        'near_20_to_40mm': [r for r in rows if .02 <= abs(r['predicted_port_tcp_pose9'][2]) < .04],
        'contact_0_to_20mm': [r for r in rows if abs(r['predicted_port_tcp_pose9'][2]) < .02],
    }
    results = {}
    for name, path in args.checkpoint:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        stats = {key: saved['model'][key] for key in ('state_mean', 'state_std', 'pose_mean', 'pose_std', 'target_mean', 'target_std')}
        model = SharedPortTCPActor(stats, pretrained=False, condition_on_pose=True,
            separate_family_orientation_head=saved.get('separate_family_orientation_head', False),
            separate_family_action_head=saved.get('separate_family_action_head', False),
            use_local_history=saved.get('use_local_history', False)).cuda()
        model.load_state_dict(saved['model'], strict=True)
        results[name] = {'checkpoint': str(path), 'update': saved.get('update'), 'phases': {}}
        for phase, subset in phases.items():
            if not subset: continue
            loader = DataLoader(LocalRows(subset, args.sfp_cache, rows), batch_size=24,
                                shuffle=False, num_workers=0, pin_memory=True)
            results[name]['phases'][phase] = {'rows': len(subset), 'metrics': evaluate(model, loader)['sfp']}
    payload = {'schema': 'aic_sfp_recovery_heldout_action_agreement/v1',
               'meaning': 'Offline recorded-action agreement; no autonomous insertion claim',
               'episodes': sorted({r['episode'] for r in rows}),
               'scene_groups': sorted(test_scenes),
               'scene_group_overlap_with_training': False,
               'rows': len(rows), 'checkpoints': results,
               'reserved_final_opened': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'episodes': len(payload['episodes']), 'rows': len(rows),
                      'results': {k: {phase: v['metrics']['first_body_command_mm']['p95']
                          for phase, v in result['phases'].items()}
                          for k, result in results.items()}}))


if __name__ == '__main__': main()
