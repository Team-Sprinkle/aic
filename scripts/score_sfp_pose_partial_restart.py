#!/usr/bin/env python3
"""Score matched actor initialization arms and whether they use frozen pose."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_sfp_later_correction_matched import evaluate_one
from train_temporal_transport_probe import LONG_AGES, TemporalTransport, build_data, predict


def p95(x): return float(np.percentile(x, 95)) if len(x) else None


def pose_use(checkpoint, data):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = TemporalTransport(data['mean'], data['std'], raw_state_skip=True,
                              feature_dim=data['features'].shape[1], state_task_dim=42,
                              route_conditioning=False, route_goal_conditioning=True).cuda().eval()
    model.load_state_dict(saved['model'])
    indices = [i for i, row in enumerate(data['rows']) if row['split'] == 'validation' and
               row['family'] == 'sfp' and row.get('corrective_teacher') and
               int(row['episode']) >= 8100000]
    offset = data['route_goal'][indices].clone()
    accepted = offset.norm(dim=-1) > 1e-8
    result = []
    with torch.inference_mode():
        for start in range(0, len(indices), 64):
            part = indices[start:start + 64]
            ind = torch.tensor(part, device='cuda')
            true_offset = data['route_goal'][ind].clone()
            action, *_ = predict(model, data, ind, False)
            data['route_goal'][ind] = 0
            zero, *_ = predict(model, data, ind, False)
            data['route_goal'][ind] = true_offset
            action = (action[:, 0] * model.target_std + model.target_mean).float()
            zero = (zero[:, 0] * model.target_std + model.target_mean).float()
            truth = data['target'][ind, 0].float()
            for j, row_index in enumerate(part):
                result.append(dict(episode=int(data['rows'][row_index]['episode']),
                                   accepted=bool(float(true_offset[j].norm()) > 1e-8),
                                   error_mm=float((action[j, :3] - truth[j, :3]).norm() * 1000),
                                   zero_error_mm=float((zero[j, :3] - truth[j, :3]).norm() * 1000),
                                   action_change_mm=float((action[j, :3] - zero[j, :3]).norm() * 1000)))
    accepted_result = [r for r in result if r['accepted']]
    return dict(rows=len(result), episodes=len({r['episode'] for r in result}),
                accepted_rows=len(accepted_result), accepted_episodes=len({r['episode'] for r in accepted_result}),
                all_error_mm_p95=p95([r['error_mm'] for r in result]),
                accepted_error_mm_p95=p95([r['error_mm'] for r in accepted_result]),
                accepted_zero_error_mm_p95=p95([r['zero_error_mm'] for r in accepted_result]),
                accepted_action_change_mm_p95=p95([r['action_change_mm'] for r in accepted_result]),
                by_episode={str(e):dict(rows=sum(r['episode'] == e for r in result),
                                        accepted=sum(r['episode'] == e and r['accepted'] for r in result),
                                        error_mm_p95=p95([r['error_mm'] for r in result if r['episode'] == e]))
                            for e in sorted({r['episode'] for r in result})})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'base-checkpoint', 'values', 'warm', 'reset-head', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('Preserve previous result')
    data = build_data(args.cache, args.base_checkpoint, 'cuda', history_ages=LONG_AGES,
                      route_goal_values=args.values)
    arms = {}
    for name, checkpoint in (('warm', args.warm), ('reset_head', args.reset_head)):
        grouped = evaluate_one(checkpoint, data, 'cuda', correction_episode_min=8100000)
        arms[name] = dict(checkpoint=str(checkpoint), groups=grouped['groups'],
                          late=pose_use(checkpoint, data))
    warm, reset = arms['warm']['late'], arms['reset_head']['late']
    result = dict(schema='aic_sfp_pose_partial_restart_score/v1', arms=arms,
                  reset_improvement_fraction=(warm['all_error_mm_p95'] - reset['all_error_mm_p95']) /
                                             warm['all_error_mm_p95'],
                  final_scenes_opened=False, autonomous_rollouts=0)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(warm=warm, reset_head=reset,
                          reset_improvement_fraction=result['reset_improvement_fraction'])))


if __name__ == '__main__': main()
