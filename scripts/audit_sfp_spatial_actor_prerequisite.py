#!/usr/bin/env python3
"""Freeze and score visual target identity on saved autonomous transport views.

Simulator port positions are used only after RGB inference for this audit.
No geometry enters a crop, actor, or acceptance decision.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_sfp_balanced_anchor import BalancedViews, evaluate
from train_sfp_structured_slot_locator import RequestedSlot
from train_target_conditioned_port_locator import ConditionalLocator
from train_two_family_port_locator import CAMERAS, Locator


def load(path, kind):
    base = Locator(None) if kind == 'structured' else ConditionalLocator(None)
    base.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
    return (RequestedSlot(base) if kind == 'structured' else base).cuda().eval()


def episode_report(predictions):
    groups = defaultdict(list)
    for row in predictions:
        groups[row['episode']].append(row)
    result = []
    for episode, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r['time_s'])
        accepted = [r for r in rows if r['accepted']]
        correct = [r for r in accepted if r['nearest_identity_label_only'] == r['requested']]
        wrong = [r for r in accepted if r['nearest_identity_label_only'] != r['requested']]
        result.append(dict(episode=episode, requested=rows[0]['requested'],
                           rows=len(rows), accepted=len(accepted), wrong=len(wrong),
                           first_correct_after_first_sample_s=(correct[0]['time_s'] - rows[0]['time_s']
                               if correct else None),
                           first_wrong_s=wrong[0]['time_s'] if wrong else None))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--selected', type=Path, required=True)
    p.add_argument('--structured', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve previous audit')
    rows = [json.loads(line) for line in (a.labels / 'rows.jsonl').open()]
    if len({r['scene_group'] for r in rows}) != 3 or not all(r['autonomous_actor'] for r in rows):
        p.error('Expected three saved autonomous reset groups')
    a.output.mkdir(parents=True)
    torch.set_num_threads(4)
    cal = json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics = {camera: np.asarray(cal['tcp_to_optical'][camera], float) for camera in CAMERAS}
    indices = [(i, c) for i in range(len(rows)) for c in range(3)]
    reports = {}
    for name, path in [('selected', a.selected), ('structured', a.structured)]:
        model = load(path, name)
        data = BalancedViews(rows, indices, False)
        loader = DataLoader(data, batch_size=6, num_workers=0, pin_memory=True)
        predictions_path = a.output / f'{name}_predictions.jsonl'
        score = evaluate(model, rows, loader, 'cuda', optics, predictions_path)
        episode = episode_report([json.loads(line) for line in predictions_path.open()])
        reports[name] = dict(checkpoint=str(path), score=score, episodes=episode,
                             episodes_with_early_correct=sum(x['first_correct_after_first_sample_s']
                                 is not None and x['first_correct_after_first_sample_s'] <= 3
                                 for x in episode))
        del model
        torch.cuda.empty_cache()
    limited = all(r['score']['accepted_wrong_identity'] == 0 and
                  r['episodes_with_early_correct'] == 3 for r in reports.values())
    summary = dict(schema='sfp_spatial_actor_prerequisite/v1',
                   source='three previously run nonfinal autonomous candidate routes',
                   labels=str(a.labels), episode_groups=3, reports=reports,
                   limited_three_scene_identity_subtest_passed=limited,
                   full_ten_request_identity_gate_passed=False,
                   full_gate_reason='Only three sparse one/two-card requests here; prior held-out five-card acquisition fails 9/10 gate',
                   geometry_in_model_input=False, geometry_for_posthoc_scoring_only=True,
                   final_scenes_opened=False)
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: dict(score=v['score'], early=v['episodes_with_early_correct'])
                      for k, v in reports.items()}), flush=True)


if __name__ == '__main__':
    main()
