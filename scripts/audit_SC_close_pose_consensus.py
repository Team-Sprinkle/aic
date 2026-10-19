#!/usr/bin/env python3
"""Score the existing five-fresh-frame consensus, without changing its gates.

Only matched held-out near episodes are used. Ground truth scores accepted
outputs; it never chooses windows or changes their predictions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from close_grasp_consensus import CloseGraspConsensus
from shared_port_tcp_geometry import matrix_from_pose9


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rotation-fit', type=Path, required=True)
    p.add_argument('--translation-fit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)

    def load(fit):
        summary = json.loads((fit/'summary.json').read_text())
        source = Path(summary['plan']['source'])
        rows = json.loads((source/'manifest.json').read_text())
        ids = np.load(fit/'heldout_indices.npy')
        predictions = np.load(fit/'heldout_pose_predictions.npy')
        labels = np.load(source/'labels.npy')
        return summary, { (rows[int(i)]['episode'], rows[int(i)]['time_s']):
            (rows[int(i)], predictions[j], matrix_from_pose9(labels[int(i)]))
            for j, i in enumerate(ids) }

    rs, rotations = load(a.rotation_fit)
    ts, positions = load(a.translation_fit)
    keys = sorted(k for k, v in rotations.items()
                  if v[0].get('source_role', '').startswith('Near25mm'))
    outputs, episodes = [], []
    for episode in sorted({k[0] for k in keys}):
        consensus = CloseGraspConsensus()
        accepted = []
        first = None
        for key in [k for k in keys if k[0] == episode]:
            row, rotation, truth = rotations[key]
            prow, position, ptruth = positions[key]
            assert row['images'] == prow['images']
            assert np.allclose(truth, ptruth, atol=1e-10)
            result = consensus.add(
                [row['image_stamps_ns'][c] for c in ['center', 'left', 'right']],
                position[:3, 3], rotation[:3, :3])
            if not result['accepted']:
                continue
            relative = truth[:3, :3].T @ np.asarray(result['rotation'])
            error = dict(episode=episode, time_s=key[1],
                orientation_error_deg=float(np.degrees(Rotation.from_matrix(relative).magnitude())),
                axial_tilt_error_deg=float(np.degrees(np.arccos(np.clip(relative[2, 2], -1, 1)))),
                translation_error_mm=float(np.linalg.norm(np.asarray(result['position_m'])-truth[:3, 3])*1000),
                rotation_span_deg=result['rotation_span_deg'],
                translation_span_mm=result['translation_span_mm'])
            accepted.append(error)
            if first is None:
                first = error
        outputs.extend(accepted)
        episodes.append(dict(episode=episode, samples=sum(k[0] == episode for k in keys),
                             accepted_windows=len(accepted), first_causal_update=first))
    metrics = {}
    for field in ['orientation_error_deg', 'axial_tilt_error_deg', 'translation_error_mm']:
        values = [o[field] for o in outputs]
        first_values = [e['first_causal_update'][field] for e in episodes if e['first_causal_update']]
        metrics[field] = dict(all_window_p50_p95=np.percentile(values, [50, 95]).tolist() if values else None,
                             first_per_episode=first_values,
                             first_episode_p50_p95=np.percentile(first_values, [50, 95]).tolist() if first_values else None)
    report = dict(episodes=episodes, metrics=metrics, accepted_outputs=outputs,
                  rotation_fit=str(a.rotation_fit), translation_fit=str(a.translation_fit),
                  both_training_plateaus=rs['plateau_reached'] and ts['plateau_reached'],
                  geometry_for_scoring_only=True, automatic_deployment=False,
                  caveat='Overlapping windows are not independent episodes. Raw-frame qualification remains separately reported.')
    (a.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(episodes=episodes, metrics=metrics)), flush=True)


if __name__ == '__main__':
    main()
