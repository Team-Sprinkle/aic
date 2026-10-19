#!/usr/bin/env python3
"""Causal temporal 3D fit of frozen RGB-predicted SFP rail endpoints."""
import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from audit_sc_port_targets import pose_matrix
from build_sc_native_pose_labels import FOCAL, WIDTH, HEIGHT
from evaluate_sc_native_triangulation import triangulate


CAMS = ('center', 'left', 'right')


def read(path):
    return [json.loads(line) for line in path.open()]


def key(row):
    return row['scene_group'], row['episode'], round(float(row['time_s']), 6)


def ray(pixel, camera):
    direction = np.array([(pixel[0] - WIDTH / 2) / FOCAL,
                          (pixel[1] - HEIGHT / 2) / FOCAL, 1.], dtype=float)
    direction = camera[:3, :3] @ direction
    return camera[:3, 3], direction / np.linalg.norm(direction)


def pair_point(a, b):
    c1, d1 = a['origin'], a['direction']
    c2, d2 = b['origin'], b['direction']
    if np.linalg.norm(c1 - c2) < .01:
        return None
    matrix = np.stack((d1, -d2), axis=1)
    if np.linalg.cond(matrix) > 100:
        return None
    distance = np.linalg.lstsq(matrix, c2 - c1, rcond=None)[0]
    if (distance <= .02).any():
        return None
    return (c1 + distance[0] * d1 + c2 + distance[1] * d2) / 2


def project(point, camera):
    q = camera[:3, :3].T @ (point - camera[:3, 3])
    if q[2] <= .02:
        return None
    return np.array([WIDTH / 2 + FOCAL * q[0] / q[2],
                     HEIGHT / 2 + FOCAL * q[1] / q[2]])


def errors(point, observations):
    result = []
    for obs in observations:
        pixel = project(point, obs['camera_pose'])
        result.append(float(np.linalg.norm(pixel - obs['pixel'])) if pixel is not None
                      else float('inf'))
    return np.asarray(result)


def fit_rays(observations):
    matrix = np.zeros((3, 3), float)
    rhs = np.zeros(3, float)
    for obs in observations:
        d = obs['direction']
        perpendicular = np.eye(3) - np.outer(d, d)
        matrix += perpendicular
        rhs += perpendicular @ obs['origin']
    if np.linalg.cond(matrix) > 1e5:
        return None
    return np.linalg.solve(matrix, rhs)


def fit_endpoint(observations, threshold=5.):
    if len(observations) < 3:
        return None
    best = None
    for ia, ib in combinations(range(len(observations)), 2):
        a, b = observations[ia], observations[ib]
        if a['cam'] == b['cam'] and a['time_s'] == b['time_s']:
            continue
        point = pair_point(a, b)
        if point is None:
            continue
        residual = errors(point, observations)
        inliers = np.where(residual <= threshold)[0]
        if len(inliers) < 3:
            continue
        used = [observations[i] for i in inliers]
        if len({x['cam'] for x in used}) < 2 or len({x['time_s'] for x in used}) < 2:
            continue
        score = (-len(inliers), float(np.median(residual[inliers])), ia, ib)
        if best is None or score < best[0]:
            best = score, inliers
    if best is None:
        return None
    inliers = best[1]
    for _ in range(2):
        point = fit_rays([observations[i] for i in inliers])
        if point is None:
            return None
        inliers = np.where(errors(point, observations) <= threshold)[0]
        used = [observations[i] for i in inliers]
        if len(inliers) < 3 or len({x['cam'] for x in used}) < 2 or \
                len({x['time_s'] for x in used}) < 2:
            return None
    return {'point': point, 'inliers': int(len(inliers)),
            'inlier_median_px': float(np.median(errors(point, observations)[inliers])),
            'cameras': sorted({observations[i]['cam'] for i in inliers}),
            'times': len({observations[i]['time_s'] for i in inliers})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rail-manifest', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--source-labels', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--projection-control-noise-px', type=float, default=None,
                        help='Privileged diagnostic: use true rail projections with fixed Gaussian pixel noise')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve previous output')
    rails = read(args.rail_manifest / 'rows.jsonl')
    predictions = read(args.predictions)
    sources = {key(row): row for row in read(args.source_labels / 'rows.jsonl')}
    calibration = json.loads(args.calibration.read_text())
    optics = {cam: np.asarray(calibration['tcp_to_optical'][cam], dtype=float)
              for cam in CAMS}
    if args.projection_control_noise_px is not None and args.projection_control_noise_px < 0:
        parser.error('Control noise must be nonnegative')
    rng = np.random.default_rng(20261007)
    by_scene = defaultdict(list)
    for pred in predictions:
        row = rails[pred['row']]
        if row['split'] != 'validation' or row['autonomous_actor_diagnostic']:
            raise ValueError('Expected only ten development groups')
        source = sources[key(row)]
        if row['images'] != source['images']:
            raise ValueError('Camera provenance mismatch')
        camera = {cam: pose_matrix(source['state']) @ optics[cam] for cam in CAMS}
        observations = [[], []]
        for ci, cam in enumerate(CAMS):
            for end in (0, 1):
                if args.projection_control_noise_px is None:
                    if pred['endpoint_inframe_prob'][ci][end] < .5:
                        continue
                    pixel = np.asarray(pred['endpoint_xy_normalized'][ci][end], float) * [1151, 1023]
                else:
                    projected = row['rail_projections_training_label_only'][cam][4 * end]
                    if not projected[3]:
                        continue
                    pixel = (np.asarray(projected[:2], float) +
                             rng.normal(0, args.projection_control_noise_px, 2))
                origin, direction = ray(pixel, camera[cam])
                observations[end].append({'time_s': row['time_s'], 'cam': cam,
                                          'pixel': pixel, 'camera_pose': camera[cam],
                                          'origin': origin, 'direction': direction})
        by_scene[row['scene_group']].append((row, camera, observations))
    if len(by_scene) != 10:
        raise ValueError(f'Expected ten grouped development scenes, got {len(by_scene)}')
    records = []
    for scene, items in sorted(by_scene.items()):
        items.sort(key=lambda x: x[0]['time_s'])
        start = items[0][0]['time_s']
        history = [[], []]
        true_rail = items[0][0]['rail_projections_training_label_only']
        truth = []
        for end in (0, 4):
            pixels = {cam: true_rail[cam][end][:2] for cam in CAMS}
            truth.append(triangulate(pixels, items[0][1]))
        for row, _, new in items:
            for end in (0, 1):
                history[end].extend(new[end])
                history[end] = [x for x in history[end]
                                if 0 <= row['time_s'] - x['time_s'] <= 3.]
            estimates = [fit_endpoint(history[end]) for end in (0, 1)]
            accepted = bool(all(x is not None for x in estimates))
            span = None
            error = None
            reversed_order = None
            if accepted:
                positions = np.stack([x['point'] for x in estimates])
                span = float(np.linalg.norm(positions[1] - positions[0]) * 1000)
                accepted = 120 <= span <= 200
                error = [float(np.linalg.norm(positions[end] - truth[end]) * 1000)
                         for end in (0, 1)]
                true_axis = truth[1] - truth[0]
                reversed_order = bool(np.dot(positions[1] - positions[0], true_axis) < 0)
            records.append({'scene_group': scene, 'episode': row['episode'],
                            'time_s': row['time_s'], 'age_s': row['time_s'] - start,
                            'observations': [len(x) for x in history],
                            'endpoint_fits': [{k: v for k, v in fit.items() if k != 'point'}
                                              if fit else None for fit in estimates],
                            'accepted': accepted, 'span_mm': span,
                            'endpoint_error_mm_label_only': error,
                            'reversed_order_label_only': reversed_order,
                            'predicted_endpoints_base_m': [x['point'].tolist() if x else None
                                                           for x in estimates]})
    episodes = []
    for scene in sorted(by_scene):
        rr = [x for x in records if x['scene_group'] == scene]
        early = [x for x in rr if x['age_s'] <= 3]
        acceptable = [x for x in early if x['accepted']]
        good = [x for x in acceptable if max(x['endpoint_error_mm_label_only']) <= 10
                and not x['reversed_order_label_only']]
        episodes.append({'scene_group': scene, 'episode': rr[0]['episode'],
                         'early_any_accepted': bool(acceptable),
                         'early_any_good_reference': bool(good),
                         'early_wrong_order': any(x['reversed_order_label_only']
                                                  for x in acceptable),
                         'first_accepted_age_s': acceptable[0]['age_s'] if acceptable else None,
                         'best_early_max_error_mm': min((max(x['endpoint_error_mm_label_only'])
                                                         for x in acceptable), default=None)})
    result = {'schema': 'sfp_temporal_board_reference/v1',
              'episodes': len(episodes), 'rows': len(records),
              'early_any_accepted': sum(x['early_any_accepted'] for x in episodes),
              'early_any_good_reference': sum(x['early_any_good_reference'] for x in episodes),
              'early_wrong_order_episodes': sum(x['early_wrong_order'] for x in episodes),
              'gate_passed': (sum(x['early_any_good_reference'] for x in episodes) >= 9
                              and not any(x['early_wrong_order'] for x in episodes)),
              'by_episode': episodes,
              'geometry_runtime_input': args.projection_control_noise_px is not None,
              'projection_control_noise_px': args.projection_control_noise_px,
              'privileged_projection_input': args.projection_control_noise_px is not None,
              'truth_used_for_scoring_only': True, 'final_scenes_opened': False}
    args.output.mkdir(parents=True)
    with (args.output / 'rows.jsonl').open('w') as target:
        for record in records:
            target.write(json.dumps(record) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('episodes', 'early_any_accepted',
          'early_any_good_reference', 'early_wrong_order_episodes', 'gate_passed')},
          indent=2))


if __name__ == '__main__':
    main()
