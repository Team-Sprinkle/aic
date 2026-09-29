#!/usr/bin/env python3
"""Admit scored SC teacher corrections from bounded or teacher-rescued windows.

This creates supervised correction labels only. Bounded interventions use a
teacher-centered clip. Direct takeover requires exact actor-proposal execution
and a full scored teacher rescue. The teacher target remains the label.
Failed episodes are retained in their source run and excluded here.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--position-clip-m', type=float, default=.01)
    p.add_argument('--rotation-clip-rad', type=float, default=.03)
    p.add_argument('--allow-unbounded-takeover', action='store_true',
                   help='Admit scored teacher-rescued direct actor takeovers; verify exact proposal execution')
    a = p.parse_args()
    max_position_gap = .20 if a.allow_unbounded_takeover else .03
    max_rotation_gap = 1.2 if a.allow_unbounded_takeover else .08
    if not 0 < a.position_clip_m <= max_position_gap or not 0 < a.rotation_clip_rad <= max_rotation_gap:
        raise ValueError('Invalid bounded student clips')
    episodes = []; ports = []; calibrations = []; sources = []; excluded = []
    seen = set(); seen_episode_indices = set()
    for run in a.runs:
        score_file = run / 'eval_collection/collection_config/attempt_0001/scoring.yaml'
        scores = yaml.safe_load(score_file.read_text())
        edges_file = run / 'edges_joined.json'
        calibrations_file = run / 'dynamic_grasp_calibration.json'
        edges = json.loads(edges_file.read_text())['rows']
        by_trial = {r['trial']: r for r in json.loads(calibrations_file.read_text())['episodes']}
        sources.append({'run': str(run), 'edges_sha256': sha(edges_file),
                        'calibration_sha256': sha(calibrations_file)})
        for item in edges:
            trial = item['trial']
            source_trial = (str(run.resolve()), trial)
            if source_trial in seen:
                raise ValueError(f'Duplicate source trial {source_trial}')
            seen.add(source_trial)
            score = float(item['official_tier3'])
            if abs(score - 75) > 1e-6:
                excluded.append({'trial': trial, 'tier3': score,
                                 'reason': 'no_full_scored_insertion'})
                continue
            categories = scores[trial]['tier_2']['categories']
            force_category = categories['insertion force']
            contact_category = categories['contacts']
            force_match = re.search(r'Max detected force:\s*([0-9.]+)\s*N',
                                    force_category['message'])
            scored_peak_force = float(force_match.group(1)) if force_match else 0.
            if float(force_category['score']) < 0 or float(contact_category['score']) < 0 or scored_peak_force > 80:
                excluded.append({'trial': trial, 'tier3': score,
                                 'reason': 'official_force_or_off_limit_contact_gate',
                                 'scored_peak_force_n': scored_peak_force,
                                 'force_score': float(force_category['score']),
                                 'contact_score': float(contact_category['score'])})
                continue
            calibration = by_trial[trial]
            drift = max(x['translation_change_from_first_mm']
                        for x in calibration['samples'])
            if drift > .25:
                excluded.append({'trial': trial, 'tier3': score,
                                 'reason': 'physical_grasp_drift_over_0.25mm',
                                 'drift_mm': drift})
                continue
            folder = Path(item['episode'])
            frames_file = folder / 'frames.jsonl'
            frames = [json.loads(x) for x in frames_file.open()]
            metadata = json.loads((folder / 'episode.json').read_text())
            lineage = metadata.get('student_correction')
            if not lineage or lineage.get('kind') != 'shared_port_tcp':
                raise ValueError(f'No shared actor student lineage: {trial}')
            if bool(lineage.get('unbounded_student_takeover', False)) != a.allow_unbounded_takeover:
                raise ValueError(f'Wrong intervention type for {trial}')
            times = np.array([float(x['sim_time']) for x in frames])
            indices = np.array([int(x['command_index']) for x in frames])
            if len(frames) < 4 or not np.all(np.diff(times) > 0) or not np.all(np.diff(indices) >= 0):
                raise ValueError(f'Noncausal corrective frames: {trial}')
            if any(set(x.get('images', {})) != {'center', 'left', 'right'} or
                   not all((folder / name).is_file() for name in x['images'].values())
                   for x in frames):
                raise ValueError(f'Missing RGB: {trial}')
            active = 0; max_position = 0.; max_rotation = 0.; max_force = 0.
            for frame in frames:
                target = np.asarray(frame['teacher_target_pose'], dtype=float)
                executed = np.asarray(frame['executed_target_pose'], dtype=float)
                d = float(np.linalg.norm(executed[:3] - target[:3]))
                r = float((Rotation.from_quat(target[3:]).inv() *
                           Rotation.from_quat(executed[3:])).magnitude())
                is_active = bool(frame.get('student_active'))
                if is_active and frame.get('student_absolute_action') is None:
                    raise ValueError(f'Active frame has no student proposal: {trial}')
                if is_active and a.allow_unbounded_takeover:
                    proposal = np.asarray(frame['student_absolute_action'], dtype=float)
                    proposal_rot = Rotation.from_rotvec(proposal[3:6])
                    executed_rot = Rotation.from_quat(executed[3:7])
                    if (np.linalg.norm(proposal[:3] - executed[:3]) > 1e-6 or
                        (proposal_rot.inv() * executed_rot).magnitude() > 1e-5):
                        raise ValueError(f'Actor proposal was not executed exactly: {trial}')
                if not is_active and (d > 1e-9 or r > 1e-8):
                    raise ValueError(f'Unmarked executed deviation: {trial}')
                if d > a.position_clip_m + 1e-7 or r > a.rotation_clip_rad + 1e-6:
                    raise ValueError(f'Executed student deviation exceeds clip: {trial}')
                active += is_active
                max_position = max(max_position, d)
                max_rotation = max(max_rotation, r)
                max_force = max(max_force, float(np.linalg.norm(frame['state'][26:29])))
            if active == 0:
                excluded.append({'trial': trial, 'tier3': score,
                                 'reason': 'successful_teacher_only_warmup; no_student_active_frames'})
                continue
            if max_force > 80:
                excluded.append({'trial': trial, 'tier3': score,
                                 'reason': 'force_over_80n', 'peak_force_n': max_force})
                continue
            port = int(item['target_module_name'].rsplit('_', 1)[-1])
            # A corrective replay may reuse an expert trial ID. Keep its
            # episode distinct for elapsed-time normalization and sampling.
            index = int.from_bytes(hashlib.sha256(
                f'{run.resolve()}::{trial}'.encode()).digest()[:6], 'big')
            if index in seen_episode_indices:
                raise ValueError(f'Corrective episode-index collision: {trial}')
            seen_episode_indices.add(index)
            task_vector = [0., 1., *([1., 0.] if port == 0 else [0., 1.]), *([0.] * 6)]
            episodes.append({'episode_index': index, 'trial_id': trial,
                             'split': 'train', 'scene_sha256': item['scene_sha256'],
                             'canonical_episode_dir': str(folder),
                             'task': {'task_family': 'sc_to_sc', 'target_port_index': port},
                             'task_vector': task_vector, 'official_tier3': score,
                             'frames': len(frames), 'frames_sha256': sha(frames_file),
                             'student_active_frames': active,
                             'student_checkpoint_sha256': lineage['sha256'],
                             'max_executed_teacher_position_gap_mm': max_position * 1000,
                             'max_executed_teacher_rotation_gap_deg': np.rad2deg(max_rotation),
                             'peak_observed_force_n': max_force,
                             'peak_scored_force_n': scored_peak_force,
                             'max_measured_grasp_drift_mm': drift,
                             'source_run': str(run),
                             'status': ('scored_corrective_supervision; teacher_rescued_direct_actor_takeover; not_clean_expert'
                                        if a.allow_unbounded_takeover else
                                        'scored_corrective_supervision; not_clean_expert')})
            ports.append({'episode_index': index, 'target_port_index': port,
                          'scene_sha256': item['scene_sha256'], 'edges': item['edges']})
            calibrations.append({**calibration, 'episode_index': index})
    a.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 'sc_bounded_student_corrective_candidates/v1',
                'training_label_only': True, 'autonomous_actor_evidence': False,
                'teacher_target_is_label': True, 'student_action_is_executed_only': True,
                'direct_actor_takeover_teacher_rescued': a.allow_unbounded_takeover,
                'position_clip_m': a.position_clip_m,
                'rotation_clip_rad': a.rotation_clip_rad,
                'gap_limit_semantics': ('maximum actor/teacher discrepancy; no teacher-centered clip'
                                        if a.allow_unbounded_takeover else 'executed teacher-centered clip'),
                'sources': sources, 'episodes': episodes, 'excluded': excluded}
    (a.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (a.output_dir / 'port_edges.json').write_text(json.dumps({
        'schema':'sc_corrective_port_edges/v1','episodes':ports}, indent=2)+'\n')
    (a.output_dir / 'physical_grasp_calibrations.json').write_text(json.dumps({
        'schema':'sc_corrective_grasp_calibrations/v1','episodes':calibrations}, indent=2)+'\n')
    print(json.dumps({'admitted_episodes': len(episodes),
                      'excluded': len(excluded),
                      'student_active_frames': sum(x['student_active_frames'] for x in episodes)}))


if __name__ == '__main__':
    main()
