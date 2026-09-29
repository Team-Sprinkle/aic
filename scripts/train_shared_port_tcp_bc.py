#!/usr/bin/env python3
"""Train one visual SFP/SC actor on complete recorded port-frame TCP targets.

The auxiliary pose head predicts current TCP-in-port from RGB/state/task. The
actor consumes that *prediction*, never the privileged label. Full recorded
TCP targets, scene geometry, and expert port frames are supervision only.
This script evaluates offline BC; it does not claim autonomous insertion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import ResNet18_Weights, resnet18

from act_cache import load_cached_images


CAMERAS = ('center', 'left', 'right')
SFP_CAMERA_KEYS = {camera: f'observation.images.{camera}_camera' for camera in CAMERAS}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pose9(pose):
    r = Rotation.from_quat(pose[3:]).as_matrix()
    return np.asarray(list(pose[:3]) + list(r[:, :2].T.reshape(-1)), dtype=np.float32)


def relative_tcp_target9(current, target):
    """Full teacher TCP target expressed in the observed TCP frame."""
    origin = Rotation.from_quat(current[3:])
    goal = Rotation.from_quat(target[3:])
    relative = origin.inv() * goal
    return pose9(np.r_[origin.inv().apply(np.asarray(target[:3]) - current[:3]),
                       relative.as_quat()])


def read_rows(paths, *, time_coordinate='sim_elapsed', target_frame='port'):
    rows = []
    first_times = {}
    nominal_times = {}
    for family, path in paths:
        with path.open() as stream:
            for line in stream:
                source = json.loads(line)
                episode_key = (family, int(source['episode_index']))
                timestamp = float(source['timestamp_s'] if family == 'sfp' else source['sim_time'])
                first_times.setdefault(episode_key, float(source.get(
                    'episode_start_sim_time', timestamp)))
                if time_coordinate == 'nominal_decision':
                    if family == 'sfp':
                        elapsed = float(source['frame_index']) / 20.
                    else:
                        frames_file = Path(source['images']['center']).parent / 'frames.jsonl'
                        if frames_file not in nominal_times:
                            nominal_times[frames_file] = {
                                int(row['frame']): float(row['nominal_expert_time'])
                                for row in (json.loads(line) for line in frames_file.open())}
                        elapsed = nominal_times[frames_file][int(source['frame'])]
                elif time_coordinate == 'sim_elapsed':
                    elapsed = max(0., timestamp - first_times[episode_key])
                else:
                    raise ValueError(f'Unknown time coordinate: {time_coordinate}')
                targets = source['teacher_tcp_target_poses_port_frame'] if family == 'sfp' else source['teacher_tcp_target_poses_opening_frame']
                current = source['observed_tcp_pose_port_frame'] if family == 'sfp' else source['observed_tcp_pose_opening_frame']
                physical_tip = (None if family == 'sfp' else
                                source['observed_sc_tip_pose_opening_frame'])
                if len(targets) != 4:
                    raise ValueError(f'Expected four full TCP targets: {path}')
                rows.append({'family': family, 'episode': int(source['episode_index']),
                             'scene': source['scene_sha256'], 'split': source['split'],
                             'corrective_episode': bool(source.get('corrective_episode', False)),
                             'local_correction': bool(source.get('local_teacher_correction_from_failed_episode', False)),
                             'rescue_actor_active': bool(source.get('teacher_rescued_direct_actor_takeover', False)
                                                         and source.get('corrective_student_active_anchor', False)),
                             'frame': int(source['frame_index'] if family == 'sfp' else source['frame']),
                             'elapsed_s': elapsed,
                             'cache_index': source.get('cache_frame_index'),
                             'images': source.get('images'),
                             'state': np.asarray(source['state'], dtype=np.float32),
                             'task': np.asarray(source['task_vector'], dtype=np.float32),
                             'pose': pose9(current),
                             'physical_tip': (np.zeros(9, dtype=np.float32) if physical_tip is None
                                              else pose9(physical_tip)),
                             'targets': np.stack([pose9(x) if target_frame == 'port'
                                                  else relative_tcp_target9(current, x)
                                                  for x in targets])})
    scenes = defaultdict(set)
    seen_frames = {}
    for r in rows:
        frame_key = (r['family'], r['episode'], r['frame'])
        if frame_key in seen_frames:
            raise ValueError(f'Duplicate dataset episode/frame {frame_key}: '
                             f'{seen_frames[frame_key]} and {r["images"]}')
        seen_frames[frame_key] = r['images']
        if not np.isfinite(r['state']).all() or not np.isfinite(r['targets']).all():
            raise ValueError('Nonfinite state/target')
        if r['state'].shape != (32,) or r['task'].shape != (10,):
            raise ValueError('State/task schema mismatch')
        if (r['family'] == 'sfp') != (r['task'][0] == 1):
            raise ValueError('Task family one-hot mismatch')
        scenes[r['split']].add(r['scene'])
    if scenes['train'] & scenes['validation']:
        raise ValueError('Train/validation scene leakage')
    return rows


class Samples(Dataset):
    def __init__(self, rows, sfp_cache):
        self.rows = rows
        self.cache = json.loads((sfp_cache / 'cache.json').read_text())
        self.images = load_cached_images(sfp_cache, self.cache)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        images = []
        for camera in CAMERAS:
            if row['family'] == 'sfp':
                image = Image.fromarray(self.images[SFP_CAMERA_KEYS[camera]][row['cache_index']], 'RGB')
            else:
                with Image.open(row['images'][camera]) as f:
                    image = f.convert('RGB')
            image = image.resize((224, 224), Image.Resampling.BILINEAR)
            images.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1))
        return (torch.stack(images), torch.from_numpy(row['state']),
                torch.from_numpy(row['task']), torch.from_numpy(row['pose']),
                torch.from_numpy(row['targets']),
                torch.tensor(row['elapsed_s'], dtype=torch.float32), row['family'],
                torch.from_numpy(row['physical_tip']))


class SharedPortTCPActor(nn.Module):
    def __init__(self, stats, *, pretrained=True, use_time=False, use_tip=False,
                 condition_on_pose=True):
        super().__init__()
        self.use_time = use_time
        self.use_tip = use_tip
        self.condition_on_pose = condition_on_pose
        for key, value in stats.items():
            self.register_buffer(key, torch.as_tensor(value, dtype=torch.float32))
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.view_embedding = nn.Parameter(torch.zeros(3, 512))
        self.view_attention = nn.MultiheadAttention(512, 8, batch_first=True)
        self.context = nn.Sequential(nn.LayerNorm(512 + 32 + 10),
                                     nn.Linear(554, 256), nn.SiLU(), nn.Linear(256, 256))
        if use_time:
            self.time_context = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 256))
            nn.init.zeros_(self.time_context[-1].weight)
            nn.init.zeros_(self.time_context[-1].bias)
        if use_tip:
            self.tip_context = nn.Sequential(nn.LayerNorm(10), nn.Linear(10, 64),
                                             nn.SiLU(), nn.Linear(64, 256))
            nn.init.zeros_(self.tip_context[-1].weight)
            nn.init.zeros_(self.tip_context[-1].bias)
        self.pose_head = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 9))
        self.pose_context = nn.Sequential(nn.LayerNorm(256 + 9), nn.Linear(265, 256), nn.SiLU())
        self.waypoint_tokens = nn.Parameter(torch.randn(1, 4, 256) * .01)
        layer = nn.TransformerEncoderLayer(256, 8, 512, dropout=.05,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2)
        self.action_head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 9))

    def forward(self, images, state, task, elapsed=None, external_pose=None,
                external_mask=None, physical_tip=None, tip_mask=None):
        n = images.shape[0]
        x = images.float().reshape(n * 3, 3, 224, 224) / 255
        mean = x.new_tensor((.485, .456, .406))[None, :, None, None]
        std = x.new_tensor((.229, .224, .225))[None, :, None, None]
        x = (x - mean) / std
        views = self.backbone(x).reshape(n, 3, 512) + self.view_embedding[None]
        views, _ = self.view_attention(views, views, views, need_weights=False)
        state = (state - self.state_mean) / self.state_std
        context = self.context(torch.cat((views.mean(1), state, task), -1))
        if self.use_time:
            if elapsed is None:
                raise ValueError('Elapsed simulation time is required for this checkpoint')
            context = context + self.time_context(torch.log1p(elapsed.clamp_min(0))[:, None] / 4)
        if self.use_tip:
            if physical_tip is None or tip_mask is None or tip_mask.shape != (n,):
                raise ValueError('Physical plug-tip pose requires an explicit mask per row')
            tip = (physical_tip - self.tip_mean) / self.tip_std
            tip = torch.cat((tip, tip_mask[:, None].to(tip.dtype)), -1)
            context = context + self.tip_context(tip) * tip_mask[:, None]
        predicted_pose_normalized = self.pose_head(context)
        condition_pose = predicted_pose_normalized
        if external_pose is not None:
            if external_mask is None or external_mask.shape != (n,):
                raise ValueError('External pose requires one explicit mask bit per row')
            normalized_external = (external_pose - self.pose_mean) / self.pose_std
            condition_pose = torch.where(external_mask[:, None], normalized_external,
                                         predicted_pose_normalized)
        if not self.condition_on_pose:
            condition_pose = torch.zeros_like(condition_pose)
        condition = self.pose_context(torch.cat((context, condition_pose), -1))
        tokens = self.waypoint_tokens + condition[:, None]
        targets = self.action_head(self.transformer(tokens)) * self.target_std + self.target_mean
        predicted_pose = predicted_pose_normalized * self.pose_std + self.pose_mean
        return targets, predicted_pose


def error_mm(pred, truth):
    return ((pred[..., :3] - truth[..., :3]).square().sum(-1).sqrt() * 1000)


def rotation_from_pose9(pose):
    first = nn.functional.normalize(pose[..., 3:6], dim=-1)
    second = pose[..., 6:9] - (pose[..., 6:9] * first).sum(-1, keepdim=True) * first
    second = nn.functional.normalize(second, dim=-1)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def body_translation_from_poses(current, target):
    rotation = rotation_from_pose9(current)
    displacement = target[..., :3] - current[..., None, :3]
    return torch.einsum('bij,bkj->bki', rotation.transpose(1, 2), displacement)


def synthetic_sc_near_pose(pose, families, noise_mm, generator=None, *, all_phases=False):
    """Optimistic training-only proxy; never an autonomous pose estimator."""
    mask = torch.as_tensor([family == 'sc' for family in families], device=pose.device)
    if not all_phases:
        mask &= pose[:, 2].abs() < .03
    noisy = pose.clone()
    noisy[:, :3] += torch.randn(noisy[:, :3].shape, device=pose.device,
                                generator=generator) * (noise_mm / 1000)
    noisy[:, 3:] += torch.randn(noisy[:, 3:].shape, device=pose.device,
                                generator=generator) * .002
    return noisy, mask


def synthetic_sc_near_tip(tip, pose, families, noise_mm, generator=None):
    """Nondeployable physical-tip label proxy for a distinct information test."""
    mask = torch.as_tensor([family == 'sc' for family in families], device=pose.device)
    mask &= pose[:, 2].abs() < .03
    noisy = tip.clone()
    noisy[:, :3] += torch.randn(noisy[:, :3].shape, device=pose.device,
                                generator=generator) * (noise_mm / 1000)
    noisy[:, 3:] += torch.randn(noisy[:, 3:].shape, device=pose.device,
                                generator=generator) * .002
    return noisy, mask


@torch.no_grad()
def evaluate(model, loader, device, max_batches, oracle_sc_noise_mm=None,
             oracle_sc_tip_noise_mm=None, oracle_sc_all_pose=False,
             target_frame='port'):
    model.eval()
    results = defaultdict(list)
    near_lateral = defaultdict(list)
    start_command = defaultdict(list)
    initial_command = defaultdict(list)
    row_offset = 0
    oracle_generator = torch.Generator(device=device).manual_seed(20260924)
    for batch_index, (images, state, task, pose, target, elapsed, families, tip) in enumerate(loader):
        if batch_index >= max_batches:
            break
        images, state, task = images.to(device), state.to(device), task.to(device)
        target, pose = target.to(device), pose.to(device)
        external_pose = external_mask = None
        if oracle_sc_noise_mm is not None:
            external_pose, external_mask = synthetic_sc_near_pose(pose, families,
                                                                  oracle_sc_noise_mm, oracle_generator,
                                                                  all_phases=oracle_sc_all_pose)
        tip_input = tip_mask = None
        if oracle_sc_tip_noise_mm is not None:
            tip_input, tip_mask = synthetic_sc_near_tip(tip.to(device), pose, families,
                                                       oracle_sc_tip_noise_mm, oracle_generator)
        pred, guessed_pose = model(images, state, task, elapsed.to(device),
                                   external_pose, external_mask, tip_input, tip_mask)
        conditioned_pose = (torch.where(external_mask[:, None], external_pose, guessed_pose)
                            if external_pose is not None else guessed_pose)
        if target_frame == 'body':
            body_error = (pred[:, 0, :3] - target[:, 0, :3]).norm(dim=-1) * 1000
        else:
            body_error = (body_translation_from_poses(conditioned_pose.float(), pred.float()) -
                          body_translation_from_poses(pose.float(), target.float())).norm(dim=-1)[:, 0] * 1000
        for i, family in enumerate(families):
            if loader.dataset.rows[row_offset + i]['frame'] < 50:
                start_command[family].append(float(body_error[i]))
            if loader.dataset.rows[row_offset + i]['frame'] < 10:
                initial_command[family].append(float(body_error[i]))
            results[family].append((float(error_mm(pred[i, 0], target[i, 0])),
                                    float(error_mm(pred[i, -1], target[i, -1])),
                                    float(error_mm(guessed_pose[i], pose[i])),
                                    float(body_error[i])))
            near_bound = .005 if family == 'sc' else .03
            if abs(float(pose[i, 2])) < near_bound:
                if target_frame == 'body':
                    rotation = rotation_from_pose9(pose[i:i+1].float())[0]
                    delta = (rotation @ (pred[i, 0, :3] - target[i, 0, :3]))[:2] * 1000
                else:
                    delta = ((pred[i, 0, :2] - conditioned_pose[i, :2]) -
                             (target[i, 0, :2] - pose[i, :2])) * 1000
                near_lateral[family].append(float(torch.linalg.vector_norm(delta)))
        row_offset += len(families)
    report = {}
    for family, errors in results.items():
        values = np.asarray(errors)
        report[family] = {'rows': len(errors), 'first_target_mm_median': float(np.median(values[:, 0])),
                          'first_target_mm_p95': float(np.quantile(values[:, 0], .95)),
                          'fourth_target_mm_median': float(np.median(values[:, 1])),
                          'pose_mm_median': float(np.median(values[:, 2])),
                          'pose_mm_p95': float(np.quantile(values[:, 2], .95)),
                          'first_body_command_mm_median': float(np.median(values[:, 3])),
                          'first_body_command_mm_p95': float(np.quantile(values[:, 3], .95))}
        report[family]['near_lateral_command_mm_p95'] = (
            float(np.quantile(near_lateral[family], .95)) if near_lateral[family] else None)
        report[family]['near_lateral_rows'] = len(near_lateral[family])
        report[family]['start_body_command_mm_p95'] = (
            float(np.quantile(start_command[family], .95)) if start_command[family] else None)
        report[family]['start_rows'] = len(start_command[family])
        report[family]['initial_body_command_mm_p95'] = (
            float(np.quantile(initial_command[family], .95)) if initial_command[family] else None)
        report[family]['initial_rows'] = len(initial_command[family])
    model.train()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sfp-labels', type=Path, required=True)
    p.add_argument('--sc-canonical', type=Path, required=True)
    p.add_argument('--sc-supplemental', type=Path, required=True)
    p.add_argument('--sc-extra', type=Path, nargs='+',
                   help='Additional scored SC successes, all episode-grouped and admitted to BC')
    p.add_argument('--sfp-cache', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--updates', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=20260924)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--val-every', type=int, default=500)
    p.add_argument('--val-batches', type=int, default=40)
    p.add_argument('--patience', type=int, default=2500)
    p.add_argument('--init-checkpoint', type=Path)
    p.add_argument('--freeze-pose-encoder', action='store_true',
                   help='After warm start, keep visual/context/pose modules fixed and train only the complete-target action head')
    p.add_argument('--command-loss-weight', type=float, default=0.)
    p.add_argument('--start-oversample', type=float, default=1.)
    p.add_argument('--include-elapsed', action='store_true')
    p.add_argument('--time-coordinate', choices=('sim_elapsed', 'nominal_decision'),
                   default='sim_elapsed',
                   help='Clock supplied to the causal time branch; nominal_decision uses teacher command time')
    p.add_argument('--target-frame', choices=('port', 'body'), default='port',
                   help='Body is a separate direct full-target ablation, without pose conditioning')
    p.add_argument('--oracle-sc-near-pose-noise-mm', type=float,
                   help='Training-only optimistic near-port pose input; requires a real RGB estimator before deployment')
    p.add_argument('--oracle-sc-pose-all-phases', action='store_true',
                   help='Nondeployable upper bound: supply the scored SC TCP/port pose during transport too')
    p.add_argument('--oracle-sc-near-tip-noise-mm', type=float,
                   help='Training-only physical plug-tip label proxy; never autonomous evidence')
    p.add_argument('--sc-near-oversample', type=float, default=1.)
    p.add_argument('--sc-sampling-fraction', type=float, default=.5,
                   help='Before phase/episode boosts, sampled SC fraction; default balances SFP and SC equally')
    p.add_argument('--corrective-episode-boost', type=float, default=1.,
                   help='Sampling multiplier for separately admitted scored corrective episodes')
    p.add_argument('--local-correction-boost', type=float, default=1.,
                   help='Sampling multiplier only for safe, actor-owned local correction rows from failed episodes')
    p.add_argument('--rescue-actor-active-boost', type=float, default=1.,
                   help='Sampling multiplier for actor-owned rows in full-success teacher-rescued episodes')
    p.add_argument('--near-oversample', type=float, default=1.,
                   help='Training-only sampling multiplier when TCP is within 30 mm axially')
    p.add_argument('--near-lateral-loss-weight', type=float, default=0.,
                   help='Weight on complete target and command lateral accuracy near the opening')
    p.add_argument('--start-command-loss-weight', type=float, default=0.,
                   help='Extra first-command loss on the first 2.5 seconds of each episode')
    p.add_argument('--initial-command-loss-weight', type=float, default=0.,
                   help='Additional first-command loss on the first 0.5 seconds of each episode')
    p.add_argument('--selection-metric', choices=('global_command_median', 'near_lateral_p95',
                                                 'start_plus_near', 'initial_plus_near'),
                   default='global_command_median')
    a = p.parse_args()
    if not 1. <= a.corrective_episode_boost <= 20.:
        raise ValueError('Corrective episode boost must be in [1, 20]')
    if not 1. <= a.local_correction_boost <= 20.:
        raise ValueError('Local correction boost must be in [1, 20]')
    if not 1. <= a.rescue_actor_active_boost <= 20.:
        raise ValueError('Rescue actor-active boost must be in [1, 20]')
    if a.oracle_sc_pose_all_phases and a.oracle_sc_near_pose_noise_mm is None:
        raise ValueError('All-phase oracle pose requires --oracle-sc-near-pose-noise-mm')
    if not 0. <= a.start_command_loss_weight <= 10.:
        raise ValueError('Start command loss weight must be in [0, 10]')
    if not 0. <= a.initial_command_loss_weight <= 20.:
        raise ValueError('Initial command loss weight must be in [0, 20]')
    if not 0. < a.sc_sampling_fraction < 1.:
        raise ValueError('SC sampling fraction must be strictly between zero and one')
    torch.set_num_threads(8)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    sources = [('sfp', a.sfp_labels), ('sc', a.sc_canonical), ('sc', a.sc_supplemental)]
    if a.sc_extra is not None:
        sources.extend(('sc', path) for path in a.sc_extra)
    source_hashes = {str(path): sha(path) for _, path in sources}
    rows = read_rows(sources, time_coordinate=a.time_coordinate,
                     target_frame=a.target_frame)
    fit = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'validation']
    if not fit or not validation or not {'sfp', 'sc'} <= {r['family'] for r in validation}:
        raise ValueError('Need nonempty episode-grouped validation for both task families')
    stats = {}
    for name, values in [('state', np.stack([r['state'] for r in fit])),
                         ('pose', np.stack([r['pose'] for r in fit])),
                         ('target', np.concatenate([r['targets'] for r in fit]))]:
        stats[name + '_mean'] = values.mean(0)
        stats[name + '_std'] = values.std(0).clip(min=1e-3)
    if a.oracle_sc_near_tip_noise_mm is not None:
        tips = np.stack([r['physical_tip'] for r in fit if r['family'] == 'sc'])
        stats['tip_mean'] = tips.mean(0)
        stats['tip_std'] = tips.std(0).clip(min=1e-3)
    episode_counts = Counter((r['family'], r['episode']) for r in fit)
    family_episodes = Counter(k[0] for k in episode_counts)
    weights = torch.tensor([((a.sc_sampling_fraction if r['family'] == 'sc' else 1. - a.sc_sampling_fraction) /
                            family_episodes[r['family']] /
                            episode_counts[(r['family'], r['episode'])]) *
                            (a.start_oversample if r['frame'] < 50 else 1.) *
                            (a.near_oversample if abs(r['pose'][2]) < .03 else 1.) *
                            (a.corrective_episode_boost if r['corrective_episode'] else 1.) *
                            (a.local_correction_boost if r['local_correction'] else 1.) *
                            (a.rescue_actor_active_boost if r['rescue_actor_active'] else 1.) *
                            (a.sc_near_oversample if r['family'] == 'sc' and abs(r['pose'][2]) < .03 else 1.)
                            for r in fit], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, a.batch_size * a.updates, replacement=True,
                                    generator=torch.Generator().manual_seed(a.seed))
    train_loader = DataLoader(Samples(fit, a.sfp_cache), batch_size=a.batch_size,
                              sampler=sampler, num_workers=a.workers, pin_memory=True,
                              persistent_workers=a.workers > 0)
    # The validation order is deterministic and episode-grouped; it never
    # chooses a checkpoint based on reserved final scenes.
    rng = np.random.default_rng(a.seed)
    val_rows = []
    for family in ('sfp', 'sc'):
        family_rows = [r for r in validation if r['family'] == family]
        family_budget = min(len(family_rows), a.val_batches*a.batch_size//2)
        if a.selection_metric in ('near_lateral_p95', 'start_plus_near', 'initial_plus_near'):
            near_bound = .005 if family == 'sc' else .03
            near_indices = np.array([i for i, r in enumerate(family_rows)
                                     if abs(r['pose'][2]) < near_bound], dtype=int)
            target_near = min(len(near_indices), family_budget//2)
            chosen_near = rng.choice(near_indices, target_near, replace=False)
            chosen_start = np.array([], dtype=int)
            if a.selection_metric in ('start_plus_near', 'initial_plus_near'):
                start_indices = np.array([i for i, r in enumerate(family_rows)
                                          if r['frame'] < (10 if a.selection_metric == 'initial_plus_near' else 50)], dtype=int)
                start_indices = np.setdiff1d(start_indices, chosen_near)
                target_start = min(len(start_indices), family_budget//4)
                chosen_start = rng.choice(start_indices, target_start, replace=False)
            rest = np.setdiff1d(np.arange(len(family_rows)),
                                np.concatenate((chosen_near, chosen_start)))
            chosen = np.concatenate((chosen_near, chosen_start, rng.choice(rest,
                family_budget-len(chosen_near)-len(chosen_start), replace=False)))
        else:
            chosen = rng.choice(len(family_rows), family_budget, replace=False)
        val_rows.extend(family_rows[int(i)] for i in chosen)
    val_loader = DataLoader(Samples(val_rows, a.sfp_cache), batch_size=a.batch_size,
                            shuffle=False, num_workers=a.workers, pin_memory=True,
                            persistent_workers=a.workers > 0)
    device = torch.device('cuda:0')
    model = SharedPortTCPActor(stats, use_time=a.include_elapsed,
                               use_tip=a.oracle_sc_near_tip_noise_mm is not None,
                               condition_on_pose=a.target_frame == 'port').to(device)
    if a.init_checkpoint:
        previous = torch.load(a.init_checkpoint, map_location='cpu', weights_only=False)
        if previous.get('target_frame', 'port') != a.target_frame:
            raise ValueError('Warm start target frame differs; train the body-target ablation from scratch')
        prior_hashes = previous.get('source_sha256', {})
        if any(source_hashes.get(path) != digest for path, digest in prior_hashes.items()):
            raise ValueError('Warm start changed or removed an existing label source')
        if set(source_hashes) != set(prior_hashes) and not a.sc_extra:
            raise ValueError('Warm start label source set changed without --sc-extra')
        missing, unexpected = model.load_state_dict(previous['model'], strict=False)
        expected_missing = ({'time_context.0.weight', 'time_context.0.bias',
                             'time_context.2.weight', 'time_context.2.bias'}
        if a.include_elapsed and not previous.get('include_elapsed', False)
                            else set())
        if a.oracle_sc_near_tip_noise_mm is not None and not previous.get('use_tip', False):
            expected_missing |= {k for k in model.state_dict() if k.startswith('tip_') or
                                 k.startswith('tip_context.')}
        if set(missing) != expected_missing or unexpected:
            raise ValueError(f'Warm-start architecture mismatch: {missing}, {unexpected}')
    if a.freeze_pose_encoder:
        if not a.init_checkpoint:
            raise ValueError('Freezing the pose encoder requires a pretrained shared actor')
        frozen_prefixes = ('backbone.', 'view_embedding', 'view_attention.',
                           'context.', 'time_context.', 'pose_head.')
        for name, parameter in model.named_parameters():
            if name.startswith(frozen_prefixes):
                parameter.requires_grad_(False)
        frozen_modules = (model.backbone, model.view_attention,
                          model.context, model.pose_head) + (
                              (model.time_context,) if model.use_time else ())
        for module in frozen_modules:
            module.eval()
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters()
                                   if parameter.requires_grad),
                                  lr=a.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    history = []; best = float('inf'); best_update = 0
    for update, (images, state, task, pose, target, elapsed, families, tip) in enumerate(train_loader, 1):
        if a.freeze_pose_encoder:
            # evaluate() restores model.train(); keep frozen BatchNorm statistics
            # and dropout behavior fixed across every update.
            for module in frozen_modules:
                module.eval()
        images, state, task = images.to(device, non_blocking=True), state.to(device, non_blocking=True), task.to(device, non_blocking=True)
        pose, target = pose.to(device, non_blocking=True), target.to(device, non_blocking=True)
        elapsed = elapsed.to(device, non_blocking=True)
        external_pose = external_mask = None
        if a.oracle_sc_near_pose_noise_mm is not None:
            external_pose, external_mask = synthetic_sc_near_pose(
                pose, families, a.oracle_sc_near_pose_noise_mm,
                all_phases=a.oracle_sc_pose_all_phases)
        tip_input = tip_mask = None
        if a.oracle_sc_near_tip_noise_mm is not None:
            tip_input, tip_mask = synthetic_sc_near_tip(
                tip.to(device, non_blocking=True), pose, families,
                a.oracle_sc_near_tip_noise_mm)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            pred, guessed_pose = model(images, state, task, elapsed,
                                       external_pose, external_mask, tip_input, tip_mask)
            target_residual = (pred - target) / model.target_std
            pose_residual = (guessed_pose - pose) / model.pose_std
            position = nn.functional.smooth_l1_loss(target_residual[..., :3],
                                                    torch.zeros_like(target_residual[..., :3]), beta=.1)
            orientation = nn.functional.smooth_l1_loss(target_residual[..., 3:],
                                                       torch.zeros_like(target_residual[..., 3:]), beta=.1)
            pose_position = nn.functional.smooth_l1_loss(pose_residual[..., :3],
                                                         torch.zeros_like(pose_residual[..., :3]), beta=.1)
            pose_orientation = nn.functional.smooth_l1_loss(pose_residual[..., 3:],
                                                            torch.zeros_like(pose_residual[..., 3:]), beta=.1)
            command = torch.zeros((), device=device)
            start_command = torch.zeros((), device=device)
            initial_command = torch.zeros((), device=device)
            if a.command_loss_weight:
                conditioned_pose = (torch.where(external_mask[:, None], external_pose, guessed_pose)
                                    if external_pose is not None else guessed_pose)
                if a.target_frame == 'body':
                    commanded, teacher_command = pred[..., :3], target[..., :3]
                else:
                    commanded = body_translation_from_poses(conditioned_pose.float(), pred.float())
                    teacher_command = body_translation_from_poses(pose.float(), target.float())
                command = nn.functional.smooth_l1_loss(
                    (commanded - teacher_command) / .02,
                    torch.zeros_like(commanded), beta=.1)
                if a.start_command_loss_weight:
                    start = elapsed < 2.5
                    if start.any():
                        start_command = nn.functional.smooth_l1_loss(
                            (commanded[start, 0] - teacher_command[start, 0]) / .02,
                            torch.zeros_like(commanded[start, 0]), beta=.1)
                if a.initial_command_loss_weight:
                    initial = elapsed < .5
                    if initial.any():
                        initial_command = nn.functional.smooth_l1_loss(
                            (commanded[initial, 0] - teacher_command[initial, 0]) / .02,
                            torch.zeros_like(commanded[initial, 0]), beta=.1)
            near_lateral = torch.zeros((), device=device)
            if a.near_lateral_loss_weight:
                near = pose[:, 2].abs() < .03
                if near.any():
                    conditioned = (torch.where(external_mask[:, None], external_pose, guessed_pose)
                                   if external_pose is not None else guessed_pose)
                    if a.target_frame == 'body':
                        rotation = rotation_from_pose9(pose[near].float())
                        delta = torch.einsum('bij,bkj->bki', rotation,
                                             (pred[near, :, :3] - target[near, :, :3]).float())
                        target_lateral = delta[..., :2] / .002
                        command_lateral = delta[:, 0, :2] / .002
                    else:
                        target_lateral = (pred[near, :, :2] - target[near, :, :2]) / .002
                        command_lateral = ((pred[near, 0, :2] - conditioned[near, :2]) -
                                           (target[near, 0, :2] - pose[near, :2])) / .002
                    near_lateral = (nn.functional.smooth_l1_loss(target_lateral,
                        torch.zeros_like(target_lateral), beta=.2) +
                        nn.functional.smooth_l1_loss(command_lateral,
                        torch.zeros_like(command_lateral), beta=.2))
            loss = (position + .1*orientation + .3*pose_position + .03*pose_orientation +
                    a.command_loss_weight*command + a.start_command_loss_weight*start_command +
                    a.initial_command_loss_weight*initial_command +
                    a.near_lateral_loss_weight*near_lateral)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 1)
        scaler.step(optimizer); scaler.update()
        if update == 1 or update % a.val_every == 0:
            report = evaluate(model, val_loader, device, a.val_batches,
                              a.oracle_sc_near_pose_noise_mm,
                              a.oracle_sc_near_tip_noise_mm,
                              a.oracle_sc_pose_all_phases, a.target_frame)
            selection_field = ('first_body_command_mm_median' if a.command_loss_weight
                               else 'first_target_mm_median')
            if a.selection_metric in ('near_lateral_p95', 'start_plus_near', 'initial_plus_near'):
                if any(report[family]['near_lateral_command_mm_p95'] is None
                       for family in ('sfp', 'sc')):
                    raise ValueError('Near-lateral checkpoint selection has no near rows')
                key = .5*(report['sfp']['near_lateral_command_mm_p95'] +
                          report['sc']['near_lateral_command_mm_p95'])
                if a.selection_metric == 'start_plus_near':
                    if any(report[family]['start_body_command_mm_p95'] is None
                           for family in ('sfp', 'sc')):
                        raise ValueError('Start-plus-near selection has no start rows')
                    key += .05*(report['sfp']['start_body_command_mm_p95'] +
                                report['sc']['start_body_command_mm_p95'])
                if a.selection_metric == 'initial_plus_near':
                    if any(report[family]['initial_body_command_mm_p95'] is None
                           for family in ('sfp', 'sc')):
                        raise ValueError('Initial-plus-near selection has no initial rows')
                    key += .3*(report['sfp']['initial_body_command_mm_p95'] +
                               report['sc']['initial_body_command_mm_p95'])
            else:
                key = .5*(report['sfp'][selection_field] + report['sc'][selection_field])
            history.append({'update': update, 'train_loss': float(loss.detach()),
                            'command_loss': float(command.detach()), 'selection_mm': key,
                            'start_command_loss': float(start_command.detach()),
                            'initial_command_loss': float(initial_command.detach()),
                            'near_lateral_loss': float(near_lateral.detach()),
                            'validation': report})
            (a.output_dir/'history.json').write_text(json.dumps(history, indent=2)+'\n')
            if key < best - .01:
                best, best_update = key, update
                torch.save({'model': model.state_dict(), 'update': update, 'validation': report,
                            'architecture': ('shared_port_tcp_rgb_resnet18_pose_aux_transformer_time_v2'
                                             if a.include_elapsed else 'shared_port_tcp_rgb_resnet18_pose_aux_transformer_v1'),
                            'include_elapsed': a.include_elapsed,
                            'target_frame': a.target_frame,
                            'time_coordinate': a.time_coordinate,
                            'runtime_inputs': 'RGB, measured state, 10D task; predicted pose only',
                            'training_command_loss_weight': a.command_loss_weight,
                            'training_start_oversample': a.start_oversample,
                            'training_start_command_loss_weight': a.start_command_loss_weight,
                            'training_initial_command_loss_weight': a.initial_command_loss_weight,
                            'training_freeze_pose_encoder': a.freeze_pose_encoder,
                            'training_near_oversample': a.near_oversample,
                            'training_sc_sampling_fraction': a.sc_sampling_fraction,
                            'training_local_correction_boost': a.local_correction_boost,
                            'training_rescue_actor_active_boost': a.rescue_actor_active_boost,
                            'training_near_lateral_loss_weight': a.near_lateral_loss_weight,
                            'selection_metric': a.selection_metric,
                            'oracle_sc_near_pose_noise_mm': a.oracle_sc_near_pose_noise_mm,
                            'oracle_sc_pose_all_phases': a.oracle_sc_pose_all_phases,
                            'oracle_sc_near_tip_noise_mm': a.oracle_sc_near_tip_noise_mm,
                            'use_tip': a.oracle_sc_near_tip_noise_mm is not None,
                            'sc_near_oversample': a.sc_near_oversample,
                            'checkpoint_selection_field': selection_field,
                            'warm_start': str(a.init_checkpoint) if a.init_checkpoint else None,
                            'target': ('four complete TCP poses in selected port-opening frame; xyz + rotation 6D'
                                       if a.target_frame == 'port' else
                                       'four complete teacher TCP targets in observed TCP frame; xyz + rotation 6D'),
                            'train_rows': len(fit), 'validation_rows': len(validation),
                            'source_sha256': source_hashes,
                            'reserved_final_opened': False, 'rl_started': False}, a.output_dir/'best.pt')
            print(json.dumps({'update': update, 'selection_mm': key, 'best_update': best_update,
                              'sfp': report['sfp'], 'sc': report['sc']}), flush=True)
            if update - best_update >= a.patience:
                break
    summary = {'schema': 'shared_port_tcp_bc_offline/v1', 'completed_updates': update,
               'best_update': best_update, 'best_selection_mm': best,
               'train_rows': len(fit), 'validation_rows': len(validation),
               'train_episodes': len({(r['family'], r['episode']) for r in fit}),
               'validation_episodes': len({(r['family'], r['episode']) for r in validation}),
               'parameters': sum(p.numel() for p in model.parameters()),
               'command_loss_weight': a.command_loss_weight,
               'near_oversample': a.near_oversample,
               'sc_sampling_fraction': a.sc_sampling_fraction,
               'near_lateral_loss_weight': a.near_lateral_loss_weight,
               'selection_metric': a.selection_metric,
               'include_elapsed': a.include_elapsed,
               'target_frame': a.target_frame,
               'time_coordinate': a.time_coordinate,
               'start_oversample': a.start_oversample,
               'start_command_loss_weight': a.start_command_loss_weight,
               'initial_command_loss_weight': a.initial_command_loss_weight,
               'freeze_pose_encoder': a.freeze_pose_encoder,
               'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
               'oracle_sc_near_pose_noise_mm': a.oracle_sc_near_pose_noise_mm,
               'oracle_sc_pose_all_phases': a.oracle_sc_pose_all_phases,
               'oracle_sc_near_tip_noise_mm': a.oracle_sc_near_tip_noise_mm,
               'sc_near_oversample': a.sc_near_oversample,
               'corrective_episode_boost': a.corrective_episode_boost,
               'local_correction_boost': a.local_correction_boost,
               'rescue_actor_active_boost': a.rescue_actor_active_boost,
               'corrective_train_episodes': len({r['episode'] for r in fit if r['corrective_episode']}),
               'warm_start': str(a.init_checkpoint) if a.init_checkpoint else None,
               'checkpoint': str(a.output_dir/'best.pt'),
               'offline_only': True, 'reserved_final_opened': False, 'rl_started': False}
    (a.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
