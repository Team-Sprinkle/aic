#!/usr/bin/env python3
"""Larger route-aware diffusion BC. Future measured motion is a label only."""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.nn import functional as F

from train_temporal_transport_probe import LONG_AGES, build_data, sha

DENSE_AGES = (70.,60.,50.,40.,30.,25.,20.,16.,14.,12.,10.,9.,8.,7.,6.,5.,
              4.,3.,2.5,2.,1.5,1.,.8,.6,.4,.2,0.)
FINE_AGES = (70.,60.,50.,40.,30.,25.,20.,17.,15.,13.,11.,10.,9.,8.,7.,6.,
             5.,4.,3.5,3.,2.5,2.,1.75,1.5,1.25,1.,.9,.8,.7,.6,.5,.4,
             .3,.2,.15,.1,.05,0.)


def route_targets(rows):
    target = np.zeros((len(rows), 3), np.float32)
    valid = np.zeros(len(rows), bool)
    episodes = defaultdict(list)
    for i, r in enumerate(rows):
        episodes[(r['split'], r['family'], r['episode'])].append(i)
    for ids in episodes.values():
        ids.sort(key=lambda i: rows[i]['time_s'])
        times = np.array([rows[i]['time_s'] for i in ids])
        for k, i in enumerate(ids):
            at = int(np.searchsorted(times, times[k] + 1.))
            candidates = [j for j in (at-1, at) if k < j < len(ids)
                          and .8 <= times[j]-times[k] <= 1.25]
            if not candidates:
                continue
            j = min(candidates, key=lambda j: abs(times[j]-times[k]-1.))
            current = np.array(rows[i]['state'][:3])
            future = np.array(rows[ids[j]]['state'][:3])
            quat = np.array(rows[i]['state'][3:7])
            target[i] = Rotation.from_quat(quat).inv().apply(future - current)
            valid[i] = True
    return target, valid


class RouteDiffusion(nn.Module):
    def __init__(self, feature_dim, target_mean, target_std, width=384,
                 history_layers=3, denoise_layers=6, steps=100, action_horizon=4,
                 handoff_head=False):
        super().__init__()
        self.steps = steps
        self.action_horizon = action_horizon
        self.register_buffer('target_mean', torch.as_tensor(target_mean, dtype=torch.float32))
        self.register_buffer('target_std', torch.as_tensor(target_std, dtype=torch.float32))
        t = torch.arange(steps + 1, dtype=torch.float64)
        alpha = torch.cos(((t / steps + .008) / 1.008) * torch.pi / 2).square()
        self.register_buffer('alpha_bar', (alpha[1:] / alpha[0]).clamp(1e-5, 1).float())
        self.history_input = nn.Sequential(nn.LayerNorm(feature_dim + 48 + 42),
                                           nn.Linear(feature_dim + 48 + 42, width), nn.GELU())
        self.task_token = nn.Linear(10, width)
        history_layer = nn.TransformerEncoderLayer(width, 8, width * 3,
            dropout=.05, activation='gelu', batch_first=True, norm_first=True)
        self.history_net = nn.TransformerEncoder(history_layer, history_layers,
                                                 enable_nested_tensor=False)
        self.goal_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                       nn.GELU(), nn.Linear(width, 3))
        self.handoff_head = (nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                          nn.GELU(), nn.Linear(width, 1))
                             if handoff_head else None)
        self.goal_token = nn.Sequential(nn.Linear(3, width), nn.GELU(),
                                        nn.Linear(width, width))
        self.noisy_input = nn.Linear(9, width)
        self.time_embedding = nn.Embedding(steps, width)
        self.action_position = nn.Embedding(action_horizon, width)
        denoise_layer = nn.TransformerEncoderLayer(width, 8, width * 3,
            dropout=.05, activation='gelu', batch_first=True, norm_first=True)
        self.denoiser = nn.TransformerEncoder(denoise_layer, denoise_layers,
                                             enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 9))

    def context(self, data, indices, prior_dropout=0., prior_lag_max=0,
                prior_lag_probability=0., eval_prior_lag=0):
        h = data['history'][indices]
        mask = data['mask'][indices]
        clock = data['camera_clock'][h].float().clone()
        clock[..., :3] /= .05
        previous = data['prev'][h].float()
        if eval_prior_lag:
            previous=data[f'prev_lag{eval_prior_lag}'][h].float()
        elif prior_lag_max and prior_lag_probability:
            choice=torch.randint(1,prior_lag_max+1,previous.shape[:2],device=previous.device)
            delayed=torch.stack([data[f'prev_lag{k}'][h].float()
                                 for k in range(1,prior_lag_max+1)],2)
            selected=torch.gather(delayed,2,
                (choice-1)[...,None,None].expand(*choice.shape,1,previous.shape[-1])).squeeze(2)
            keep_delayed=(torch.rand((*previous.shape[:2],1),device=previous.device)
                          <prior_lag_probability)
            previous=torch.where(keep_delayed,selected,previous)
        if prior_dropout:
            keep = torch.rand((*previous.shape[:2], 1), device=previous.device) >= prior_dropout
            previous = previous * keep
        x = torch.cat((data['features'][h].float(), previous,
                       data['motion'][h].float(), data['age'][indices][..., None],
                       data['gap'][indices][..., None], clock,
                       data['state_task_full'][h].float()), -1)
        x = self.history_net(self.history_input(x), src_key_padding_mask=~mask)
        return x[:, -1] + self.task_token(data['state_task_full'][indices, -10:].float())

    def goal(self, context):
        return self.goal_head(context)

    def forward(self, context, goal, noisy, timestep):
        time_emb = self.time_embedding(timestep)
        pos = self.action_position(torch.arange(self.action_horizon, device=noisy.device))[None]
        action = self.noisy_input(noisy) + time_emb[:, None] + pos
        tokens = torch.cat((context[:, None] + time_emb[:, None],
                            self.goal_token(goal)[:, None], action), 1)
        return self.output(self.denoiser(tokens)[:, 2:])

    @torch.no_grad()
    def sample(self, context, seed=1234, sample_steps=12, goal_override=None):
        generator = torch.Generator(device=context.device).manual_seed(seed)
        x = torch.randn((len(context), self.action_horizon, 9), generator=generator, device=context.device)
        goal = self.goal(context) if goal_override is None else goal_override
        schedule = np.linspace(self.steps - 1, 0, sample_steps, dtype=int).tolist()
        for k, step in enumerate(schedule):
            t = torch.full((len(context),), step, device=context.device, dtype=torch.long)
            alpha = self.alpha_bar[step]
            velocity = self(context, goal, x, t)
            clean = alpha.sqrt() * x - (1 - alpha).sqrt() * velocity
            eps = (1 - alpha).sqrt() * x + alpha.sqrt() * velocity
            next_step = schedule[k+1] if k+1 < len(schedule) else -1
            next_alpha = self.alpha_bar[next_step] if next_step >= 0 else alpha.new_tensor(1.)
            x = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * eps
        return x, goal


def sampling_weights(rows, train, corrective_mass=0., anchor_mass=0.,
                     sc_early_weight=1., sc_late_weight=1.):
    episode_rows = Counter((rows[i]['family'], rows[i]['episode']) for i in train)
    request_episodes = defaultdict(set)
    for i in train:
        r = rows[i]
        if r['family'] == 'sfp':
            code = (int(np.argmax(r['task'][4:9])), int(np.argmax(r['task'][2:4])))
            request_episodes[code].add(r['episode'])
    weights = []
    for i in train:
        r = rows[i]
        w = 1 / episode_rows[(r['family'], r['episode'])]
        if r['family'] == 'sfp':
            code = (int(np.argmax(r['task'][4:9])), int(np.argmax(r['task'][2:4])))
            w /= len(request_episodes[code])
            if 4 <= r['time_s'] <= 12:
                w *= 2
        elif r['family'] == 'sc':
            if 3 <= r['time_s'] < 12:
                w *= sc_early_weight
            elif 35 <= r['time_s'] < 70:
                w *= sc_late_weight
        weights.append(w)
    weights = np.asarray(weights, float)
    if corrective_mass:
        sfp=np.asarray([rows[i]['family']=='sfp' for i in train])
        corrective=np.asarray([bool(rows[i].get('corrective_teacher')) for i in train])
        if not 0<corrective_mass<1 or not np.any(sfp & corrective):
            raise ValueError('Corrective SFP mass requires valid corrective teacher rows')
        ordinary=sfp & ~corrective
        weights[sfp & corrective]*=corrective_mass/weights[sfp & corrective].sum()
        weights[ordinary]*=(1-corrective_mass)/weights[ordinary].sum()
    if anchor_mass:
        if not 0<anchor_mass<1:raise ValueError('anchor_mass must be in (0,1)')
        family=np.asarray([rows[i]['family'] for i in train])
        anchor=np.asarray([bool(rows[i].get('official_anchor_teacher')) for i in train])
        for name in ('sfp','sc'):
            selected=family==name
            if not np.any(selected & anchor) or not np.any(selected & ~anchor):
                raise ValueError(f'Missing old or official-anchor {name} rows')
            weights[selected & anchor]*=anchor_mass/weights[selected & anchor].sum()
            weights[selected & ~anchor]*=(1-anchor_mass)/weights[selected & ~anchor].sum()
    present=[family for family in ('sfp','sc') if any(rows[i]['family']==family for i in train)]
    for family in present:
        mask = np.asarray([rows[i]['family'] == family for i in train])
        weights[mask] *= (1/len(present)) / weights[mask].sum()
    return weights / weights.sum()


@torch.inference_mode()
def evaluate(model, data, selected, route, valid, sample_steps, seed,
             prior_dropout=0., eval_prior_lag=0, execute_steps=4):
    model.eval()
    groups = defaultdict(lambda: defaultdict(list))
    for offset in range(0, len(selected), 48):
        ids = selected[offset:offset+48]
        idx = torch.as_tensor(ids, device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            context = model.context(data, idx, prior_dropout=prior_dropout,
                                    eval_prior_lag=eval_prior_lag)
            action, goal = model.sample(context, seed+offset, sample_steps)
        predicted = action.float() * model.target_std + model.target_mean
        step_error = (predicted[:, :, :3] - data['target'][idx, :, :3]).norm(dim=-1)*1000
        slot_mask = data.get('action_valid')
        slots = slot_mask[idx] if slot_mask is not None else torch.ones_like(step_error, dtype=torch.bool)
        first_valid=slots.to(torch.int64).argmax(1)
        action_error=step_error.gather(1,first_valid[:,None]).squeeze(1)
        def rotation_matrix(x):
            first = F.normalize(x[..., 3:6], dim=-1)
            second = F.normalize(x[..., 6:9]-(first*x[..., 6:9]).sum(-1,keepdim=True)*first, dim=-1)
            return torch.stack((first, second, torch.cross(first,second,dim=-1)), -1)
        relative = rotation_matrix(predicted).transpose(-1,-2) @ rotation_matrix(data['target'][idx])
        orientation_error = torch.acos(((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1))*180/torch.pi
        goal_error = (goal.float() - route[idx]).norm(dim=-1)*50
        for j, i in enumerate(ids):
            row = data['rows'][i]
            phase = 'early' if row['family']=='sfp' and 4 <= row['time_s'] <= 12 else 'other'
            key = f"{row['family']}_{phase}"
            keys=(key,f"{row['family']}_{'official_anchor' if row.get('official_anchor_teacher') else 'existing'}")
            if row['family']=='sc':
                t=row['time_s']
                sc_phase=('early' if 3<=t<12 else 'middle' if 12<=t<35
                          else 'late' if 35<=t<70 else 'other_time')
                keys=(*keys,f'sc_{sc_phase}')
            for group_key in keys:
                if not bool(slots[j].any()):
                    continue
                groups[group_key]['action'].append(float(action_error[j]))
                executed = slots[j,:execute_steps]
                groups[group_key]['executed'].append(float(step_error[j,:execute_steps][executed].mean())
                                                     if bool(executed.any()) else float(step_error[j][slots[j]][0]))
                groups[group_key]['full'].append(float(step_error[j][slots[j]].mean()))
                groups[group_key]['last'].append(float(step_error[j][slots[j]][-1]))
                groups[group_key]['orientation'].append(float(orientation_error[j][slots[j]].mean()))
                if valid[i]:
                    groups[group_key]['goal'].append(float(goal_error[j]))
    return {key:dict(rows=len(v['action']),
                     action_first_mm_p95=float(np.quantile(v['action'], .95)),
                     executed_mean_mm_p95=float(np.quantile(v['executed'], .95)),
                     full_horizon_mean_mm_p95=float(np.quantile(v['full'], .95)),
                     final_slot_mm_p95=float(np.quantile(v['last'], .95)),
                     orientation_mean_deg_p95=float(np.quantile(v['orientation'], .95)),
                     goal_1s_mm_p95=(float(np.quantile(v['goal'], .95)) if v['goal'] else None),
                     goal_valid_rows=len(v['goal']))
            for key, v in groups.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=8000)
    p.add_argument('--budget-minutes', type=float, default=240)
    p.add_argument('--batch-size', type=int, default=48)
    p.add_argument('--sample-steps', type=int, default=12)
    p.add_argument('--action-horizon',type=int,choices=(4,8,12),default=4)
    p.add_argument('--execute-steps',type=int,default=4)
    p.add_argument('--train-family',choices=('both','sc','sfp'),default='both')
    p.add_argument('--history-context',choices=('long','dense70','fine70'),default='long')
    p.add_argument('--select-on',choices=('validation','train'),default='validation',
                   help='Use train only for an intentional memorization diagnostic')
    p.add_argument('--extended-labels',type=Path)
    p.add_argument('--partial-action-supervision',action='store_true',
                   help='Use valid future slots without excluding terminal anchors')
    p.add_argument('--handoff-aux-loss',type=float,default=0.,
                   help='Independent stop-head BCE on all non-context-only family rows')
    p.add_argument('--evaluation-labels',type=Path,
                   help='Common timestamp-repaired reference and masks for both comparison arms')
    p.add_argument('--min-updates',type=int,default=3000)
    p.add_argument('--plateau-patience-updates',type=int,default=2500)
    p.add_argument('--seed', type=int, default=20261011)
    p.add_argument('--init', type=Path)
    p.add_argument('--paired-audit', type=Path)
    p.add_argument('--pair-loss-weight', type=float, default=0.)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--corrective-mass', type=float, default=0.)
    p.add_argument('--anchor-mass', type=float, default=0.,
                   help='Per-family mass for verified public-qualification-family anchor episodes')
    p.add_argument('--sc-early-weight',type=float,default=1.)
    p.add_argument('--sc-late-weight',type=float,default=1.)
    p.add_argument('--sc-phase-selection',action='store_true',
                   help='Select SC checkpoint by held-out early plus late phase errors')
    p.add_argument('--prior-dropout', type=float, default=.1,
                   help='Chance of hiding each prior teacher chunk during training; 1.0 removes the shortcut')
    p.add_argument('--prior-lag-max',type=int,default=0,
                   help='Maximum extra teacher-decision delay in prior target during training')
    p.add_argument('--prior-lag-probability',type=float,default=0.,
                   help='Chance each history token uses a delayed prior target')
    a = p.parse_args()
    if a.output.exists():
        p.error('Output already exists')
    if not 0<=a.prior_dropout<=1:p.error('prior-dropout must be in [0,1]')
    if a.prior_lag_max not in (0,1,2):p.error('prior-lag-max must be 0, 1, or 2')
    if not 0<=a.prior_lag_probability<=1:p.error('prior-lag-probability must be in [0,1]')
    if bool(a.prior_lag_max)!=bool(a.prior_lag_probability):
        p.error('Set both prior-lag-max and prior-lag-probability, or neither')
    if not 1<=a.execute_steps<=a.action_horizon:p.error('Invalid execute steps')
    if a.sc_early_weight<=0 or a.sc_late_weight<=0:p.error('SC phase weights must be positive')
    if a.sc_phase_selection and (a.train_family!='sc' or a.select_on!='validation'):
        p.error('SC phase selection requires SC-only validation selection')
    if a.action_horizon>4 and a.extended_labels is None:p.error('Long horizon requires extended labels')
    a.output.mkdir(parents=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(8)
    history_ages = {'long':LONG_AGES,'dense70':DENSE_AGES,
                    'fine70':FINE_AGES}[a.history_context]
    data = build_data(a.cache, a.checkpoint, 'cuda', history_ages)
    longer_valid=None
    if a.extended_labels is not None:
        extended_manifest=json.loads((a.extended_labels/'manifest.json').read_text())
        if extended_manifest['rows']!=len(data['rows']):raise ValueError('Extended label row count mismatch')
        extended=np.load(a.extended_labels/'targets12.npy',mmap_mode='r')
        valid_slots=np.load(a.extended_labels/'valid12.npy',mmap_mode='r')
        if extended_manifest['slot_s']==.05 and not np.allclose(extended[:,:4],data['target'].cpu().numpy(),atol=1e-5):
            raise ValueError('Extended original slots do not match frozen cache')
        longer_valid=np.asarray(valid_slots[:,:a.action_horizon].any(1) if a.partial_action_supervision
                                else valid_slots[:,a.action_horizon-1],bool)
        data['target']=torch.as_tensor(np.array(extended[:,:a.action_horizon],copy=True),device='cuda')
        data['action_valid']=torch.as_tensor(np.array(valid_slots[:,:a.action_horizon],copy=True),device='cuda')
        if not a.partial_action_supervision:
            # Legacy fully valid anchors remain the matched control.
            data['action_valid']=torch.ones_like(data['action_valid'])
        if extended_manifest['slot_s']!=.05:
            # Previous-command history must have the same time spacing as the
            # newly supervised action chunk. build_data initially used 50 ms.
            normalized=((data['target'][:,:4]-torch.as_tensor(data['mean'],device='cuda')) /
                        torch.as_tensor(data['std'],device='cuda'))
            previous=torch.zeros_like(data['prev'])
            groups=defaultdict(list)
            for i,row in enumerate(data['rows']):groups[(row['family'],row['episode'])].append(i)
            for ids in groups.values():
                for earlier,later in zip(ids,ids[1:]):
                    if data['rows'][later]['time_s']-data['rows'][earlier]['time_s']<=4.:
                        previous[later]=normalized[earlier].reshape(-1)
            data['prev']=previous
    if a.prior_lag_max:
        groups=defaultdict(list)
        for i,row in enumerate(data['rows']):groups[(row['family'],row['episode'])].append(i)
        for lag in range(1,a.prior_lag_max+1):
            delayed=torch.zeros_like(data['prev'])
            for ids in groups.values():
                if len(ids)>lag:
                    delayed[ids[lag:]]=data['prev'][ids[:-lag]]
            data[f'prev_lag{lag}']=delayed
    rows = data['rows']
    stop_train=np.array([i for i,r in enumerate(rows) if r['split']=='train'
                         and not r.get('actor_prefix_context_only',False)
                         and r['handoff']>=0
                         and (a.train_family=='both' or r['family']==a.train_family)],np.int64)
    stop_val=np.array([i for i,r in enumerate(rows) if r['split']=='validation'
                       and not r.get('actor_prefix_context_only',False)
                       and r['handoff']>=0
                       and (a.train_family=='both' or r['family']==a.train_family)],np.int64)
    stop_labels=torch.as_tensor([float(r['handoff']) for r in rows],device='cuda')
    train = np.array([i for i,r in enumerate(rows) if r['split']=='train'
                      and not r.get('actor_prefix_context_only',False)], np.int64)
    val = np.array([i for i,r in enumerate(rows) if r['split']=='validation'
                    and not r.get('actor_prefix_context_only',False)], np.int64)
    if longer_valid is not None:
        train=train[longer_valid[train]];val=val[longer_valid[val]]
    if a.train_family!='both':
        train=np.asarray([i for i in train if rows[i]['family']==a.train_family],dtype=np.int64)
        val=np.asarray([i for i in val if rows[i]['family']==a.train_family],dtype=np.int64)
    if not len(train) or not len(val):raise ValueError('Empty selected family split')
    assert not {rows[i]['scene'] for i in train} & {rows[i]['scene'] for i in val}
    target, valid = route_targets(rows)
    if valid[train].mean() < .1 or valid[val].mean() < .1:
        raise ValueError('Too few valid future route labels for this decision cadence')
    route = torch.as_tensor(target/.05, device='cuda')
    valid_t = torch.as_tensor(valid, device='cuda')
    weights = sampling_weights(rows, train, a.corrective_mass, a.anchor_mass,
                               a.sc_early_weight, a.sc_late_weight)
    rng = np.random.default_rng(a.seed)
    episodes = defaultdict(list)
    evaluation_ids=(np.array([i for i,r in enumerate(rows) if r['split']=='validation'
                             and not r.get('actor_prefix_context_only',False)
                             and (a.train_family=='both' or r['family']==a.train_family)],np.int64)
                    if a.evaluation_labels is not None else val)
    evaluation_data=data
    if a.evaluation_labels is not None:
        reference_manifest=json.loads((a.evaluation_labels/'manifest.json').read_text())
        if reference_manifest['rows']!=len(rows) or reference_manifest['slot_s']!=extended_manifest['slot_s']:
            raise ValueError('Common evaluation reference is misaligned')
        evaluation_data=dict(data)
        evaluation_data['target']=torch.as_tensor(np.array(np.load(a.evaluation_labels/'targets12.npy')[:,:a.action_horizon],copy=True),device='cuda')
        evaluation_data['action_valid']=torch.as_tensor(np.array(np.load(a.evaluation_labels/'valid12.npy')[:,:a.action_horizon],copy=True),device='cuda')
    for i in evaluation_ids:
        episodes[(rows[i]['family'],rows[i]['episode'])].append(int(i))
    selected = []
    for ids in episodes.values():
        ids.sort(key=lambda i: rows[i]['time_s'])
        selected += np.asarray(ids)[np.linspace(0,len(ids)-1,min(12,len(ids))).astype(int)].tolist()
    if a.evaluation_labels is not None:
        selected += [int(i) for i in evaluation_ids if rows[i]['handoff']]
    selected = sorted(set(selected))
    train_selected=[]
    train_episodes=defaultdict(list)
    for i in train:train_episodes[(rows[i]['family'],rows[i]['episode'])].append(int(i))
    for ids in train_episodes.values():
        ids.sort(key=lambda i:rows[i]['time_s'])
        train_selected+=np.asarray(ids)[np.linspace(0,len(ids)-1,min(12,len(ids))).astype(int)].tolist()
    train_selected=sorted(set(train_selected))
    model = RouteDiffusion(data['features'].shape[1], data['mean'], data['std'],
                           action_horizon=a.action_horizon,handoff_head=a.handoff_aux_loss>0).cuda()
    if a.init:
        pretrained=torch.load(a.init,map_location='cpu',weights_only=False)['model']
        if pretrained['action_position.weight'].shape[0]!=a.action_horizon:
            old=pretrained['action_position.weight']
            if old.shape[0]!=4:raise ValueError('Expected four-slot initialization')
            grown=model.action_position.weight.detach().cpu().clone()
            grown[:4]=old
            for j in range(4,a.action_horizon):
                grown[j]=old[-1]+.01*torch.randn_like(old[-1])
            pretrained['action_position.weight']=grown
        missing,unexpected=model.load_state_dict(pretrained,strict=False)
        if unexpected or any(not k.startswith('handoff_head.') for k in missing):
            raise ValueError(f'Unexpected warm-start mismatch: {missing}, {unexpected}')
    ema = copy.deepcopy(model).eval()
    pairs=[]
    if a.paired_audit:
        audit=json.loads(a.paired_audit.read_text())
        for record in audit['records']:
            if record['split']!='train' or not record['same_observation'] or record['future_path_delta_mm']<10:
                continue
            left,right=record['index_a'],record['index_b']
            assert rows[left]['scene']==rows[right]['scene']==record['scene']
            if valid[left] and valid[right]:pairs.append((left,right))
        if not pairs:raise ValueError('No valid training-only request pairs')
    opt = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=1e-4)
    manifest = dict(schema='aic_route_aware_diffusion_transport/v1',
        cache=str(a.cache.resolve()), cache_manifest_sha256=sha(a.cache/'manifest.json'),
        checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=sha(a.checkpoint),
        train_rows=len(train), development_rows=len(val),
        train_reset_groups=len({rows[i]['scene'] for i in train}),
        development_reset_groups=len({rows[i]['scene'] for i in val}),
        trainable_parameters=sum(v.numel() for v in model.parameters()),
        frozen_encoder_parameters=11176512,
        route_label='nearest measured future 0.8-1.25s TCP translation in current TCP frame',
        action_label=f'{a.action_horizon} complete recorded TCP-body pose9 targets; no residual adapter',
        action_horizon=a.action_horizon,execute_steps=a.execute_steps,
        action_slot_s=(extended_manifest['slot_s'] if a.extended_labels else .05),
        history_ages_s=list(history_ages),
        train_family=a.train_family,selection_split=a.select_on,
        extended_labels=str(a.extended_labels) if a.extended_labels else None,
        model='3-layer 384D history + 6-layer 384D action denoiser; frozen three-view ImageNet ResNet18',
        noise_steps=100, prediction_type='velocity', inference_steps=a.sample_steps, update_cap=a.updates,
        budget_minutes=a.budget_minutes, seed=a.seed, privileged_runtime_input=False,
        final_scenes_opened=False,init_checkpoint=str(a.init) if a.init else None,
        paired_audit=str(a.paired_audit) if a.paired_audit else None,
        train_request_pairs=len(pairs),pair_loss_weight=a.pair_loss_weight,
        learning_rate=a.learning_rate,corrective_mass_within_sfp=a.corrective_mass,
        official_anchor_mass_within_each_family=a.anchor_mass,
        sc_early_weight=a.sc_early_weight,sc_late_weight=a.sc_late_weight,
        sc_phase_selection=a.sc_phase_selection,
        prior_dropout=a.prior_dropout,
        validation_prior_dropout=(1. if a.prior_dropout==1. else 0.),
        prior_lag_max=a.prior_lag_max,prior_lag_probability=a.prior_lag_probability,
        validation_prior_lag=(1 if a.prior_lag_max else 0),
        excluded_actor_prefix_context_rows=sum(bool(r.get('actor_prefix_context_only')) for r in rows),
        min_updates=a.min_updates,plateau_patience_updates=a.plateau_patience_updates)
    manifest.update(partial_action_supervision=a.partial_action_supervision,
                    evaluation_labels=str(a.evaluation_labels) if a.evaluation_labels else None,
                    handoff_aux_loss=a.handoff_aux_loss,stage_aux_loss=0.,
                    handoff_supported_families=([a.train_family] if a.train_family!='both' else ['sc','sfp']),
                    stop_train_rows=len(stop_train),stop_val_rows=len(stop_val),
                    stop_train_positives=int(stop_labels[stop_train].sum()),
                    stop_val_positives=int(stop_labels[stop_val].sum()))
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.output/'selected_validation_rows.json').write_text(json.dumps(selected)+'\n')
    print(json.dumps(dict(event='initialized',params=manifest['trainable_parameters'],
                          train=len(train),val=len(val),valid_route_train=int(valid[train].sum()),
                          valid_route_val=int(valid[val].sum()))),flush=True)
    started=time.monotonic(); best=float('inf'); best_update=0; history=[]; stop='update_cap'
    for update in range(1,a.updates+1):
        if time.monotonic()-started>a.budget_minutes*60:
            stop='wall_budget';break
        ids=rng.choice(train,a.batch_size,replace=True,p=weights)
        idx=torch.as_tensor(ids,device='cuda')
        model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            context=model.context(data,idx,prior_dropout=a.prior_dropout,
                                  prior_lag_max=a.prior_lag_max,
                                  prior_lag_probability=a.prior_lag_probability)
            goal=model.goal(context)
            action=(data['target'][idx]-model.target_mean)/model.target_std
            slot_mask = data.get('action_valid')
            action_mask = slot_mask[idx] if slot_mask is not None else torch.ones(action.shape[:2],device='cuda',dtype=torch.bool)
            action=torch.where(action_mask[...,None],action,torch.zeros_like(action))
            step=torch.randint(0,model.steps,(len(idx),),device='cuda')
            noise=torch.randn_like(action)
            alpha=model.alpha_bar[step,None,None]
            noisy=alpha.sqrt()*action+(1-alpha).sqrt()*noise
            predicted=model(context,goal,noisy,step)
            velocity_target=alpha.sqrt()*noise-(1-alpha).sqrt()*action
            errors=(predicted.float()-velocity_target.float()).square()
            diffusion_loss=(errors*action_mask[...,None]).sum()/(action_mask.sum().clamp_min(1)*9)
            mask=valid_t[idx]
            goal_loss=(F.smooth_l1_loss(goal[mask].float(),route[idx][mask])
                       if mask.any() else goal.sum()*0)
            loss=diffusion_loss+3*goal_loss
            handoff_loss=loss.new_zeros(())
            if a.handoff_aux_loss:
                positive=stop_train[np.asarray([rows[i]['handoff']>0 for i in stop_train])]
                negative=stop_train[np.asarray([rows[i]['handoff']<=0 for i in stop_train])]
                if not len(positive) or not len(negative):
                    raise ValueError('Stop supervision needs positives and negatives')
                stop_ids=np.r_[rng.choice(positive,max(1,a.batch_size//2),replace=True),
                               rng.choice(negative,max(1,a.batch_size//2),replace=True)]
                stop_idx=torch.as_tensor(stop_ids,device='cuda')
                stop_context=model.context(data,stop_idx,prior_dropout=a.prior_dropout)
                handoff_loss=F.binary_cross_entropy_with_logits(model.handoff_head(stop_context).float().squeeze(-1),stop_labels[stop_idx])
                loss=loss+a.handoff_aux_loss*handoff_loss
            if pairs and a.pair_loss_weight:
                pair_ids=rng.choice(len(pairs),min(8,len(pairs)),replace=True)
                left=torch.as_tensor([pairs[j][0] for j in pair_ids],device='cuda')
                right=torch.as_tensor([pairs[j][1] for j in pair_ids],device='cuda')
                left_context=model.context(data,left,prior_dropout=a.prior_dropout,
                    prior_lag_max=a.prior_lag_max,prior_lag_probability=a.prior_lag_probability)
                right_context=model.context(data,right,prior_dropout=a.prior_dropout,
                    prior_lag_max=a.prior_lag_max,prior_lag_probability=a.prior_lag_probability)
                left_goal=model.goal(left_context)
                right_goal=model.goal(right_context)
                goal_difference=route[left]-route[right]
                pair_goal_loss=F.smooth_l1_loss((left_goal-right_goal).float(),goal_difference)
                left_action=(data['target'][left]-model.target_mean)/model.target_std
                right_action=(data['target'][right]-model.target_mean)/model.target_std
                pair_step=torch.randint(75,model.steps,(len(left),),device='cuda')
                pair_alpha=model.alpha_bar[pair_step,None,None]
                shared_noise=torch.randn_like(left_action)
                left_noisy=pair_alpha.sqrt()*left_action+(1-pair_alpha).sqrt()*shared_noise
                right_noisy=pair_alpha.sqrt()*right_action+(1-pair_alpha).sqrt()*shared_noise
                left_v=model(left_context,left_goal,left_noisy,pair_step)
                right_v=model(right_context,right_goal,right_noisy,pair_step)
                left_clean=pair_alpha.sqrt()*left_noisy-(1-pair_alpha).sqrt()*left_v
                right_clean=pair_alpha.sqrt()*right_noisy-(1-pair_alpha).sqrt()*right_v
                action_difference=left_action[:,0,:3]-right_action[:,0,:3]
                pair_action_loss=F.smooth_l1_loss((left_clean-right_clean)[:,0,:3].float(),action_difference)
                loss=loss+a.pair_loss_weight*(pair_goal_loss+pair_action_loss)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        with torch.no_grad():
            for shadow,current in zip(ema.parameters(),model.parameters()):
                shadow.lerp_(current,.005)
            for shadow,current in zip(ema.buffers(),model.buffers()):
                shadow.copy_(current)
        if update%500==0 or update==a.updates:
            metrics=evaluate(ema,evaluation_data,selected,route,valid,a.sample_steps,a.seed,
                             prior_dropout=manifest['validation_prior_dropout'],
                             eval_prior_lag=manifest['validation_prior_lag'],execute_steps=a.execute_steps)
            train_metrics=(evaluate(ema,data,train_selected,route,valid,a.sample_steps,a.seed,
                                    prior_dropout=manifest['validation_prior_dropout'],
                                    eval_prior_lag=manifest['validation_prior_lag'],execute_steps=a.execute_steps)
                           if a.select_on=='train' else None)
            scoring_metrics=train_metrics if train_metrics is not None else metrics
            stop_metrics=None
            if a.handoff_aux_loss:
                probabilities=[]
                with torch.inference_mode():
                    for start in range(0,len(stop_val),48):
                        stop_idx=torch.as_tensor(stop_val[start:start+48],device='cuda')
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            c=ema.context(data,stop_idx,prior_dropout=manifest['validation_prior_dropout'])
                            probabilities.extend(ema.handoff_head(c).float().squeeze(-1).sigmoid().cpu().tolist())
                labels=stop_labels[stop_val].cpu().numpy();probabilities=np.asarray(probabilities)
                bce=float(-(labels*np.log(np.maximum(probabilities,1e-7))+(1-labels)*np.log(np.maximum(1-probabilities,1e-7))).mean())
                stop_metrics=dict(rows=len(labels),positives=int(labels.sum()),bce=bce,thresholds={})
                for threshold in (.5,.8,.9,.95,.98,.99):
                    accepts=probabilities>=threshold
                    stop_metrics['thresholds'][str(threshold)]=dict(tp=int((accepts & (labels>0)).sum()),fp=int((accepts & (labels<=0)).sum()),fn=int((~accepts & (labels>0)).sum()))
            if a.sc_phase_selection:
                score=(metrics['sc_early']['executed_mean_mm_p95']+
                       metrics['sc_late']['executed_mean_mm_p95'])
            elif a.anchor_mass:
                keys=('sfp_existing','sfp_official_anchor','sc_existing','sc_official_anchor')
                measure='action_first_mm_p95' if a.action_horizon==4 else 'executed_mean_mm_p95'
                score=sum(metrics[k][measure] for k in keys)
            else:
                measure='action_first_mm_p95' if a.action_horizon==4 else 'executed_mean_mm_p95'
                score=sum(v[measure] for k,v in scoring_metrics.items()
                          if k in ('sfp_early','sfp_other','sc_other'))
            entry=dict(update=update,elapsed_minutes=(time.monotonic()-started)/60,
                       loss=float(loss),diffusion_loss=float(diffusion_loss),
                       goal_loss=float(goal_loss),score=score,validation=metrics)
            if stop_metrics is not None:
                score+=10*stop_metrics['bce']
                entry.update(score=score,handoff_loss=float(handoff_loss),stop_validation=stop_metrics)
            if train_metrics is not None:entry['train_diagnostic']=train_metrics
            if pairs and a.pair_loss_weight:
                entry.update(pair_goal_loss=float(pair_goal_loss),pair_action_loss=float(pair_action_loss))
            history.append(entry);print(json.dumps(entry),flush=True)
            if score<best:
                best=score;best_update=update
                torch.save(dict(model=ema.state_dict(),update=update,
                                source_manifest=manifest,validation=metrics),
                           a.output/'best.pt')
            (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
            if len(history)>=6 and update>=a.min_updates:
                recent=[e for e in history if e['update']>update-a.plateau_patience_updates]
                if recent and min(e['score'] for e in recent)>=best*.995 and best_update<=update-a.plateau_patience_updates:
                    stop='validation_plateau';break
    result=dict(best_update=best_update,updates_completed=update,stop_reason=stop,
                best_score=best,best_validation=next((e['validation'] for e in history
                                                      if e['update']==best_update),None),
                autonomous_gate_passed=False,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(event='finished',**result)),flush=True)


if __name__=='__main__':
    main()
