#!/usr/bin/env python3
"""Fit an SFP arrival proposal without changing the transport actor's actions."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from handoff_region_prior import allows, load_prior
from train_temporal_transport_probe import LONG_AGES, TemporalTransport, build_data, predict, sha


def event_metrics(model, data, indices, prior, threshold=.95, streak=3):
    groups = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for chunk in np.array_split(indices, max(1, (len(indices)+127)//128)):
            if not len(chunk): continue
            logits = predict(model, data, torch.as_tensor(chunk, device='cuda'), False)[3]
            for i, probability in zip(chunk, logits.sigmoid().cpu().tolist()):
                row = data['rows'][int(i)]
                permitted = allows(prior, row['task'], row['state'][:3])
                groups[row['episode']].append((row['time_s'], bool(row['handoff']),
                                               probability, permitted))
    summary = Counter(); per_episode = []
    for episode, rows in groups.items():
        rows.sort()
        positives = [t for t, truth, _, _ in rows if truth]
        if not positives: raise ValueError(f'No positive safe window: {episode}')
        onset, end = min(positives), max(positives)
        run = 0; trigger = None; raw_false_rows = 0
        for t, truth, prob, permitted in rows:
            raw_false_rows += int(prob >= threshold and not truth)
            run = run+1 if prob >= threshold and permitted else 0
            if run >= streak and trigger is None: trigger = t
        outcome = ('missed' if trigger is None else 'early' if trigger < onset
                   else 'timely' if trigger <= end else 'late')
        summary[outcome] += 1
        summary['raw_false_positive_rows'] += raw_false_rows
        per_episode.append(dict(episode=episode, outcome=outcome, trigger_s=trigger,
                                safe_onset_s=onset, safe_end_s=end,
                                raw_false_positive_rows=raw_false_rows))
    return dict(episodes=len(groups), **summary), per_episode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=800)
    p.add_argument('--hard-negative-mass',type=float,default=0.,
                   help='Fraction of sampled negatives from post-cutoff head-only rows')
    a=p.parse_args()
    if not 0<=a.hard_negative_mass<1:p.error('Hard-negative mass must be in [0,1)')
    if a.output.exists():p.error('Preserve existing output')
    a.output.mkdir(parents=True)
    np.random.seed(20261005);torch.manual_seed(20261005);torch.set_num_threads(8)
    saved=torch.load(a.actor,map_location='cpu',weights_only=False)
    old=saved['source_manifest']
    if not old['long_history'] or not old['full_state_skip'] or old['route_conditioning']:
        p.error('Expected long-history, full-state, action-only actor')
    data=build_data(a.cache,a.base,'cuda',LONG_AGES)
    model=TemporalTransport(data['mean'],data['std'],raw_state_skip=True,
                            feature_dim=data['features'].shape[1],state_task_dim=42,
                            route_conditioning=False).cuda()
    model.load_state_dict(saved['model'])
    for name,param in model.named_parameters():param.requires_grad_(name.startswith('handoff.'))
    opt=torch.optim.AdamW(model.handoff.parameters(),lr=.001,weight_decay=.001)
    rows=data['rows'];prior=load_prior(a.prior)
    tr=lambda truth:np.asarray([i for i,r in enumerate(rows) if r['family']=='sfp' and
        r['split']=='train' and bool(r['handoff'])==truth],dtype=np.int64)
    positive=tr(True);negative=tr(False)
    hard_negative=np.asarray([i for i in negative if rows[i].get('past_transport_cutoff')],dtype=np.int64)
    ordinary_negative=np.asarray([i for i in negative if not rows[i].get('past_transport_cutoff')],dtype=np.int64)
    if a.hard_negative_mass and (not len(hard_negative) or not len(ordinary_negative)):
        p.error('Hard-negative sampling requires both post-cutoff and ordinary negatives')
    val=np.asarray([i for i,r in enumerate(rows) if r['family']=='sfp' and
                    r['split']=='validation'],dtype=np.int64)
    if not len(positive) or not len(negative) or not len(val):raise ValueError('Missing split')
    manifest=dict(old)
    manifest.update(handoff_aux_loss=1.,handoff_prior_required=True,
                    handoff_supported_families=['sfp'],handoff_training='frozen action-only actor; SFP linear handoff head only',
                    handoff_source_cache=str(a.cache),handoff_cache_manifest_sha256=sha(a.cache/'manifest.json'),
                    handoff_actor_sha256=sha(a.actor),handoff_prior_sha256=sha(a.prior),
                    handoff_train_positive_rows=len(positive),handoff_train_negative_rows=len(negative),
                    handoff_train_post_cutoff_negative_rows=len(hard_negative),
                    handoff_hard_negative_mass=a.hard_negative_mass,
                    handoff_validation_rows=len(val),handoff_threshold=.95,handoff_streak=3,
                    handoff_prior_json=str(a.prior),handoff_actor_checkpoint=str(a.actor),
                    handoff_update_cap=a.updates,final_scenes_opened=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    history=[];best=None;best_score=(float('inf'),float('inf'))
    for update in range(1,a.updates+1):
        n_hard=int(round(32*a.hard_negative_mass))
        sampled_negatives=(np.r_[np.random.choice(hard_negative,n_hard),
                                 np.random.choice(ordinary_negative,32-n_hard)]
                           if n_hard else np.random.choice(negative,32))
        indices=np.r_[np.random.choice(positive,32),sampled_negatives]
        np.random.shuffle(indices)
        ind=torch.as_tensor(indices,device='cuda')
        model.eval();opt.zero_grad(set_to_none=True)
        logits=predict(model,data,ind,False)[3]
        loss=nn.functional.binary_cross_entropy_with_logits(logits,data['handoff'][ind])
        loss.backward();opt.step()
        if update%100==0 or update==a.updates:
            metrics,per_episode=event_metrics(model,data,val,prior)
            # Raw false positives are unsafe even when a later streak lands
            # inside the labeled proposal window. Keep them in checkpoint
            # selection rather than reporting them only after selection.
            score=(metrics.get('early',0),metrics.get('raw_false_positive_rows',0),
                   metrics.get('missed',0)+metrics.get('late',0))
            entry=dict(update=update,loss=float(loss.detach()),metrics=metrics)
            history.append(entry);print(json.dumps(entry),flush=True)
            if score<best_score:
                best_score=score;best=dict(entry,per_episode=per_episode)
                torch.save(dict(model=model.state_dict(),update=update,
                    history_enabled=True,source_manifest=manifest,
                    final_scenes_opened=False,rl_started=False),a.output/'best.pt')
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(dict(best=best,
        action_weights_frozen=True,train_episodes=len(set(rows[i]['episode'] for i in np.r_[positive,negative])),
        validation_episodes=len(set(rows[i]['episode'] for i in val)),
        final_scenes_opened=False),indent=2)+'\n')


if __name__=='__main__':main()
