#!/usr/bin/env python3
"""Grouped offline full-action and withheld recovery-direction comparison."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from train_temporal_transport_probe import (
    LONG_AGES, TemporalTransport, build_data, predict, rotation_from_pose9)


def family_group(row,correction_episode_min=None):
    if row['family']=='sc':return 'sc'
    if row.get('corrective_teacher',False):
        if correction_episode_min is not None and int(row['episode'])<correction_episode_min:
            return 'previous_correction'
        return 'withheld_later_correction'
    episode=int(row['episode'])
    if episode>=2400000:return 'new_sfp'
    return 'old_sfp'


def metrics(values):
    errors=np.array([x['translation_error_mm'] for x in values],float)
    angles=np.array([x['orientation_error_deg'] for x in values],float)
    useful=[x for x in values if x['teacher_translation_mm']>=5]
    return dict(rows=len(values),episodes=len({x['episode'] for x in values}),
                translation_mm_p50=float(np.median(errors)),
                translation_mm_p95=float(np.percentile(errors,95)),
                orientation_deg_p95=float(np.percentile(angles,95)),
                direction_cosine_p50=(float(np.median([x['translation_cosine'] for x in useful]))
                                      if useful else None),
                direction_cosine_ge_0p7_fraction=(float(np.mean([x['translation_cosine']>=.7
                                                               for x in useful]))
                                                if useful else None),
                useful_direction_rows=len(useful))


@torch.inference_mode()
def evaluate_one(path,data,device,correction_episode_min=None):
    cp=torch.load(path,map_location='cpu',weights_only=False)
    m=cp['source_manifest']
    if not m['long_history'] or not m['full_state_skip'] or m['route_conditioning']:
        raise ValueError('Expected full-state long-history action-only checkpoint')
    model=TemporalTransport(data['mean'],data['std'],raw_state_skip=True,
                            feature_dim=data['features'].shape[1],state_task_dim=42,
                            route_conditioning=False,
                            route_goal_conditioning=m.get('route_goal_conditioning',False)).to(device).eval()
    model.load_state_dict(cp['model'])
    indices=np.array([i for i,r in enumerate(data['rows']) if r['split']=='validation' and
                      not r.get('actor_prefix_context_only',False)],int)
    groups=defaultdict(list);episodes=defaultdict(list)
    for part in np.array_split(indices,max(1,(len(indices)+127)//128)):
        if not len(part):continue
        ind=torch.as_tensor(part,device=device)
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
            action,*_=predict(model,data,ind,False)
        pred=(action[:,0]*model.target_std+model.target_mean).float()
        truth=data['target'][ind,0].float()
        delta=(pred[:,:3]-truth[:,:3]).norm(dim=-1).cpu().numpy()*1000
        teacher_norm=truth[:,:3].norm(dim=-1).cpu().numpy()*1000
        cosine=torch.nn.functional.cosine_similarity(pred[:,:3],truth[:,:3],dim=-1).cpu().numpy()
        relative=rotation_from_pose9(pred).transpose(-1,-2)@rotation_from_pose9(truth)
        angular=(((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
                 .acos()*180/np.pi).cpu().numpy()
        for k,i in enumerate(part):
            row=data['rows'][int(i)];group=family_group(row,correction_episode_min)
            entry=dict(episode=int(row['episode']),frame=int(row['frame']),
                       translation_error_mm=float(delta[k]),
                       orientation_error_deg=float(angular[k]),
                       teacher_translation_mm=float(teacher_norm[k]),
                       translation_cosine=float(cosine[k]),
                       card=(int(np.argmax(row['task'][4:9])) if row['family']=='sfp' else None),
                       port=int(np.argmax(row['task'][2:4])))
            groups[group].append(entry)
            episodes[(group,entry['episode'])].append(entry)
    return dict(checkpoint=str(path),best_update=cp['update'],
                groups={k:metrics(v) for k,v in sorted(groups.items())},
                withheld_episodes={str(k):metrics(v) for k,v in sorted(episodes.items())
                                   if k[0]=='withheld_later_correction'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--base-checkpoint',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--withheld-scenes-unseen-to-prior',action='store_true')
    p.add_argument('--correction-episode-min',type=int,
                   help='Score only newer correction episodes in the correction gate')
    p.add_argument('--route-goal-values',type=Path,
                   help='Frozen selected-port offsets for a matched zero/pose-input comparison')
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous comparison')
    control_manifest=torch.load(a.control,map_location='cpu',weights_only=False)['source_manifest']
    candidate_manifest=torch.load(a.candidate,map_location='cpu',weights_only=False)['source_manifest']
    if a.route_goal_values:
        if not (control_manifest.get('route_goal_conditioning') and
                candidate_manifest.get('route_goal_conditioning') and
                control_manifest.get('route_goal_zero_control') and
                not candidate_manifest.get('route_goal_zero_control')):
            p.error('Expected matched zero/selected-port route-goal checkpoints')
        control_data=build_data(a.cache,a.base_checkpoint,a.device,history_ages=LONG_AGES,
                                route_goal_values=a.route_goal_values,route_goal_zero_control=True)
        candidate_data=build_data(a.cache,a.base_checkpoint,a.device,history_ages=LONG_AGES,
                                  route_goal_values=a.route_goal_values)
    else:
        control_data=candidate_data=build_data(a.cache,a.base_checkpoint,a.device,
                                                history_ages=LONG_AGES)
    control=evaluate_one(a.control,control_data,a.device,a.correction_episode_min)
    candidate=evaluate_one(a.candidate,candidate_data,a.device,a.correction_episode_min)
    regressions={}
    for name in ('old_sfp','new_sfp','sc'):
        before=control['groups'][name]['translation_mm_p95']
        after=candidate['groups'][name]['translation_mm_p95']
        regressions[name]=dict(control_mm=before,candidate_mm=after,
                               relative_change=(after-before)/before)
    prior_dir=control['groups']['withheld_later_correction']['direction_cosine_ge_0p7_fraction']
    next_dir=candidate['groups']['withheld_later_correction']['direction_cosine_ge_0p7_fraction']
    correction_control_p95=control['groups']['withheld_later_correction']['translation_mm_p95']
    correction_candidate_p95=candidate['groups']['withheld_later_correction']['translation_mm_p95']
    gate=dict(no_group_regression_gt_10_percent=all(x['relative_change']<=.10
                                                    for x in regressions.values()),
              correction_direction_reliable=bool(next_dir>=.8 and next_dir>=prior_dir-.02),
              correction_action_p95_improves_10_percent=bool(
                  correction_candidate_p95<=.9*correction_control_p95),
              passed=False)
    gate['passed']=all((gate['no_group_regression_gt_10_percent'],
                        gate['correction_direction_reliable'],
                        gate['correction_action_p95_improves_10_percent']))
    result=dict(schema='aic_sfp_later_correction_offline_gate/v1',
                cache=str(a.cache),control=control,candidate=candidate,
                regressions=regressions,gate=gate,
                correction_validation_excluded_from_checkpoint_selection=True,
                correction_episode_min=a.correction_episode_min,
                route_goal_values=str(a.route_goal_values) if a.route_goal_values else None,
                prior_checkpoint_saw_original_teacher_episodes_on_withheld_scenes=
                    not a.withheld_scenes_unseen_to_prior,
                final_scenes_opened=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(regressions=regressions,gate=gate,
                          correction_control=control['groups']['withheld_later_correction'],
                          correction_candidate=candidate['groups']['withheld_later_correction'])))


if __name__=='__main__':main()
