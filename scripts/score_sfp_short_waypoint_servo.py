#!/usr/bin/env python3
"""Offline safety/command audit before any SFP waypoint-servo rollout."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from train_temporal_transport_probe import LONG_AGES, TemporalTransport, build_data, predict


def direction(a,b):
    return np.sum(a*b,axis=1)/np.maximum(1e-9,np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))


def summary(pred,teacher,future,episodes):
    truth_mm=np.linalg.norm(teacher,axis=1)*1000
    future_mm=np.linalg.norm(future,axis=1)*1000
    useful=truth_mm>=5
    moving=future_mm>=10
    error=np.linalg.norm((pred-teacher)*1000,axis=1)
    return dict(rows=len(pred),episodes=len(set(episodes)),
                first_action_error_mm_p95=float(np.percentile(error,95)),
                first_action_direction_ge0p7_fraction=float(np.mean(direction(pred[useful],teacher[useful])>=.7)) if useful.any() else None,
                future_direction_ge0p7_all_fraction=float(np.mean(direction(pred,future)>=.7)),
                future_direction_ge0p7_fraction=float(np.mean(direction(pred[moving],future[moving])>=.7)) if moving.any() else None,
                moving_rows=int(moving.sum()),command_norm_mm_p95=float(np.percentile(np.linalg.norm(pred,axis=1)*1000,95)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','base-checkpoint','labels','actor','waypoint','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--step-mm',type=float,default=10.)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior result')
    if not 1<=a.step_mm<=20:p.error('Bound waypoint step to 1..20 mm')
    data=build_data(a.cache,a.base_checkpoint,'cuda',history_ages=LONG_AGES)
    rows=data['rows'];source=np.load(a.labels)
    truth_waypoint=np.asarray(source['waypoint_tcp'],float)
    teacher_valid=np.asarray(source['teacher_valid'],bool)
    boundary_valid=np.asarray(source['actor_boundary_valid'],bool)
    actor_cp=torch.load(a.actor,map_location='cpu',weights_only=False)
    way_cp=torch.load(a.waypoint,map_location='cpu',weights_only=False)
    def model():
        return TemporalTransport(data['mean'],data['std'],raw_state_skip=True,
                                 feature_dim=data['features'].shape[1],state_task_dim=42,
                                 route_conditioning=False).cuda().eval()
    actor=model();actor.load_state_dict(actor_cp['model'])
    waypoint=model();waypoint.load_state_dict(way_cp['model'])
    scores={}
    for group,valid in (('teacher',teacher_valid),('actor_boundary',boundary_valid)):
        ids=np.asarray([i for i,r in enumerate(rows) if r['split']=='validation' and valid[i]],int)
        proposed=[];base=[]
        with torch.inference_mode():
            for part in np.array_split(ids,max(1,(len(ids)+127)//128)):
                if not len(part):continue
                ind=torch.as_tensor(part,device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    action,*_=predict(actor,data,ind,False)
                    _,_,route,_=predict(waypoint,data,ind,False,force_zero_previous=True)
                base.append((action[:,0]*actor.target_std+actor.target_mean)[:,:3].float().cpu().numpy())
                proposed.append(route[:,:3].float().cpu().numpy())
        route=np.concatenate(proposed);base=np.concatenate(base)
        norm=np.linalg.norm(route,axis=1)
        servo=route*np.minimum(1,a.step_mm/1000/np.maximum(norm,1e-9))[:,None]
        truth=np.asarray([rows[i]['target'][0][:3] for i in ids],float)
        future=truth_waypoint[ids]
        # Actor-prefix targets are the actor's own commands, not teacher labels.
        if group=='actor_boundary':truth=future
        episode=np.asarray([rows[i]['episode'] for i in ids])
        scores[group]=dict(action_only=summary(base,truth,future,episode),
                           waypoint_servo=summary(servo,truth,future,episode),
                           route_norm_mm_p95=float(np.percentile(norm*1000,95)),
                           route_predictions_over_120mm=int((norm>.12).sum()))
        if group=='actor_boundary':
            scores[group]['records']=[dict(episode=int(episode[j]),frame=int(rows[i]['frame']),
                                          actor_time_s=float(rows[i]['time_s']),
                                          future_teacher_motion_tcp_mm=(future[j]*1000).tolist(),
                                          predicted_waypoint_tcp_mm=(route[j]*1000).tolist(),
                                          servo_step_tcp_mm=(servo[j]*1000).tolist(),
                                          previous_actor_command_tcp_mm=(base[j]*1000).tolist(),
                                          servo_future_cosine=float(direction(servo[j:j+1],future[j:j+1])[0]))
                                      for j,i in enumerate(ids)]
        if group=='teacher':
            by={}
            for phase,mask in (('early',np.asarray([rows[i]['time_s']<3 and not rows[i].get('corrective_teacher') for i in ids])),
                               ('middle',np.asarray([3<=rows[i]['time_s']<8 and not rows[i].get('corrective_teacher') for i in ids])),
                               ('late',np.asarray([rows[i]['time_s']>=8 and not rows[i].get('corrective_teacher') for i in ids])),
                               ('corrective',np.asarray([rows[i].get('corrective_teacher',False) for i in ids]))):
                if mask.any():by[phase]=dict(action_only=summary(base[mask],truth[mask],future[mask],episode[mask]),
                                        waypoint_servo=summary(servo[mask],truth[mask],future[mask],episode[mask]))
            scores[group]['by_phase']=by
    ordinary=scores['teacher'];boundary=scores['actor_boundary']
    gate=dict(ordinary_future_direction_ge80pct=
              ordinary['waypoint_servo']['future_direction_ge0p7_fraction']>=.8,
              boundary_future_direction_ge_4of6=
              boundary['waypoint_servo']['future_direction_ge0p7_all_fraction']>=4/6,
              no_extreme_route_predictions=ordinary['route_predictions_over_120mm']==0 and
                                           boundary['route_predictions_over_120mm']==0,
              command_norm_capped=ordinary['waypoint_servo']['command_norm_mm_p95']<=a.step_mm+.01)
    gate['passed']=all(gate.values())
    result=dict(schema='aic_sfp_short_waypoint_servo_offline/v1',step_mm=a.step_mm,
                groups=scores,gate=gate,baseline_orientation_preserved_in_proposed_runtime=True,
                true_geometry_actor_input=False,final_scenes_opened=False,autonomous_rollouts=0)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(gate=gate,teacher=scores['teacher']['waypoint_servo'],
                          actor_boundary=scores['actor_boundary']['waypoint_servo'])))


if __name__=='__main__':main()
