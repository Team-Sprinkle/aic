#!/usr/bin/env python3
"""Matched stage-level summary of the four frozen SFP visual heads."""
import argparse
from collections import Counter
import json
from pathlib import Path


CASES=(('card1_port0','card1_port0_v2'),('card2_port0','card2_port0_v3'),
       ('card0_port1','card0_port1_v3'),('card4_port1','card4_port1_v3'))
STAGES=('baseline_view','positive_view','negative_view')
MODELS=('allport','structured','selected','crop')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('Preserve previous summary')
    labels=[json.loads(s) for s in (args.root/'event_labels_v1/rows.jsonl').open()]
    lookup={(r['episode'],r['time_s']):r for r in labels}
    predictions={}
    for model in MODELS:
        path=args.root/'view_model_scores_v1'/f'{model}.jsonl'
        predictions[model]={}
        for s in path.open():
            r=json.loads(s);key=(r['episode'],r['time_s'])
            stage=lookup[key]['stage'];predictions[model][(r['episode'],stage)]=r
    episodes=[]
    for name,folder in CASES:
        event=json.loads((args.root/folder/'side_view_events.json').read_text())
        events=event['events'];stages={e['stage']:e for e in events if e['stage'] in STAGES}
        baseline=stages['baseline_view']['tcp_base_m'][1]
        stage_results=[]
        for stage in STAGES:
            item=stages[stage]
            models={}
            for model in MODELS:
                rec=predictions[model].get((name,stage))
                models[model]=(dict(proposed=rec is not None,accepted=bool(rec and rec['accepted']),
                                    error_mm=rec['error_mm'] if rec and rec['accepted'] else None,
                                    wrong_accepted=bool(rec and rec['accepted'] and
                                                        rec['nearest_identity_label_only']!=rec['requested'])))
            stage_results.append(dict(stage=stage,sim_time_s=item['sim_time_s'],
                                      actual_y_from_baseline_mm=round((item['tcp_base_m'][1]-baseline)*1000,2),
                                      force_n=item['force_n'],contact_pairs=item['contact_pairs'],models=models))
        first_force=next(e for e in events if e['stage']=='baseline_before_hold')['force_n']
        episodes.append(dict(episode=name,folder=str(args.root/folder),
                             measured_max_force_rise_n=round(max(e['force_n']-first_force for e in events),3),
                             measured_max_force_n=round(max(e['force_n'] for e in events),3),
                             unsafe_abort=any(e['stage']=='unsafe_abort' for e in events),
                             stages=stage_results))
    counts={}
    for model in MODELS:
        counts[model]={}
        for stage in STAGES:
            results=[next(x for x in e['stages'] if x['stage']==stage)['models'][model] for e in episodes]
            counts[model][stage]=dict(proposed=sum(r['proposed'] for r in results),
                                      accepted=sum(r['accepted'] for r in results),
                                      wrong_accepted=sum(r['wrong_accepted'] for r in results))
    report=dict(schema='sfp_side_view_matched_diagnostic/v1',episode_count=len(episodes),
                events_per_complete_scene=3,models_frozen=True,independent_scenes=False,
                no_model_training=True,final_scenes_opened=False,
                complete_live_inference_latency_measured=False,
                counts=counts,episodes=episodes)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(counts=counts,force_rises={x['episode']:x['measured_max_force_rise_n'] for x in episodes}),
                     indent=2))


if __name__=='__main__':main()
