#!/usr/bin/env python3
"""Record the fixed data-diversity, actor and pause-head decision gates."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--retry',type=Path,required=True)
    p.add_argument('--initial-admission',type=Path,required=True)
    p.add_argument('--final-admission',type=Path,required=True)
    p.add_argument('--experiment-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous summary')
    first=read(a.initial/'progress.json')['results']
    second=read(a.retry/'progress.json')['results']
    initial_admission=read(a.initial_admission/'manifest.json')
    admission=read(a.final_admission/'manifest.json')
    split=read(a.experiment_root/'split_score_v1.json')['results']
    sensitivity={name:read(a.experiment_root/f'{name}_task_sensitivity_v1.json')
                 for name in ('control','diverse')}
    control=next(v for k,v in split.items() if 'control_matched_v1' in k)
    diverse=next(v for k,v in split.items() if 'diverse_matched_v1' in k)
    first_p95=lambda result,section,family:result[section]['metrics'][family]['action_first_mm_p95']
    new_reduction=1-first_p95(diverse,'new_diverse','sfp')/first_p95(control,'new_diverse','sfp')
    old_regression=first_p95(diverse,'old_sfp','sfp')/first_p95(control,'old_sfp','sfp')-1
    sc_regression=first_p95(diverse,'sc','sc')/first_p95(control,'sc','sc')-1
    diameter_ratio=(sensitivity['diverse']['median_ten_code_action_diameter_mm']/
                    sensitivity['control']['median_ten_code_action_diameter_mm'])
    gates=dict(new_development_action_reduction_ge_10pct=new_reduction>=.10,
               old_sfp_action_regression_le_10pct=old_regression<=.10,
               sc_action_regression_le_10pct=sc_regression<=.10,
               requested_code_diameter_ge_2x=diameter_ratio>=2.)
    heads=read(a.experiment_root/'pause_proposal_split_score_v1.json')['results']
    head={k:v['all_sfp']['summary'] for k,v in heads.items()}
    replay=read(a.experiment_root/'pause_proposal_diverse_offroute_replay_v1.json')['runs']
    result=dict(schema='aic_sfp_diverse_transport_decision/v1',
        collection=dict(initial_attempts=len(first),initial_full=sum(
            x['official_score'].get('tier3',0)>=75 for x in first.values()),
            retry_attempts=len(second),retry_full=sum(
            x['official_score'].get('tier3',0)>=75 for x in second.values()),
            retry_infrastructure_failure_preserved=True,
            new_admitted_episodes=initial_admission['new_episodes']+admission['new_episodes'],
            new_admitted_transport_chunks=initial_admission['new_rows']+admission['new_rows'],
            new_development_validation_episodes=initial_admission['new_development_validation_episodes'],
            final_admission=str(a.final_admission)),
        actor_comparison=dict(control=str(a.experiment_root/'control_matched_v1/best.pt'),
            diverse=str(a.experiment_root/'diverse_matched_v1/best.pt'),
            matched_updates=3000,new_development_action_reduction_fraction=new_reduction,
            old_sfp_action_regression_fraction=old_regression,
            sc_action_regression_fraction=sc_regression,
            requested_code_action_diameter_ratio=diameter_ratio,
            gates=gates,offline_gate_passed=all(gates.values()),
            live_rollout_run=False,reason_no_live='requested-code sensitivity gate failed'),
        pause_head=dict(split_scores=head,
            diverse_offroute_first_triggers=[dict(run=x['run'],trigger=x['trigger']) for x in replay],
            shadow_only=True,
            reason='one of three saved actor routes had a false first trigger; post-window overconfidence remains'),
        final_scenes_opened=False,rl_started=False,automatic_handoff_enabled=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(actor=result['actor_comparison'],pause_head=result['pause_head'])),flush=True)


if __name__=='__main__':main()
