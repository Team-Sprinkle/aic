#!/usr/bin/env python3
"""Audit a balanced SFP command-label collection gate."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--progress',type=Path,required=True)
    p.add_argument('--materialized-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-per-target',type=int,default=3)
    p.add_argument('--min-score-success',type=int,default=27)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve existing gate result')
    plan=json.loads(a.plan.read_text());progress=json.loads(a.progress.read_text())['results']
    materialized=json.loads(a.materialized_manifest.read_text())['episodes']
    by_trial={e['trial_id']:e for e in materialized}
    verdicts=[];targets=Counter();target_success=Counter()
    for row in plan['scene_trials']:
        tid=Path(row['source_trial']).stem
        report=by_trial.get(tid);control=progress.get(tid)
        checks=dict(official_insertion=bool(control and control['official_score'].get('tier3')==75),
            materialized=report is not None)
        if report:
            checks.update(known_target=(report['card']==row['card'] and report['port']==row['port']),
                joined_fraction=report['joined']/report['commands']>=.95,
                enough_transport_chunks=report['transport_chunks']>=200,
                command_reference_median_mm=report['reference_target_error_mm']['median']<1,
                command_reference_p95_mm=report['reference_target_error_mm']['p95']<5,
                command_reference_orientation_p95_deg=report['reference_orientation_error_deg']['p95']<2,
                transport_cutoff_axial_mm=-75<=report['cutoff_axial_mm']<=-40,
                transport_cutoff_lateral_mm=report['cutoff_lateral_mm']<=10)
        targets[(row['card'],row['port'])]+=1
        target_success[(row['card'],row['port'])]+=int(checks['official_insertion'])
        verdicts.append(dict(trial=tid,card=row['card'],port=row['port'],checks=checks,
                             passed=all(checks.values())))
    score_success=sum(v['checks']['official_insertion'] for v in verdicts)
    all_labels_valid=all(v['passed'] for v in verdicts if v['checks']['official_insertion'])
    gate=(len(verdicts)==10*a.expected_per_target and
          set(targets.values())=={a.expected_per_target} and
          score_success>=a.min_score_success and
          min(target_success.values(),default=0)>=max(2,a.expected_per_target-1) and
          all_labels_valid)
    summary=dict(schema='aic_sfp_balanced_teacher_gate/v2',gate_passed=gate,
        planned=len(verdicts),score_success=score_success,all_labels_valid=all_labels_valid,
        expected_per_target=a.expected_per_target,min_score_success=a.min_score_success,
        per_target={str(k):v for k,v in sorted(targets.items())},episodes=verdicts,
        successful_per_target={str(k):v for k,v in sorted(target_success.items())},
        expansion_authorized_by_predeclared_gate=(gate and a.expected_per_target==3),
        reserved_final_opened=False,invalid_agent_actions_used=False)
    a.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='episodes'}))


if __name__=='__main__':main()
