#!/usr/bin/env python3
"""Per-round score losses and public-state phase times for a completed cycle."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
summary=json.loads((a.root/'analysis_summary.json').read_text());results=[]
for trial in summary['results']:
    path=Path(trial['analysis_view']);rows=[json.loads(s) for s in (path/'runtime_trace.jsonl').open()]
    def stamp(event,last=False):
        data=[r['sim_time'] for r in rows if r['event']==event and 'sim_time' in r]
        return data[-1 if last else 0] if data else None
    force=stamp('initial_force_reference_frozen');lock=stamp('shared_visual_lock');near=stamp('near_handoff_ready');servo=stamp('visual_servo_started');end=stamp('terminal_before_return',True)
    stages={'transport_after_force_calibration':None if force is None or lock is None else lock-force,
            'visual_approach':None if lock is None or near is None else near-lock,
            'near_reestimation_and_hold':None if near is None or servo is None else servo-near,
            'insertion_and_dwell':None if servo is None or end is None else end-servo}
    categories=trial['toolkit_score']['tier_2']['categories']
    losses={key:maximum-categories[key]['score'] for key,maximum in [('duration',12),('trajectory efficiency',6),('trajectory smoothness',6)]}
    score=sum(trial['toolkit_score'][k]['score'] for k in ('tier_1','tier_2','tier_3'))
    results.append(dict(trial=trial['trial'],task=trial['task'],score=score,physical=trial['physical'],score_losses=losses,public_phase_duration_s=stages,categories=categories))
report=dict(total=summary['total'],strict_insertions=summary['strict_physical_insertions'],rounds=results,unchanged_scoring=True,geometry_diagnosis_only=True)
(a.root/'gap_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Per-round measured gaps','','| Round | Score | Strict insertion | Transport s | Approach s | Insertion+dwell s |','| --- | --- | --- | --- | --- | --- |']
for r in results:
    t=r['public_phase_duration_s'];lines.append(f"| {r['trial']} | {r['score']:.3f} | {r['physical']['qualified']} | {t['transport_after_force_calibration']} | {t['visual_approach']} | {t['insertion_and_dwell']} |")
(a.root/'gap_analysis.md').write_text('\n'.join(lines)+'\n');print(json.dumps(report),flush=True)
