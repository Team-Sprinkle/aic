#!/usr/bin/env python3
"""Grouped coverage counts; no tuning or transition-random confidence claims."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import math
import numpy as np


def interval(k,n):
    if not n:return None
    z=1.96;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-radius),min(1.,center+radius)]


def percentiles(values):
    return dict(zip(['p50','p90'],map(float,np.percentile(values,[50,90])))) if values else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    plan=json.loads((a.root/'plan.json').read_text());rows=[]
    for c in plan['cases']:
        for arm in c['arms']:
            f=a.root/'runs'/f'{c["case"]}__{arm}'/'coverage_metrics.json'
            if f.exists():rows.append(json.loads(f.read_text()))
    groups=defaultdict(list)
    for r in rows:groups[(r['domain'],r['family'],r['arm'],r['card_count'],r['port'])].append(r)
    cells=[]
    for (domain,family,arm,n,port),g in sorted(groups.items()):
        started=sum(r['insertion_actor_ran'] for r in g);full=sum(bool(r['SC_full_physical_insertion']) for r in g) if family=='sc' else None
        entered=[r for r in g if r['insertion_actor_ran']]
        cell={'domain':domain,'family':family,'arm':arm,'cards':n,'port':port,'episodes':len(g),
            'visual_approved':sum(bool(r['rule_visual_acquisition']) for r in g) if family=='sc' else None,
            'near_approach_ready':sum(bool(r['near_approach_completed']) for r in g) if arm=='near_approach' else None,
            'insertion_actor_started':started,'full_physical_insertions':full,
            'full_rate_Wilson95':interval(full,len(g)) if full is not None else None,
            'no_scored_harm':sum(r['no_scored_harm'] for r in g),
            'wrong_visual_accepts':sum(bool(r['wrong_accepted_port']) for r in g),
            'broad_diagnostic_near_visits':sum(r['posthoc_broad_near_port_visit'] for r in g),
            'terminal_depth_actor_started_mm':percentiles([r['terminal']['axial_mm'] for r in entered]),
            'terminal_lateral_mm':percentiles([r['terminal']['lateral_mm'] for r in g]),
            'closest_requested_tip_distance_mm':percentiles([r['closest_geometric_tip_visit']['distance_mm'] for r in g if r['closest_geometric_tip_visit']]),
            'rule_failure_counts':dict(Counter(r.get('rule_failure') for r in g if r.get('rule_failure'))),
            'last_visual_reason_counts':dict(Counter(r.get('last_visual_reason') for r in g if r.get('last_visual_reason')))}
        cells.append(cell)
    pair=defaultdict(dict)
    for r in rows:
        if r['family']=='sc' and r['domain']=='primary':pair[r['case']][r['arm']]=r
    pairs=[g for g in pair.values() if 'near_approach' in g and 'far_actor_control' in g]
    matched={'paired_configs':len(pairs),'near_approach_insertions':sum(g['near_approach']['SC_full_physical_insertion'] for g in pairs),
        'far_control_insertions':sum(g['far_actor_control']['SC_full_physical_insertion'] for g in pairs),
        'both_visual_approved':sum(g['near_approach']['rule_visual_acquisition'] and g['far_actor_control']['rule_visual_acquisition'] for g in pairs),
        'note':'Same frozen scene/model and desired reset; realized physics/perception can vary. Control cells only1/5card extremes.'}
    completed=len(rows);full=completed==plan['rollouts']
    if full and (len(cells)!=28 or any(c['episodes']!=5 for c in cells)):
        raise ValueError('Completed marker would violate five-per-cell matrix')
    result={'schema':'aic_frozen_coverage_summary/v1','completed':completed,'planned':plan['rollouts'],'matrix_complete':full,
        'cells':cells,'matched_SC_extremes':matched,'final_scenes_opened':False,'SFP_full_pipeline':'not_evaluated_not_integrated',
        'statistical_limit':'Five grouped resets per cell meet requested minimum but have wide uncertainty.5/5 has Wilson95 lower bound~0.566. Not a reliability certificate.'}
    (a.root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    field=['domain','family','arm','cards','port','episodes','visual_approved','near_approach_ready','insertion_actor_started','full_physical_insertions','no_scored_harm','wrong_visual_accepts','broad_diagnostic_near_visits']
    with (a.root/'coverage_cells.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=field);writer.writeheader();writer.writerows({k:r[k] for k in field} for r in cells)
    lines=['# Frozen coverage results',f'\nCompleted **{completed}/{plan["rollouts"]}**. Final scenes sealed; no model updates.\n',
        '| Domain | Family | Arm | Cards | Port | Episodes | Visual approval | Approach ready | Actor ran | Full SC insertion |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for c in cells:
        val=lambda k:'N/A' if c[k] is None else str(c[k])
        lines.append('| '+' | '.join([c['domain'],c['family'],c['arm'],str(c['cards']),str(c['port']),str(c['episodes']),val('visual_approved'),val('near_approach_ready'),str(c['insertion_actor_started']),val('full_physical_insertions')])+' |')
    lines+=['\nSFP is transport only: its full-pipeline handoff/insertion is N/A, not0% or a success claim.',
        '\nNear-port geometric visits are diagnostic opportunities, not approved visual handoffs.',
        '\nFive cases per cell have wide uncertainty; even5/5 is not proof of high reliability.']
    (a.root/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'completed':completed,'planned':plan['rollouts'],'complete':full,'matched':matched}))


if __name__=='__main__':main()
