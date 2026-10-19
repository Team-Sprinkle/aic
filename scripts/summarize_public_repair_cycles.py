#!/usr/bin/env python3
"""Make a compact, auditable public-sample cycle comparison and plot."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
results = json.loads((args.root/'cycle_results.json').read_text())
rows = []
for result in results:
    folder = Path(result['artifact_root'])
    analysis = json.loads((folder/'analysis_summary.json').read_text())
    trials = []
    for trial in analysis['results']:
        physical = trial['physical']
        trials.append(dict(trial=trial['trial'], family=physical['family'],
            task=trial['task'], score=sum(trial['toolkit_score'][key]['score']
                                        for key in ('tier_1','tier_2','tier_3')),
            **{key:physical[key] for key in ('depth_mm','lateral_mm','orientation_error_deg',
                                            'official_tier3','qualified','no_scored_harm')},
            video=trial['video']))
    rows.append(dict(**result, trials=trials))
report = dict(schema='public_sample_repair_cycles/v1', target_score_exclusive=290,
    target_reached=any(row['target_reached'] for row in rows),
    current_state=json.loads((args.root/'state.json').read_text()),
    completed_cycles=len(rows), initial_requested_cycles=5,
    episodes=3*len(rows), unique_public_sample_scenarios=3,
    final_scenes_opened=False, RL=False, learned_actor_inference=False,
    runtime_inputs='RGB, task code, public robot state; no true reset geometry',
    scope='Repeated original public sample, not broad setting coverage or final certification',
    cycles=rows)
report['best_toolkit_score']=max(row['score'] for row in rows)
report['best_full_gate_score']=max((row['score'] for row in rows
    if row['physical_all_passed'] and row['latency_gate_passed']),default=None)
args.output.write_text(json.dumps(report, indent=2)+'\n')
fig, axes = plt.subplots(1,3,figsize=(13,3.6),constrained_layout=True)
cycles = [row['cycle'] for row in rows]
axes[0].plot(cycles,[row['score'] for row in rows],'o-')
axes[0].axhline(290,color='red',linestyle='--',label='Target: above290')
axes[0].set(ylabel='Toolkit score / 300',xlabel='Complete public sample cycle',ylim=(0,302))
axes[0].legend()
axes[1].bar(cycles,[row['strict_insertions'] for row in rows])
axes[1].set(ylabel='Strict physical + toolkit insertions / 3',xlabel='Cycle',ylim=(0,3.3),yticks=[0,1,2,3])
axes[2].bar(cycles,[max((v['p50_p95_p99_ms'][1] for v in row['latencies'].values()
                        if v['p50_p95_p99_ms'] is not None),default=0.) for row in rows],
            color=['steelblue' if row['latency_gate_passed'] else 'darkorange' for row in rows])
axes[2].axhline(300,color='red',linestyle='--')
axes[2].set(ylabel='Largest observed-stage p95 / ms',xlabel='Cycle; orange also includes missing stages')
for axis in axes:
    axis.set_xticks(cycles)
    axis.grid(axis='y',alpha=.25)
fig.savefig(args.root/'repair_cycle_comparison.png',dpi=160)
plt.close(fig)
print(json.dumps(dict(completed_cycles=len(rows),target_reached=report['target_reached'],
                      best_toolkit_score=report['best_toolkit_score'],
                      best_full_gate_score=report['best_full_gate_score'])),flush=True)
