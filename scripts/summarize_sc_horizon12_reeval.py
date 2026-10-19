#!/usr/bin/env python3
"""Summarize frozen SC 12/6 replays and plot measured axial route progress."""
import argparse
import json
import math
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np

from materialize_sc_scored_route import port_opening


def read(path):
    data=json.loads(path.read_text())
    return {row['name']:row for row in data['rows'] if row['family']=='sc'}


def metrics(row,cohort):
    trace=row['matched_trace']
    best=min(trace,key=lambda x:math.hypot(x['actor_axial_mm'],x['actor_lateral_mm']))
    departure=row['first_sustained_20mm_departure']
    events=[json.loads(line) for line in (Path(row['actor'])/'runtime_trace.jsonl').open()]
    proposals=[x['sim_time'] for x in events if x.get('event')=='proposal']
    commands=[x for x in events if x.get('event')=='command']
    chunks=[commands[i:i+6] for i in range(0,len(commands),6)]
    if len(proposals)!=len(chunks) or any(len(chunk)!=6 for chunk in chunks):
        raise ValueError(f'Incomplete six-command chunk: {row["name"]}')
    late=[x['first_command_port_mm'][2] for x in trace if x['sim_s']>=40]
    stream=json.loads((Path(row['actor'])/'scored_stream.json').read_text())
    selected_port=int(stream['task']['target_module_name'].split('_')[-1])
    opening=port_opening(stream,selected_port)
    late_issued=[]
    for command in commands:
        if command['sim_time']<40:continue
        actual_target=np.asarray(command['absolute_tcp_target'])[:3,3]
        observed=np.asarray(command['state'][:3])
        late_issued.append(float((opening[:3,:3].T@(actual_target-observed))[2]*1000))
    return dict(cohort=cohort,scene=row['name'],teacher_full_insertion=True,
        safe_buffer=bool(row['actor_score']['safe_buffer_visit']),
        full_insertion=float(row['actor_score']['official_tier3'])>=75,
        scored_contact=row['actor_score']['official_contact_message']!='No contact detected.',
        axial_band_samples=int(row['actor_score']['axial_band_samples']),
        departure_time_s=(departure['sim_s'] if departure else None),
        best_tcp_to_port_3d_mm=math.hypot(best['actor_axial_mm'],best['actor_lateral_mm']),
        best_tcp_axial_mm=best['actor_axial_mm'],best_tcp_lateral_mm=best['actor_lateral_mm'],
        best_time_s=best['sim_s'],last_tcp_axial_mm=trace[-1]['actor_axial_mm'],
        teacher_last_tcp_axial_mm=trace[-1]['teacher_axial_mm'],
        late_first_command_axial_median_mm=statistics.median(late),
        late_first_command_away_count=sum(v<0 for v in late),late_first_command_count=len(late),
        late_issued_command_axial_median_mm=statistics.median(late_issued),
        late_issued_command_away_count=sum(v<0 for v in late_issued),
        late_issued_command_count=len(late_issued),
        proposal_count=len(proposals),command_count=len(commands),
        proposal_interval_p50_s=statistics.median(b-a for a,b in zip(proposals,proposals[1:])),
        command_gap_p50_s=statistics.median(chunks[i+1][0]['sim_time']-chunks[i][-1]['sim_time']
                                         for i in range(len(chunks)-1)),
        audit=row['actor'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--repeat',type=Path,required=True)
    p.add_argument('--new',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    sources={key:read(getattr(a,key)) for key in ('original','repeat','new')}
    if sources['original'].keys()!=sources['repeat'].keys():
        raise ValueError('Repeated scenes differ from original comparison')
    rows=[metrics(row,key) for key in ('original','repeat','new')
          for row in sources[key].values()]
    summary=dict(schema='aic_sc_horizon12_reeval/v1',checkpoint_frozen=True,
        final_scenes_opened=False,actor_privileged_inputs=False,
        geometry_posthoc_only=True,source_analyses={key:str(getattr(a,key)) for key in sources},
        original_repeated_scene_count=len(sources['original']),
        new_development_scene_count=len(sources['new']),
        safe_buffers=sum(row['safe_buffer'] for row in rows),
        full_insertions=sum(row['full_insertion'] for row in rows),
        scored_contacts=sum(row['scored_contact'] for row in rows),runs=rows)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(15,7),sharey=True)
    for j,name in enumerate(sources['original']):
        ax=axes[0,j]
        for key,color in [('original','C0'),('repeat','C1')]:
            trace=sources[key][name]['matched_trace']
            ax.plot([x['sim_s'] for x in trace],[x['actor_axial_mm'] for x in trace],
                    color=color,label=key)
        trace=sources['repeat'][name]['matched_trace']
        ax.plot([x['sim_s'] for x in trace],[x['teacher_axial_mm'] for x in trace],
                color='black',ls='--',label='successful teacher')
        ax.axhspan(-100,-70,color='green',alpha=.13);ax.set_title(name)
    for j,(name,row) in enumerate(sources['new'].items()):
        ax=axes[1,j];trace=row['matched_trace']
        ax.plot([x['sim_s'] for x in trace],[x['actor_axial_mm'] for x in trace],
                color='C2',label='12/6 actor')
        ax.plot([x['sim_s'] for x in trace],[x['teacher_axial_mm'] for x in trace],
                color='black',ls='--',label='successful teacher')
        ax.axhspan(-100,-70,color='green',alpha=.13);ax.set_title(name+' (new dev)')
    for ax in axes.flat:
        ax.set_xlabel('Simulation time (s)');ax.grid(alpha=.2)
    for ax in axes[:,0]:ax.set_ylabel('TCP axial distance to port (mm)')
    axes[0,0].legend(loc='lower right',fontsize=8)
    axes[1,0].legend(loc='lower right',fontsize=8)
    fig.tight_layout();fig.savefig(a.output/'axial_progress.png',dpi=150);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ('safe_buffers','full_insertions','scored_contacts')}))


if __name__=='__main__':main()
