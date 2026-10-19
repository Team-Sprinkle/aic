#!/usr/bin/env python3
"""Copy command derivatives with unique episode IDs and pre-completion labels.

Post-event observations remain in originals, but frozen-simulator commands
must not teach physical action dynamics. Predicted port frames remain a
separate gate before BC training.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from prepare_verified_insertion_command_rows import episode_identity


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--cutoffs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    summary=json.loads((a.input/'summary.json').read_text());cutoffs={r['source_stream']:r for r in map(json.loads,a.cutoffs.open())}
    reports=[];counts=Counter();profiles=Counter();groups={'train':set(),'validation':set()};seen_episodes={}
    for i,e in enumerate(summary['episodes']):
        cutoff=cutoffs[e['source']]
        if not cutoff['valid']:raise ValueError('No observed completion cutoff for an admitted source')
        d=json.loads(Path(e['source']).read_text());commands=[c for c in d['commands'] if c['kind']=='pose']
        name=e.get('source_episode_name',e['episode']);episode=episode_identity(e['source'],name)
        if episode in seen_episodes and seen_episodes[episode]!=e['source']:raise ValueError('Episode identity collision')
        seen_episodes[episode]=e['source'];rejected=Counter();rows=0;filename=f'{i:04d}_{episode}.jsonl'
        with (a.output/filename).open('w') as writer:
            for line in (a.input/e['rows_file']).open():
                r=json.loads(line);stamp=round(r['time_s']*1e9)
                if stamp+200_000_000>cutoff['conservative_BC_cutoff_sim_ns']:
                    rejected['future_horizon_crosses_first_completion']+=1;continue
                cs=[commands[j] for j in r['source_command_indices']]
                if any(c['message']['trajectory_generation_mode']['mode']!=2 for c in cs):
                    rejected['non_position_mode']+=1;continue
                stiffness=[[c['message']['target_stiffness'][j*7] for j in range(6)] for c in cs]
                damping=[[c['message']['target_damping'][j*7] for j in range(6)] for c in cs]
                if any(not np.allclose(g,[90,90,90,50,50,50]) for g in stiffness):
                    rejected['incompatible_stiffness_profile']+=1;continue
                if any(not np.allclose(g,[50,50,50,20,20,20]) for g in damping):
                    rejected['incompatible_damping_profile']+=1;continue
                r.update(episode=episode,source_episode_name=name,frame=rows,
                    conservative_completion_cutoff_sim_ns=cutoff['conservative_BC_cutoff_sim_ns'],
                    completion_event_actor_input=False,post_completion_commands_excluded=True,
                    recorded_controller_stiffness=stiffness[0],recorded_controller_damping=damping[0],
                    body_command_conversion='header-time preceding measured TCP; RT-consumption timing approximation separately audited',
                    actor_training_ready=False)
                writer.write(json.dumps(r,separators=(',',':'))+'\n');rows+=1
                groups[r['split']].add(r['scene']);profiles[str(stiffness[0])]+=1
        result=dict(e,episode=episode,source_episode_name=name,rows=rows,rows_file=filename,finalization_exclusions=dict(rejected))
        reports.append(result);counts[f'{e["family"]}/{e["split"]}']+=rows
        with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        if (i+1)%25==0:print(json.dumps(dict(completed=i+1,planned=len(summary['episodes']),rows=sum(counts.values()))),flush=True)
    assert groups['train'].isdisjoint(groups['validation'])
    report=dict(schema='verified_precompletion_command_derivative/v1',episodes=reports,counts=dict(counts),
        sources_with_rows=sum(e['rows']>0 for e in reports),controller_stiffness_profiles=dict(profiles),
        input_summary_sha256=hashlib.sha256((a.input/'summary.json').read_bytes()).hexdigest(),
        completion_cutoffs_sha256=hashlib.sha256(a.cutoffs.read_bytes()).hexdigest(),
        original_data_modified=False,actor_training_ready=False,remaining_gate='causal RGB-only port locks and estimated-frame targets',final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='episodes'}),flush=True)


if __name__=='__main__':main()
