#!/usr/bin/env python3
"""Split a completed full sample by causal raw-bag wall windows for offline review."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

from audit_wide_approach_teacher_terminal import audit


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args=parser.parse_args()
    root=args.root
    source=root/'rollout'
    cfg=yaml.safe_load((source/'eval_config.yaml').read_text())
    scoring=yaml.safe_load((source/'results/scoring.yaml').read_text())
    events=[json.loads(line) for line in (source/'runtime_trace.jsonl').open()]
    ends=[(i,e) for i,e in enumerate(events) if e['event']=='terminal_before_return'
          and e.get('observation_source')=='retained_public_observation_before_return']
    frames=list(csv.DictReader((source/'frame_manifest.csv').open()))
    analysis=root/'per_trial'
    analysis.mkdir(exist_ok=False)
    results=[]
    consumed=set()
    for trial_id,trial in cfg['trials'].items():
        bags=list((source/'results').glob(f'bag_{trial_id}_*'))
        assert len(bags)==1, bags
        bag=bags[0]
        meta=yaml.safe_load((bag/'metadata.yaml').read_text())['rosbag2_bagfile_information']
        wall0=meta['starting_time']['nanoseconds_since_epoch']
        wall1=wall0+meta['duration']['nanoseconds']
        candidates=[(i,e) for i,e in ends if wall0<=e['observation_wall_time_ns']<=wall1]
        if len(candidates)!=1:
            raise ValueError(f'{trial_id}: expected one causal retained terminal, found{len(candidates)}')
        end_index,terminal=candidates[0]
        assert end_index not in consumed
        consumed.add(end_index)
        previous=[i for i,_ in ends if i<end_index]
        begin=max(previous)+1 if previous else 0
        trace=events[begin:end_index+1]
        view=analysis/trial_id
        view.mkdir()
        (view/'results').mkdir()
        (view/'results'/bag.name).symlink_to(bag.resolve(), target_is_directory=True)
        # These are analysis-only views. The actual launched YAML remains untouched.
        local_cfg=dict(cfg,trials={trial_id:trial})
        (view/'eval_config.yaml').write_text(yaml.safe_dump(local_cfg, sort_keys=False))
        (view/'results/scoring.yaml').write_text(yaml.safe_dump(scoring, sort_keys=False))
        (view/'frames').symlink_to((source/'frames').resolve(), target_is_directory=True)
        for filename in ['engine.log','policy.log']:
            (view/filename).symlink_to((source/filename).resolve())
        (view/'runtime_trace.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in trace))
        selected=[f for f in frames if wall0<=int(f['wall_ns'])<=terminal['observation_wall_time_ns']]
        assert selected
        with (view/'frame_manifest.csv').open('w') as output:
            writer=csv.DictWriter(output, fieldnames=list(frames[0]))
            writer.writeheader();writer.writerows(selected)
        save(view/'causal_manifest.json',dict(trial=trial_id, raw_bag=str(bag),
            wall_bounds_ns=[wall0,wall1], retained_terminal_wall_ns=terminal['observation_wall_time_ns'],
            source_trace_indices=[begin,end_index], episode_index=terminal['episode_index'],
            analysis_only_single_trial_yaml=True, original_full_yaml_unchanged=True,
            post_reset_geometry_used=False))
        with (view/'extract.log').open('w') as log:
            subprocess.run([sys.executable,'scripts/extract_moveit_training_stream.py',str(view),str(view/'scored_stream.json')],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        physical=audit(view, terminal_event='terminal_before_return')
        save(view/'physical_terminal.json', physical)
        task=next(iter(trial['tasks'].values()))
        counts={event:sum(e['event']==event for e in trace) for event in (
            'shared_visual_lock','near_handoff_ready','near_handoff_failed','locked_local_perception',
            'connector_rotation_acquisition','full_connector_grasp_bound','pipeline_exception',
            'measured_retreat_complete','lateral_retry_complete')}
        row=dict(trial=trial_id, task=task, toolkit_score=scoring[trial_id], physical=physical,
                 stage_event_counts=counts, terminal_reason=terminal.get('underlying_actor_terminal_reason',terminal.get('reason')),
                 analysis_view=str(view), runtime_geometry_input=any(e.get('geometry_input') is True for e in trace))
        commands=[e for e in trace if e['event']=='command' and 'absolute_tcp_target' in e]
        local=[e for e in trace if e['event']=='locked_local_perception']
        row['actor_ran']=bool(local)
        if local:
            jumps=[]
            for a,b in zip(commands,commands[1:]):
                if b.get('chunk_index')==0 and a.get('sim_time',-1)>=local[0]['sim_time']:
                    jumps.append(float(np.linalg.norm(np.asarray(a['absolute_tcp_target'])[:3,3]-np.asarray(b['absolute_tcp_target'])[:3,3])*1000))
            row['command_boundary_jump_mm_p50_p95_max']=np.percentile(jumps,[50,95,100]).tolist() if jumps else None
        results.append(row)
        save(root/'analysis_progress.json',dict(results=results, completed=len(results)))
        with (view/'video.log').open('w') as log:
            subprocess.run([sys.executable,'scripts/render_transport_five_camera_video.py',str(view),'--fps','20','--grid'],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        row['video']=str(view/'five_cameras_20fps_grid.mp4')
    summary=dict(total=scoring['total'], maximum=300, complete_original_yaml=True,
                 strict_physical_insertions=sum(r['physical']['qualified'] for r in results),
                 trials=len(results), results=results, recovery=json.loads((root/'plan.json').read_text()).get('recovery','none'), RL=False,
                 reserved_final_scenes_opened=False, post_reset_geometry_used=False,
                 note='Whole-actor boundary statistics may include simulator freeze; do not use them to claim smoothness improvement.')
    save(root/'analysis_summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('total','maximum','strict_physical_insertions','trials')}),flush=True)


if __name__=='__main__':main()
