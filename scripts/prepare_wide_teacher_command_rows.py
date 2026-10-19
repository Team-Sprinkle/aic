#!/usr/bin/env python3
"""Admit only recorded local teacher commands, excluding privileged resets.

Raw-bag terminal qualification is required separately. Simulator poses never
become actor inputs here: output is complete recorded base-frame commands and
causal public observations, still awaiting RGB-only port locks.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import json
from pathlib import Path

import numpy as np

from prepare_verified_insertion_command_rows import (
    episode_identity, measured_state, message_pose, native_anchors, pose9)
from prepare_verified_insertion_source_index import task_vector


def match_owned_commands(commands, events, *, audit=None):
    times=[c['sim_ns'] for c in commands]
    owned=[];previous=-1;unmatched=[];offsets=[]
    for event in events:
        if event.get('event')!='wide_teacher_local_command':continue
        if event['command_owner']!='teacher':raise ValueError('Unexpected ownership')
        target=np.asarray(event['command_base'],float)
        stamp=round(event['time_sec']*1e9)
        candidates=[]
        # The trace callback can run one control tick after publication. Keep
        # exact pose equality and monotonic ownership; timestamps only bound
        # the search. Unmatched commands are NEVER action-label candidates.
        for j in range(max(previous+1,bisect_left(times,stamp-50_000_000)),
                       bisect_right(times,stamp+50_000_000)):
            if commands[j]['message']['header']['frame_id']!='base_link':continue
            if np.allclose(message_pose(commands[j]['message']['pose']),target,
                           atol=1e-8,rtol=0):
                candidates.append(j)
        if not candidates:
            unmatched.append(stamp);continue
        previous=min(candidates,key=lambda j:abs(times[j]-stamp));owned.append(previous)
        offsets.append((times[previous]-stamp)/1e6)
    if not owned:raise ValueError('No owned local teacher commands')
    if audit is not None:
        audit.update(matched=len(owned),unmatched_trace_events=len(unmatched),
                     unmatched_event_stamps_ns=unmatched,search_window_ms=50,
                     matched_stamp_offset_min_max_ms=[min(offsets),max(offsets)],
                     unmatched_commands_admitted=False)
    return set(owned)


def prepare(case, run, output):
    qualification=json.loads((run/'terminal_qualification.json').read_text())
    if qualification['qualified']!=1:raise ValueError('Unqualified teacher is not BC expert')
    d=json.loads((run/'scored_stream.json').read_text())
    events=[json.loads(l) for l in (run/'runtime_trace.jsonl').open()]
    commands=[c for c in d['commands'] if c['kind']=='pose']
    times=[c['sim_ns'] for c in commands]
    if times!=sorted(times):raise ValueError('Unsorted commands')
    ownership_audit={}
    owned=match_owned_commands(commands,events,audit=ownership_audit)
    grasp_start=max(e['time_sec'] for e in events if e.get('event')=='supervisor_approach_reset_released')
    completion=next(e for e in events if e.get('event')=='wide_teacher_completion_observed')
    cutoff=min(round(completion['time_sec']*1e9),max(times[j] for j in owned))
    family='sfp' if d['task']['plug_name'].startswith('sfp') else 'sc'
    task=task_vector(d['task'])
    episode=episode_identity(run/'scored_stream.json',case['name'])
    anchors=native_anchors(d,{},cutoff);excluded=Counter();rows=0;last_stationary=-float('inf')
    # Perception can retain public observations before actor ownership. The
    # privileged reset's commands are still excluded from every action label.
    # A deployment rule transport similarly observes the port before handoff.
    context_file=output.with_suffix('.observations.jsonl')
    context_anchors=native_anchors(d,{},cutoff,period_ns=50_000_000)
    with context_file.open('w') as context:
        for anchor in context_anchors:
            stamp=max(anchor['image_sim_ns'].values())
            try:state=measured_state(d,stamp)
            except ValueError:continue
            context.write(json.dumps(dict(family=family,episode=episode,source_episode_name=case['name'],
                scene=case['physical_reset_group'],split=case['split'],time_s=stamp/1e9,
                images=anchor['images'],state=state,task=task,
                image_kind='native_1152x1024_rgb_jpeg',observation_only=True,
                supervisor_action_labels_included=False))+'\n')
    with output.open('w') as writer:
        for anchor in anchors:
            stamp=max(anchor['image_sim_ns'].values())
            if stamp+200_000_000>cutoff:
                excluded['future_horizon_crosses_last_owned_or_completion']+=1;continue
            indices=[bisect_right(times,stamp+int(dt*1e9))-1 for dt in [.05,.1,.15,.2]]
            if any(j not in owned for j in indices):
                excluded['supervisor_or_transport_command']+=1;continue
            try:state=measured_state(d,stamp)
            except ValueError:
                excluded['stale_causal_observation']+=1;continue
            stiffness=[[commands[j]['message']['target_stiffness'][k*7] for k in range(6)] for j in indices]
            damping=[[commands[j]['message']['target_damping'][k*7] for k in range(6)] for j in indices]
            if (not all(np.allclose(g,[90,90,90,50,50,50]) for g in stiffness) or
                    not all(np.allclose(g,[50,50,50,20,20,20]) for g in damping)):
                raise ValueError('Teacher controller differs from required insertion profile')
            for j in indices:
                message=commands[j]['message'];wrench=message['feedforward_wrench_at_tip']
                if (not np.allclose(message['wrench_feedback_gains_at_tip'],[.5,.5,.5,0,0,0]) or
                        not np.allclose([wrench[k][axis] for k in ('force','torque') for axis in 'xyz'],0) or
                        message['trajectory_generation_mode']['mode']!=2):
                    raise ValueError('Wide teacher force/trajectory profile differs from deployment')
            transforms=[message_pose(commands[j]['message']['pose']) for j in indices]
            span=max(np.linalg.norm(t[:3,3]-transforms[0][:3,3]) for t in transforms)
            angle=np.arccos(np.clip((np.trace(transforms[0][:3,:3].T@transforms[-1][:3,:3])-1)/2,-1,1))
            stationary=span<.00001 and angle<np.deg2rad(.02) and np.linalg.norm(state[7:10])<.0002
            if stationary:
                if stamp/1e9-last_stationary<.5:
                    excluded['duplicate_static_hold_thinned_copy_only']+=1;continue
                last_stationary=stamp/1e9
            row=dict(schema='verified_insertion_recorded_commands/v2',family=family,
                episode=episode,source_episode_name=case['name'],scene=case['physical_reset_group'],
                split=case['split'],frame=rows,time_s=stamp/1e9,images=anchor['images'],
                image_kind='native_1152x1024_rgb_jpeg',state=state,task=task,
                recorded_base_tcp_targets_pose9=[pose9(message_pose(commands[j]['message']['pose'])) for j in indices],
                source_command_indices=indices,source_command_stamps_ns=[times[j] for j in indices],
                source_command_frames=['base_link']*4,action_target_offsets_s=[.05,.10,.15,.20],
                image_stamps_ns=anchor['image_sim_ns'],source_stream=str(run/'scored_stream.json'),
                action_owner='teacher',labels_are_recorded_targets_not_measured_future_motion=True,
                post_completion_commands_excluded=True,conservative_completion_cutoff_sim_ns=cutoff,
                completion_event_actor_input=False,recorded_controller_stiffness=stiffness[0],
                recorded_controller_wrench_feedback_gains=[.5,.5,.5,0.,0.,0.],
                recorded_feedforward_wrench=[0.]*6,recorded_trajectory_mode=2,
                full_recorded_controller_contract_verified=True,
                static_hold_retained=bool(stationary),
                recorded_controller_damping=damping[0],body_command_conversion='base-frame target recorded exactly; no body conversion',
                ownership_proof='monotonic exact pose match to wide_teacher_local_command within50ms; unmatched commands excluded',
                teacher_privileged_collection=True,predicted_port_input_ready=False,actor_training_ready=False)
            writer.write(json.dumps(row,separators=(',',':'))+'\n');rows+=1
    if rows==0:raise ValueError('No valid owned image/action samples')
    return dict(source=str(run/'scored_stream.json'),episode=episode,source_episode_name=case['name'],
        family=family,split=case['split'],scene=case['physical_reset_group'],rows=rows,
        rows_file=output.name,observation_prefix_file=context_file.name,grasp_acquisition_start_s=grasp_start,
        matched_owned_commands=len(owned),ownership_audit=ownership_audit,rejected=dict(excluded))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--case',help='One already-qualified pilot for dataset QA')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    plan=json.loads(a.plan.read_text());reports=[];excluded=[]
    for i,case in enumerate(plan['cases']):
        if a.case and case['name']!=a.case:continue
        run=Path(case['run']) if case.get('run') else a.runs/case['name'];q=run/'terminal_qualification.json'
        if not q.exists() or json.loads(q.read_text())['qualified']!=1:
            excluded.append(case['name']);continue
        reports.append(prepare(case,run,a.output/f'{i:03d}_{case["name"]}.jsonl'))
    if not reports:raise ValueError('No qualified sources')
    summary=dict(schema='verified_owned_wide_commands/v1',episodes=reports,excluded_cases=excluded,
        sources_with_rows=len(reports),rows=sum(r['rows'] for r in reports),
        original_data_modified=False,supervisor_commands_excluded=True,
        actor_training_ready=False,remaining_gate='causal RGB locks',final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
