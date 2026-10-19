#!/usr/bin/env python3
"""Recover first matching completion events for BC exclusion, not actor inputs.

Targets after CablePlugin freezes motion are not physical action supervision.
Use the preceding recorded command as a conservative simulation-time cutoff.
Retain post-event observations in originals for separate terminal analysis.
"""
import argparse
from bisect import bisect_left
import json
from pathlib import Path
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sources=[json.loads(l) for l in a.sources.open()]
    if a.limit:sources=sources[:a.limit]
    results=[];typ=get_message('std_msgs/msg/String')
    for s in sources:
        result=dict(source_stream=s['source_stream'],family=s['family'],split=s['split'])
        try:
            d=json.loads(Path(s['source_stream']).read_text());task=d['task'];bags=list((Path(s['source_audit'])/'results').glob('bag_*/*.mcap'))
            if len(bags)!=1:raise ValueError('Expected one original raw bag')
            commands=sorted([c for c in d['commands'] if c['kind']=='pose'],key=lambda c:c['wall_ns']);walls=[c['wall_ns'] for c in commands]
            event=None
            with bags[0].open('rb') as raw:
                reader=make_reader(raw)
                topics={c.topic for c in reader.get_summary().channels.values()}
                if '/scoring/insertion_event' not in topics:raise ValueError('Completion topic not recorded')
                for _,_,message in reader.iter_messages(topics=['/scoring/insertion_event']):
                    namespace=deserialize_message(message.data,typ).data.strip('/')
                    tokens=namespace.split('/')
                    if tokens[-2:]==[task['target_module_name'],task['port_name']]:
                        event=message.log_time,namespace;break
            if event is None:raise ValueError('No matching observed completion event')
            i=bisect_left(walls,event[0])-1
            if i<0:raise ValueError('No pre-completion command')
            result.update(valid=True,first_event_wall_ns=event[0],event_namespace=event[1],
                conservative_BC_cutoff_sim_ns=commands[i]['sim_ns'],
                pose_commands_after_first_event=len(commands)-i-1,
                cutoff_is_teacher_label_only=True,raw_post_event_observations_retained=True)
        except Exception as exc:result.update(valid=False,error=f'{type(exc).__name__}: {exc}')
        results.append(result)
        with (a.output/'results.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        if len(results)%25==0:print(json.dumps(dict(completed=len(results),planned=len(sources),valid=sum(r['valid'] for r in results))),flush=True)
    summary=dict(sources=len(results),valid=sum(r['valid'] for r in results),invalid=sum(not r['valid'] for r in results),
        post_completion_pose_commands=sum(r.get('pose_commands_after_first_event',0) for r in results),
        originals_modified=False,scoring_event_actor_input=False,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
