#!/usr/bin/env python3
"""Score privileged recorded transport replays; never autonomous evidence."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser(description=__doc__);p.add_argument('audit',type=Path);a=p.parse_args()
records=[json.loads(s) for s in (a.audit/'runtime_trace.jsonl').open()]
rows=[r for r in records if r['event']=='recorded_teacher_command']
terminal=[r for r in records if r['event']=='terminal_before_return'];assert rows
source=Path(json.loads((a.audit/'eval_manifest.json').read_text())['teacher_replay_source'])
goal=json.loads(source.read_text())['commands'][-1]['pose'];state=terminal[-1]['state'] if terminal else rows[-1]['state']
result=dict(schema='aic_recorded_teacher_clip_diagnostic/v1',privileged_recorded_route=True,
 autonomous=False,cap_mm=rows[0]['cap_m']*1000,commands=len(rows),
 clipped_fraction=float(np.mean([r['capped'] for r in rows])),
 final_recorded_target_distance_mm=float(np.linalg.norm(np.array(state[:3])-goal[:3])*1000),
 final_orientation_error_deg=float((Rotation.from_quat(state[3:7]).inv()*Rotation.from_quat(goal[3:7])).magnitude()*180/np.pi),
 raw_wrist_force_max_n=float(max(np.linalg.norm(r['state'][26:29]) for r in rows)),
 terminal_observation_retained=bool(terminal),final_scenes_opened=False,
 endpoint_sample='terminal pre-reset' if terminal else 'last recorded pre-abort command; terminal missing; never post-reset geometry',
 note='Recorded target distance, not insertion or perception error. Raw wrench includes gravity. Video/route review additionally required.')
(a.audit/'diagnostic_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
