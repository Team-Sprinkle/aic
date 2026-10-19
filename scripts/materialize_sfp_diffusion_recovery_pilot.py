#!/usr/bin/env python3
"""Join safe actor-prefix/teacher trajectories and preserve actor context as input only."""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

from materialize_sfp_scored_transport import materialize


ROOT=Path(__file__).resolve().parents[1]
EXTRACTOR=ROOT/'scripts/extract_moveit_training_stream.py'
BASE=Path('/var/tmp/chmin_aic_20260918_act/sfp_paired_request_20261005/sfp_divergent_merged_round2_v1/sfp_labels.jsonl')
CALIBRATION=ROOT/'configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior label set')
    pilot=json.loads((a.pilot/'summary.json').read_text())
    if not pilot['gate_passed'] or pilot['accepted']<8 or pilot['final_scenes_opened']:
        p.error('Recovery pilot gate did not pass')
    rows=json.loads((a.pilot/'progress.json').read_text())
    chosen=[r for r in rows.values() if r['accepted']]
    if len(chosen)!=pilot['accepted']:p.error('Pilot progress mismatch')
    a.output.mkdir(parents=True)
    correction=a.output/'correction_labels.jsonl'
    calibration=json.loads(CALIBRATION.read_text())
    reports=[];failures=[]
    with correction.open('w') as out:
        for i,row in enumerate(chosen):
            audit=Path(row['audit'])
            scored=audit/'scored_stream.json'
            if not scored.exists():
                env=os.environ.copy()
                env['PYTHONPATH']='/var/tmp/chmin_aic_mcap_reader_py312'
                env['LD_LIBRARY_PATH']=str(ROOT/'.pixi/envs/default/lib')+':'+env.get('LD_LIBRARY_PATH','')
                subprocess.run([str(ROOT/'.pixi/envs/default/bin/python'),str(EXTRACTOR),
                                str(audit),str(scored)],cwd=ROOT,env=env,check=True,
                               stdout=subprocess.DEVNULL)
            buffer=io.StringIO()
            try:
                report=materialize(audit,calibration,2800000+i,buffer,'train',
                    corrective=True,full_insert_corrective=True,
                    preserve_actor_prefix_context=True)
                if (report['teacher_transport_chunks']<30 or
                    report['actor_prefix_context_chunks']<5 or
                    report['reference_target_error_mm']['p95']>2 or
                    report['reference_orientation_error_deg']['p95']>1 or
                    report['actor_prefix_used_as_supervision']):
                    raise ValueError('Causal label or reference gate failed')
                out.write(buffer.getvalue());reports.append(report)
                print(json.dumps(dict(audit=str(audit),teacher_chunks=report['teacher_transport_chunks'],
                    context_chunks=report['actor_prefix_context_chunks'],
                    target_ref_p95_mm=report['reference_target_error_mm']['p95'])),flush=True)
            except ValueError as exc:
                failures.append(dict(audit=str(audit),error=str(exc)))
                print(json.dumps(dict(audit=str(audit),rejected=str(exc))),flush=True)
    merged=a.output/'sfp_labels.jsonl'
    train_scenes=set();dev_scenes=set();counts=Counter();indices=set()
    with merged.open('w') as out:
        for source,path in [('ordinary',BASE),('correction',correction)]:
            for line in path.open():
                row=json.loads(line);split=row['split'];scene=row['scene_sha256']
                key=(row['episode_index'],row['frame_index'])
                if key in indices:raise ValueError(f'Duplicate label row {key}')
                indices.add(key)
                (train_scenes if split=='train' else dev_scenes).add(scene)
                if source=='correction' and row.get('actor_prefix_context_only'):
                    if row.get('privileged_teacher_after_takeover') or row.get('actor_prefix_used_as_supervision'):
                        raise ValueError('Actor context incorrectly marked for supervised loss')
                counts[(source,split)]+=1
                out.write(line)
    if train_scenes & dev_scenes:raise ValueError('Train/development scene overlap')
    result=dict(schema='aic_sfp_diffusion_recovery_labels/v1',
        pilot=str(a.pilot),pilot_summary_sha256=sha(a.pilot/'summary.json'),
        base=str(BASE),base_sha256=sha(BASE),
        calibration=str(CALIBRATION),calibration_sha256=sha(CALIBRATION),
        reports=reports,failures=failures,
        audited_recovery_episodes=len(reports),label_gate_passed=len(reports)>=8,
        rows_by_source_split={str(k):v for k,v in counts.items()},
        train_development_scene_overlap=0,
        actor_prefix_used_as_supervision=False,final_scenes_opened=False)
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(audited=len(reports),failed=len(failures),
                          label_gate_passed=result['label_gate_passed'])),flush=True)
    if not result['label_gate_passed']:sys.exit(2)


if __name__=='__main__':main()
