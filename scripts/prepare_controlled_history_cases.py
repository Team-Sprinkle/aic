#!/usr/bin/env python3
"""Select proven25mm development starts after the active preflight finishes.

An unused SC three-card preflight fails before actor execution. The controlled
actor comparison therefore reuses the two documented successful25mm start
configurations, with fresh-reset generalization deferred until an actor works.
"""
import json
from pathlib import Path
import subprocess
import time


def main():
    repo=Path.cwd();root=Path('/home/nas/chmin/aic_archives/2026-10-17_history_chunk_continuation_v1')
    integration=Path('/home/nas/chmin/aic_archives/2026-10-17_history_chunk_integration_v1')
    pid=int((integration/'coordinator.pid').read_text())
    while True:
        status=Path(f'/proc/{pid}/stat')
        if not status.exists() or status.read_text().split()[2]=='Z':break
        time.sleep(20)
    if (root/'live_comparison').exists():raise RuntimeError('Do not change frozen live scenes after comparison starts')
    source=repo/'scripts/run_history_chunk_live_comparison.py'
    text=source.read_text();anchor="    variants=json.loads((a.root/'evaluation_variants.json').read_text())['variants']\n"
    addition="    if (a.root/'evaluation_cases.json').exists():\n        cases=json.loads((a.root/'evaluation_cases.json').read_text())['cases']\n"
    if addition not in text:
        assert text.count(anchor)==1
        (root/'live_runner_before_controlled_case_selection.py').write_text(text)
        text=text.replace(anchor,addition+anchor)
        text=text.replace('Unused replicate2sample-derived development configurations; no official or reserved final scenes.',
            'Explicit controlled development cases; evaluation_cases.json selects reused25mm integration starts. No official/final scenes.')
        text=text.replace('Two new development configurations per variant are a diagnostic screen, not a generalization certificate.',
            'Two controlled development configurations per variant are a diagnostic screen. Reused starts do not establish generalization.')
        compile(text,str(source),'exec');source.write_text(text)
    prior=Path('/home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1')
    plan=json.loads((prior/'plan.json').read_text())
    cases=[next(c for c in plan['scenarios'] if c['family']==f and c['card_count']==1 and c['port']==0
                and (f=='sc' or c['requested_card']==0)) for f in ['sc','sfp']]
    data=dict(cases=cases,scope='Reused development25mm integration configurations; not evidence of generalization.',
        reason='Separate actor effects from a newly observed upstream SC route failure before actor execution.',
        model_or_rule_or_scene_geometry_changed=False,official_or_final_scenes_used=False)
    (root/'evaluation_cases.json').write_text(json.dumps(data,indent=2)+'\n')
    check=Path('/home/nas/chmin/aic_archives/2026-10-17_history_known_start_preflight_v2');check.mkdir()
    (check/'evaluation_cases.json').write_text(json.dumps(data,indent=2)+'\n')
    (check/'evaluation_variants.json').write_text((integration/'evaluation_variants.json').read_text())
    with (check/'coordinator.log').open('w') as log:
        p=subprocess.Popen([str(repo/'.pixi/envs/default/bin/python'),'-u',str(source),'--root',str(check),'--output',str(check/'runs')],
            cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (check/'coordinator.pid').write_text(str(p.pid)+'\n')
    (root/'controlled_case_revision.json').write_text(json.dumps(dict(initial_preflight=str(integration),
        revised_runtime_preflight=str(check),active_jobs_interrupted=False,
        comparison_is_reused_development_not_generalization=True),indent=2)+'\n')


if __name__=='__main__':main()
