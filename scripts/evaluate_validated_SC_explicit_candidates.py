#!/usr/bin/env python3
"""Complete unchanged public sample after a candidate's strict SC pilot passes."""
import json
from pathlib import Path
import shlex
import subprocess
import time

repo = Path.cwd()
python = repo/'.pixi/envs/default/bin/python'
root = Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1')
state = root/'candidate_sample_followup.json'
if state.exists():
    raise FileExistsError('Inspect current coordinator; do not duplicate')


def save(data):
    temporary = state.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(state)


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


results = []
for version, pilot in [(18, 'V18_seating_margin_development'),
                       (19, 'V19_precontact_calibration_development')]:
    save(dict(stage='waiting_for_strict_SC_pilot', version=version, results=results))
    path = root/pilot/'runs/summary.json'
    while not read(path):
        time.sleep(15)
    report = read(path)
    passed = len(report['results']) == 1 and all(
        row['physical_full_insertion'] and row['raw_terminal']['no_scored_harm']
        and row['requested_port_locks_passed'] for row in report['results'])
    if not passed:
        results.append(dict(version=version, pilot_passed=False, sample_run=False))
        continue
    output = root/f'sample_V{version}'
    save(dict(stage='running_unchanged_full_sample', version=version, results=results))
    command = [python, repo/'scripts/run_selected_recovery_sample_configuration.py',
               '--policy', f'RunVisualConnectorServoRecoveryV{version}',
               '--recovery', 'local', '--output', output]
    with (root/f'sample_V{version}.coordinator.log').open('w') as log:
        subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, check=True)
    save(dict(stage='auditing_causal_geometry_and_videos', version=version, results=results))
    command = ('source .pixi/envs/default/setup.bash\n'
               'export PYTHONPATH=/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/host_reader_deps:"${PYTHONPATH:-}"\n'
               + shlex.join([str(python), 'scripts/analyze_full_connector_sample_configuration.py', str(output)]))
    with (root/f'sample_V{version}.analysis.log').open('w') as log:
        subprocess.run(['bash', '-c', command], stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = read(output/'analysis_summary.json')
    results.append(dict(version=version, pilot_passed=True, sample_run=True,
                        score=summary['total'], strict_passes=summary['strict_physical_insertions']))
save(dict(stage='completed_candidates_need_review_or_targeted_repair', results=results,
          final_scenes_opened=False, RL=False))
