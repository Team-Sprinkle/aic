#!/usr/bin/env python3
"""Read-only status of explicit SC repair, full scores and bounded pose fit."""
import json
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


root = Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1')
print(datetime.now(timezone.utc).strftime('%H:%M:%S UTC'))
print('Current controller checks (not learned BC/RL):')
for name in ['V21_three_sample_state.json','V26_three_sample_state.json','V27_three_sample_state.json']:
    value=read(root/name)
    if value:
        print(name,value['stage'],'round',value.get('round'),
              'finished',[(round(r['total'],2),r['strict_successes']) for r in value.get('results',[])])
for name in ['V28_context_fusion_failed_case','V29_tight_alignment_failed_case',
             'V30_lateral_tracking_failed_case','V31_lateral_bias_seed_failed_case',
             'V32_lateral_stiffness_failed_case','V33_lateral_stiffness_failed_case']:
    result=read(root/name/'runs/summary.json') or read(root/name/'runs/progress.json')
    if result:
        print(name,'finished',result['completed'],
              'physical passes',sum(r['physical_full_insertion'] for r in result['results']))
    elif (root/name/'runs/current_job.json').exists():
        print(name,'running or awaiting scoring')
    else:
        continue
current=read(root/'V34_SC_matched_coverage/state.json')
if current:
    print('V34',current['stage'])
    for row in current.get('results',[]):
        print(' ',row.get('case'), 'PASS' if row.get('gate',{}).get('explicit_trial_pass') else 'FAIL',row.get('physical',row.get('error')))
    for row in current.get('sample_results',[]):
        print(' ',row['root'],round(row['total'],2),row['strict_successes'],'strict; timing',row['timing']['p95_gate_passed'])
print('Preserved earlier comparisons:')
for name in ['sample_V17', 'sample_V17_repeat2', 'sample_V21', 'sample_V18', 'sample_V19', 'sample_V20']:
    state = read(root/name/'state.json')
    if state:
        summary = read(root/name/'analysis_summary.json') or read(root/name/'analysis_progress.json')
        print(name, state)
        for row in (summary or {}).get('results', []):
            g = row['physical']
            print(' ', row['trial'], 'PASS' if g['qualified'] else 'FAIL',
                  'depth/lateral/R', *[round(g[k], 3) for k in
                                     ['depth_mm', 'lateral_mm', 'orientation_error_deg']])
for name in ['V17_fresh_SC_development', 'V18_seating_margin_development',
             'V19_precontact_calibration_development', 'V20_calibrated_speed_development',
             'V22_contact_compliance_development', 'V21_close_grasp_development']:
    data = read(root/name/'runs/summary.json') or read(root/name/'runs/progress.json')
    if data:
        print(name, data['completed'], 'scored;',
              sum(r['physical_full_insertion'] for r in data['results']), 'strict passes')
        for row in data['results']:
            g = row['raw_terminal']
            print(' ', row['case'], 'depth/lateral/R',
                  *[round(g[k], 3) for k in ['depth_mm', 'lateral_mm', 'orientation_error_deg']])
for filename in ['state.json', 'prefill_progress.json']:
    data = read(root/'acquisition_captures'/filename)
    if data:
        print('captures', filename, {k: v for k, v in data.items() if k != 'results'})
for label, filename in [('pose followup', 'pose_repair_state.json'), ('near followup', 'near_pose_repair_state.json')]:
    data = read(root/filename)
    print(label, {k: v for k, v in (data or {}).items() if k != 'metrics'})
print('conditional compact ablation', read(root/'compact_crop_ablation_state.json'))
for name in ['acquisition_spatial_joint', 'far_near_spatial_joint', 'compact_far_near_spatial_joint',
             'expanded_near_fused_orientation_warm', 'spacefilled_near_fused_orientation_warm',
             'fine_feature_near_orientation', 'fine_feature_near_orientation_frozen_base']:
    history = read(root/name/'history.json')
    if history:
        best = min(history, key=lambda row: row['score'])
        summary = read(root/name/'summary.json')
        print(name, 'updates', history[-1]['step'], 'best normalized gate score', round(best['score'], 3),
              'best step', best['step'], 'finished', summary is not None,
              'plateau', (summary or {}).get('plateau_reached'),
              'qualified', (summary or {}).get('gate_passed'))
