#!/usr/bin/env python3
"""Read-only progress dashboard for the October 17 insertion repairs."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def last_training_record(path):
    try:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size-32768))
            lines=stream.read().decode(errors='replace').splitlines()
        for line in reversed(lines):
            try:
                record=json.loads(line)
                if isinstance(record,dict) and 'step' in record:
                    return record
            except json.JSONDecodeError:
                pass
    except FileNotFoundError:
        pass
    return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/home/nas/chmin/aic_archives/2026-10-17_insertion_control_repair_v1'))
    root=parser.parse_args().root
    print('AIC insertion repairs — '+datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'))
    print('Read-only snapshot; completion is not a performance/promotion claim.\n')
    for name in ['context_masked_current_pose_v1','context_public_current_pose_v1',
                 'sc_nominal_joint_calibrated_v3','sc_nominal_orientation_calibrated_v3',
                 'sc_spatial_grasp_pixel_v6','sc_spatial_grasp_joint_v6']:
        report=read(root/name/'summary.json')
        if report:
            status=f"FINISHED | updates {report.get('updates_completed', '?')} | {report.get('stop_reason', '?')}"
            if 'gate_passed' in report:
                status+=' | component gate '+('PASS' if report['gate_passed'] else 'FAIL')
        else:
            row=last_training_record(root/(name+'.log'))
            status=f"latest logged update {row['step']} (see process/log for liveness)" if row else 'waiting/not started; see scheduler'
        print(f'{name:43s} {status}')
    print()
    for label,folder in [('Fresh SFP visual controller (not BC)','visual_servo_fresh_SFP_coverage_v2/runs'),
                         ('Matched BC context comparison','context_live_current_pose_v1/runs'),
                         ('BC backoff comparison','post_repair_backoff_comparison_v1/runs')]:
        report=read(root/folder/'summary.json') or read(root/folder/'progress.json')
        if report:
            rows=report['results']
            print(f"{label}: {report['completed']}/{report.get('planned',report['completed'])} scored; "
                  f"{sum(bool(row['physical_full_insertion']) for row in rows)} combined successes"+
                  (' [finished]' if (root/folder/'summary.json').exists() else ' [partial]'))
            job=read(root/folder/'current_job.json')
            if job and not (root/folder/'summary.json').exists():
                print('  Latest dispatched: '+job['case']['case'])
        else:
            print(label+': awaiting scored results')
    state=read(root/'post_repair_backoff_comparison_v1/scheduler_state.json')
    if state:
        print('\nBackoff scheduler: '+state['stage'])
        print('Pending precursor summaries: '+str(len(state.get('missing',[]))))
    cycle=Path('/home/nas/chmin/aic_archives/2026-10-18_three_trial_repair_cycle_v1')
    print('\nThree-trial sample cycle:')
    for label,filename in [('Baseline','state.json'),('Repair/rescore','repair_followup_state.json')]:
        current=read(cycle/filename)
        if current:print('  '+label+': '+current['stage'])
    for label,folder in [('First','baseline'),('Second','repaired'),('Recovery candidate','recovery_candidate')]:
        scored=read(cycle/folder/'analysis_summary.json')
        if scored:print('  '+label+' score: '+str(round(scored['total'],2))+'/300; '+str(scored['strict_physical_insertions'])+'/3 combined successes')
        else:
            scored=read(cycle/folder/'summary.json')
            if scored:print('  '+label+' toolkit score: '+str(round(scored['score']['total'],2))+'/300 (physical audit pending)')
    for label,folder in [('Seat dwell repair','dwell_repair_development/runs'),('SC loaded-stall recovery','SC_visual_recovery_diagnostic_v1/runs')]:
        scored=read(cycle/folder/'summary.json') or read(cycle/folder/'progress.json')
        if scored:
            rr=scored['results']
            print('  '+label+': '+str(scored['completed'])+'/'+str(scored.get('planned',scored['completed']))+' scored; '+str(sum(x['physical_full_insertion'] for x in rr))+' combined successes; '+str(sum(x.get('backoff_observations',0) for x in rr))+' completed retreats')
    followup=read(cycle/'SC_recovery_followup_state.json')
    if followup:print('  SC followup: '+followup['stage'])
    print('\nArtifacts: '+str(root))


if __name__=='__main__':
    main()
