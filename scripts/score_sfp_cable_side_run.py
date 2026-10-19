#!/usr/bin/env python3
"""Score one predeclared privileged cable-side route replay."""
import argparse
import json
from pathlib import Path

import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit',type=Path)
    a=p.parse_args()
    out=a.audit/'cable_side_gate.json'
    if out.exists():p.error('Preserve previous score')
    replay=[json.loads(x) for x in (a.audit/'moveit_gazebo_replay.jsonl').read_text().splitlines()]
    take=next(x for x in replay if x['event']=='takeover')
    steps=[x for x in replay if x['event']=='waypoint']
    final=replay[-1]
    count=0;max_streak=0
    for x in steps:
        count=count+1 if x['force_change_n']>6 else 0
        max_streak=max(max_streak,count)
    nic_peak=0.;nic_peak_t=None;board_peak=0.
    timeline=a.audit/'corrected_contact_timeline.jsonl'
    for line in timeline.open():
        row=json.loads(line);t=row['sim_ns']/1e9
        if t < take['sim_s']:continue
        for c in row['contacts']:
            pair=c['pair'];force=c['max_point_normal_force_n']
            if 'cable_0' not in pair:continue
            if 'nic_card' in pair and force>nic_peak:
                nic_peak=force;nic_peak_t=t
            if 'task_board_base' in pair:board_peak=max(board_peak,force)
    scoring=yaml.safe_load((a.audit/'results/scoring.yaml').read_text())
    trial=next(k for k in scoring if k.startswith('trial_'))
    contact_msg=scoring[trial]['tier_2']['categories']['contacts']['message']
    official_clear=contact_msg=='No contact detected.'
    criteria=dict(buffer_arrival=(final['event']=='finished' and
                                  final.get('goal_tcp_error_mm',float('inf'))<=10),
                  no_sustained_wrist_force=max_streak<3,
                  cable_nic_point_force=nic_peak<=10,
                  official_contact_clear=official_clear)
    result=dict(schema='aic_sfp_cable_side_gate/v1',audit=str(a.audit),
                criteria=criteria,pass_all=all(criteria.values()),
                final=final,takeover_sim_s=take['sim_s'],
                max_wrist_force_change_n=max(x['force_change_n'] for x in steps),
                max_over_6n_waypoint_streak=max_streak,
                peak_cable_nic_point_normal_n=nic_peak,
                peak_cable_nic_sim_s=nic_peak_t,
                peak_cable_board_point_normal_n=board_peak,
                official_contact_message=contact_msg,
                threshold_mm=10,threshold_wrist_n=6,
                threshold_wrist_streak=3,threshold_nic_point_n=10,
                privileged_route=True,autonomous_success_claim=False)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['pass_all','criteria','max_wrist_force_change_n',
                                           'max_over_6n_waypoint_streak',
                                           'peak_cable_nic_point_normal_n',
                                           'peak_cable_board_point_normal_n']}))


if __name__=='__main__':main()
