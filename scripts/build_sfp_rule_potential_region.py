#!/usr/bin/env python3
"""Conservative workspace prior from training regions and fixed calibration.

This region is a safety plausibility bound, not a requested-port classifier.
Neither development target poses nor final configurations set its boundaries.
"""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--asset-calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous prior versions')
    data=json.loads(a.input.read_text());cal=json.loads(a.asset_calibration.read_text())
    tasks=list(data['tasks'].values())
    low=[min(t['potential_min_base_m'][j] for t in tasks) for j in range(3)]
    high=[max(t['potential_max_base_m'][j] for t in tasks) for j in range(3)]
    # Nominal flat board and robot base have the same configured Z datum.
    # Only rough ray-plane guidance uses this fixture-height assumption.
    height=cal['board_to_nic_mount']['z_m']+cal['nic_mount_to_port_opening']['0']['xyz_m'][2]
    for task in tasks:
        task['original_task_specific_potential_min_base_m']=task['potential_min_base_m']
        task['original_task_specific_potential_max_base_m']=task['potential_max_base_m']
        task['potential_min_base_m']=low;task['potential_max_base_m']=high
        task['guidance_nominal_height_m']=height;task['guidance_height_uncertainty_m']=.020
    data.update(revision='Union of training task regions; not an identity classifier',
        guidance_height_assumption='nominal flat fixture; rough centering only, never insertion lock',
        guidance_height_calibration=str(a.asset_calibration),original_priors_preserved=True,
        development_geometry_used=False,final_scenes_opened=False)
    a.output.write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
