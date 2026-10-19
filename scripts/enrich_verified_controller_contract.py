#!/usr/bin/env python3
"""Audit every selected recorded command and copy its complete control profile."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commands',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((a.commands/'summary.json').read_text());reports=[]
    for e in source['episodes']:
        raw=json.loads(Path(e['source']).read_text());commands=[c for c in raw['commands'] if c['kind']=='pose'];count=0
        with (a.output/e['rows_file']).open('w') as f:
            for row in map(json.loads,(a.commands/e['rows_file']).open()):
                for index in row['source_command_indices']:
                    m=commands[index]['message'];w=m['feedforward_wrench_at_tip']
                    force=[w[k][axis] for k in ('force','torque') for axis in 'xyz']
                    if (not np.allclose(force,0) or not np.allclose(m['wrench_feedback_gains_at_tip'],[.5,.5,.5,0,0,0])
                            or m['trajectory_generation_mode']['mode']!=2):raise ValueError('Recorded controller differs from the deployable position helper')
                row.update(recorded_controller_wrench_feedback_gains=[.5,.5,.5,0.,0.,0.],
                           recorded_feedforward_wrench=[0.]*6,recorded_trajectory_mode=2,
                           full_recorded_controller_contract_verified=True)
                f.write(json.dumps(row,separators=(',',':'))+'\n');count+=1
        if count!=e['rows']:raise ValueError('Derivative row count changed')
        reports.append(e)
        print(json.dumps(dict(completed=len(reports),planned=len(source['episodes']))),flush=True)
    report=dict(source,episodes=reports,source_command_derivative=str(a.commands),
                source_summary_sha256=hashlib.sha256((a.commands/'summary.json').read_bytes()).hexdigest(),
                full_recorded_controller_contract_verified=True,original_modified=False)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
