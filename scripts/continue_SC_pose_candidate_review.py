#!/usr/bin/env python3
"""Complete both matched fits' CPU prediction QA without deploying a model."""
import json
from pathlib import Path
import subprocess
import time
root=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1')
state=root/'pose_prediction_review_state.json'
if state.exists():
    raise FileExistsError('Inspect the existing reviewer; do not duplicate it')
repo=Path.cwd()
pending=['acquisition_spatial_joint','far_near_spatial_joint']
results={}
while pending:
    state.write_text(json.dumps(dict(stage='waiting_for_completed_fits',pending=pending,results=results),indent=2)+'\n')
    for name in list(pending):
        fit=root/name
        if not (fit/'summary.json').exists():
            continue
        output=root/(name+'_prediction_review')
        command=[str(repo/'.pixi/envs/default/bin/python'),str(repo/'scripts/review_sc_spatial_pose_candidate.py'),'--fit',str(fit),'--output',str(output)]
        (root/(name+'_prediction_review.command.json')).write_text(json.dumps(command)+'\n')
        with (root/(name+'_prediction_review.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        results[name]=json.loads((output/'approval.json').read_text())
        pending.remove(name)
    if pending:
        time.sleep(30)
state.write_text(json.dumps(dict(stage='prediction_QA_complete_requires_actual_visual_inspection',results=results,automatic_deployment=False),indent=2)+'\n')
