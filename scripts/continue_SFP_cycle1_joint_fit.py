#!/usr/bin/env python3
"""Matched joint-pose objective: test whether old translation also needs repair."""
import json
import os
from pathlib import Path
import subprocess
import time
repo=Path.cwd();py=repo/'.pixi/envs/default/bin/python'
root=Path('/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1');data=root/'SFP_wide_orientation_dataset'
state=root/'joint_pose_fit_state.json';assert not state.exists()
state.write_text(json.dumps(dict(stage='waiting_for_same_frozen_merged_corpus'))+'\n')
while not (data/'summary.json').exists():time.sleep(20)
command=[py,repo/'scripts/train_connector_orientation_to_plateau.py','--family','sfp','--source',data,
         '--pixels',data/'native_crops.npy','--checkpoint','/home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1/frozen/models/sfp_grasp.pt',
         '--output',root/'SFP_wide_joint_pose_fit','--balance-new-angles','--crop-jitter','2',
         '--updates','80000','--minutes','180','--min-updates','6000','--patience','2400','--seed','20261018102']
(root/'joint_pose_fit.command.json').write_text(json.dumps(list(map(str,command)),indent=2)+'\n')
state.write_text(json.dumps(dict(stage='training_joint_pose_to_LR_floor_plateau',physical_gpu=4))+'\n')
with (root/'joint_pose_fit.log').open('w') as stream:
    subprocess.run(list(map(str,command)),cwd=repo,env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=stream,stderr=subprocess.STDOUT,check=True)
summary=json.loads((root/'SFP_wide_joint_pose_fit/summary.json').read_text())
state.write_text(json.dumps(dict(stage='fit_complete_requires_domain_QA' if summary['plateau_reached'] else 'incomplete_training_requires_resume',summary=summary,automatic_deployment=False),indent=2)+'\n')
