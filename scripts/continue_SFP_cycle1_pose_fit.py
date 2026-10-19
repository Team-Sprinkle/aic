#!/usr/bin/env python3
"""Wait for both bounded captures, merge unchanged data, fit to LR-floor plateau."""
import json
import os
from pathlib import Path
import subprocess
import time

repo=Path.cwd();py=repo/'.pixi/envs/default/bin/python'
root=Path('/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1')
state=root/'pose_fit_state.json'
assert not state.exists()
def save(data):state.write_text(json.dumps(data,indent=2)+'\n')
def run(cmd,log,env=None):
    log.with_suffix('.command.json').write_text(json.dumps(list(map(str,cmd)),indent=2)+'\n')
    with log.open('w') as output:subprocess.run(list(map(str,cmd)),cwd=repo,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
save(dict(stage='waiting_for_32_QA_approved_captures'))
while not all((root/f'SFP_orientation_capture_worker{i}/summary.json').exists() for i in range(2)):time.sleep(20)
run([py,repo/'scripts/merge_SFP_cycle_orientation_data.py','--root',root],root/'merge_pose_data.log')
data=root/'SFP_wide_orientation_dataset'
save(dict(stage='training_SFP_rotation_to_LR_floor_plateau',source=str(data),physical_gpu=5))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
run([py,repo/'scripts/train_connector_orientation_to_plateau.py','--family','sfp',
     '--source',data,'--pixels',data/'native_crops.npy',
     '--checkpoint','/home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1/frozen/models/sfp_grasp.pt',
     '--output',root/'SFP_wide_orientation_fit','--rotation-only','--balance-new-angles',
     '--crop-jitter','2','--updates','80000','--minutes','180','--min-updates','6000',
     '--patience','2400','--seed','20261018102'],root/'pose_fit.log',env)
summary=json.loads((root/'SFP_wide_orientation_fit/summary.json').read_text())
save(dict(stage='fit_complete_requires_domain_and_visual_QA' if summary['plateau_reached'] else 'incomplete_training_requires_resume',summary=summary,automatic_deployment=False,final_scenes_sealed=True))
