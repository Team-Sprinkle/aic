#!/usr/bin/env python3
"""If fixed-crop regression fails, use pixel landmarks with calibrated geometry."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

repo=Path.cwd();py=repo/'.pixi/envs/default/bin/python';root=Path('/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1')
state=root/'geometry_pose_fit_state.json';assert not state.exists()
def save(data):state.write_text(json.dumps(data,indent=2)+'\n')
save(dict(stage='waiting_for_fixed_crop_fit_convergence'))
while not (root/'SFP_fixed_crop_joint_fit/summary.json').exists():time.sleep(20)
prior=json.loads((root/'SFP_fixed_crop_joint_fit/summary.json').read_text());assert prior['plateau_reached']
wide=prior['expanded_development_validation']
if prior['gate_passed'] and wide['orientation_p50_p95_deg'][1]<=1 and wide['translation_p50_p95_mm'][1]<=.4:
    save(dict(stage='not_needed_fixed_crop_fit_passed'));raise SystemExit(0)
source=root/'SFP_wide_orientation_dataset';dataset=root/'SFP_geometry_dataset';dataset.mkdir(exist_ok=False)
rows=json.loads((source/'manifest.json').read_text());pixels=np.load(source/'native_crops.npy',mmap_mode='r');poses=np.load(source/'body_tip_matrices.npy')
(dataset/'manifest.json').symlink_to(source/'manifest.json')
np.save(dataset/'labels.npy',np.c_[poses[:,:3,3],poses[:,:3,:2].transpose(0,2,1).reshape(-1,6)])
output=np.lib.format.open_memmap(dataset/'native_crops.npy',mode='w+',dtype=np.uint8,shape=(len(rows),3,192,192,3))
for i in range(0,len(rows),32):output[i:i+32]=pixels[i:i+32,:,8:200,8:200]
output.flush();del output
trainer=(repo/'scripts/train_sc_spatial_grasp_to_plateau.py').read_text()
trainer=trainer.replace('from train_sc_grasp_keypoint_probe import CAD,rigid','from prepare_sfp_native_grasp_labels import REFERENCE as CAD')
trainer=trainer.replace("model=PretrainedKeypointRefiner(None,False,point_count=5,condition_camera=True);model.load_state_dict(initial['model'],strict=True);model.backbone.layer4=nn.Identity();model.backbone.avgpool=nn.Identity();model.backbone.fc=nn.Identity();model.cuda()", "model=PretrainedKeypointRefiner(None,False,point_count=5,condition_camera=True);model.backbone.layer4=nn.Identity();model.backbone.avgpool=nn.Identity();model.backbone.fc=nn.Identity();model.load_state_dict(initial['model'],strict=True);model.cuda()")
trainer=trainer.replace("'nominal_home'","'wide'").replace("startswith('Nominal-centred')","startswith('Wide SFP')")
trainer=trainer.replace("schema='sc_spatial_grasp_training/v1'", "schema='sfp_spatial_grasp_training/v1',family='sfp'")
trainer=trainer.replace("schema='sc_rgb_grasp_geometry/v1'", "schema='sfp_rgb_grasp_geometry/v1'")
fit=root/'SFP_calibrated_geometry_fit';snapshot=fit.with_suffix('.trainer_snapshot.py');snapshot.write_text(trainer)
cmd=[py,snapshot,'--source',dataset,'--pixels',dataset/'native_crops.npy','--checkpoint','/home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1/frozen/models/sfp_grasp.pt','--output',fit,'--mode','joint','--pose-solver','kabsch','--selection-domains','original','wide','--crop-jitter','4','--updates','100000','--minutes','180','--min-updates','6000','--patience','2400']
(root/'geometry_pose_fit.command.json').write_text(json.dumps(list(map(str,cmd)),indent=2)+'\n');save(dict(stage='training_calibrated_keypoint_geometry_to_plateau',physical_gpu=3,source=str(dataset)))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONPATH=str(repo/'scripts')+':'+os.environ.get('PYTHONPATH',''),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
with (root/'geometry_pose_fit.log').open('w') as log:subprocess.run(list(map(str,cmd)),cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
summary=json.loads((fit/'summary.json').read_text());save(dict(stage='geometry_fit_complete_requires_review',summary=summary,automatic_deployment=False))
