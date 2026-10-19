#!/usr/bin/env python3
"""Controlled repair: fixed calibrated native crops, same photometric augmentation."""
from pathlib import Path
import sys
root=Path('/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1')
base=root/'SFP_wide_joint_pose_fit.converged_trainer_snapshot.py'
source=base.read_text()
old="shifts = rng.integers(0,17,size=(len(ids),3,2)) if jitter else np.full((len(ids),3,2),8)"
assert source.count(old)==1;source=source.replace(old,"shifts = np.full((len(ids),3,2),8)")
a=source.index('        if jitter and a.crop_jitter:');b=source.index('        return torch.from_numpy(array)',a)
source=source[:a]+'''        if jitter:
            # Preserve the previous colour augmentation. Only unobserved
            # independent camera crop shifts are removed in this ablation.
            gain=rng.uniform(.85,1.15,size=(len(ids),3,1,1,3))
            array=np.clip(array*gain,0,255).astype(np.uint8)
'''+source[b:]
source=source.replace("rotation_only=a.rotation_only,freeze_encoder=a.freeze_encoder,crop_jitter=a.crop_jitter,", "rotation_only=a.rotation_only,freeze_encoder=a.freeze_encoder,crop_jitter=0,fixed_camera_crop_origins=True,photometric_augmentation=True,")
out=Path(sys.argv[sys.argv.index('--output')+1]);snapshot=out.with_suffix('.fixed_crop_trainer_snapshot.py');assert not snapshot.exists();snapshot.write_text(source)
exec(compile(source,str(snapshot),'exec'),dict(__name__='__main__',__file__=str(snapshot)))
