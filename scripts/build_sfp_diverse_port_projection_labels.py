#!/usr/bin/env python3
"""Posthoc port projections for new grouped coarse RGB experiments only."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import pose_matrix
from build_sc_native_pose_labels import project
from build_two_family_port_projection_labels import sfp_ports


CAMERAS=('center','left','right')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--camera-calibration',type=Path,required=True)
    p.add_argument('--sfp-calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve previous output')
    camera=json.loads(a.camera_calibration.read_text())
    optics={c:np.asarray(camera['tcp_to_optical'][c],float) for c in CAMERAS}
    sfp=json.loads(a.sfp_calibration.read_text())
    cached={};written=0;groups={}
    a.output.mkdir(parents=True)
    with (a.output/'rows.jsonl').open('w') as out:
        for line in (a.source/'rows.jsonl').open():
            row=json.loads(line)
            stream=row['source_scored_stream']
            if stream not in cached:
                cfg=Path(stream).parent/'eval_config.yaml'
                import yaml
                tid=next(iter(yaml.safe_load(cfg.read_text())['trials']))
                cached[stream]=sfp_ports({'trial_yaml':str(cfg),'trial_id':tid},sfp)
            tcp=pose_matrix(np.asarray(row['state'][:7],float))
            targets=[]
            for card,port,m in cached[stream]:
                targets.append(dict(identity=f'sfp_card_{card}_port_{port}',
                                    position_base_m=m[:3,3].tolist(),
                                    orientation_base_xyzw=Rotation.from_matrix(m[:3,:3]).as_quat().tolist(),
                                    cameras={c:project(m,tcp@optics[c]) for c in CAMERAS}))
            row['targets']=targets
            row['fit_split']=row['split']
            row['projection_is_not_visibility']=True
            groups[row['scene_group']]=row['split']
            out.write(json.dumps(row,separators=(',',':'))+'\n')
            written+=1
    summary=dict(schema='aic_sfp_diverse_port_projection_labels/v1',
                 source=str(a.source),rows=written,groups=len(groups),
                 projections_training_or_posthoc_score_only=True,
                 geometry_runtime_input=False,final_scenes_opened=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
