#!/usr/bin/env python3
"""Label fresh scored SFP RGB for offline, frozen model evaluation only."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import pose_matrix
from build_sc_native_pose_labels import project
from build_two_family_port_projection_labels import CAMERAS, from_controller, sfp_ports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--stride-seconds', type=float, default=.5)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Preserve previous label output')
    plan = json.loads(args.plan.read_text())
    root = args.plan.parent
    cal = json.loads(Path('configs/hierarchical_recovery/sfp_port_opening_gazebo_tf_20260924.json').read_text())
    optics_json = json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics = {c: np.asarray(optics_json['tcp_to_optical'][c], float) for c in CAMERAS}
    rows = []
    summary = []
    for spec in plan['scenes']:
        audit = root/f"teacher_{spec['name']}"
        path = audit/'scored_stream.json'
        if not path.exists():
            summary.append(dict(name=spec['name'], scored=False, rows=0))
            continue
        stream = json.loads(path.read_text())
        ports = sfp_ports({'trial_yaml': str(audit/'eval_config.yaml'),
                           'trial_id': stream['trial_id']}, cal)
        target_id = f"sfp_card_{spec['card']}_port_{spec['port']}"
        if target_id not in {f'sfp_card_{c}_port_{p}' for c,p,_ in ports}:
            raise ValueError(f'Missing requested port: {spec["name"]}')
        controllers = stream['streams']['/aic_controller/controller_state']
        cameras = stream['cameras']
        last = -1e10
        start = len(rows)
        for stamp, _ in cameras['center']:
            if stamp/1e9-last < args.stride_seconds:
                continue
            images = {}
            good = True
            for cam in CAMERAS:
                seq = cameras[cam]
                j = bisect_right(seq, stamp, key=lambda x:x[0])-1
                if j < 0 or stamp-seq[j][0] > 50_000_000:
                    good = False; break
                virtual = seq[j][1]
                if virtual.startswith('/audit/'):
                    image = audit/virtual[len('/audit/'):]
                else:
                    image = Path(virtual)
                    if audit.resolve() not in image.resolve().parents:
                        raise ValueError(f'Image outside rollout: {virtual}')
                if not image.is_file():
                    raise FileNotFoundError(image)
                images[cam] = str(image)
            j = bisect_right(controllers, stamp, key=lambda x:x[0])-1
            if not good or j < 0 or stamp-controllers[j][0] > 100_000_000:
                continue
            state = from_controller(controllers[j][1])
            tcp = pose_matrix(state)
            targets = []
            for card, port, m in ports:
                targets.append(dict(identity=f'sfp_card_{card}_port_{port}',
                                    position_base_m=m[:3,3].tolist(),
                                    orientation_base_xyzw=Rotation.from_matrix(m[:3,:3]).as_quat().tolist(),
                                    cameras={cam:project(m,tcp@optics[cam]) for cam in CAMERAS}))
            last = stamp/1e9
            rows.append(dict(family='sfp', episode=spec['name'],
                             scene_group=spec['reset_sha256'], split='validation',
                             time_s=last, state=state.tolist(), images=images,
                             image_kind='native_1152x1024_rgb_jpeg', targets=targets,
                             selected_identity=target_id,
                             card_count=spec['card_count'],
                             geometry_training_label_only=True,
                             source_scored_stream=str(path)))
        summary.append(dict(name=spec['name'], scored=True, rows=len(rows)-start,
                            official_tier3=stream['official_tier3']))
    args.output.mkdir(parents=True)
    with (args.output/'rows.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row,separators=(',',':'))+'\n')
    meta = dict(schema='sfp_fresh_anchor_labels/v1', episodes=summary,
                rows=len(rows), stride_seconds=args.stride_seconds,
                simulator_geometry_is_label_only=True, final_scenes_opened=False)
    (args.output/'summary.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
