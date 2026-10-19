#!/usr/bin/env python3
"""Render an explicitly offline video of a saved MoveIt SFP TCP corridor."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--variant', choices=('direct', 'align_then_approach', 'retreat_align_approach'),
                   default='retreat_align_approach')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fps', type=int, default=20)
    args = p.parse_args()
    if args.output.exists():
        p.error('Preserve prior video')
    plan = json.loads(args.plan.read_text())
    route = plan['route_variants'][args.variant]
    if not route['success'] or len(route['samples']) < 2:
        p.error('Missing completed saved path')
    scored = json.loads((Path(plan['audit']) / 'scored_stream.json').read_text())
    takeover = plan['takeover_sim_s']
    nearest = min(scored['cameras']['center'], key=lambda row: abs(row[0] * 1e-9 - takeover))
    if abs(nearest[0] * 1e-9 - takeover) > .05:
        p.error('No camera frame within 50 ms of saved takeover')
    image = cv2.cvtColor(cv2.imread(nearest[1]), cv2.COLOR_BGR2RGB)
    origin = np.asarray(plan['selected_port_base_m'], float)
    rotation = Rotation.from_quat(plan['selected_port_quat_xyzw']).as_matrix()
    path_base = np.asarray([row['tcp_base_m'] for row in route['samples']], float)
    relative = (rotation.T @ (path_base - origin).T).T * 1000
    path_mm = float(np.linalg.norm(np.diff(path_base, axis=0), axis=1).sum() * 1000)
    frame_count = max(args.fps * 6, len(relative) * 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), args.fps,
                             (1600, 900))
    if not writer.isOpened():
        raise RuntimeError('Could not open MP4 writer')
    try:
        for frame in range(frame_count):
            progress = max(0., min(1., (frame / args.fps - .6) / (frame_count / args.fps - 1.2)))
            index = min(len(relative) - 1, round(progress * (len(relative) - 1)))
            fig = plt.figure(figsize=(16, 9), dpi=100, facecolor='#f8f9fc')
            layout = fig.add_gridspec(2, 2, width_ratios=[1.12, 1], hspace=.27, wspace=.2,
                                      left=.04, right=.98, top=.85, bottom=.11)
            camera = fig.add_subplot(layout[:, 0])
            camera.imshow(image)
            camera.axis('off')
            camera.set_title('Frozen center-camera frame at actor takeover', fontsize=13)
            for panel, dimension, label in (
                    (fig.add_subplot(layout[0, 1]), 0, 'Port-frame lateral x (mm)'),
                    (fig.add_subplot(layout[1, 1]), 1, 'Port-frame lateral y (mm)')):
                panel.plot(relative[:, 2], relative[:, dimension], color='#a8afbd',
                           linewidth=2.5, label='Complete planned TCP route')
                panel.plot(relative[:index + 1, 2], relative[:index + 1, dimension],
                           color='#1976d2', linewidth=3.2, label='Traversed in animation')
                panel.scatter(relative[0, 2], relative[0, dimension], s=90, color='#229954',
                              zorder=5, label='Saved actor state')
                panel.scatter(relative[index, 2], relative[index, dimension], s=130,
                              color='#1976d2', edgecolor='white', linewidth=1.5, zorder=6,
                              label='Planned TCP')
                panel.scatter([-85], [0], marker='*', s=240, color='#e67e22', zorder=6,
                              label='85 mm buffered goal')
                panel.scatter([0], [0], marker='X', s=100, color='#b03a2e', zorder=6,
                              label='Requested port opening')
                panel.set_xlim(min(-260, relative[:, 2].min() - 12), 15)
                panel.set_ylim(min(-70, relative[:, dimension].min() - 12),
                               max(70, relative[:, dimension].max() + 12))
                panel.set_xlabel('Axial distance to opening (mm)')
                panel.set_ylabel(label)
                panel.grid(alpha=.25)
            fig.suptitle('OFFLINE MOVEIT PLAN — NOT GAZEBO EXECUTION', fontsize=20,
                         fontweight='bold', color='#a93226', y=.97)
            fig.text(.5, .895,
                     f"Requested card {plan['request_card']}, port {plan['request_port']}  |  "
                     f"{args.variant.replace('_', ' ')}  |  planned TCP travel {path_mm:.1f} mm  |  "
                     f"sample {index + 1}/{len(relative)}", ha='center', fontsize=12)
            fig.text(.5, .035,
                     'The camera image is static. The moving dot follows saved planned TCP poses; '
                     'cable motion and contact are not simulated.', ha='center', fontsize=11,
                     color='#555555')
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            plt.close(fig)
    finally:
        writer.release()
    print(json.dumps({'output': str(args.output), 'case': str(args.plan),
                      'variant': args.variant, 'fps': args.fps, 'frames': frame_count,
                      'path_mm': path_mm, 'gazebo_executed': False}))


if __name__ == '__main__':
    main()
