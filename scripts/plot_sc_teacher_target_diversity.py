#!/usr/bin/env python3
"""Show episode-level SC near-port TCP and recorded target variation."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--canonical', type=Path, required=True)
    p.add_argument('--supplemental', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--summary', type=Path, required=True)
    a = p.parse_args()
    groups = defaultdict(list)
    for source_kind, path in [('canonical', a.canonical), ('supplemental', a.supplemental)]:
        for line in path.open():
            row = json.loads(line)
            tcp = np.asarray(row['observed_tcp_pose_opening_frame'][:3], dtype=float)
            if abs(tcp[2]) >= .03:
                continue
            target = np.asarray(row['teacher_tcp_target_poses_opening_frame'][0][:3], dtype=float)
            port = int(np.argmax(row['task_vector'][2:4]))
            groups[(int(row['episode_index']), source_kind, row['split'], port)].append((tcp, target))
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {0: '#1565c0', 1: '#ef6c00'}
    rows = []
    for (episode, kind, split, port), samples in sorted(groups.items()):
        tcp = np.median(np.stack([x[0] for x in samples]), 0) * 1000
        target = np.median(np.stack([x[1] for x in samples]), 0) * 1000
        color = colors[port]
        ax.annotate('', xy=target[:2], xytext=tcp[:2],
                    arrowprops={'arrowstyle': '->', 'lw': 1, 'alpha': .5, 'color': color})
        ax.scatter(*target[:2], color=color, marker='o' if kind == 'canonical' else 's',
                   s=45, edgecolors='black' if split == 'validation' else 'none',
                   linewidths=.7, zorder=3)
        ax.text(target[0]+.1, target[1]+.1, str(episode), fontsize=6)
        rows.append({'episode': episode, 'kind': kind, 'split': split, 'target_port': port,
                     'near_rows': len(samples), 'tcp_median_xy_mm': tcp[:2].tolist(),
                     'teacher_first_target_median_xy_mm': target[:2].tolist()})
    ax.axhline(0, color='.8', lw=.6); ax.axvline(0, color='.8', lw=.6)
    ax.set_xlabel('Selected opening-frame x (mm)')
    ax.set_ylabel('Selected opening-frame y (mm)')
    ax.set_title('SC teacher TCP targets within 30 mm axially\n'
                 'Blue/orange: selected port 0/1; squares: supplemental; black rim: held out')
    ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.2)
    fig.tight_layout()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=180); plt.close(fig)
    a.summary.write_text(json.dumps({'schema':'sc_teacher_target_diversity/v1',
        'analysis_only':True, 'axial_condition':'abs(observed TCP z in selected opening frame) < 30 mm',
        'episodes':rows}, indent=2)+'\n')
    print(json.dumps({'episodes':len(rows),'image':str(a.output)}))


if __name__ == '__main__':
    main()
