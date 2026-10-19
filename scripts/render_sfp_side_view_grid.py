#!/usr/bin/env python3
"""Save original RGB view grids before any geometry-marker inspection."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np


CAMERAS=('center','left','right')
STAGES=('baseline_view','positive_view','negative_view')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--labels',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():ap.error('Preserve previous grids')
    rows=defaultdict(dict)
    for s in (args.labels/'rows.jsonl').open():
        r=json.loads(s);rows[r['episode']][r['stage']]=r
    args.output.mkdir(parents=True)
    summary=[]
    for episode,stages in sorted(rows.items()):
        if set(stages)!=set(STAGES):raise ValueError(f'Missing stage {episode}')
        strips=[]
        for stage in STAGES:
            r=stages[stage];tiles=[]
            for camera in CAMERAS:
                src=cv2.imread(r['images'][camera],cv2.IMREAD_COLOR)
                if src is None:raise FileNotFoundError(r['images'][camera])
                tile=cv2.resize(src,(384,341),interpolation=cv2.INTER_AREA)
                cv2.putText(tile,f'{stage} / {camera}',(8,24),cv2.FONT_HERSHEY_SIMPLEX,
                            .55,(0,0,0),3,cv2.LINE_AA)
                cv2.putText(tile,f'{stage} / {camera}',(8,24),cv2.FONT_HERSHEY_SIMPLEX,
                            .55,(255,255,255),1,cv2.LINE_AA)
                tiles.append(tile)
            strips.append(np.concatenate(tiles,axis=1))
        grid=np.concatenate(strips,axis=0)
        path=args.output/f'{episode}_original_rgb_grid.jpg'
        cv2.imwrite(str(path),grid,[cv2.IMWRITE_JPEG_QUALITY,94])
        summary.append(dict(episode=episode,request=stages['baseline_view']['selected_identity'],
                            grid=str(path),unmarked_original_rgb=True))
    (args.output/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
