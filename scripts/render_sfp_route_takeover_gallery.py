#!/usr/bin/env python3
"""Render original camera views nearest each saved route takeover for inspection."""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve earlier gallery')
    names = ['side_down_1', 'side_down_2', 'side_sweep_1',
             'side_sweep_2', 'side_sweep_3']
    cameras = ['overhead', 'side', 'left', 'center', 'right']
    width, height, margin = 320, 220, 28
    sheet = Image.new('RGB', (width * len(cameras), height * len(names) + margin), 'white')
    draw = ImageDraw.Draw(sheet)
    selected = []
    for col, camera in enumerate(cameras):
        draw.text((col * width + 4, 5), camera, fill='black')
    for row_idx, name in enumerate(names):
        folder = a.source / name
        event = next(json.loads(line) for line in (folder / 'moveit_gazebo_replay.jsonl').open()
                     if json.loads(line).get('event') == 'takeover')
        take_s = float(event['sim_s'])
        frames = list(csv.DictReader((folder / 'frame_manifest.csv').open()))
        for col, camera in enumerate(cameras):
            choices = [x for x in frames if x['camera'] == camera]
            closest = min(choices, key=lambda x: abs(int(x['sim_ns']) * 1e-9 - take_s))
            image = Image.open(folder / 'frames' / closest['file']).convert('RGB')
            image.thumbnail((width, height - 24))
            x = col * width + (width - image.width) // 2
            y = margin + row_idx * height
            sheet.paste(image, (x, y + 24))
            draw.text((col * width + 4, y + 4),
                      f'{name}  t={int(closest["sim_ns"]) * 1e-9:.2f}', fill='black')
            selected.append({'run': name, 'camera': camera, 'takeover_s': take_s,
                             'frame_sim_s': int(closest['sim_ns']) * 1e-9,
                             'path': str(folder / 'frames' / closest['file'])})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.output)
    a.output.with_suffix('.json').write_text(json.dumps(selected, indent=2) + '\n')
    print(a.output)


if __name__ == '__main__':
    main()
