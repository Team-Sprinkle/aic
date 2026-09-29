#!/usr/bin/env python3
"""Predict physical SC plug-tip pixels from RGB-selected plug crops."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from train_sc_native_keypoints import CAMERAS, PretrainedKeypointRefiner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--crop-labels', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    rows = [json.loads(x) for x in args.crop_labels.open()]
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if saved.get('architecture') != 'pretrained_resnet18_tip_only' or saved.get('point_count') != 1:
        raise ValueError('Not a tip-only checkpoint')
    size = int(saved['crop_size'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PretrainedKeypointRefiner(None, False, point_count=1).to(device).eval()
    model.load_state_dict(saved['model'], strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as out, torch.inference_mode():
        for row in rows:
            if row.get('predicted_crop_center_kind') != 'tip':
                raise ValueError('Tip-only predictor requires tip-centered crops')
            images = []
            for cam in CAMERAS:
                with Image.open(row['images'][cam]) as image:
                    rgb = image.convert('RGB')
                    if rgb.size != (size, size):
                        raise ValueError('Tip crop size mismatch')
                    images.append(torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1))
            local = model(torch.stack(images).to(device).float() / 255).cpu().numpy() * (size - 1)
            result = {'trial': row['trial'], 'frame': row['frame'],
                      'episode_id': row['episode_id'], 'sim_time': row['sim_time'], 'cameras': {}}
            for cam, pixel in zip(CAMERAS, local):
                cx, cy = row['predicted_crop_centers'][cam]
                left, top = round(cx - size / 2), round(cy - size / 2)
                result['cameras'][cam] = {'tip_xy': [float(pixel[0] + left), float(pixel[1] + top)]}
            out.write(json.dumps(result, separators=(',', ':')) + '\n')
    print(json.dumps({'rows': len(rows), 'output': str(args.output),
                      'runtime_inputs': 'RGB-only tip crop and frozen RGB locator'}))


if __name__ == '__main__':
    main()
