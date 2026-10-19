#!/usr/bin/env python3
"""Cache a native-crop refiner's RGB-only selected-port predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_two_family_sc_native_refiner as refiner
from train_sc_native_keypoints import PretrainedKeypointRefiner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--coarse-predictions', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve existing predictions')
    torch.set_num_threads(8)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    refiner.SIZE = checkpoint['crop_size']
    refiner.NETWORK_INPUT_SIZE = checkpoint['network_input_size']
    rows = [json.loads(s) for s in (a.labels / 'rows.jsonl').open()]
    coarse = {(q['row'], q['camera']): q for q in map(json.loads, a.coarse_predictions.open())}
    examples = []
    for i, row in enumerate(rows):
        target = next(t for t in row['targets'] if t['identity'] == row['selected_identity'])
        for camera in refiner.CAMERAS:
            q = coarse[i, camera]['predicted'][row['selected_identity']]
            port=int(row['selected_identity'].split('_')[-1]) if checkpoint.get('condition_target_port',False) else 0
            examples.append((row, camera, q['xy'], target['cameras'][camera][:2],port,True))
    loader = DataLoader(refiner.Crops(examples, False), batch_size=64, shuffle=False, num_workers=0)
    condition_port=bool(checkpoint.get('condition_target_port',False))
    model = PretrainedKeypointRefiner(None, condition_port, point_count=1).cuda().eval()
    model.load_state_dict(checkpoint['model'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    with a.output.open('w') as file, torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for image, _, center, _, _, _, port in loader:
            pred = model(image.cuda(non_blocking=True).float() / 255.,
                         port.cuda(non_blocking=True) if condition_port else None).float().cpu().numpy()
            pixel = pred.reshape(-1, 2) * (refiner.SIZE - 1) + np.round(center.numpy() - refiner.SIZE / 2)
            for j in range(len(pixel)):
                i = (offset + j) // 3
                cam = refiner.CAMERAS[(offset + j) % 3]
                row = rows[i]
                old = coarse[i, cam]['predicted'][row['selected_identity']]
                item = {'row': i, 'camera': cam, 'episode': row['episode'], 'split': row['split'],
                        'selected_identity': row['selected_identity'],
                        'predicted': {row['selected_identity']: {
                            'xy': pixel[j].tolist(),
                            'projected_inframe_probability': old['projected_inframe_probability']}},
                        'runtime_crop_from_rgb': True,
                        'runtime_geometry_input': False, 'visibility_certified': False}
                file.write(json.dumps(item, separators=(',', ':')) + '\n')
            offset += len(pixel)
    print(json.dumps({'rows': len(examples), 'episodes': len({r['episode'] for r in rows}),
                      'coarse_checkpoint': str(a.coarse_predictions), 'refiner': str(a.checkpoint),
                      'output': str(a.output), 'runtime_geometry_input': False}))


if __name__ == '__main__':
    main()
