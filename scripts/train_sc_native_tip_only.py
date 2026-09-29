#!/usr/bin/env python3
"""Train an RGB plug-tip refiner on a plug-centered native-resolution crop.

Scored physical-tip projections supervise training only. The crop is selected
by a frozen RGB locator prediction, including at evaluation time.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from train_sc_native_keypoints import CAMERAS, PretrainedKeypointRefiner, percentile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-labels', type=Path, required=True)
    p.add_argument('--crop-labels', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--pretrained-resnet', type=Path, required=True)
    p.add_argument('--updates', type=int, default=3000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=20260924)
    args = p.parse_args()
    torch.set_num_threads(8); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    truth = [json.loads(x) for x in args.native_labels.open()]
    crops = [json.loads(x) for x in args.crop_labels.open()]
    if [(r['trial'], r['frame']) for r in truth] != [(r['trial'], r['frame']) for r in crops]:
        raise ValueError('Unmatched native/crop rows')
    if any(r.get('predicted_crop_center_kind') != 'tip' for r in crops):
        raise ValueError('Tip-only refiner requires observation-selected tip-centered crops')
    size = 224
    images = []; labels = []; splits = []; keys = []; rejected = {'out_of_frame': 0, 'out_of_crop': 0}
    for row, crop in zip(truth, crops):
        for cam in CAMERAS:
            tip = row['projected_training_pixels'][cam]['tip']
            if not tip[3]:
                rejected['out_of_frame'] += 1
                continue
            cx, cy = crop['predicted_crop_centers'][cam]
            left, top = round(cx - size / 2), round(cy - size / 2)
            x, y = tip[0] - left, tip[1] - top
            if not (0 <= x < size and 0 <= y < size):
                rejected['out_of_crop'] += 1
                continue
            with Image.open(crop['images'][cam]) as image:
                rgb = image.convert('RGB')
                if rgb.size != (size, size):
                    raise ValueError('Wrong tip crop shape')
                images.append(torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1))
            labels.append([x / (size - 1), y / (size - 1)])
            splits.append(row['split']); keys.append((row['trial'], row['frame'], cam))
    x = torch.stack(images); y = torch.tensor(labels, dtype=torch.float32)
    train = torch.tensor([i for i, split in enumerate(splits) if split == 'train'])
    val = torch.tensor([i for i, split in enumerate(splits) if split == 'validation'])
    if not len(train) or not len(val):
        raise ValueError('Empty grouped split')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PretrainedKeypointRefiner(args.pretrained_resnet, False, point_count=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    best = None; best_metric = float('inf'); best_step = 0; history = []
    def predict(indices):
        return model(x[indices].to(device).float() / 255)
    for step in range(1, args.updates + 1):
        indices = train[torch.randint(len(train), (args.batch_size,), generator=generator)]
        prediction = predict(indices)
        loss = F.smooth_l1_loss(prediction, y[indices].to(device), beta=.01)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step == 1 or step % 100 == 0:
            model.eval(); outputs = []
            with torch.inference_mode():
                for batch in val.split(128):
                    outputs.append(predict(batch).cpu())
            error = ((torch.cat(outputs) - y[val]) * (size - 1)).norm(dim=1).numpy()
            metric = float(np.percentile(error, 95))
            history.append({'step': step, 'train_loss': float(loss), 'validation_tip_p95_px': metric})
            print(json.dumps(history[-1]), flush=True)
            if metric < best_metric:
                best = copy.deepcopy(model.state_dict()); best_metric = metric; best_step = step
            model.train()
            if step - best_step >= 700:
                break
    model.load_state_dict(best); model.eval()
    errors = {'train': [], 'validation': []}
    with torch.inference_mode():
        for batch in torch.arange(len(x)).split(128):
            pred = predict(batch).cpu()
            for index, value in zip(batch.tolist(), pred):
                errors[splits[index]].append(float(((value - y[index]) * (size - 1)).norm()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.cpu().state_dict(), 'crop_size': size,
                'architecture': 'pretrained_resnet18_tip_only', 'point_count': 1,
                'condition_target_port': False}, args.output_dir / 'tip_refiner.pt')
    report = {'schema': 'sc_native_tip_only_refiner/v1', 'train_views': len(train),
              'validation_views': len(val), 'train_episodes': len({keys[i][0] for i in train.tolist()}),
              'validation_episodes': len({keys[i][0] for i in val.tolist()}),
              'rejected': rejected, 'best_step': best_step, 'completed_step': step,
              'errors_px': {split: {'median': percentile(values, 50), 'p95': percentile(values, 95)}
                            for split, values in errors.items()}, 'history': history,
              'runtime_crop_selection': 'frozen RGB locator tip pixel; no scored TF'}
    (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
