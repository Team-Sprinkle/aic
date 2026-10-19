#!/usr/bin/env python3
"""Bounded RGB-only SFP rail-reference probe on complete reset groups.

The actor never receives the projected rail geometry, occupancy labels, or
scene YAML. This diagnostic predicts two fixed rail endpoints in each wrist
camera and a five-bit occupied-slot map from all three native RGB views.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from train_two_family_port_locator import Locator


CAMS = ('center', 'left', 'right')
SIZE = (576, 512)


class RailViews(Dataset):
    def __init__(self, rows, indices):
        self.rows, self.indices = rows, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        r = self.rows[self.indices[item]]
        images = []
        points = np.zeros((3, 2, 2), np.float32)
        visible = np.zeros((3, 2), np.float32)
        for ci, cam in enumerate(CAMS):
            frame = cv2.imread(r['images'][cam], cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(r['images'][cam])
            frame = cv2.cvtColor(cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA),
                                 cv2.COLOR_BGR2RGB)
            images.append(np.transpose(frame.copy(), (2, 0, 1)))
            for j, rail in enumerate((0, 4)):
                p = r['rail_projections_training_label_only'][cam][rail]
                if p[3]:
                    points[ci, j] = [p[0] / 1151, p[1] / 1023]
                    visible[ci, j] = 1
        return (torch.from_numpy(np.stack(images)), torch.from_numpy(points),
                torch.from_numpy(visible),
                torch.tensor(r['occupied_slots_training_label_only'], dtype=torch.float32),
                self.indices[item])


class BoardReference(nn.Module):
    def __init__(self, init_path):
        super().__init__()
        base = Locator(None)
        base.load_state_dict(torch.load(init_path, map_location='cpu', weights_only=False)['model'])
        self.backbone = base.backbone
        for name, param in self.backbone.named_parameters():
            param.requires_grad_(name.startswith('layer3.') or name.startswith('layer4.'))
        self.keypoints = nn.Sequential(nn.Conv2d(768, 192, 3, padding=1), nn.GELU(),
                                       nn.Conv2d(192, 96, 3, padding=1), nn.GELU(),
                                       nn.Conv2d(96, 2, 1))
        self.inframe = nn.Linear(512, 2)
        self.occupancy = nn.Sequential(nn.Linear(1536, 256), nn.GELU(),
                                       nn.Linear(256, 5))
        self.register_buffer('mean', torch.tensor((.485, .456, .406))[None, :, None, None])
        self.register_buffer('std', torch.tensor((.229, .224, .225))[None, :, None, None])

    def train(self, mode=True):
        super().train(mode)
        self.backbone.bn1.eval()
        self.backbone.layer1.eval()
        self.backbone.layer2.eval()
        return self

    def forward(self, images):
        batch = images.shape[0]
        x = images.reshape(batch * 3, 3, *images.shape[-2:])
        x = (x - self.mean) / self.std
        x = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        mid = self.backbone.layer3(x)
        deep = self.backbone.layer4(mid)
        heat = self.keypoints(torch.cat((mid, F.interpolate(deep, size=mid.shape[-2:],
                                                        mode='bilinear', align_corners=False)), dim=1))
        _, _, h, w = heat.shape
        probs = F.softmax(heat.flatten(2).float() * 4, dim=-1)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, h, device=x.device),
                                torch.linspace(0, 1, w, device=x.device), indexing='ij')
        xy = torch.stack(((probs * xx.flatten()).sum(-1),
                          (probs * yy.flatten()).sum(-1)), dim=-1).reshape(batch, 3, 2, 2)
        pooled = deep.mean((-1, -2))
        return (xy, self.inframe(pooled).reshape(batch, 3, 2),
                self.occupancy(pooled.reshape(batch, 1536)), heat.reshape(batch, 3, 2, h, w))


class BoardReferenceLocal(BoardReference):
    """Read every slot at RGB-predicted rail positions instead of global pooling."""
    def __init__(self, init_path):
        super().__init__(init_path)
        del self.occupancy
        self.slot_view = nn.Sequential(nn.Linear(256 * 9, 128), nn.GELU())
        self.slot_global = nn.Sequential(nn.Linear(1536, 256), nn.GELU())
        self.slot_head = nn.Sequential(nn.Linear(128 * 3 + 256 + 5 + 3, 256),
                                       nn.GELU(), nn.Linear(256, 1))

    def forward(self, images, return_slot_features=False):
        batch = images.shape[0]
        x = images.reshape(batch * 3, 3, *images.shape[-2:])
        x = (x - self.mean) / self.std
        x = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        mid = self.backbone.layer3(x)
        deep = self.backbone.layer4(mid)
        heat = self.keypoints(torch.cat((mid, F.interpolate(deep, size=mid.shape[-2:],
                                                        mode='bilinear', align_corners=False)), dim=1))
        _, _, h, w = heat.shape
        probs = F.softmax(heat.flatten(2).float() * 4, dim=-1)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, h, device=x.device),
                                torch.linspace(0, 1, w, device=x.device), indexing='ij')
        xy = torch.stack(((probs * xx.flatten()).sum(-1),
                          (probs * yy.flatten()).sum(-1)), dim=-1).reshape(batch, 3, 2, 2)
        flags = self.inframe(deep.mean((-1, -2))).reshape(batch, 3, 2)
        # Detach the landmark coordinates: occupancy may learn from local RGB,
        # but must not move the endpoint estimates merely to ease classification.
        fraction = torch.linspace(0, 1, 5, device=x.device)[None, None, :, None]
        slots_xy = xy[:, :, :1, :].detach() * (1 - fraction) + xy[:, :, 1:, :].detach() * fraction
        offsets = torch.tensor(((-.045, -.045), (0, -.045), (.045, -.045),
                                (-.045, 0), (0, 0), (.045, 0),
                                (-.045, .045), (0, .045), (.045, .045)),
                               device=x.device, dtype=slots_xy.dtype)
        grid = (slots_xy[:, :, :, None, :] + offsets).reshape(batch * 3, 45, 1, 2) * 2 - 1
        sampled = F.grid_sample(mid, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
        sampled = sampled.reshape(batch, 3, 256, 5, 9).permute(0, 3, 1, 2, 4)
        local = self.slot_view(sampled.flatten(-2)).reshape(batch, 5, 384)
        global_features = self.slot_global(deep.mean((-1, -2)).reshape(batch, 1536))
        global_features = global_features[:, None, :].expand(-1, 5, -1)
        code = F.one_hot(torch.arange(5, device=x.device), num_classes=5).float()
        code = code[None].expand(batch, -1, -1)
        camera_support = ((slots_xy >= 0) & (slots_xy <= 1)).all(-1).float().permute(0, 2, 1)
        occupied = self.slot_head(torch.cat((local, global_features, code,
                                             camera_support), dim=-1)).squeeze(-1)
        outputs = (xy, flags, occupied, heat.reshape(batch, 3, 2, h, w))
        if return_slot_features:
            return outputs, local, global_features[:, 0]
        return outputs


def loss_terms(output, point, inframe, occupied):
    xy, flags, slots, heat = output
    point_mask = inframe.bool()
    coord = F.smooth_l1_loss(xy, point, beta=.02, reduction='none').sum(-1)
    coord = (coord * inframe).sum() / inframe.sum().clamp_min(1)
    h, w = heat.shape[-2:]
    cell = ((point[..., 1] * (h - 1)).round().long().clamp(0, h - 1) * w +
            (point[..., 0] * (w - 1)).round().long().clamp(0, w - 1))
    spatial = F.cross_entropy((heat.reshape(-1, h * w) * 4).float(),
                              cell.reshape(-1), reduction='none').reshape_as(inframe)
    spatial = (spatial * point_mask).sum() / inframe.sum().clamp_min(1)
    flag = F.binary_cross_entropy_with_logits(flags.float(), inframe)
    occupancy = F.binary_cross_entropy_with_logits(slots.float(), occupied)
    return coord + .01 * spatial + .1 * flag + .5 * occupancy, {
        'coord': float(coord.detach()), 'spatial': float(spatial.detach()),
        'inframe': float(flag.detach()), 'occupancy': float(occupancy.detach())}


@torch.inference_mode()
def evaluate(model, loader, rows, device, save=None):
    model.eval()
    records = []
    losses = []
    for images, point, inframe, occupied, indices in loader:
        images = images.to(device, non_blocking=True).float() / 255
        point = point.to(device, non_blocking=True)
        inframe = inframe.to(device, non_blocking=True)
        occupied = occupied.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(images)
        loss, _ = loss_terms(output, point, inframe, occupied)
        losses.append((float(loss), len(indices)))
        xy, flags, slots, _ = output
        for i, index in enumerate(indices.tolist()):
            r = rows[index]
            records.append({'row': index, 'episode': r['episode'], 'scene_group': r['scene_group'],
                            'card_count': r['card_count'], 'time_s': r['time_s'],
                            'autonomous_actor_diagnostic': r['autonomous_actor_diagnostic'],
                            'occupied_true_label_only': r['occupied_slots_training_label_only'],
                            'occupied_prob': slots[i].float().sigmoid().cpu().tolist(),
                            'endpoint_xy_normalized': xy[i].float().cpu().tolist(),
                            'endpoint_inframe_prob': flags[i].float().sigmoid().cpu().tolist()})
    groups = defaultdict(list)
    for r in records:
        groups[r['scene_group']].append(r)
    exact = 0
    sparse_exact = 0
    sparse = 0
    confident_false = 0
    episodes = []
    for scene, rr in sorted(groups.items()):
        rr.sort(key=lambda q: q['time_s'])
        early = [q for q in rr if q['time_s'] - rr[0]['time_s'] <= 3]
        if not early:
            early = [rr[0]]
        probabilities = np.median([q['occupied_prob'] for q in early], axis=0)
        truth = np.asarray(rr[0]['occupied_true_label_only'], dtype=bool)
        pred = probabilities >= .5
        good = bool(np.array_equal(pred, truth))
        exact += good
        if sum(truth) <= 2:
            sparse += 1
            sparse_exact += good
        false = int(np.sum((probabilities >= .8) & ~truth))
        confident_false += false
        episodes.append({'scene_group': scene, 'episode': rr[0]['episode'],
                         'card_count': int(sum(truth)), 'early_rows': len(early),
                         'exact_occupancy': good, 'confident_empty_slot_errors': false,
                         'predicted_occupied_prob': probabilities.tolist(),
                         'true_occupied_label_only': truth.astype(int).tolist()})
    report = {'rows': len(records), 'episodes': len(groups), 'exact_occupancy_episodes': exact,
              'sparse_episodes': sparse, 'sparse_exact_episodes': sparse_exact,
              'confident_empty_slot_errors': confident_false,
              'mean_loss': sum(v*n for v,n in losses)/sum(n for _,n in losses),
              'episode_details': episodes}
    if save:
        with save.open('w') as file:
            for r in records:
                file.write(json.dumps(r, separators=(',', ':')) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--init', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=2000)
    p.add_argument('--minutes', type=float, default=120)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--head', choices=('pooled', 'slot_local'), default='pooled')
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve existing output')
    if a.updates > 2000:
        p.error('Maximum 2000 updates')
    a.output.mkdir(parents=True)
    rows = [json.loads(s) for s in (a.labels / 'rows.jsonl').open()]
    train = [i for i, r in enumerate(rows) if r['split'] == 'train']
    dev = [i for i, r in enumerate(rows) if r['split'] == 'validation'
           and not r['autonomous_actor_diagnostic']]
    actor = [i for i, r in enumerate(rows) if r['autonomous_actor_diagnostic']]
    if {rows[i]['scene_group'] for i in train} & {rows[i]['scene_group'] for i in dev+actor}:
        raise ValueError('Reset leakage')
    torch.manual_seed(20261005)
    np.random.seed(20261005)
    torch.set_num_threads(8)
    rng = random.Random(20261005)
    by_count = defaultdict(lambda: defaultdict(list))
    for index in train:
        r = rows[index]
        by_count[r['card_count']][r['scene_group']].append(index)
    if set(by_count) != set(range(1, 6)):
        raise ValueError('Missing one-to-five-card train categories')
    draw = []
    for _ in range(a.updates * a.batch_size):
        count = rng.choice(tuple(sorted(by_count)))
        scenes = by_count[count]
        group = rng.choice(tuple(scenes))
        draw.append(rng.choice(scenes[group]))
    train_loader = DataLoader(RailViews(rows, draw), batch_size=a.batch_size,
                              num_workers=a.workers, pin_memory=True,
                              persistent_workers=a.workers > 0,
                              multiprocessing_context='spawn' if a.workers > 0 else None)
    dev_loader = DataLoader(RailViews(rows, dev), batch_size=a.batch_size,
                            num_workers=a.workers, pin_memory=True,
                            persistent_workers=a.workers > 0,
                            multiprocessing_context='spawn' if a.workers > 0 else None)
    actor_loader = DataLoader(RailViews(rows, actor), batch_size=a.batch_size,
                              num_workers=a.workers, pin_memory=True,
                              persistent_workers=a.workers > 0,
                              multiprocessing_context='spawn' if a.workers > 0 else None)
    model = (BoardReferenceLocal(a.init) if a.head == 'slot_local'
             else BoardReference(a.init)).cuda()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=5e-5, weight_decay=1e-4)
    start = time.monotonic()
    history = []
    best = float('inf')
    best_update = None
    stopped = 'update_cap'
    completed = 0
    for update, batch in enumerate(train_loader, 1):
        if update > a.updates:
            break
        if time.monotonic() - start > a.minutes * 60:
            stopped = 'time_cap'
            break
        images, points, inframe, occupied, _ = batch
        images = images.cuda(non_blocking=True).float() / 255
        points = points.cuda(non_blocking=True)
        inframe = inframe.cuda(non_blocking=True)
        occupied = occupied.cuda(non_blocking=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(images)
            loss, terms = loss_terms(output, points, inframe, occupied)
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite loss at {update}')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        completed = update
        if update % 200 == 0 or update == 1:
            dev_report = evaluate(model, dev_loader, rows, 'cuda')
            point = {'update': update, 'train_loss': float(loss.detach()),
                     'terms': terms, 'development': {k:v for k,v in dev_report.items()
                                                  if k != 'episode_details'},
                     'elapsed_s': time.monotonic()-start}
            history.append(point)
            (a.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
            if dev_report['mean_loss'] < best:
                best = dev_report['mean_loss']
                best_update = update
                torch.save({'model': model.state_dict(), 'update': update,
                            'development_loss': best, 'labels': str(a.labels),
                            'init': str(a.init), 'head': a.head}, a.output / 'best.pt')
            print(json.dumps(point), flush=True)
    if best_update is None:
        raise RuntimeError('No development checkpoint')
    state = torch.load(a.output / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    model.cuda()
    dev_result = evaluate(model, dev_loader, rows, 'cuda', a.output / 'development_predictions.jsonl')
    actor_result = evaluate(model, actor_loader, rows, 'cuda', a.output / 'actor_diagnostic_predictions.jsonl')
    result = {'schema': 'sfp_board_rail_reference_probe/v1', 'head': a.head,
              'best_update': best_update,
              'completed_updates': completed,
              'stop_reason': stopped, 'elapsed_s': time.monotonic()-start,
              'train_groups': len({rows[i]['scene_group'] for i in train}),
              'development_groups': len({rows[i]['scene_group'] for i in dev}),
              'actor_diagnostic_groups': len({rows[i]['scene_group'] for i in actor}),
              'development': dev_result, 'actor_diagnostic': actor_result,
              'geometry_runtime_input': False, 'final_scenes_opened': False}
    (a.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:result[k] for k in ('best_update','completed_updates',
          'stop_reason','development_groups','actor_diagnostic_groups')}), flush=True)


if __name__ == '__main__':
    main()
