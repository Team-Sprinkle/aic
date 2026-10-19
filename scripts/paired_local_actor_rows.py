"""Same-episode causal observation pairs for soft overlapping-plan consistency."""
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F

from train_predicted_port_local_actor import LocalRows
from shared_port_tcp_geometry import matrix_from_pose9, matrix_from_xyz_xyzw


class PairedLocalRows(LocalRows):
    def __init__(self, rows, *args, **kwargs):
        super().__init__(rows, *args, **kwargs)
        self.pairs = {}
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[r['family'], r['episode']].append(i)
        for indices in groups.values():
            indices.sort(key=lambda i: rows[i]['time_s'])
            for current, following in zip(indices, indices[1:]):
                a, b = rows[current], rows[following]
                shift = round((b['time_s']-a['time_s'])/.05)
                h = len(a['teacher_tcp_targets_predicted_port_pose9'])
                if not (1 <= shift < h and abs(b['time_s']-a['time_s']-shift*.05) < 1e-5):
                    continue
                pa = matrix_from_xyz_xyzw(a['state'][:7]) @ np.linalg.inv(matrix_from_pose9(a['predicted_port_tcp_pose9']))
                pb = matrix_from_xyz_xyzw(b['state'][:7]) @ np.linalg.inv(matrix_from_pose9(b['predicted_port_tcp_pose9']))
                transform = np.linalg.inv(pa) @ pb
                matches = True
                for av, bv in zip(a['recorded_tcp_targets_predicted_port_pose9'][shift:],
                                  b['recorded_tcp_targets_predicted_port_pose9'][:h-shift]):
                    am = matrix_from_pose9(av)
                    bm = transform @ matrix_from_pose9(bv)
                    if not np.allclose(am, bm, atol=2e-6):
                        matches = False
                        break
                if matches:
                    self.pairs[current] = (following, transform.astype(np.float32), shift)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        partner, transform, shift = self.pairs.get(index, (index, np.eye(4, dtype=np.float32), 1))
        return (*item, super().__getitem__(partner), torch.from_numpy(transform),
                torch.tensor(shift), torch.tensor(index in self.pairs))


def overlapping_plan_loss(first, second, transform, shift, valid, rotation_from_pose9):
    """Compare decoded TCP plans at identical absolute future timestamps.

    `transform` maps the second estimated port frame into the first. Invalid
    pairs contribute no gradient. This is a soft objective, not a forced hold.
    """
    horizon = first.shape[1]
    indexes = torch.arange(horizon, device=first.device)[None].expand(len(first), -1)
    partner_indexes = (indexes-shift[:, None]).clamp(0, horizon-1)
    matching = valid[:, None] & (indexes >= shift[:, None])
    if not matching.any():
        return first.sum()*0
    other = second.gather(1, partner_indexes[..., None].expand(-1, -1, 9)).float()
    points = (transform[:, None, :3, :3] @ other[..., :3, None]).squeeze(-1) + transform[:, None, :3, 3]
    residual = (first.float()[..., :3]-points)/.002
    translation = F.smooth_l1_loss(residual[matching], torch.zeros_like(residual[matching]), beta=.5)
    ra = rotation_from_pose9(first.float())
    rb = transform[:, None, :3, :3] @ rotation_from_pose9(other)
    cosine = ((ra.transpose(-1, -2) @ rb).diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2
    rotation = (1-cosine.clamp(-1, 1))[matching].mean()
    return translation + rotation
