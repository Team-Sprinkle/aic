#!/usr/bin/env python3
"""Episode-grouped SC visual pose probe on scored Gazebo observations.

Privileged port geometry generates training targets only. Runtime inputs are
the three recorded wrist images and ordinary 32D robot state. This diagnostic
is not a deployable actor or a claim of 0.5 mm insertion accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CAMERAS = ("left", "center", "right")


class SCFrames(Dataset):
    def __init__(self, records, repo_host: str, repo_container: str, image_size: int, images: bool,
                 state_mean: torch.Tensor, state_std: torch.Tensor, label_frame: str):
        self.records = records
        self.repo_host = repo_host.rstrip("/")
        self.repo_container = repo_container.rstrip("/")
        self.images = images
        self.state_mean = state_mean
        self.state_std = state_std
        self.label_frame = label_frame
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        row = self.records[i]
        images = torch.empty(0)
        if self.images:
            arrays = []
            for camera in CAMERAS:
                path = row["images"][camera]
                if path.startswith(self.repo_host + "/"):
                    path = self.repo_container + path[len(self.repo_host):]
                with Image.open(path) as im:
                    arrays.append(self.transform(im.convert("RGB")))
            images = torch.stack(arrays)
        target = row["observed_sc_tip_pose_opening_frame" if self.label_frame == "opening" else
                     "observed_sc_tip_pose_port_frame"]
        state = (torch.tensor(row["state"], dtype=torch.float32)-self.state_mean)/self.state_std
        task = torch.tensor(row["task_vector"], dtype=torch.float32)
        if task.numel() != 10:
            raise ValueError("Expected the canonical 10D task/target encoding")
        state = torch.cat((state, task))
        xyz = torch.tensor(target[:3], dtype=torch.float32)
        quat = torch.tensor(target[3:7], dtype=torch.float32)
        near = abs(float(target[2]) + (0.0 if self.label_frame == "opening" else 0.01564)) < 0.03
        return images, state, xyz, quat, near, int(row["episode_index"]), int(row["frame"])


class Probe(nn.Module):
    def __init__(self, *, images: bool, pretrained_path: Path | None):
        super().__init__()
        self.images = images
        if images:
            self.backbone = models.resnet18(weights=None)
            if pretrained_path:
                self.backbone.load_state_dict(torch.load(pretrained_path, map_location="cpu", weights_only=True))
            self.backbone.fc = nn.Identity()
            # Adapt the final block and projection; keep lower visual features.
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad = name.startswith("layer4.")
            visual_dim = 512 * len(CAMERAS)
        else:
            visual_dim = 0
        self.state_net = nn.Sequential(nn.Linear(42, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(visual_dim + 128, 512), nn.ReLU(),
                                  nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 7))

    def forward(self, images, state):
        context = self.state_net(state)
        if self.images:
            batch, views = images.shape[:2]
            visual = self.backbone(images.reshape(batch * views, *images.shape[2:]))
            visual = visual.reshape(batch, views * 512)
            context = torch.cat((context, visual), dim=-1)
        prediction = self.head(context)
        return prediction[:, :3], nn.functional.normalize(prediction[:, 3:], dim=-1)


def metrics(model, loader, device):
    model.eval()
    rows = []
    with torch.inference_mode():
        for images, state, xyz, quat, near, episode, frame in loader:
            if images.numel(): images = images.to(device, non_blocking=True)
            pred_xyz, pred_quat = model(images, state.to(device, non_blocking=True))
            translation = (pred_xyz.cpu() - xyz).numpy() * 1000
            dots = (pred_quat.cpu() * quat).sum(-1).abs().clamp(max=1).numpy()
            for j in range(len(episode)):
                lateral = math.hypot(float(translation[j][0]), float(translation[j][1]))
                axial = abs(float(translation[j][2]))
                rows.append({"episode": int(episode[j]), "frame": int(frame[j]),
                             "near": bool(near[j]), "translation_mm": float(math.sqrt(sum(float(v*v) for v in translation[j]))),
                             "lateral_mm": lateral, "axial_mm": axial,
                             "orientation_deg": math.degrees(2*math.acos(float(dots[j]))),
                             "lateral_sign_correct": bool((pred_xyz[j, :2].cpu() * xyz[j, :2]).sum() >= 0)})
    def aggregate(items):
        if not items: return {"count": 0}
        result = {"count": len(items), "episodes": len({r["episode"] for r in items})}
        for key in ("translation_mm", "lateral_mm", "axial_mm", "orientation_deg"):
            values = sorted(r[key] for r in items)
            result[key] = {"median": values[len(values)//2], "p95": values[min(len(values)-1, int(0.95*len(values)))]}
        result["lateral_sign_accuracy"] = sum(r["lateral_sign_correct"] for r in items)/len(items)
        return result
    return {"all": aggregate(rows), "near_port": aggregate([r for r in rows if r["near"]]),
            "per_episode_near_port": {str(ep): aggregate([r for r in rows if r["near"] and r["episode"] == ep])
                                      for ep in sorted({r["episode"] for r in rows})}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--repo-host", default="/data1/chmin/yj/ws_aic/src/aic")
    p.add_argument("--repo-container", default="/workspace/isaaclab/aic")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--pretrained-resnet", type=Path)
    p.add_argument("--init-checkpoint", type=Path,
                   help="Exact compatible pose-probe checkpoint; no partial loading")
    p.add_argument("--state-only", action="store_true")
    p.add_argument("--label-frame", choices=("opening", "port_base"), default="opening")
    p.add_argument("--orientation-only", action="store_true",
                   help="Train and select the visual head for quaternion accuracy; translation is diagnostic only")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260924)
    a = p.parse_args()
    torch.set_num_threads(a.cpu_threads)
    random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = True
    records = [json.loads(line) for line in a.labels.open()]
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "validation"]
    train_scenes = {r["scene_sha256"] for r in train}; val_scenes = {r["scene_sha256"] for r in val}
    if train_scenes & val_scenes: raise ValueError("Scene leakage")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    use_images = not a.state_only
    checkpoint = torch.load(a.init_checkpoint, map_location="cpu", weights_only=False) if a.init_checkpoint else None
    if checkpoint:
        if bool(checkpoint["state_only"]) != a.state_only or checkpoint["label_frame"] != a.label_frame:
            raise ValueError("Initialization checkpoint modality/frame mismatch")
        if checkpoint["image_size"] != a.image_size:
            raise ValueError("Initialization checkpoint image-size mismatch")
        state_mean = checkpoint["state_mean"]
        state_std = checkpoint["state_std"]
    else:
        all_train_state = torch.tensor([r["state"] for r in train], dtype=torch.float32)
        state_mean = all_train_state.mean(0)
        state_std = all_train_state.std(0).clamp_min(1e-3)
    ds_train = SCFrames(train, a.repo_host, a.repo_container, a.image_size, use_images, state_mean, state_std, a.label_frame)
    ds_val = SCFrames(val, a.repo_host, a.repo_container, a.image_size, use_images, state_mean, state_std, a.label_frame)
    train_loader = DataLoader(ds_train, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                              pin_memory=True, persistent_workers=a.workers > 0)
    val_loader = DataLoader(ds_val, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                            pin_memory=True, persistent_workers=a.workers > 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Probe(images=use_images, pretrained_path=a.pretrained_resnet).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=1e-4)
    history = []
    best = float("inf")
    if checkpoint:
        initial = metrics(model, val_loader, device)
        selected_key = "orientation_deg" if a.orientation_only else "lateral_mm"
        best = initial["near_port"][selected_key]["p95"]
        history.append({"epoch": 0, "train_loss": None, "validation": initial, "elapsed_s": 0.0})
        torch.save({"model": model.state_dict(), "epoch": 0, "state_only": a.state_only,
                    "image_size": a.image_size, "label_frame": a.label_frame,
                    "validation": initial, "state_mean": state_mean, "state_std": state_std},
                   a.output_dir / "best.pt")
        print(json.dumps({"epoch": 0, "selected_metric": selected_key, "near_p95": best}), flush=True)
    for epoch in range(1, a.epochs + 1):
        started = time.time();model.train();loss_sum=0.0;steps=0
        for images,state,xyz,quat,near,_,_ in train_loader:
            if images.numel(): images=images.to(device,non_blocking=True)
            state=state.to(device,non_blocking=True);xyz=xyz.to(device,non_blocking=True)
            quat=quat.to(device,non_blocking=True);near=near.to(device,non_blocking=True)
            pred_xyz,pred_quat=model(images,state)
            diff=(pred_xyz-xyz)*10.0
            lateral=(diff[:,:2]**2).sum(-1)
            axial=diff[:,2]**2
            position_loss=((1+3*near.float())*lateral+axial).mean()
            angle_loss=(1-(pred_quat*quat).sum(-1).square()).mean()
            loss=angle_loss if a.orientation_only else position_loss+0.2*angle_loss
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            loss_sum+=float(loss.detach());steps+=1
        result=metrics(model,val_loader,device)
        score=result["near_port"]["orientation_deg" if a.orientation_only else "lateral_mm"]["p95"]
        history.append({"epoch":epoch,"train_loss":loss_sum/max(steps,1),"validation":result,
                        "elapsed_s":time.time()-started})
        print(json.dumps({"epoch":epoch,"loss":history[-1]["train_loss"],
                          "near_lateral_median_mm":result["near_port"]["lateral_mm"]["median"],
                          "near_lateral_p95_mm":result["near_port"]["lateral_mm"]["p95"],
                          "near_orientation_p95_deg":result["near_port"]["orientation_deg"]["p95"],
                          "selected_score":score,"elapsed_s":history[-1]["elapsed_s"]}),flush=True)
        if score<best:
            best=score
            torch.save({"model":model.state_dict(),"epoch":epoch,"state_only":a.state_only,
                        "image_size":a.image_size,"label_frame":a.label_frame,"validation":result,
                        "state_mean":state_mean,"state_std":state_std},a.output_dir/"best.pt")
        (a.output_dir/"metrics.json").write_text(json.dumps({"schema":"aic_sc_pose_probe/v1",
            "config":vars(a)|{"labels":str(a.labels),"output_dir":str(a.output_dir),
                              "pretrained_resnet":str(a.pretrained_resnet) if a.pretrained_resnet else None},
            "train_episodes":len({r['episode_index'] for r in train}),
            "validation_episodes":len({r['episode_index'] for r in val}),
            "train_scene_groups":len(train_scenes),"validation_scene_groups":len(val_scenes),
            "best_near_lateral_p95_mm":best,"history":history},indent=2,default=str)+"\n")
    if a.epochs == 0 and checkpoint:
        (a.output_dir/"metrics.json").write_text(json.dumps({"schema":"aic_sc_pose_probe/v1",
            "config":vars(a)|{"labels":str(a.labels),"output_dir":str(a.output_dir),
                              "pretrained_resnet":str(a.pretrained_resnet) if a.pretrained_resnet else None},
            "train_episodes":len({r['episode_index'] for r in train}),
            "validation_episodes":len({r['episode_index'] for r in val}),
            "train_scene_groups":len(train_scenes),"validation_scene_groups":len(val_scenes),
            "best_near_lateral_p95_mm":best,"history":history},indent=2,default=str)+"\n")


if __name__ == "__main__":
    main()
