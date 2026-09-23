#!/usr/bin/env python3
"""Benchmark selected pretrained landmark perception plus the frozen trunk."""
from __future__ import annotations
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
def module(name,file):
 spec=importlib.util.spec_from_file_location(name,HERE/file);m=importlib.util.module_from_spec(spec);assert spec.loader is not None;spec.loader.exec_module(m);return m
opening=module("opening_pretrained_latency","train_opening_landmark_pose_probe.py")
pretrained=module("pretrained_opening_latency","calibrate_pretrained_opening_landmarks.py")
def q(v): return {"p50":float(np.quantile(v,.5)),"p95":float(np.quantile(v,.95)),"p99":float(np.quantile(v,.99))}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--evaluation-replay",type=Path,required=True);p.add_argument("--scene-manifest",type=Path,required=True);p.add_argument("--metrics",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--crop-size",type=int,default=160);p.add_argument("--device",default="cuda");a=p.parse_args();device=torch.device(a.device)
 ids={x["episode_id"] for x in json.loads(a.scene_manifest.read_text())["development"]};rows,audit=opening.world.load([a.evaluation_replay],ids);rows,rejected=opening.attach(rows,a.evaluation_replay)
 saved=torch.load(a.checkpoint,map_location="cpu",weights_only=False);selection=saved["summary"]["selection"]["selected"]
 locator=opening.crop.Locator().to(device).eval();locator.load_state_dict(saved["locator"]);landmark=pretrained.MobileNetLandmarks().to(device).eval();landmark.load_state_dict(saved["landmark"])
 images=[torch.stack([torch.from_numpy(np.asarray(Image.open(v["path"]).convert("RGB")).copy()).permute(2,0,1) for v in row["highres"]]) for row in rows]
 def run(full_cpu):
  full=full_cpu.to(device).float()/255;coarse=locator(F.interpolate(full,size=(256,288),mode="bilinear",align_corners=False));height,width=full.shape[-2:];centers=coarse.reshape(3,2,2).mean(1);size=a.crop_size;gx=torch.linspace(-(size-1)/(width-1),(size-1)/(width-1),size,device=device);gy=torch.linspace(-(size-1)/(height-1),(size-1)/(height-1),size,device=device);yy,xx=torch.meshgrid(gy,gx,indexing="ij");grids=torch.stack([torch.stack(((c[0]*2-1)+xx,(c[1]*2-1)+yy),-1) for c in centers]);return opening.decode(landmark(F.grid_sample(full,grids,mode="bilinear",padding_mode="zeros",align_corners=True)))
 with torch.inference_mode():
  for _ in range(10):run(images[0])
  elapsed=[]
  for image in images:
   torch.cuda.synchronize();start=time.perf_counter();run(image);torch.cuda.synchronize();elapsed.append((time.perf_counter()-start)*1000)
 representation=np.asarray(elapsed);trunk=np.asarray([float(r["model_inference_s"])*1000 for r in rows]);complete=representation+trunk
 report={"sample_count":len(rows),"selected_representation":selection,"landmark_network":"ImageNet MobileNetV3-small FPN","scope":"RGB locator, native crop, pretrained landmark model plus measured frozen trunk; sensor acquisition excluded","representation_ms":q(representation),"world_trunk_ms":q(trunk),"complete_inference_ms":q(complete),"p95_below_300ms":bool(np.quantile(complete,.95)<300),"audit":{"causal":audit,"rejected":rejected},"sources":{"checkpoint":opening.world.file_id(a.checkpoint),"evaluation_replay":opening.world.file_id(a.evaluation_replay)}}
 a.output.write_text(json.dumps(report,indent=2)+"\n");metrics=json.loads(a.metrics.read_text());metrics["live_latency"]=report;a.metrics.write_text(json.dumps(metrics,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
