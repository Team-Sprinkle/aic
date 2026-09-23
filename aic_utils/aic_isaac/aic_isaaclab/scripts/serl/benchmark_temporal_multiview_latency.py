#!/usr/bin/env python3
"""Benchmark frozen native-crop perception plus the causal residual head."""
from __future__ import annotations
import argparse, importlib.util, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
def module(name,file):
 spec=importlib.util.spec_from_file_location(name,HERE/file);value=importlib.util.module_from_spec(spec);assert spec.loader is not None;spec.loader.exec_module(value);return value
base=module('temporal_latency_base','train_temporal_multiview_pose_ablation.py');opening=base.opening;pretrained=base.pretrained
def q(v):return {'p50':float(np.quantile(v,.5)),'p95':float(np.quantile(v,.95)),'p99':float(np.quantile(v,.99))}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spatial-checkpoint',type=Path,required=True);p.add_argument('--ablation-checkpoint',type=Path,required=True);p.add_argument('--evaluation-replay',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--metrics',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--crop-size',type=int,default=160);p.add_argument('--device',default='cuda');a=p.parse_args();device=torch.device(a.device)
 ids={x['episode_id'] for x in json.loads(a.manifest.read_text())['development']};rows,audit=opening.world.load([a.evaluation_replay],ids);rows,rejected=opening.attach(rows,a.evaluation_replay)
 spatial=torch.load(a.spatial_checkpoint,map_location='cpu',weights_only=False);saved=torch.load(a.ablation_checkpoint,map_location='cpu',weights_only=False);selected=saved['selected'];affine=saved['translation_affine'];cfg=saved['temporal'];history_size=int((cfg['input_dim']-6)//33)
 locator=opening.crop.Locator().to(device).eval();locator.load_state_dict(spatial['locator']);landmark=pretrained.MobileNetLandmarks().to(device).eval();landmark.load_state_dict(spatial['landmark']);members=[]
 for state in cfg['state_dicts']:
  model=base.ResidualPose(cfg['input_dim']);model.load_state_dict(state);members.append(model.to(device).eval())
 images=[torch.stack([torch.from_numpy(np.asarray(Image.open(v['path']).convert('RGB')).copy()).permute(2,0,1) for v in row['highres']]) for row in rows]
 histories=defaultdict(list)
 def run(image_cpu,row,record_history=True):
  full=image_cpu.to(device).float()/255;coarse=locator(F.interpolate(full,size=(256,288),mode='bilinear',align_corners=False));height,width=full.shape[-2:];centers=coarse.reshape(3,2,2).mean(1);size=a.crop_size;gx=torch.linspace(-(size-1)/(width-1),(size-1)/(width-1),size,device=device);gy=torch.linspace(-(size-1)/(height-1),(size-1)/(height-1),size,device=device);yy,xx=torch.meshgrid(gy,gx,indexing='ij');grids=torch.stack([torch.stack(((c[0]*2-1)+xx,(c[1]*2-1)+yy),-1) for c in centers]);local=opening.decode(landmark(F.grid_sample(full,grids,mode='bilinear',padding_mode='zeros',align_corners=True)))
  center_px=centers*torch.tensor([width-1,height-1],device=device);left_top=torch.round(center_px-size/2);pixels=local*(size-1)+left_top[:,None,:];landmarks=pixels/torch.tensor([width-1,height-1],device=device);coarse_cpu=coarse.detach().cpu().reshape(1,3,4);land_cpu=landmarks.detach().cpu().reshape(1,3,6,2);points=opening.pair_points(coarse_cpu,land_cpu,selected['plug'],selected['port']);raw=np.c_[opening.relative_world(points,[row]),np.ones(1)]@affine;feature=torch.cat((coarse_cpu[:,:,:2].reshape(1,-1),land_cpu[:,:,2:6].reshape(1,-1),torch.tensor(raw,dtype=torch.float32)),1)[0]
  past=histories[row['sequence_id']];indices=(past+[feature])[-history_size:];valid=[0.]*(history_size-len(indices))+[1.]*len(indices);indices=[indices[0]]*(history_size-len(indices))+indices;window=torch.cat((*indices,torch.tensor(valid)));xn=((window-cfg['input_mean'])/cfg['input_std']).to(device);prediction=torch.stack([model(xn) for model in members]).mean(0)*cfg['target_std'].to(device)+cfg['target_mean'].to(device)+torch.tensor(raw[0],device=device)
  if record_history:past.append(feature)
  return prediction
 with torch.inference_mode():
  for _ in range(10):run(images[0],rows[0],False)
  histories.clear();elapsed=[]
  for image,row in zip(images,rows):
   torch.cuda.synchronize();start=time.perf_counter();run(image,row);torch.cuda.synchronize();elapsed.append((time.perf_counter()-start)*1000)
 perception=np.asarray(elapsed);trunk=np.asarray([float(r['model_inference_s'])*1000 for r in rows]);complete=perception+trunk
 report={'schema_version':1,'sample_count':len(rows),'scope':'three full RGB views, locator, native 160x160 crops, landmark model, calibrated triangulation, six-frame causal ensemble; sensor acquisition excluded','perception_and_temporal_head_ms':q(perception),'existing_frozen_policy_trunk_ms':q(trunk),'complete_if_paired_with_existing_trunk_ms':q(complete),'complete_p95_below_300ms':bool(np.quantile(complete,.95)<300),'history_frames':history_size,'audit':{'causal':audit,'rejected':rejected}}
 a.output.write_text(json.dumps(report,indent=2)+'\n');metrics=json.loads(a.metrics.read_text());metrics['live_latency']=report;a.metrics.write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
