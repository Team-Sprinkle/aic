#!/usr/bin/env python3
"""One-shot development evaluation of calibrated temporal/multiview heads."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
def module(name,file):
 spec=importlib.util.spec_from_file_location(name,HERE/file);value=importlib.util.module_from_spec(spec);assert spec.loader is not None;spec.loader.exec_module(value);return value
ablation=module('ablation_eval','train_temporal_multiview_pose_ablation.py')
opening=ablation.opening;pretrained=ablation.pretrained

def predict(saved,key,x,raw,device):
 config=saved[key];xn=((x-config['input_mean'])/config['input_std']).to(device);values=[]
 for state in config['state_dicts']:
  model=ablation.ResidualPose(x.shape[1]);model.load_state_dict(state);model=model.to(device).eval()
  with torch.no_grad():values.append(raw+model(xn).cpu()*config['target_std']+config['target_mean'])
 stack=torch.stack(values);return stack.mean(0).numpy(),stack.var(0,unbiased=False).numpy()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--spatial-checkpoint',type=Path,required=True);p.add_argument('--ablation-checkpoint',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--crop-size',type=int,default=160);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);device=torch.device(a.device)
 manifest=json.loads(a.manifest.read_text());ids=[x['episode_id'] for x in manifest['development']]
 rows,audit=opening.world.load([a.replay],set(ids));rows,rejected=opening.attach(rows,a.replay)
 spatial=torch.load(a.spatial_checkpoint,map_location='cpu',weights_only=False);saved=torch.load(a.ablation_checkpoint,map_location='cpu',weights_only=False);selected=saved['selected']
 locator=opening.crop.Locator();locator.load_state_dict(spatial['locator']);landmark=pretrained.MobileNetLandmarks();landmark.load_state_dict(spatial['landmark'])
 coarse,locator_metrics=opening.crop.locator_predict(locator,opening.coarse_rows(rows),device);landmarks,landmark_metrics=opening.landmark_predict(landmark,rows,coarse,a.crop_size,device)
 features,raw=ablation.predicted_features(rows,coarse,landmarks,selected,saved['translation_affine']);one=torch.cat((features,torch.ones(len(rows),1)),1);history=int((saved['temporal']['input_dim']-6)//features.shape[1]);windows=ablation.causal_windows(rows,features,history)
 multi,multi_var=predict(saved,'multiview',one,raw,device);temporal,temporal_var=predict(saved,'temporal',windows,raw,device);zero=np.zeros((len(rows),3))
 report={'schema_version':1,'status':'development_complete','evaluation_policy':'one-shot new reset configurations after calibration-only selection','scope':'translation-only residual ablation','split':{'episode_ids':ids,'rows':len(rows),'sequence_count':len({r['sequence_id'] for r in rows}),'reserved_final_opened':False},'audit':audit,'rejected':rejected,'frozen_perception':{'locator':locator_metrics,'landmarks':landmark_metrics},'models':{'frozen_spatial_baseline':ablation.model_report(rows,raw.numpy(),zero),'learned_multiview_current_frame':ablation.model_report(rows,multi,multi_var),'causal_temporal_multiview':ablation.model_report(rows,temporal,temporal_var)},'limitations':['New reset positions, but the same fixed scene/cable initialization family.','No simulator geometry, segmentation, or labels enter inference.','Orientation/contact use the unchanged frozen auxiliary probe and are outside this translation residual comparison.'],'sources':{'replay':opening.world.file_id(a.replay),'manifest':opening.world.file_id(a.manifest),'spatial_checkpoint':opening.world.file_id(a.spatial_checkpoint),'ablation_checkpoint':opening.world.file_id(a.ablation_checkpoint)}}
 for value in report['models'].values():value['gate']['passed']=all(value['gate'].values())
 report['decision']='perception translation gate passed; benchmark full latency and preserve as candidate' if report['models']['causal_temporal_multiview']['gate']['passed'] else 'gate failed; stop before policy and RL'
 (a.output_dir/'development_metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'output':str(a.output_dir),'decision':report['decision'],'near_port':{k:v['near_port'] for k,v in report['models'].items()}},indent=2))
if __name__=='__main__':main()
