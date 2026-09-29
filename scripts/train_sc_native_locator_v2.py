#!/usr/bin/env python3
"""Train or run a task-conditioned spatial SC locator on global RGB views.

Simulator projections supervise training and score the saved predictions only.
Every runtime crop is centered on RGB-predicted tip and target opening pixels.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from train_sc_native_keypoints import PretrainedKeypointRefiner
from train_sc_native_locator import CAMERAS, HEIGHT, WIDTH, percentile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('labels',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--pretrained-resnet',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--updates',type=int,default=2200)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--crop-size',type=int,default=224)
    p.add_argument('--crop-center-point', choices=('mean', 'tip', 'opening'), default='mean',
                   help='Choose a learned RGB point for the crop center; mean preserves the original paired crop.')
    p.add_argument('--seed',type=int,default=20260924)
    p.add_argument('--far-boost',type=float,default=1.,
                   help='Sampling weight for views with visible opening and plug axial distance >200 mm')
    p.add_argument('--select-far',action='store_true',
                   help='Select checkpoint by held-out far-opening p95 instead of near-opening p95')
    p.add_argument('--select-offscreen',action='store_true',
                   help='Select checkpoint by held-out in-front but offscreen opening p95')
    p.add_argument('--offscreen-boost',type=float,default=1.,
                   help='Sampling weight for in-front but offscreen opening labels')
    p.add_argument('--offscreen-offset-head',action='store_true',
                   help='Add an unconstrained opening offset so the locator can predict outside the image')
    p.add_argument('--supervise-offscreen-opening',action='store_true',
                   help='Train projected port pixel even outside the image when it remains in front of the camera')
    p.add_argument('--allow-lowres-images', action='store_true',
                   help='Accept recorded 288x256 RGB for coarse global localization; projected errors stay in native pixel units')
    p.add_argument('--cameras', nargs='+', choices=CAMERAS, default=list(CAMERAS),
                   help='Train/evaluate selected camera views using shared locator weights')
    p.add_argument('--skip-crops', action='store_true',
                   help='Report global pixel predictions only; useful for low-resolution far-view diagnosis')
    args=p.parse_args()
    if not args.checkpoint and not args.pretrained_resnet:
        raise ValueError('ImageNet weights are required for training')
    torch.set_num_threads(8);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    rows=[json.loads(line) for line in args.labels.open()]
    train_scenes={r['scene_sha256'] for r in rows if r['split']=='train'}
    val_scenes={r['scene_sha256'] for r in rows if r['split']=='validation'}
    if train_scenes & val_scenes:raise ValueError('Scene leakage')
    x=[];y=[];mask=[];task=[];splits=[];near=[];keys=[]
    for index,row in enumerate(rows):
        is_near=abs(row['observed_sc_tip_pose_opening_frame'][2])<.03
        for cam in args.cameras:
            with Image.open(row['images'][cam]) as image:
                rgb=image.convert('RGB')
                if rgb.size==(WIDTH,HEIGHT):
                    coarse=rgb.resize((288,256),Image.Resampling.BILINEAR)
                elif args.allow_lowres_images and rgb.size==(288,256):
                    coarse=rgb
                else:
                    raise ValueError(f'Unexpected RGB size {rgb.size}: {row["images"][cam]}')
                x.append(torch.from_numpy(np.asarray(coarse).copy()).permute(2,0,1))
            projected=row['projected_training_pixels'][cam]
            y.append([(projected[name][i] or 0)/((WIDTH-1) if i==0 else (HEIGHT-1))
                      for name in ('tip','opening') for i in (0,1)])
            opening_observable=(projected['opening'][2] is not None and
                                projected['opening'][2]>0 and
                                all(v is not None for v in projected['opening'][:2]))
            opening_mask=float(opening_observable if args.supervise_offscreen_opening
                               else projected['opening'][3])
            mask.append([float(projected['tip'][3]),float(projected['tip'][3]),
                         opening_mask,opening_mask])
            task.append(row['task_vector'][2:4]);splits.append(row['split']);near.append(is_near);keys.append((index,cam))
    x=torch.stack(x);y=torch.tensor(y,dtype=torch.float32)
    mask=torch.tensor(mask,dtype=torch.float32);task=torch.tensor(task,dtype=torch.float32)
    fit=torch.tensor([i for i,s in enumerate(splits) if s=='train'],dtype=torch.long)
    val=torch.tensor([i for i,s in enumerate(splits) if s=='validation'],dtype=torch.long)
    if not len(val) and not args.checkpoint:
        raise ValueError('Training requires validation rows; frozen prediction may use a single split')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.far_boost < 1:
        raise ValueError('--far-boost must be >= 1')
    if args.offscreen_boost < 1:
        raise ValueError('--offscreen-boost must be >= 1')
    if args.select_far and args.select_offscreen:
        raise ValueError('Select either far-visible or offscreen validation, not both')
    far=torch.tensor([abs(rows[row_index]['observed_sc_tip_pose_opening_frame'][2])>.2
                      for row_index,_cam in keys],dtype=torch.bool)
    offscreen=torch.tensor([
        rows[row_index]['projected_training_pixels'][cam]['opening'][2]>0 and
        not rows[row_index]['projected_training_pixels'][cam]['opening'][3]
        for row_index,cam in keys],dtype=torch.bool)
    weights=(torch.where(far[fit] & ~offscreen[fit] & (mask[fit,2]>0),args.far_boost,1.)
             *torch.where(offscreen[fit],args.offscreen_boost,1.)).float()
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False) if args.checkpoint else None
    use_offset=(bool(saved.get('offscreen_offset_head',False)) if saved is not None
                else args.offscreen_offset_head)
    model=PretrainedKeypointRefiner(args.pretrained_resnet if not args.checkpoint else None,
                                   True,offscreen_opening_head=use_offset).to(device)
    best_step=0;history=[];step=0
    if args.checkpoint:
        if saved['crop_size']!=args.crop_size:raise ValueError('Crop size mismatch')
        model.load_state_dict(saved['model'])
    else:
        if not len(fit):raise ValueError('No training rows')
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        generator=torch.Generator().manual_seed(args.seed)
        best,best_metric=None,float('inf')
        for step in range(1,args.updates+1):
            indices=fit[torch.multinomial(weights,args.batch_size,replacement=True,generator=generator)]
            prediction=model(x[indices].to(device).float()/255,task[indices].to(device))
            loss=(F.smooth_l1_loss(prediction,y[indices].to(device),beta=.01,reduction='none')
                  *mask[indices].to(device)).sum()/mask[indices].sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if step==1 or step%100==0:
                model.eval();pieces=[]
                with torch.inference_mode():
                    for batch in val.split(128):
                        pieces.append(model(x[batch].to(device).float()/255,task[batch].to(device)).cpu())
                pred=torch.cat(pieces)
                chosen=torch.tensor([j for j,i in enumerate(val.tolist())
                                     if (bool(offscreen[i]) if args.select_offscreen else
                                         bool(far[i] and not offscreen[i]) if args.select_far else near[i])
                                     and bool(mask[i,2])])
                if not len(chosen):raise ValueError('No selected-phase validation opening')
                error=((pred[chosen,2:]-y[val[chosen],2:])*torch.tensor([WIDTH-1,HEIGHT-1])).norm(dim=1).numpy()
                metric=float(np.percentile(error,95))
                history.append({'step':step,'train_loss':float(loss),
                                'val_selection_opening_p95_px':metric,
                                'selection_phase':('offscreen' if args.select_offscreen else
                                                   'far' if args.select_far else 'near')})
                print(json.dumps(history[-1]),flush=True)
                if metric<best_metric:
                    best,best_metric,best_step=copy.deepcopy(model.state_dict()),metric,step
                model.train()
                if step-best_step>=700:break
        model.load_state_dict(best)
    model.eval();predicted=[]
    with torch.inference_mode():
        for chunk in torch.arange(len(x)).split(128):
            predicted.append(model(x[chunk].to(device).float()/255,task[chunk].to(device)).cpu())
    predicted=torch.cat(predicted)
    errors={split:{point:[] for point in ('tip','opening')} for split in ('train','validation')}
    near_errors={split:{point:[] for point in ('tip','opening')} for split in ('train','validation')}
    far_errors={split:{point:[] for point in ('tip','opening')} for split in ('train','validation')}
    offscreen_errors={split:[] for split in ('train','validation')}
    for i,(prediction,truth,visible,split) in enumerate(zip(predicted,y,mask,splits)):
        for point,start in (('tip',0),('opening',2)):
            if visible[start]:
                delta=(prediction[start:start+2]-truth[start:start+2])*torch.tensor([WIDTH-1,HEIGHT-1])
                errors[split][point].append(float(delta.norm()))
                if near[i]:near_errors[split][point].append(float(delta.norm()))
                if far[i]:far_errors[split][point].append(float(delta.norm()))
                if point=='opening' and not rows[keys[i][0]]['projected_training_pixels'][keys[i][1]]['opening'][3]:
                    offscreen_errors[split].append(float(delta.norm()))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    prediction_path=args.output_dir/'locator_predictions.jsonl'
    with prediction_path.open('w') as out:
        scale=torch.tensor([WIDTH-1,HEIGHT-1]*2)
        for i,(row_index,cam) in enumerate(keys):
            row=rows[row_index]
            out.write(json.dumps({
                'trial':row['trial'],'scene_sha256':row['scene_sha256'],
                'split':row['split'],'frame':row['frame'],'sim_time':row['sim_time'],
                'camera':cam,'predicted_pixels_native':(predicted[i]*scale).tolist(),
                'label_pixels_native':(y[i]*scale).tolist(),
                'visible_mask':mask[i].tolist(),
                'observed_tip_axial_m':row['observed_sc_tip_pose_opening_frame'][2],
            },separators=(',',':'))+'\n')
    path = None
    if not args.skip_crops:
        crop_root=args.output_dir/'predicted_crops';crop_root.mkdir(exist_ok=True)
        crop_rows=[{**row,'images':dict(row['images'])} for row in rows]
        for i,(row_index,cam) in enumerate(keys):
            row=rows[row_index]
            pixels=predicted[i]*torch.tensor([WIDTH-1,HEIGHT-1]*2)
            center=(pixels.reshape(2,2).mean(0) if args.crop_center_point=='mean'
                    else pixels.reshape(2,2)[0 if args.crop_center_point=='tip' else 1])
            left=round(float(center[0])-args.crop_size/2);top=round(float(center[1])-args.crop_size/2)
            with Image.open(row['images'][cam]) as image:
                crop=image.convert('RGB').crop((left,top,left+args.crop_size,top+args.crop_size))
                crop_path=crop_root/f"{row['trial']}_{row['frame']:06d}_{cam}.jpg"
                crop.save(crop_path,quality=93)
            crop_rows[row_index]['images'][cam]=str(crop_path)
            crop_rows[row_index].setdefault('predicted_crop_centers',{})[cam]=center.tolist()
            crop_rows[row_index].setdefault('predicted_locator_pixels',{})[cam]={
                'tip':pixels[:2].tolist(),'opening':pixels[2:4].tolist()}
            crop_rows[row_index]['predicted_crop_center_kind']=args.crop_center_point
        path=args.output_dir/'pose_labels_predicted_crops.jsonl'
        with path.open('w') as out:
            for row in crop_rows:
                row.pop('projected_training_pixels')
                out.write(json.dumps(row,separators=(',',':'))+'\n')
    if not args.checkpoint:
        torch.save({'model':model.cpu().state_dict(),'crop_size':args.crop_size,
                    'offscreen_offset_head':use_offset,
                    'architecture':'task_conditioned_resnet18_global_locator'},args.output_dir/'locator.pt')
    summarize=lambda dictionary:{split:{point:{'count':len(values),'median':percentile(values,50),
                                            'p95':percentile(values,95)} for point,values in section.items()}
                                  for split,section in dictionary.items()}
    report={'schema':'sc_native_locator_v2/v1','train_episodes':len({r['episode_id'] for r in rows if r['split']=='train'}),
            'validation_episodes':len({r['episode_id'] for r in rows if r['split']=='validation'}),
            'training_views':len(fit),'validation_views':len(val),'best_step':best_step,'completed_step':step,
            'history':history,'heldout_error_px':summarize(errors),'near_error_px':summarize(near_errors),
            'far_error_px':summarize(far_errors),'far_boost':args.far_boost,
            'offscreen_boost':args.offscreen_boost,
            'offscreen_offset_head':use_offset,
            'offscreen_opening_error_px':{split:{'count':len(values),
                'median':percentile(values,50),'p95':percentile(values,95)}
                for split,values in offscreen_errors.items()},
            'supervise_offscreen_opening':args.supervise_offscreen_opening,
            'checkpoint_selection_phase':('offscreen' if args.select_offscreen else
                                          'far' if args.select_far else 'near'),
            'crop_labels':str(path) if path else None,
            'pixel_predictions':str(prediction_path),
            'cameras':args.cameras,'allow_lowres_images':args.allow_lowres_images,
            'frozen_checkpoint':str(args.checkpoint) if args.checkpoint else None,
            'runtime_crop_selection':'task one-hot plus RGB only; no TF or projected labels',
            'crop_center_point':args.crop_center_point}
    (args.output_dir/'locator_metrics.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
