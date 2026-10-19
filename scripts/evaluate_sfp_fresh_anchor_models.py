#!/usr/bin/env python3
"""Score four frozen SFP anchor checkpoints on independent non-final RGB."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_sfp_balanced_anchor import BalancedViews, evaluate
from train_sfp_structured_slot_locator import RequestedSlot
from train_sfp_observed_rail_crop import Crops, HistoryConditionalLocator, evaluate as evaluate_crop
from train_target_conditioned_port_locator import ConditionalLocator
from train_two_family_port_locator import CAMERAS, Locator


class SmallViews(BalancedViews):
    def __getitem__(self,i):
        import cv2
        row=self.rows[self.indices[i][0]]
        ci=self.indices[i][1];cam=CAMERAS[ci]
        image=cv2.imread(row['images'][cam]);
        if image is None:raise FileNotFoundError(row['images'][cam])
        image=cv2.cvtColor(cv2.resize(image,(288,256),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        from train_two_family_port_locator import LOOKUP
        target=next(t for t in row['targets'] if t['identity']==row['selected_identity'])
        uv=target['cameras'][cam];xy=[uv[0]/1151,uv[1]/1023] if uv[3] else [0.,0.]
        meta=np.zeros(5,np.float32);meta[1]=1;meta[2+ci]=1
        return (torch.from_numpy(image.copy()).permute(2,0,1),torch.from_numpy(meta),
                LOOKUP[target['identity']],torch.tensor(xy,dtype=torch.float32),
                float(uv[3]),self.indices[i][0],ci)


def load(model,path):
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
    return model.cuda().eval()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--labels',type=Path,required=True)
    ap.add_argument('--proposals',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--expected-scenes',type=int,default=10)
    ap.add_argument('--crop-checkpoint',type=Path,
                    help='Evaluate a specified corrected crop fit')
    ap.add_argument('--crop-only',action='store_true')
    ap.add_argument('--rail-band-mm',type=float,default=0.,
                    help='Observation-derived requested rail y half-width in mm')
    ap.add_argument('--rail-band-adjacent-only',action='store_true',
                    help='Apply rail band only to adjacent-card rail fallbacks')
    args=ap.parse_args()
    if args.output.exists():ap.error('Preserve existing result')
    torch.set_num_threads(4)
    rows=[json.loads(s) for s in (args.labels/'rows.jsonl').open()]
    if len({r['scene_group'] for r in rows})!=args.expected_scenes:
        raise ValueError(f'Expected {args.expected_scenes} fresh reset groups')
    idx=[(i,c) for i in range(len(rows)) for c in range(3)]
    cal=json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())
    optics={c:np.asarray(cal['tcp_to_optical'][c],float) for c in CAMERAS}
    base=Path('/home/nas/chmin/aic_archives')
    checkpoints={
        'allport':Path('outputs/experiments/2026-10-02_two_family_perception/joint_locator_native_sfp_v5_heatmap/best.pt'),
        'structured':base/'2026-10-04_sfp_structured_slot_v1/best.pt',
        'selected':base/'2026-10-03_sfp_balanced_arrival_v1/balanced_bc/selected_anchor_native576_v1/best.pt',
        'crop':base/'2026-10-04_sfp_observed_rail_crop_fit_v2/best.pt'}
    if args.crop_checkpoint:checkpoints['crop']=args.crop_checkpoint
    args.output.mkdir(parents=True)
    report={}
    for name in (() if args.crop_only else ('allport','structured','selected')):
        model=(RequestedSlot(load(Locator(None),checkpoints[name]))
               if name in ('structured','allport')
               else load(ConditionalLocator(None),checkpoints[name]))
        loader=DataLoader(SmallViews(rows,idx,False) if name=='allport' else BalancedViews(rows,idx,False),
                          batch_size=6,num_workers=0,pin_memory=True)
        report[name]=evaluate(model,rows,loader,'cuda',optics,args.output/f'{name}.jsonl')
        print(json.dumps(dict(model=name,metrics=report[name])),flush=True)
        del model
        torch.cuda.empty_cache()
    proposals=[json.loads(s) for s in (args.proposals/'rows.jsonl').open()]
    by_key={(r['episode'],r['time_s']):r for r in rows}
    crop_rows=[]
    for item in proposals:
        if not item['crop']:continue
        row=by_key[(item['episode'],item['time_s'])]
        crop_rows.append(dict(episode=item['episode'],time_s=item['time_s'],
                              images=row['images'],state=row['state'],targets=row['targets'],
                              requested=item['requested'],proposal=item['crop']))
    crop_state=torch.load(checkpoints['crop'],map_location='cpu',weights_only=False)
    source_offset_meta=bool(crop_state.get('source_offset_meta',False))
    history_context=bool(crop_state.get('history_context',False))
    if history_context:
        context_path=Path(crop_state['history_context_path'])
        manifest=json.loads((context_path/'manifest.json').read_text())
        context=np.load(context_path/'context.npy',mmap_mode='r')
        by_key={(m['episode'],m['time_s']):i for i,m in enumerate(manifest)}
        for row in crop_rows:
            index=by_key[(row['episode'],row['time_s'])]
            if manifest[index]['images']!=row['images']:
                raise ValueError('History context image mismatch')
            row['history_context']=np.asarray(context[index])
        model=load(HistoryConditionalLocator(None),checkpoints['crop'])
    else:
        model=load(ConditionalLocator(None),checkpoints['crop'])
    loader=DataLoader(Crops(crop_rows,[(i,c) for i in range(len(crop_rows)) for c in range(3)],False,
                            source_offset_meta=source_offset_meta,history_context=history_context),
                      batch_size=12,num_workers=0,pin_memory=True)
    report['crop']=evaluate_crop(model,crop_rows,loader,optics,args.output/'crop.jsonl',
                                 rail_band_mm=args.rail_band_mm,
                                 rail_band_adjacent_only=args.rail_band_adjacent_only)
    print(json.dumps(dict(model='crop',metrics=report['crop'])),flush=True)
    summary=dict(schema='sfp_fresh_anchor_frozen_model_scores/v1',
                 labels=str(args.labels),proposals=str(args.proposals),
                 checkpoint_paths={k:str(v) for k,v in checkpoints.items()},
                 rail_band_mm=args.rail_band_mm,
                 rail_band_adjacent_only=args.rail_band_adjacent_only,
                 metrics=report,final_scenes_opened=False,
                 runtime_true_geometry_input=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
