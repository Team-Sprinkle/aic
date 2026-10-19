#!/usr/bin/env python3
"""Controlled SC proposal association repair on cached RGB spatial features.

Fixed frozen-RGB proposals, not true geometry, are the inputs. Geometry labels
whether each fixed candidate is near the requested opening; this is explicitly
NOT physical visibility, uncertainty, or a safe-handoff label. Complete physical
reset splits are inherited unchanged. No reserved/frozen evaluation data read.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from audit_sc_port_targets import pose_matrix
from evaluate_sc_native_triangulation import triangulate,reprojection_residual_px
from build_sc_native_pose_labels import project

CAMS=('center','left','right')


def distance_history(rows, spacing=.04, length=10, turn_deg=10):
    """Past measured-motion anchors plus initial view; stationary time adds none."""
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['episode']].append(i)
    histories=np.zeros((len(rows),length),np.int64);masks=np.zeros_like(histories,bool)
    report=[]
    for episode,indices in groups.items():
        indices.sort(key=lambda i:rows[i]['time_s']);anchors=[];path=0.;last_anchor_path=0.;previous=None
        for i in indices:
            state=np.asarray(rows[i]['state'][:7],float)
            if previous is not None:path+=float(np.linalg.norm(state[:3]-previous[:3]))
            angle=0.
            if anchors:
                q=np.asarray(rows[anchors[-1]]['state'][3:7],float);q/=np.linalg.norm(q)
                now=state[3:7]/np.linalg.norm(state[3:7])
                angle=float(np.degrees(2*np.arccos(np.clip(abs(q@now),0,1))))
            if not anchors or path-last_anchor_path>=spacing or angle>=turn_deg:
                anchors.append(i);last_anchor_path=path
            chosen=anchors[-length:]
            # Keep the initial route/identity anchor even after the moving window fills.
            if len(anchors)>length:chosen=[anchors[0]]+anchors[-(length-1):]
            histories[i,:len(chosen)]=chosen;masks[i,:len(chosen)]=True
            previous=state
        report.append(dict(episode=episode,rows=len(indices),anchors=len(anchors),path_m=path))
    return histories,masks,report


class CandidateHistoryAssociation(nn.Module):
    def __init__(self,history=True,always_current_global=False):
        super().__init__();self.history=history;self.always_current_global=always_current_global
        self.visual=nn.Linear(256,96)
        self.context=nn.Linear(7+3,96)
        self.site=nn.Parameter(torch.randn(3,6,96)*.01)
        self.age=nn.Parameter(torch.randn(10,96)*.01)
        layer=nn.TransformerEncoderLayer(96,4,192,dropout=.1,batch_first=True,norm_first=True)
        self.memory=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        # candidate ROI + requested port + camera + native normalized proposal + public TCP
        self.query=nn.Linear(256+2+3+2+7,96)
        self.attend=nn.MultiheadAttention(96,4,dropout=.1,batch_first=True)
        self.out=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,96),nn.GELU(),nn.Linear(96,1))

    def forward(self,current,small_history,history_states,history_mask,state,camera,point,request):
        b=current.shape[0]
        if not self.history:
            small_history=F.adaptive_avg_pool2d(current.flatten(0,1),(2,3)).reshape(b,1,3,256,6).permute(0,1,2,4,3)
            history_states=state[:,None];history_mask=torch.ones(b,1,dtype=torch.bool,device=state.device)
        # B,T,3,6,256 -> spatial memory tokens retain view and cell identity.
        t=small_history.shape[1]
        motion=torch.cat((history_states,history_states[...,:3]-state[:,None,:3]),-1)
        memory=self.visual(small_history)+self.site[None,None]+self.age[None,:t,None,None]
        memory=memory+self.context(motion)[:,:,None,None,:]
        memory=memory.flatten(1,3);mask=(~history_mask)[:,:,None].expand(-1,-1,18).flatten(1)
        if self.history and self.always_current_global:
            now=F.adaptive_avg_pool2d(current.flatten(0,1),(2,3)).reshape(b,3,256,6).permute(0,1,3,2)
            now=self.visual(now)+self.site[None]+self.context(torch.cat((state,torch.zeros_like(state[:,:3])),-1))[:,None,None,:]
            memory=torch.cat((memory,now.flatten(1,2)),1)
            mask=torch.cat((mask,torch.zeros(b,18,dtype=torch.bool,device=state.device)),1)
        memory=self.memory(memory,src_key_padding_mask=mask)
        local=F.grid_sample(current[torch.arange(b,device=state.device),camera],
                            (point*2-1)[:,None,None],align_corners=True).flatten(1)
        condition=torch.cat((local,F.one_hot(request,2),F.one_hot(camera,3),point,state),-1).float()
        query=self.query(condition)[:,None]
        answer,_=self.attend(query,memory,memory,key_padding_mask=mask,need_weights=False)
        return self.out(torch.cat((query[:,0],answer[:,0]),-1)).squeeze(-1)


class StateOnlyAssociation(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(14,96),nn.GELU(),nn.Linear(96,96),nn.GELU(),nn.Linear(96,1))
    def forward(self,current,small_history,history_states,history_mask,state,camera,point,request):
        return self.net(torch.cat((state,F.one_hot(camera,3),point,F.one_hot(request,2)),-1).float()).squeeze(-1)


def causal_tracker_bank(rows):
    """Fixed past RGB -> world -> current image proposals, with NO true target access."""
    optics={c:np.asarray(m,float) for c,m in json.loads(Path('configs/hierarchical_recovery/sc_visual_standoff_20261001.json').read_text())['tcp_to_optical'].items()}
    bank=np.full((len(rows),2,3,2),np.nan,np.float32);groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['episode']].append(i)
    for indices in groups.values():
        anchors={}
        for i in sorted(indices,key=lambda i:rows[i]['time_s']):
            r=rows[i];cameras={c:pose_matrix(r['state'][:7])@optics[c] for c in CAMS}
            for candidate in range(2):
                if candidate in anchors:
                    m=np.eye(4);m[:3,3]=anchors[candidate]
                    bank[i,candidate]=np.asarray([project(m,cameras[c])[:2] for c in CAMS])
                if not all(r['coarse_inframe_probabilities'][ci][candidate]>=.5 for ci in range(3)):continue
                points={c:r['coarse_pixels_all_slots'][ci][candidate] for ci,c in enumerate(CAMS)}
                try:
                    point=triangulate(points,cameras)
                    if np.isfinite(point).all() and reprojection_residual_px(point,points,cameras)<=2:
                        anchors[candidate]=point
                except np.linalg.LinAlgError:pass
    return bank


def build_examples(rows,tracker=None,extra_candidates=None,positive_radius=12.,negative_radius=24.,crop_separation=0.):
    result=[];counts=Counter()
    for i,r in enumerate(rows):
        truth={t['identity']:t for t in r['targets']}
        for camera,c in enumerate(CAMS):
            bank=[np.asarray(r['coarse_pixels_all_slots'][camera][j],float) for j in range(2)]
            if tracker is not None:bank.extend(tracker[i,:,camera])
            if extra_candidates is not None:bank.extend(extra_candidates[i,:,camera])
            for candidate,point in enumerate(bank):
                if not np.isfinite(point).all() or np.any(point<0) or np.any(point>[1151,1023]):continue
                for request in range(2):
                    name=f'sc_port_{request}';target=truth.get(name)
                    gt=target['cameras'][c] if target is not None else None
                    distance=(float(np.linalg.norm(point-np.asarray(gt[:2],float)))
                              if gt is not None and all(v is not None for v in gt[:2]) else float('inf'))
                    # Ambiguous geometric boundaries are masked, never assigned arbitrary binary labels.
                    other=truth.get(f'sc_port_{1-request}')
                    wrong_distance=(float(np.linalg.norm(point-other['cameras'][c][:2])) if other is not None and other['cameras'][c][3] else float('inf'))
                    if gt is not None and gt[3] and distance<=positive_radius and wrong_distance>max(distance+6,crop_separation):label=1
                    elif gt is None or (not gt[3]) or distance>=negative_radius:label=0
                    else:counts['ambiguous_masked']+=1;continue
                    result.append(dict(row=i,camera=camera,candidate=candidate,request=request,
                        point=(point/[1151,1023]).tolist(),label=label,distance_px=distance if np.isfinite(distance) else None,
                        near_wrong_port=wrong_distance<=12,requested_projected_inframe=bool(gt is not None and gt[3]),
                        training_source=r.get('training_source','original'),
                        episode=r['episode'],scene_group=r['scene_group'],split=r['split']))
                    counts[f"{r['split']}/label{label}"]+=1
    return result,dict(counts)


def metrics(examples,prob,threshold=.9):
    truth=np.array([e['label'] for e in examples],bool);accepted=prob>=threshold
    wrong=np.array([e['near_wrong_port'] and not e['label'] for e in examples])
    return dict(samples=len(examples),episodes=len({e['episode'] for e in examples}),
        physical_resets=len({e['scene_group'] for e in examples}),threshold=threshold,
        positive_total=int(truth.sum()),positive_accepted=int((accepted&truth).sum()),
        negative_total=int((~truth).sum()),false_accepts=int((accepted&~truth).sum()),
        wrong_port_negative_total=int(wrong.sum()),wrong_port_false_accepts=int((accepted&wrong).sum()),
        recall=float((accepted&truth).sum()/max(1,truth.sum())),
        bce=float(-(truth*np.log(prob.clip(1e-7,1))+(~truth)*np.log((1-prob).clip(1e-7,1))).mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=5000);p.add_argument('--minutes',type=float,default=45)
    p.add_argument('--batch',type=int,default=64);p.add_argument('--spacing-mm',type=float,default=40)
    p.add_argument('--variant',choices=('current','distance_history','state_only'),required=True)
    p.add_argument('--always-current-global',action='store_true',help='Keep current full-view context even between distance anchors')
    p.add_argument('--include-causal-tracker-candidates',action='store_true',help='Repair the gap between current locator and remembered RGB proposal distributions')
    p.add_argument('--extra-candidate-bank',type=Path,help='Frozen observation-only requested-locator pixel bank: rows,2,3,2')
    p.add_argument('--positive-radius-px',type=float,default=12.)
    p.add_argument('--negative-radius-px',type=float,default=24.)
    p.add_argument('--crop-separation-px',type=float,default=0.)
    p.add_argument('--balance-training-sources',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(6);torch.manual_seed(20261015);rng=np.random.default_rng(20261015)
    rows=[]
    with (a.cache/'frame_context.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r['family']=='sc':rows.append(r)
            elif rows:break
    completion=a.cache/'cache_summary.json'
    expected=(json.loads(completion.read_text())['valid_feature_rows'] if completion.exists()
              else sum(1 for l in (a.cache/'../refiner_corpus_final_v3/rows.jsonl').open() if json.loads(l)['family']=='sc'))
    if len(rows)!=expected:raise ValueError(f'Incomplete SC feature cache: {len(rows)} != {expected}')
    assert len({r['feature_row'] for r in rows})==len(rows)
    train_groups={r['scene_group'] for r in rows if r['split']=='train'}
    assert train_groups.isdisjoint({r['scene_group'] for r in rows if r['split']=='validation'})
    history,mask,history_report=distance_history(rows,a.spacing_mm/1000)
    audits={}
    for spacing in (20,40,60):
        _,_,rr=distance_history(rows,spacing/1000)
        audits[str(spacing)]={'anchors_p50':float(np.median([x['anchors'] for x in rr])),
                              'anchors_p90':float(np.percentile([x['anchors'] for x in rr],90))}
    tracker=causal_tracker_bank(rows) if a.include_causal_tracker_candidates else None
    if tracker is not None:np.save(a.output/'fixed_RGB_tracker_candidates.npy',tracker)
    extra=np.load(a.extra_candidate_bank) if a.extra_candidate_bank else None
    if extra is not None and extra.shape!=(len(rows),2,3,2):raise ValueError('Candidate bank row/shape mismatch')
    if not 0<a.positive_radius_px<a.negative_radius_px:raise ValueError('Invalid candidate crop radii')
    examples,counts=build_examples(rows,tracker,extra,a.positive_radius_px,a.negative_radius_px,a.crop_separation_px)
    (a.output/'history_audit.json').write_text(json.dumps(dict(spacing_mm=a.spacing_mm,rule='measured path length OR 10 degree rotation; initial anchor retained; at most 10',spacing_sweep=audits,episodes=history_report),indent=2)+'\n')
    with (a.output/'manifest.jsonl').open('w') as f:
        for e in examples:f.write(json.dumps(e)+'\n')
    with (a.output/'history_indices.jsonl').open('w') as f:
        for i,r in enumerate(rows):f.write(json.dumps(dict(episode=r['episode'],time_s=r['time_s'],source_feature_row=r['feature_row'],history_rows=history[i][mask[i]].tolist()))+'\n')
    maps=np.load(a.cache/'coarse_spatial_8x9.npy',mmap_mode='r')
    # Slice ONLY completed SC rows. Unwritten SFP ranges are never read.
    current=torch.from_numpy(np.stack([maps[r['feature_row']] for r in rows])).cuda().float()
    pooled=F.adaptive_avg_pool2d(current.flatten(0,1),(2,3)).reshape(len(rows),3,256,6).permute(0,1,3,2)
    states=torch.tensor([r['state'][:7] for r in rows],device='cuda',dtype=torch.float32)
    # Canonical quaternion sign prevents equivalent q/-q from becoming a history signal.
    sign=torch.where(states[:,6:7]<0,-1.,1.);states[:,3:7]*=sign
    train_rows=torch.tensor([r['split']=='train' for r in rows],device='cuda')
    mean=states[train_rows].mean(0);std=states[train_rows].std(0).clamp_min(.02);states=(states-mean)/std
    history=torch.from_numpy(history).cuda();mask=torch.from_numpy(mask).cuda()
    row_index=torch.tensor([e['row'] for e in examples],device='cuda')
    camera=torch.tensor([e['camera'] for e in examples],device='cuda')
    point=torch.tensor([e['point'] for e in examples],device='cuda',dtype=torch.float32)
    request=torch.tensor([e['request'] for e in examples],device='cuda')
    labels=torch.tensor([e['label'] for e in examples],device='cuda',dtype=torch.float32)
    train=np.array([i for i,e in enumerate(examples) if e['split']=='train']);val=np.array([i for i,e in enumerate(examples) if e['split']=='validation'])
    # Balance physical reset and positive/negative class, not wall-clock stalls.
    cells=Counter((examples[i]['scene_group'],examples[i]['label']) for i in train)
    weights=np.array([1/cells[(examples[i]['scene_group'],examples[i]['label'])] for i in train]);weights/=weights.sum()
    if a.balance_training_sources:
        source_cells=defaultdict(set)
        for i in train:
            e=examples[i];source_cells[e['training_source']].add((e['scene_group'],e['label']))
        weights=np.array([1/cells[(examples[i]['scene_group'],examples[i]['label'])]/len(source_cells[examples[i]['training_source']]) for i in train]);weights/=weights.sum()
    # Fixed balanced validation screen for early stopping; full held-out report afterward.
    groups=defaultdict(list)
    for i in val:groups[(examples[i]['episode'],examples[i]['camera'],examples[i]['label'])].append(i)
    screen=np.array([v[j] for v in groups.values() for j in np.linspace(0,len(v)-1,min(16,len(v)),dtype=int)])
    model=(StateOnlyAssociation() if a.variant=='state_only' else CandidateHistoryAssociation(a.variant=='distance_history',a.always_current_global)).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    def forward(indices):
        ix=torch.as_tensor(indices,device='cuda');ri=row_index[ix];hi=history[ri]
        return model(current[ri],pooled[hi],states[hi],mask[ri],states[ri],camera[ix],point[ix],request[ix])
    @torch.inference_mode()
    def evaluate(indices):
        model.eval();probs=[]
        for j in range(0,len(indices),128):probs.extend(forward(indices[j:j+128]).sigmoid().cpu().tolist())
        probs=np.array(probs);selected=[examples[i] for i in indices];report=metrics(selected,probs)
        report['by_source']={source:metrics([e for e in selected if e['training_source']==source],probs[[e['training_source']==source for e in selected]]) for source in sorted({e['training_source'] for e in selected})}
        return report,probs
    best=float('inf');best_step=0;stale=0;reason='update_cap';started=time.monotonic()
    for step in range(1,a.updates+1):
        if time.monotonic()-started>a.minutes*60:reason='time_cap_not_proven_plateau';break
        indices=rng.choice(train,a.batch,replace=True,p=weights);model.train();opt.zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(forward(indices),labels[indices]);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%250==0:
            report,_=evaluate(screen);record=dict(step=step,train_loss=float(loss.detach()),screen=report,elapsed_s=time.monotonic()-started)
            with (a.output/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            checkpoint=dict(model=model.state_dict(),variant=a.variant,always_current_global=a.always_current_global,state_mean=mean.cpu(),state_std=std.cpu(),spacing_mm=a.spacing_mm,history_length=10,step=step,
                positive_radius_px=a.positive_radius_px,negative_radius_px=a.negative_radius_px,crop_separation_px=a.crop_separation_px,
                geometry_input=False,label_kind='fixed RGB candidate association, NOT visibility or handoff safety')
            torch.save(checkpoint,a.output/'last.pt')
            score=max(v['bce'] for v in report['by_source'].values()) if a.balance_training_sources else report['bce']
            if score<best-1e-4:best=score;best_step=step;stale=0;torch.save(checkpoint,a.output/'best.pt')
            else:stale+=1
            if step>=2000 and stale>=8:reason='validation_plateau';break
    model.load_state_dict(torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)['model'])
    full,prob=evaluate(val)
    with (a.output/'validation_predictions.jsonl').open('w') as f:
        for i,pred in zip(val,prob):f.write(json.dumps(dict(examples[i],association_probability=float(pred)))+'\n')
    summary=dict(schema='candidate_history_association/v1',variant=a.variant,always_current_global=a.always_current_global,causal_tracker_training_candidates=a.include_causal_tracker_candidates,counts=counts,best_step=best_step,updates_completed=step,stop_reason=reason,
        extra_candidate_bank=str(a.extra_candidate_bank) if a.extra_candidate_bank else None,
        positive_radius_px=a.positive_radius_px,negative_radius_px=a.negative_radius_px,crop_separation_px=a.crop_separation_px,
        balanced_training_sources=a.balance_training_sources,
        full_validation=full,parameters=sum(p.numel() for p in model.parameters()),
        label_is_not_visibility=True,safe_handoff_not_certified=True,final_scenes_opened=False,
        train_episodes=len({r['episode'] for r in rows if r['split']=='train'}),validation_episodes=len({r['episode'] for r in rows if r['split']=='validation'}))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
