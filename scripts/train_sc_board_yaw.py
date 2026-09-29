#!/usr/bin/env python3
"""Estimate the one varying SC board yaw from global RGB and robot state.

Qualification-like SC boards keep roll/pitch and the port mounting rotation
fixed. Scored TF supplies yaw labels and a fixed board-to-port calibration
during training/scoring only. The predictor consumes RGB, state, and task.
"""

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from audit_sc_port_targets import matrix, pose_matrix
from build_sc_native_pose_labels import chain
from train_sc_pose_probe import SCFrames


class YawFrames(Dataset):
    def __init__(self,rows,yaw_by_trial,mean,std):
        self.rows=rows;self.yaw_by_trial=yaw_by_trial
        runtime=[{**r,'images':r['lowres_images']} for r in rows]
        self.frames=SCFrames(runtime,'/data1/chmin/yj/ws_aic/src/aic','/workspace/isaaclab/aic',
                             224,True,mean,std,'opening')

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        image,state,_,_,near,*_=self.frames[index]
        return image,state,torch.tensor(self.yaw_by_trial[self.rows[index]['trial']],dtype=torch.float32),near,index


class BoardYaw(nn.Module):
    def __init__(self,pretrained):
        super().__init__()
        self.backbone=models.resnet18(weights=None)
        if pretrained is not None:
            self.backbone.load_state_dict(torch.load(pretrained,map_location='cpu',weights_only=True))
        self.backbone.fc=nn.Identity()
        for name,param in self.backbone.named_parameters():
            param.requires_grad_(name.startswith('layer4.'))
        self.state_net=nn.Sequential(nn.Linear(42,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(3*512+128,256),nn.ReLU(),nn.Linear(256,128),
                                nn.ReLU(),nn.Linear(128,1))

    def forward(self,image,state):
        batch=image.shape[0]
        visual=self.backbone(image.reshape(batch*3,3,224,224)).reshape(batch,-1)
        return .25*torch.tanh(self.head(torch.cat((visual,self.state_net(state)),dim=-1)))[:,0]


def edge_labels(edges):
    offset=matrix(json.loads(Path('configs/hierarchical_recovery/sc_port_opening_gazebo_tf_251.json').read_text())['base_to_opening'])
    yaw={};fixed=[]
    for item in edges:
        trial=item['trial'];e=item['edges'];port=item['target_module_name'].rsplit('_',1)[-1]
        parent=f'task_board/sc_port_{port}'
        world_base=chain(e,['world','tabletop','base_link'])
        world_board=chain(e,['aic_world','task_board'])
        world_port=chain(e,['aic_world','task_board',parent,parent+'/sc_port_base_link'])@offset
        base_board=np.linalg.inv(world_base)@world_board
        roll,pitch,angle=Rotation.from_matrix(base_board[:3,:3]).as_euler('xyz')
        if max(abs(roll),abs(pitch))>1e-4:
            raise ValueError('Board roll/pitch not fixed; yaw-only calibration invalid')
        yaw[trial]=angle
        fixed.append(Rotation.from_matrix((np.linalg.inv(world_board)@world_port)[:3,:3]))
    first=fixed[0]
    if max((first.inv()*r).magnitude() for r in fixed)>1e-4:
        raise ValueError('SC port mounting rotation varies across this dataset')
    return yaw,first


def evaluate(model,loader,rows,yaw_truth,board_port,device):
    model.eval();predictions={}
    with torch.inference_mode():
        for image,state,_,_,indices in loader:
            output=model(image.to(device),state.to(device)).cpu().numpy()
            for i,value in zip(indices.tolist(),output):predictions[i]=float(value)
    records=[]
    for i,pred in predictions.items():
        row=rows[i]
        near=abs(row['observed_sc_tip_pose_opening_frame'][2])<.03
        yaw_error=abs(math.atan2(math.sin(pred-yaw_truth[row['trial']]),math.cos(pred-yaw_truth[row['trial']])))
        base_tcp=pose_matrix(row['state'][:7])
        tip=base_tcp@matrix(json.loads(Path('configs/hierarchical_recovery/sc_tcp_tip_gazebo_tf_251.json').read_text())['tcp_to_sc_tip'])
        base_port=Rotation.from_euler('z',pred)*board_port
        relative=base_port.inv()*Rotation.from_matrix(tip[:3,:3])
        true=Rotation.from_quat(row['observed_sc_tip_pose_opening_frame'][3:7])
        orientation_error=(true.inv()*relative).magnitude()
        records.append({'trial':row['trial'],'frame':row['frame'],'near':near,
                        'yaw_error_deg':float(yaw_error*180/np.pi),
                        'orientation_error_deg':float(orientation_error*180/np.pi),
                        'predicted_board_yaw_rad':pred})
    near=[r for r in records if r['near']]
    stat=lambda key,subset:{'count':len(subset),'median':float(np.median([r[key] for r in subset])) if subset else None,
                            'p95':float(np.percentile([r[key] for r in subset],95)) if subset else None}
    return {'episodes':len({r['trial'] for r in near}),'near_count':len(near),
            'yaw_error_deg':stat('yaw_error_deg',near),'orientation_error_deg':stat('orientation_error_deg',near),
            'per_episode_orientation_p95_deg':{trial:stat('orientation_error_deg',[r for r in near if r['trial']==trial])['p95']
                                               for trial in sorted({r['trial'] for r in near})}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--edges',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--pretrained-resnet',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=20260924)
    args=p.parse_args()
    if not args.checkpoint and not args.pretrained_resnet:raise ValueError('ImageNet weights required')
    torch.set_num_threads(8);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    rows=[json.loads(x) for x in args.labels.open()]
    yaw_truth,calibration=edge_labels(json.loads(args.edges.read_text())['rows'])
    if not all(r['trial'] in yaw_truth for r in rows):raise ValueError('Missing scored yaw label')
    train=[i for i,r in enumerate(rows) if r['split']=='train']
    val=[i for i,r in enumerate(rows) if r['split']=='validation']
    if {rows[i]['scene_sha256'] for i in train}&{rows[i]['scene_sha256'] for i in val}:
        raise ValueError('Scene leakage')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False) if args.checkpoint else None
    if saved:
        mean,std=saved['state_mean'],saved['state_std']
        calibration=Rotation.from_quat(saved['board_to_port_xyzw'])
    else:
        state=torch.tensor([rows[i]['state'] for i in train],dtype=torch.float32)
        mean=state.mean(0);std=state.std(0).clamp_min(1e-3)
    dataset=YawFrames(rows,yaw_truth,mean,std)
    fit=DataLoader(torch.utils.data.Subset(dataset,train),batch_size=args.batch_size,
        shuffle=bool(train),num_workers=args.workers,pin_memory=True)
    validation=DataLoader(torch.utils.data.Subset(dataset,val),batch_size=args.batch_size,
        num_workers=args.workers,pin_memory=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=BoardYaw(args.pretrained_resnet if not saved else None).to(device)
    if saved:model.load_state_dict(saved['model'],strict=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if saved:
        result=evaluate(model,validation,rows,yaw_truth,calibration,device)
        (args.output_dir/'metrics.json').write_text(json.dumps({'schema':'sc_board_yaw/v1',
            'eval_only':True,'checkpoint':str(args.checkpoint),'validation':result},indent=2)+'\n')
        print(json.dumps(result));return
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=1e-4)
    best=float('inf');history=[]
    for epoch in range(1,args.epochs+1):
        model.train();losses=[]
        for image,state,target,near,_ in fit:
            prediction=model(image.to(device),state.to(device))
            weight=1+3*near.to(device).float()
            loss=(weight*(prediction-target.to(device))**2).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();losses.append(float(loss))
        result=evaluate(model,validation,rows,yaw_truth,calibration,device)
        score=result['orientation_error_deg']['p95']
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),
                        'validation_orientation_p95_deg':score})
        print(json.dumps(history[-1]),flush=True)
        if score<best:
            best=score
            torch.save({'model':copy.deepcopy(model.state_dict()),'state_mean':mean,'state_std':std,
                        'board_to_port_xyzw':calibration.as_quat().tolist(),'epoch':epoch,
                        'validation':result},args.output_dir/'best.pt')
    (args.output_dir/'metrics.json').write_text(json.dumps({'schema':'sc_board_yaw/v1',
        'train_scenes':len({rows[i]['scene_sha256'] for i in train}),
        'validation_scenes':len({rows[i]['scene_sha256'] for i in val}),
        'fixed_roll_pitch_and_port_rotation':True,'history':history,'best_p95_deg':best},indent=2)+'\n')


if __name__=='__main__':main()
