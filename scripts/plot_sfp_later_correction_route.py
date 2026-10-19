#!/usr/bin/env python3
"""Plot measured SFP TCP XY routes and offline port geometry for diagnosis."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from build_two_family_port_projection_labels import sfp_ports


def actor(run):
    rows=[json.loads(line) for line in (run/'runtime_trace.jsonl').open()]
    rows=[r for r in rows if r.get('event')=='proposal']
    return np.array([r['state'][:3] for r in rows]),np.array([r['sim_time'] for r in rows])


def teacher(run):
    s=json.loads((run/'scored_stream.json').read_text())
    rows=s['streams']['/aic_controller/controller_state']
    return np.array([[r['tcp_pose']['position'][k] for k in 'xyz'] for _,r in rows]),\
           np.array([t/1e9 for t,_ in rows])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior route plot')
    cfg=yaml.safe_load((a.candidate/'eval_config.yaml').read_text())
    for path in (a.teacher,a.control):
        if yaml.safe_load((path/'eval_config.yaml').read_text())!=cfg:
            raise ValueError('Route scenes must match exactly')
    cal=json.loads(a.calibration.read_text())
    ports={(c,p):m for c,p,m in sfp_ports(
        {'trial_yaml':str(a.candidate/'eval_config.yaml'),'trial_id':next(iter(cfg['trials']))},cal)}
    fig,ax=plt.subplots(figsize=(8,6))
    for name,path,color in (('successful teacher',a.teacher,'#228833'),
                            ('control',a.control,'#aa3377'),
                            ('corrected actor',a.candidate,'#3366cc')):
        xyz,t=(teacher(path) if name=='successful teacher' else actor(path))
        use=(t>=3)&(t<=25)
        ax.plot(xyz[use,0]*1000,xyz[use,1]*1000,color=color,label=name,linewidth=2)
        for stamp in (4,8,12,16,20):
            j=np.argmin(np.abs(t-stamp));ax.scatter(xyz[j,0]*1000,xyz[j,1]*1000,
                                                     c=color,s=20,zorder=3)
            if name=='corrected actor':
                ax.annotate(str(stamp),xy=xyz[j,:2]*1000,xytext=(4,4),
                            textcoords='offset points',fontsize=8,color=color)
    for (card,port),m in ports.items():
        pos=m[:2,3]*1000
        ax.scatter(*pos,marker='x',color='black',s=25)
        if port==1:ax.annotate(f'card {card}',pos,xytext=(3,-11),textcoords='offset points',fontsize=8)
    target=ports[1,1][:2,3]*1000
    ax.scatter(*target,marker='*',color='red',s=210,label='requested card 1, port 1')
    ax.set(xlabel='base-link X (mm)',ylabel='base-link Y (mm)',
           title='Five-card SFP: measured TCP routes and offline port locations')
    ax.grid(alpha=.3);ax.legend();ax.set_aspect('equal',adjustable='datalim')
    fig.tight_layout();fig.savefig(a.output,dpi=180)
    print(a.output)


if __name__=='__main__':main()
