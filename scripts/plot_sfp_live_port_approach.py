#!/usr/bin/env python3
"""Read-only axial/lateral actor approach trace for paired Gazebo rollouts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

from build_two_family_port_projection_labels import sfp_ports


def rows(run,cal,teacher=None):
    cfg=yaml.safe_load((run/'eval_config.yaml').read_text())
    tid=next(iter(cfg['trials']))
    ports={(c,p):m for c,p,m in sfp_ports({'trial_yaml':str(run/'eval_config.yaml'),
                                            'trial_id':tid},cal)}
    raw=[json.loads(line) for line in (run/'runtime_trace.jsonl').open()]
    raw=[x for x in raw if x.get('event')=='proposal' and len(x.get('state',[]))==32]
    task=np.asarray(raw[0]['task_vector'])
    card=int(task[4:9].argmax());port_id=int(task[2:4].argmax())
    port=ports[card,port_id]
    if teacher is not None:
        stream=json.loads((teacher/'scored_stream.json').read_text())
        records=[dict(sim_time=ns/1e9,state=[v['tcp_pose']['position'][k] for k in 'xyz'])
                 for ns,v in stream['streams']['/aic_controller/controller_state']]
    else:
        records=raw
    output=[]
    for r in records:
        rel=port[:3,:3].T@(np.asarray(r['state'][:3])-port[:3,3])
        output.append(dict(time_s=float(r['sim_time']),axial_mm=float(rel[2]*1000),
                           lateral_mm=float(np.linalg.norm(rel[:2])*1000)))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--posthoc',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Preserve prior trace')
    data=json.loads(a.posthoc.read_text())
    cal=json.loads(a.calibration.read_text());runs=[]
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    for axis,count in zip(axes,(2,3,5)):
        pair=[x for x in data['runs'] if f'cards{count}_' in Path(x['run']).name]
        if len(pair)!=2:raise ValueError(f'Expected matched {count}-card pair')
        teacher_trace=rows(Path(pair[0]['run']),cal,Path(pair[0]['teacher']))
        axis.plot([x['axial_mm'] for x in teacher_trace],
                  [x['lateral_mm'] for x in teacher_trace],color='black',
                  linestyle='--',linewidth=1.5,label='successful teacher')
        for item in pair:
            path=Path(item['run']);trace=rows(path,cal)
            in_band=[x for x in trace if -100<=x['axial_mm']<=-70]
            best=min(trace,key=lambda x:x['lateral_mm'])
            band=min(in_band,key=lambda x:x['lateral_mm']) if in_band else None
            label=path.name.rsplit('_',1)[-1]
            color='tab:blue' if label=='baseline' else 'tab:orange'
            axis.plot([x['axial_mm'] for x in trace],
                      [x['lateral_mm'] for x in trace],color=color,label=label,alpha=.75)
            axis.scatter([trace[0]['axial_mm']],[trace[0]['lateral_mm']],color=color,s=22)
            runs.append(dict(run=str(path),card_count=count,variant=label,
                             first_axial_band_time_s=in_band[0]['time_s'] if in_band else None,
                             first_axial_band_lateral_mm=in_band[0]['lateral_mm'] if in_band else None,
                             best_lateral_anywhere=best,
                             best_lateral_in_axial_band=band))
        axis.axvspan(-100,-70,color='green',alpha=.08)
        axis.axhspan(0,10,color='green',alpha=.08)
        axis.set_title(f'{count} cards');axis.set_xlabel('Axial distance to opening (mm)')
        axis.grid(alpha=.25)
    axes[0].set_ylabel('Lateral offset from opening (mm)')
    axes[-1].legend(loc='upper right')
    fig.suptitle('Measured TCP relative to requested port; green overlap is safe transport buffer')
    fig.tight_layout()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=170);plt.close(fig)
    details=a.output.with_suffix('.json')
    if details.exists():p.error('Preserve prior trace details')
    details.write_text(json.dumps(dict(schema='aic_sfp_live_port_approach_trace/v1',
                                       posthoc=str(a.posthoc),runs=runs,
                                       geometry_actor_input=False,final_scenes_opened=False),indent=2)+'\n')
    print(json.dumps(runs))


if __name__=='__main__':main()
