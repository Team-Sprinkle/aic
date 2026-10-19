#!/usr/bin/env python3
"""Score frozen three-camera selected-port identity on one-port reset groups."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


CAMERAS=('center','left','right')


def grouped(path):
    d=json.loads(path.read_text())
    result=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for item in d['development_predictions']:
        t=int(Path(item['image']).stem.rsplit('_',2)[1])
        item=dict(item,sim_ns=t)
        result[item['group']][item['requested_port']][item['camera']].append(item)
    return result


def evaluate(path,tolerance_ns):
    groups=grouped(path);records=[]
    for group,ports in groups.items():
        for request,cameras in ports.items():
            if any(camera not in cameras for camera in CAMERAS):continue
            selected=0 if 'development_p0' in cameras['center'][0]['image'] else 1
            counts={'triplets':0,'accepted_all_three':0,'accepted_two_of_three':0,
                    'all_three_projected_inframe':0,'accepted_and_all_three_projected_inframe':0}
            for center in cameras['center']:
                nearby=[min(cameras[c],key=lambda x:abs(x['sim_ns']-center['sim_ns']))
                        for c in ('left','right')]
                if any(abs(x['sim_ns']-center['sim_ns'])>tolerance_ns for x in nearby):continue
                views=[center,*nearby]
                votes=sum(x['probability']>=.5 for x in views)
                clear=all(x['projected_inframe_label'] for x in views)
                counts['triplets']+=1
                counts['accepted_all_three']+=votes==3
                counts['accepted_two_of_three']+=votes>=2
                counts['all_three_projected_inframe']+=clear
                counts['accepted_and_all_three_projected_inframe']+=votes==3 and clear
            records.append({'group':group,'scene':Path(cameras['center'][0]['image']).parent.parent.name,
                            'requested_port':request,'request_kind':'real' if request==selected else 'absent',
                            **counts})
    total={}
    for kind in ('real','absent'):
        subset=[r for r in records if r['request_kind']==kind]
        total[kind]={k:sum(r[k] for r in subset) for k in ('triplets','accepted_all_three',
            'accepted_two_of_three','all_three_projected_inframe',
            'accepted_and_all_three_projected_inframe')}
    return {'source':str(path),'by_group':records,'total':total}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spatial',type=Path,required=True)
    p.add_argument('--pooled',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tolerance-ms',type=float,default=150)
    a=p.parse_args()
    result={'schema':'sc_identity_three_camera/v1','threshold_per_camera':.5,
            'acceptance_rule':'all three requested-port probabilities >= 0.5',
            'max_timestamp_gap_ms':a.tolerance_ms,
            'labels_are_projected_inframe_not_true_occlusion_visibility':True,
            'spatial':evaluate(a.spatial,int(a.tolerance_ms*1e6)),
            'pooled':evaluate(a.pooled,int(a.tolerance_ms*1e6)),
            'final_scenes_opened':False}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k]['total'] for k in ('spatial','pooled')},indent=2))


if __name__=='__main__':main()
