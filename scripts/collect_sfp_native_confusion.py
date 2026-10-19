#!/usr/bin/env python3
"""Sequential rootless native SFP RGB collection for the wrong-port hypothesis.

One Gazebo container/GPU at a time; keep scored failures as perception-only
evidence. Existing complete audits are never overwritten.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

IMAGE='ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection',type=Path,required=True)
    ap.add_argument('--storage',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--gpu',type=int,default=3)
    ap.add_argument('--phase',choices=('first_ten','remaining','all'),default='first_ten')
    a=ap.parse_args();repo=Path.cwd().resolve()
    selection=a.selection.resolve();storage=a.storage.resolve();storage.mkdir(parents=True,exist_ok=True)
    rows=json.loads((selection/'collection_manifest.json').read_text())['rows']
    selected=[r for r in rows if (True if a.phase=='all' else
                                  r['case'].endswith('extra0_train') if a.phase=='first_ten'
                                  else not r['case'].endswith('extra0_train'))]
    env=os.environ.copy();env.update(DOCKER_HOST=f'unix:///run/user/{os.getuid()}/docker.sock',
        AIC_TRANSPORT_EVAL_GPU=str(a.gpu),AIC_TRANSPORT_EVAL_POLICY_GPU=str(a.gpu),
        AIC_TRANSPORT_EVAL_POLICY='aic_example_policies.ros.CheatCode',
        AIC_TRANSPORT_EVAL_TEACHER_PREFIX='true',AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.02')
    statuses=[]
    for n,row in enumerate(selected,1):
        case=row['case'];audit=storage/case;env['AIC_TRANSPORT_EVAL_SCENE_DIR']=str(selection/case)
        if not (audit/'results/scoring.yaml').exists():
            if audit.exists():raise RuntimeError(f'Incomplete audit needs diagnosis: {audit}')
            command=['bash','scripts/run_clean_transport_eval.sh',case,str(a.checkpoint),str(audit)]
            with (storage/f'{case}.launch.log').open('w') as log:
                code=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            if code:raise RuntimeError(f'Gazebo failed {case}; see {storage}/{case}.launch.log')
        scoring=yaml.safe_load((audit/'results/scoring.yaml').read_text())
        trial=next(k for k in scoring if k.startswith('trial_'))
        tier3=float(scoring[trial]['tier_3']['score'])
        stream=audit/'scored_stream.json'
        if not stream.exists():
            docker=['docker','run','--rm','--net=host','--entrypoint','/bin/bash',
                '-v',f'{repo}:{repo}:ro','-v',f'{audit}:/audit','-w',str(repo),IMAGE,
                '-lc','source /ws_aic/install/setup.bash; python3 scripts/extract_moveit_training_stream.py /audit /audit/scored_stream.json']
            with (storage/f'{case}.extract.log').open('w') as log:
                code=subprocess.run(docker,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            if code:raise RuntimeError(f'Stream extraction failed {case}')
        text=stream.read_text()
        if '/audit/frames/' in text:
            text=text.replace('/audit/frames/',str(audit/'frames')+'/')
            temporary=audit/'scored_stream.hostpaths.json'
            temporary.write_text(text)
            temporary.replace(stream)
        frame_count=sum(1 for _ in (audit/'frame_manifest.csv').open())-1
        record={'case':case,'split':row['split'],'tier3':tier3,'native_frames':frame_count,
                'audit':str(audit),'scored_stream':str(stream)}
        statuses.append(record)
        print(json.dumps({'progress':f'{n}/{len(selected)}',**record}),flush=True)
        (storage/f'collection_{a.phase}_summary.json').write_text(json.dumps({
            'schema':'sfp_native_confusion_collection/v1','phase':a.phase,'completed':statuses,
            'gpu':a.gpu,'rootless':True,'reserved_final_opened':False},indent=2)+'\n')


if __name__=='__main__':main()
