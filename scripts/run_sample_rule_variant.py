#!/usr/bin/env python3
"""Retain a scoped runner specialization for independent rule variants.

Only selection, policy, ROS domain and unique audit-folder names change.
The scene reader, scoring and strict qualification are the same base runner.
"""
import hashlib
import json
from pathlib import Path
import sys

source=Path(__file__).with_name('run_sample_rule_development.py')
text=source.read_text()
changes={
 "p.add_argument('--case');p.add_argument('--spec',type=Path)":
 "p.add_argument('--case');p.add_argument('--spec',type=Path);p.add_argument('--family',choices=['sc','sfp']);p.add_argument('--domain',default='92');p.add_argument('--policy',default='aic_example_policies.ros.RunSampleConservativeTransport')",
 "files=[Path(__file__),repo/'aic_example_policies/aic_example_policies/ros/RunSampleConservativeTransport.py',a.spec or Path(plan['rule_spec'])]":
 "files=[Path(__file__),repo/'aic_example_policies/aic_example_policies/ros/RunSampleConservativeTransport.py',repo/'aic_example_policies/aic_example_policies/ros'/(a.policy.rsplit('.',1)[1]+'.py'),a.spec or Path(plan['rule_spec'])]",
 "cases=[c for c in plan['cases'] if not a.case or c['case']==a.case]":
 "cases=[c for c in plan['cases'] if (not a.case or c['case']==a.case) and (not a.family or c['family']==a.family)]",
 "policy='aic_example_policies.ros.RunSampleConservativeTransport',domain='91'":
 "policy=a.policy,domain=a.domain",
 "run=a.output/c['case'];cmd=":
 "run=a.output/(c['case']+'__'+a.domain);cmd=",
}
for old,new in changes.items():
    if text.count(old)!=1:raise ValueError('Base runner changed: '+old)
    text=text.replace(old,new)
output=Path(sys.argv[sys.argv.index('--output')+1]);snapshot=output.with_suffix('.runner_snapshot.py')
if snapshot.exists():raise FileExistsError(snapshot)
snapshot.write_text(text)
output.with_suffix('.specialization.json').write_text(json.dumps(dict(base=str(source),base_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    snapshot=str(snapshot),snapshot_sha256=hashlib.sha256(text.encode()).hexdigest(),changes=list(changes),scoring_changed=False),indent=2)+'\n')
exec(compile(text,str(snapshot),'exec'),dict(__name__='__main__',__file__=str(snapshot)))
