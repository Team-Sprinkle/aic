#!/usr/bin/env python3
"""Explicit retained specialization of the frozen sample development runner."""
from pathlib import Path
import hashlib
import json
import sys

source=Path(__file__).with_name('run_sample_rule_development.py')
text=source.read_text()
old="policy='aic_example_policies.ros.RunSampleConservativeTransport'"
assert text.count(old)==1
text=text.replace(old,"policy='aic_example_policies.ros.RunSampleTransportRecovery'")
old="files=[Path(__file__),repo/'aic_example_policies/aic_example_policies/ros/RunSampleConservativeTransport.py',a.spec or Path(plan['rule_spec'])]"
assert text.count(old)==1
text=text.replace(old,"files=[Path(__file__),repo/'aic_example_policies/aic_example_policies/ros/RunSampleConservativeTransport.py',repo/'aic_example_policies/aic_example_policies/ros/RunSampleTransportRecovery.py',a.spec or Path(plan['rule_spec'])]")
output=Path(sys.argv[sys.argv.index('--output')+1])
snapshot=output.with_suffix('.runner_snapshot.py')
if snapshot.exists():raise FileExistsError(snapshot)
snapshot.write_text(text)
output.with_suffix('.specialization.json').write_text(json.dumps(dict(base=str(source),base_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    snapshot=str(snapshot),snapshot_sha256=hashlib.sha256(text.encode()).hexdigest(),only_changes='Deployment class plus inclusion of recovery source in frozen hashes; same scenes/scoring/gates.'),indent=2)+'\n')
exec(compile(text,str(snapshot),'exec'),dict(__name__='__main__',__file__=str(snapshot)))
