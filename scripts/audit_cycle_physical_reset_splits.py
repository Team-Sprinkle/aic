#!/usr/bin/env python3
"""Reconstruct canonical reset identities from source YAML, not opaque hash labels."""
import hashlib
import json
from pathlib import Path
import re
import yaml
root=Path('/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1')
corpora={'SFP_pose':json.loads((root/'SFP_wide_orientation_dataset/manifest.json').read_text()),'empty_row':json.loads((root/'empty_row_visual_ablation/manifest.json').read_text())}
cache={};reports={};reset_tables={}
for name,rows in corpora.items():
    groups={};episodes={}
    for row in rows:
        key=row['episode']
        if key in episodes:continue
        source=row.get('source_stream');audit=Path(row['images']['center']).parent.parent
        if source:
            with Path(source).open() as f:prefix=f.read(20000)
            match=re.search(r'"trial_id"\s*:\s*"([^"]+)"',prefix);assert match
            trial_id=match[1]
        else:
            # The detector stores image provenance; its admitted source holds
            # and teacher episodes each use one trial in the capture YAML.
            trial_id=None
        cfg=yaml.safe_load((audit/'eval_config.yaml').read_text())
        if trial_id is None:
            if len(cfg['trials'])!=1:raise ValueError('Ambiguous detector trial; recover its source manifest')
            trial_id=next(iter(cfg['trials']))
        trial=cfg['trials'][trial_id];task=next(iter(trial['tasks'].values()))
        identity=dict(scene=trial['scene'],robot=cfg.get('robot'),task={k:task[k] for k in ['plug_type','port_type','target_module_name','cable_name']})
        digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        episodes[key]=dict(episode=key,split=row['split'],canonical_reset_sha256=digest)
        groups.setdefault(digest,set()).add(row['split'])
    conflicts=[key for key,values in groups.items() if len(values)>1]
    reports[name]=dict(episodes=len(episodes),canonical_reset_groups=len(groups),cross_split_conflicts=conflicts,split_gate_passed=not conflicts)
    reset_tables[name]=list(episodes.values())
(root/'canonical_reset_split_audit.json').write_text(json.dumps(dict(reports=reports,reset_tables=reset_tables),indent=2)+'\n')
print(json.dumps(reports),flush=True)
