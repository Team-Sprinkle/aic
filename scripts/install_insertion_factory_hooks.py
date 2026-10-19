#!/usr/bin/env python3
"""Wait for the active four-case pilot, then install legacy-compatible hooks.

No active simulation/source hash changes. Preserve pre-change code and record
the exact implementation before starting a separately versioned experiment.
"""
from pathlib import Path
import hashlib
import json
import shutil
import time
import subprocess


def main():
    repo=Path.cwd()
    pilot=Path('/home/nas/chmin/aic_archives/2026-10-17_25mm_handoff_diagnostic_v1')
    output=Path('/home/nas/chmin/aic_archives/2026-10-17_history_chunk_continuation_v1')
    output.mkdir(exist_ok=True)
    while not (pilot/'summary.json').exists():
        time.sleep(20)
    ros=repo/'aic_example_policies/aic_example_policies/ros'
    changes=[
        (ros/'RunSharedPortTCP.py',[
            ('    supports_sc_tip_target_reference=False\n',
             '    supports_sc_tip_target_reference=False\n\n'
             '    @staticmethod\n'
             '    def _make_actor(checkpoint, stats, **kwargs):\n'
             '        return SharedPortTCPActor(stats, **kwargs)\n'),
            ('self.actor = SharedPortTCPActor(stats, pretrained=False,',
             'self.actor = self._make_actor(checkpoint, stats, pretrained=False,')]),
        (ros/'RunRuleThenSharedRecovery.py',[
            ('class RunRuleThenSharedRecovery(Policy):\n',
             'class RunRuleThenSharedRecovery(Policy):\n'
             '    def _make_local_actor(self, checkpoint):\n'
             '        return SupervisedLockedActor(self.parent, checkpoint)\n\n'),
            ("self.local=SupervisedLockedActor(parent_node,os.environ['AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_CHECKPOINT'])",
             "self.local=self._make_local_actor(os.environ['AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_CHECKPOINT'])")])]
    report=[]
    for path, replacements in changes:
        text=path.read_text()
        if ('def _make_actor(' in text if path.name=='RunSharedPortTCP.py' else 'def _make_local_actor(' in text):
            continue
        before=hashlib.sha256(text.encode()).hexdigest()
        backup=output/'pre_hook_sources'/path.name
        backup.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,backup)
        for old,new in replacements:
            if text.count(old)!=1:raise ValueError('Source changed: '+str(path)+' '+old)
            text=text.replace(old,new)
        compile(text,str(path),'exec')
        path.write_text(text)
        report.append(dict(path=str(path),before=before,after=hashlib.sha256(text.encode()).hexdigest()))
    subprocess.run([str(repo/'.pixi/envs/default/bin/python'),'-m','py_compile',
        *[str(x[0]) for x in changes],str(ros/'RunHistoryChunkedInsertion.py')],check=True)
    (output/'runtime_hooks_ready.json').write_text(json.dumps(dict(changes=report,
        prior_pilot_complete=True,legacy_four_query_default_unchanged=True,
        active_job_interrupted=False),indent=2)+'\n')


if __name__=='__main__':main()
