"""Cooperate with rootless rollout GPU locks for explicitly selected host GPUs."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path


@contextmanager
def selected_gpu_reservation():
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    ids=visible.split(',') if visible else []
    if not ids or any(not x.strip().isdigit() for x in ids):
        yield
        return
    held=set(os.environ.get('AIC_PROJECT_GPU_LOCKS_HELD','').split(','));files=[]
    try:
        for gpu in sorted(set(map(int,ids))):
            if str(gpu) in held:continue
            f=Path(f'/var/tmp/chmin_aic_gpu{gpu}.lock').open('a')
            fcntl.flock(f,fcntl.LOCK_EX);files.append(f)
        yield
    finally:
        for f in reversed(files):fcntl.flock(f,fcntl.LOCK_UN);f.close()
