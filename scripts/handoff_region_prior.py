"""Observation-only TCP base-frame veto for learned handoff proposals."""
import json
from pathlib import Path

import numpy as np


def load_prior(path):
    value = json.loads(Path(path).read_text())
    if value.get('schema') != 'aic_handoff_tcp_base_region/v1':
        raise ValueError('Unexpected handoff prior schema')
    if value.get('training_geometry_runtime_input') is not False:
        raise ValueError('Runtime handoff prior must contain only fitted bounds')
    return value


def requested_key(task_vector):
    task = np.asarray(task_vector, float)
    if task.shape != (10,):
        raise ValueError('Expected public 10D task vector')
    port = int(task[2:4].argmax())
    if task[0] > .5:
        return f'sfp_card_{int(task[4:9].argmax())}_port_{port}'
    if task[1] > .5:
        return f'sc_port_{port}'
    raise ValueError('Unknown task family')


def allows(prior, task_vector, measured_tcp_base):
    key = requested_key(task_vector)
    region = prior['regions'].get(key)
    if region is None:
        raise ValueError(f'Missing handoff region for {key}')
    xyz = np.asarray(measured_tcp_base, float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        return False
    return bool(np.all(xyz >= region['min_base_m']) and
                np.all(xyz <= region['max_base_m']))
