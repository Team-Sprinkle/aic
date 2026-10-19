"""Inference definitions extracted from task_encoding.py; no training entry point."""
from typing import Any
import numpy as np
TASK_VECTOR_DIM = 10

def validate_task_vector(vector: Any, *, dtype: Any=np.float32) -> np.ndarray:
    values = np.asarray(vector, dtype=dtype).reshape(-1)
    if values.shape != (TASK_VECTOR_DIM,):
        raise ValueError(f'task_vector must have dim {TASK_VECTOR_DIM}, got {values.shape[0]}')
    if not np.all(np.isin(values, [0, 1])):
        raise ValueError(f'task_vector must contain only 0/1 values, got {values.tolist()!r}')
    if int(values[0:2].sum()) != 1:
        raise ValueError(f'task_family one-hot is invalid: {values[0:2].tolist()!r}')
    if int(values[2:4].sum()) != 1:
        raise ValueError(f'target_port one-hot is invalid: {values[2:4].tolist()!r}')
    target_card_valid = int(values[9])
    card_sum = int(values[4:9].sum())
    if target_card_valid == 0 and card_sum != 0:
        raise ValueError('target_card one-hot must be all zeros when target_card_valid=0')
    if target_card_valid == 1 and card_sum != 1:
        raise ValueError('target_card one-hot must contain exactly one 1 when valid')
    return values.astype(dtype, copy=False)
