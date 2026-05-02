"""Canonical task-conditioning vector encoding for AIC hybrid training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


TASK_VECTOR_DIM = 10
TASK_FAMILIES = ("sfp_to_nic", "sc_to_sc")
TARGET_PORT_COUNT = 2
TARGET_CARD_COUNT = 5
TASK_VECTOR_NAMES = [
    "task_family_sfp_to_nic",
    "task_family_sc_to_sc",
    "target_port_0",
    "target_port_1",
    "target_card_0",
    "target_card_1",
    "target_card_2",
    "target_card_3",
    "target_card_4",
    "target_card_valid",
]


@dataclass(frozen=True)
class TaskFields:
    task_family: str
    target_port_index: int
    target_card_index: int
    target_card_valid: int


def _as_int(value: Any, field: str) -> int:
    if value is None or value == "":
        raise ValueError(f"Missing required task field: {field}")
    return int(value)


def validate_task_fields(
    *,
    task_family: str,
    target_port_index: int,
    target_card_index: int,
    target_card_valid: int,
) -> TaskFields:
    if task_family not in TASK_FAMILIES:
        raise ValueError(f"task_family must be one of {TASK_FAMILIES}, got {task_family!r}")
    if target_port_index not in range(TARGET_PORT_COUNT):
        raise ValueError(f"target_port_index must be 0 or 1, got {target_port_index!r}")
    if target_card_valid not in (0, 1, False, True):
        raise ValueError(f"target_card_valid must be 0 or 1, got {target_card_valid!r}")
    target_card_valid = int(target_card_valid)
    if task_family == "sfp_to_nic":
        if target_card_valid != 1:
            raise ValueError("sfp_to_nic requires target_card_valid=1")
        if target_card_index not in range(TARGET_CARD_COUNT):
            raise ValueError(f"sfp_to_nic target_card_index must be in 0..4, got {target_card_index!r}")
    if task_family == "sc_to_sc":
        if target_card_valid != 0:
            raise ValueError("sc_to_sc requires target_card_valid=0")
        if target_card_index != -1:
            raise ValueError(f"sc_to_sc target_card_index must be -1, got {target_card_index!r}")
    return TaskFields(task_family, int(target_port_index), int(target_card_index), target_card_valid)


def encode_task_vector(
    *,
    task_family: str,
    target_port_index: int,
    target_card_index: int,
    target_card_valid: int | None = None,
    dtype: Any = np.float32,
) -> np.ndarray:
    if target_card_valid is None:
        target_card_valid = 1 if task_family == "sfp_to_nic" else 0
    fields = validate_task_fields(
        task_family=task_family,
        target_port_index=int(target_port_index),
        target_card_index=int(target_card_index),
        target_card_valid=int(target_card_valid),
    )
    vector = np.zeros((TASK_VECTOR_DIM,), dtype=dtype)
    vector[0 if fields.task_family == "sfp_to_nic" else 1] = 1
    vector[2 + fields.target_port_index] = 1
    if fields.target_card_valid:
        vector[4 + fields.target_card_index] = 1
    vector[9] = fields.target_card_valid
    validate_task_vector(vector)
    return vector


def encode_task_vector_from_metadata(metadata: dict[str, Any], *, dtype: Any = np.float32) -> np.ndarray:
    task = metadata.get("task") if isinstance(metadata.get("task"), dict) else metadata
    if not isinstance(task, dict):
        raise ValueError("Task metadata must be a dict or contain a dict field named 'task'")
    if "task_vector" in task and task["task_vector"] not in (None, ""):
        return validate_task_vector(task["task_vector"], dtype=dtype)
    return encode_task_vector(
        task_family=str(task["task_family"]),
        target_port_index=_as_int(task.get("target_port_index"), "target_port_index"),
        target_card_index=_as_int(task.get("target_card_index"), "target_card_index"),
        target_card_valid=_as_int(task.get("target_card_valid"), "target_card_valid"),
        dtype=dtype,
    )


def validate_task_vector(vector: Any, *, dtype: Any = np.float32) -> np.ndarray:
    values = np.asarray(vector, dtype=dtype).reshape(-1)
    if values.shape != (TASK_VECTOR_DIM,):
        raise ValueError(f"task_vector must have dim {TASK_VECTOR_DIM}, got {values.shape[0]}")
    if not np.all(np.isin(values, [0, 1])):
        raise ValueError(f"task_vector must contain only 0/1 values, got {values.tolist()!r}")
    if int(values[0:2].sum()) != 1:
        raise ValueError(f"task_family one-hot is invalid: {values[0:2].tolist()!r}")
    if int(values[2:4].sum()) != 1:
        raise ValueError(f"target_port one-hot is invalid: {values[2:4].tolist()!r}")
    target_card_valid = int(values[9])
    card_sum = int(values[4:9].sum())
    if target_card_valid == 0 and card_sum != 0:
        raise ValueError("target_card one-hot must be all zeros when target_card_valid=0")
    if target_card_valid == 1 and card_sum != 1:
        raise ValueError("target_card one-hot must contain exactly one 1 when valid")
    return values.astype(dtype, copy=False)


def task_encoding_schema() -> dict[str, Any]:
    return {
        "name": "aic_option_a_task_vector",
        "dim": TASK_VECTOR_DIM,
        "dtype": "float32",
        "fields": [
            {"name": "task_family_onehot", "offset": 0, "size": 2, "values": {"sfp_to_nic": [1, 0], "sc_to_sc": [0, 1]}},
            {"name": "target_port_index_onehot", "offset": 2, "size": 2, "values": {"0": [1, 0], "1": [0, 1]}},
            {"name": "target_card_index_onehot", "offset": 4, "size": 5, "valid_when": "target_card_valid == 1"},
            {"name": "target_card_valid", "offset": 9, "size": 1},
        ],
        "names": TASK_VECTOR_NAMES,
        "examples": {
            "sfp_to_nic_card3_port1": [1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            "sc_to_sc_port0": [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        },
    }
