"""Validate the ACT state layout shared by training, export, and ROS inference."""

from __future__ import annotations

from typing import Any

from .runtime_features import AIC_STATE_DIMS
from .task_encoding import TASK_VECTOR_DIM, task_encoding_schema


def validate_act_state_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return explicit layout fields, preserving legacy checkpoints without time.

    New task-conditioned checkpoints with elapsed time must declare the task
    slice and canonical encoding: silently inferring a different feature order
    changes the policy's input while leaving its tensor shape valid.
    """
    state_shape = metadata.get("state_shape")
    if not isinstance(state_shape, (list, tuple)) or len(state_shape) != 1:
        raise ValueError("ACT metadata must declare one-dimensional state_shape")
    state_dim = int(state_shape[0])
    include_time = metadata.get("include_elapsed_sim_time", False)
    if not isinstance(include_time, bool):
        raise ValueError("include_elapsed_sim_time must be a boolean")
    base_dim = int(metadata.get("base_state_dim", state_dim - int(include_time)))
    if base_dim not in AIC_STATE_DIMS or state_dim != base_dim + int(include_time):
        raise ValueError("ACT state_shape/base_state_dim/elapsed-time layout disagrees")
    has_task = base_dim in (42, 82)
    if "task_conditioned" in metadata and metadata["task_conditioned"] is not has_task:
        raise ValueError("ACT task_conditioned disagrees with base_state_dim")
    expected_indices = [base_dim - TASK_VECTOR_DIM, base_dim] if has_task else None
    if has_task and include_time:
        for key in ("task_conditioned", "task_vector_indices", "task_encoding"):
            if key not in metadata:
                raise ValueError(f"Task-conditioned ACT with time requires explicit {key}")
    if metadata.get("task_vector_indices", expected_indices) != expected_indices:
        raise ValueError("ACT task_vector_indices must identify the ten task values before elapsed time")
    expected_encoding = task_encoding_schema() if has_task else None
    if metadata.get("task_encoding", expected_encoding) != expected_encoding:
        raise ValueError("ACT task_encoding differs from the canonical ten-dimensional schema")
    return {
        "state_shape": [state_dim],
        "base_state_dim": base_dim,
        "include_elapsed_sim_time": include_time,
        "task_conditioned": has_task,
        "task_vector_indices": expected_indices,
        "task_encoding": expected_encoding,
    }
