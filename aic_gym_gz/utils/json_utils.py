"""Helpers for converting runtime objects into JSON-safe structures."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy-backed containers and scalars into JSON-safe values."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.ndarray):
        return [to_jsonable(item) for item in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(item) for item in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(item) for item in obj]
    return obj
