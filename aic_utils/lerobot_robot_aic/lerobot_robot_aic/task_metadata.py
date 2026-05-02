"""Task-conditioning metadata helpers for AIC LeRobot datasets."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

from .task_encoding import (
    TASK_VECTOR_DIM,
    TASK_VECTOR_NAMES,
    encode_task_vector,
    task_encoding_schema,
    validate_task_vector as validate_task_vector_array,
)


def task_vector_from_fields(
    task_family: str,
    target_port_index: int,
    target_card_index: int,
) -> list[int]:
    """Return Option A task vector for one episode."""
    return [int(v) for v in encode_task_vector(
        task_family=task_family,
        target_port_index=target_port_index,
        target_card_index=target_card_index,
    ).tolist()]


def validate_task_vector(
    vector: Iterable[Any],
    *,
    task_family: str | None = None,
    target_port_index: int | None = None,
    target_card_index: int | None = None,
    target_card_valid: int | None = None,
) -> list[int]:
    values = [int(v) for v in vector]
    if len(values) != TASK_VECTOR_DIM:
        raise ValueError(f"task_vector must have dim {TASK_VECTOR_DIM}, got {len(values)}")
    if any(v not in (0, 1) for v in values):
        raise ValueError(f"task_vector must contain only 0/1 values, got {values!r}")
    if sum(values[0:2]) != 1:
        raise ValueError(f"task_family one-hot is invalid: {values[0:2]!r}")
    if sum(values[2:4]) != 1:
        raise ValueError(f"target_port one-hot is invalid: {values[2:4]!r}")
    if values[9] not in (0, 1):
        raise ValueError("target_card_valid must be 0 or 1")
    if values[9] == 0 and any(values[4:9]):
        raise ValueError("target_card one-hot must be all zeros when target_card_valid=0")
    if values[9] == 1 and sum(values[4:9]) != 1:
        raise ValueError("target_card one-hot must contain exactly one 1 when valid")

    if task_family is not None:
        expected_family = "sfp_to_nic" if values[0] == 1 else "sc_to_sc"
        if task_family != expected_family:
            raise ValueError(f"task_family {task_family!r} does not match vector {values!r}")
    if target_port_index is not None and values[2 + target_port_index] != 1:
        raise ValueError(f"target_port_index {target_port_index} does not match vector {values!r}")
    if target_card_valid is not None and values[9] != int(target_card_valid):
        raise ValueError("target_card_valid does not match vector")
    if target_card_index is not None:
        if target_card_index == -1:
            if any(values[4:9]):
                raise ValueError("target_card_index -1 requires zero card vector")
        elif target_card_index in range(5):
            if values[4 + target_card_index] != 1:
                raise ValueError("target_card_index does not match vector")
        else:
            raise ValueError(f"target_card_index must be -1 or 0..4, got {target_card_index}")
    return values


def _parse_int(value: Any, field: str) -> int:
    if value in (None, ""):
        raise ValueError(f"Missing required integer field {field}")
    if isinstance(value, float) and math.isnan(value):
        raise ValueError(f"Missing required integer field {field}")
    return int(value)


def _parse_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return int(value)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _row_task_vector(row: dict[str, Any]) -> list[int]:
    raw = row.get("task_vector")
    if raw not in (None, ""):
        vector = json.loads(str(raw))
    else:
        vector = task_vector_from_fields(
            str(row["task_family"]),
            _parse_int(row.get("target_port_index"), "target_port_index"),
            _parse_int(row.get("target_card_index"), "target_card_index"),
        )
    return validate_task_vector(
        vector,
        task_family=str(row["task_family"]),
        target_port_index=_parse_int(row.get("target_port_index"), "target_port_index"),
        target_card_index=_parse_int(row.get("target_card_index"), "target_card_index"),
        target_card_valid=_parse_int(row.get("target_card_valid"), "target_card_valid"),
    )


def parse_task_metadata_csv(path: Path) -> list[dict[str, Any]]:
    """Read and validate attempts.csv or accepted.csv task metadata."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            parsed = dict(row)
            parsed["target_port_index"] = _parse_int(row.get("target_port_index"), "target_port_index")
            parsed["target_card_index"] = _parse_int(row.get("target_card_index"), "target_card_index")
            parsed["target_card_valid"] = _parse_int(row.get("target_card_valid"), "target_card_valid")
            parsed["task_vector"] = _row_task_vector(row)
            parsed["selected"] = _parse_bool(row.get("selected", False))
            for key in ("run_index", "source_episode_index", "accepted_episode_index"):
                parsed[key] = _parse_optional_int(row.get(key))
            rows.append(parsed)
    return rows


def load_episode_task_vectors(manifest_path: Path) -> dict[int, list[int]]:
    """Return accepted episode index -> task vector from accepted.csv or JSONL."""
    vectors: dict[int, list[int]] = {}
    if manifest_path.suffix == ".jsonl":
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                accepted_idx = obj.get("accepted_episode_index")
                if accepted_idx is None:
                    continue
                task = obj["task"]
                vectors[int(accepted_idx)] = validate_task_vector(
                    task["task_vector"],
                    task_family=task["task_family"],
                    target_port_index=int(task["target_port_index"]),
                    target_card_index=int(task["target_card_index"]),
                    target_card_valid=int(task["target_card_valid"]),
                )
        return vectors

    for row in parse_task_metadata_csv(manifest_path):
        accepted_idx = row.get("accepted_episode_index")
        if accepted_idx is None:
            continue
        vectors[int(accepted_idx)] = row["task_vector"]
    return vectors


def infer_task_metadata_path(dataset_root: Path) -> Path | None:
    candidates = [
        dataset_root / "manifests" / "accepted.csv",
        dataset_root.parent / "manifests" / "accepted.csv",
        dataset_root.parent / "accepted.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def join_task_vector_to_frames(
    frames: Iterable[dict[str, Any]],
    episode_vectors: dict[int, list[int]],
    *,
    episode_key: str = "episode_index",
    output_key: str = "task_vector",
    missing: str = "zeros",
) -> list[dict[str, Any]]:
    """Attach per-episode vectors to frame dictionaries."""
    joined: list[dict[str, Any]] = []
    zero = [0] * TASK_VECTOR_DIM
    for frame in frames:
        episode = int(frame[episode_key])
        vector = episode_vectors.get(episode)
        if vector is None:
            if missing == "error":
                raise KeyError(f"Missing task vector for episode {episode}")
            vector = zero
        new_frame = dict(frame)
        new_frame[output_key] = list(vector)
        joined.append(new_frame)
    return joined


def append_task_vectors_to_observation_state_dataset(
    dataset_root: Path,
    manifest_path: Path | None,
    output_root: Path,
    *,
    missing: str = "zeros",
    overwrite: bool = True,
) -> Path:
    """Copy a LeRobot dataset and append task vector columns to observation.state.

    This leaves the source dataset unchanged and creates a derived local dataset
    that standard LeRobot ACT can consume as a wider low-dimensional state.
    """
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Task-conditioned dataset preparation requires pandas and numpy.") from exc

    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(output_root)
    shutil.copytree(dataset_root, output_root)

    episode_vectors = load_episode_task_vectors(manifest_path) if manifest_path else {}
    zero = np.zeros((TASK_VECTOR_DIM,), dtype=np.float32)
    data_files = sorted((output_root / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No LeRobot parquet data files found under {output_root / 'data'}")

    appended_states: list[Any] = []
    for data_file in data_files:
        df = pd.read_parquet(data_file)
        if "episode_index" not in df.columns or "observation.state" not in df.columns:
            raise ValueError(f"{data_file} must contain episode_index and observation.state")

        def append_vector(row: Any) -> Any:
            episode = int(row["episode_index"])
            vector = episode_vectors.get(episode)
            if vector is None:
                if missing == "error":
                    raise KeyError(f"Missing task vector for episode {episode}")
                vector_arr = zero
            else:
                vector_arr = np.asarray(validate_task_vector(vector), dtype=np.float32)
            state = np.asarray(row["observation.state"], dtype=np.float32).reshape(-1)
            return np.concatenate([state, vector_arr]).astype(np.float32)

        df["observation.state"] = df.apply(append_vector, axis=1)
        appended_states.extend(df["observation.state"].tolist())
        df.to_parquet(data_file, index=False)

    info_path = output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    state_spec = info.get("features", {}).get("observation.state")
    if not isinstance(state_spec, dict):
        raise ValueError(f"Missing observation.state feature spec in {info_path}")
    old_shape = state_spec.get("shape")
    if not isinstance(old_shape, list) or len(old_shape) != 1:
        raise ValueError(f"Expected 1D observation.state shape in {info_path}, got {old_shape!r}")
    state_spec["shape"] = [int(old_shape[0]) + TASK_VECTOR_DIM]
    names = state_spec.get("names")
    if isinstance(names, list):
        state_spec["names"] = [*names, *TASK_VECTOR_NAMES]
    info.setdefault("aic_task_conditioning", {})["method"] = "append_task_vector_to_observation.state"
    info["aic_task_conditioning"]["task_vector_dim"] = TASK_VECTOR_DIM
    info["aic_task_conditioning"]["manifest_path"] = str(manifest_path) if manifest_path else None
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")

    stats_path = output_root / "meta" / "stats.json"
    if stats_path.exists() and appended_states:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        matrix = np.stack([np.asarray(state, dtype=np.float32).reshape(-1) for state in appended_states], axis=0)
        stats["observation.state"] = {
            "min": matrix.min(axis=0).tolist(),
            "max": matrix.max(axis=0).tolist(),
            "mean": matrix.mean(axis=0).tolist(),
            "std": matrix.std(axis=0).tolist(),
            "count": [int(matrix.shape[0])],
            "q01": np.quantile(matrix, 0.01, axis=0).tolist(),
            "q10": np.quantile(matrix, 0.10, axis=0).tolist(),
            "q50": np.quantile(matrix, 0.50, axis=0).tolist(),
            "q90": np.quantile(matrix, 0.90, axis=0).tolist(),
            "q99": np.quantile(matrix, 0.99, axis=0).tolist(),
        }
        stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return output_root
