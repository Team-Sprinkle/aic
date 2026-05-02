"""Hardware selection helpers for training launchers."""

from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
from typing import Iterable


@dataclass(frozen=True)
class GPUInfo:
    index: int
    memory_total_mb: int
    memory_used_mb: int

    @property
    def memory_free_mb(self) -> int:
        return self.memory_total_mb - self.memory_used_mb


def parse_nvidia_smi_csv(text: str) -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip().replace(" MiB", "") for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Unexpected nvidia-smi CSV row: {line!r}")
        gpus.append(GPUInfo(index=int(parts[0]), memory_total_mb=int(parts[1]), memory_used_mb=int(parts[2])))
    return gpus


def query_nvidia_smi() -> list[GPUInfo]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi was not found; cannot auto-select CUDA devices") from exc
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed with code {result.returncode}: {result.stderr.strip()}")
    return parse_nvidia_smi_csv(result.stdout)


def select_cuda_devices(
    *,
    cuda_devices: Iterable[int] | None = None,
    auto_select_free_devices: bool = False,
    num_devices: int = 1,
    min_free_memory_gb: float | None = None,
    gpus: list[GPUInfo] | None = None,
) -> list[int]:
    if cuda_devices is not None:
        selected = [int(device) for device in cuda_devices]
        if not selected:
            raise ValueError("cuda_devices must not be empty")
        return selected
    if not auto_select_free_devices:
        return [0]
    if num_devices < 1:
        raise ValueError("num_devices must be >= 1")
    gpus = query_nvidia_smi() if gpus is None else gpus
    min_free_mb = 0 if min_free_memory_gb is None else int(float(min_free_memory_gb) * 1024)
    candidates = [gpu for gpu in gpus if gpu.memory_free_mb >= min_free_mb]
    candidates.sort(key=lambda gpu: gpu.memory_free_mb, reverse=True)
    if len(candidates) < num_devices:
        raise RuntimeError(
            f"Could not find {num_devices} CUDA device(s) with >= {min_free_mb} MiB free; "
            f"available={[(gpu.index, gpu.memory_free_mb) for gpu in gpus]}"
        )
    return [gpu.index for gpu in candidates[:num_devices]]


def apply_cuda_visible_devices(selected: list[int]) -> dict[str, str]:
    value = ",".join(str(device) for device in selected)
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    return {
        "selected_physical_gpus": selected,
        "CUDA_VISIBLE_DEVICES": value,
        "effective_training_device": "cuda:0",
    }
