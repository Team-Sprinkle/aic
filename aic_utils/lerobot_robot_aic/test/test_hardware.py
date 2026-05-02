from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

from lerobot_robot_aic.hardware import GPUInfo, parse_nvidia_smi_csv, select_cuda_devices  # noqa: E402


def test_parse_nvidia_smi_csv() -> None:
    gpus = parse_nvidia_smi_csv("0, 24576, 1024\n1, 24576, 12000 MiB\n")

    assert gpus == [
        GPUInfo(index=0, memory_total_mb=24576, memory_used_mb=1024),
        GPUInfo(index=1, memory_total_mb=24576, memory_used_mb=12000),
    ]
    assert gpus[0].memory_free_mb == 23552


def test_select_cuda_devices_auto_uses_free_memory_order() -> None:
    selected = select_cuda_devices(
        auto_select_free_devices=True,
        num_devices=2,
        min_free_memory_gb=8,
        gpus=[
            GPUInfo(index=0, memory_total_mb=24576, memory_used_mb=20000),
            GPUInfo(index=1, memory_total_mb=24576, memory_used_mb=4000),
            GPUInfo(index=2, memory_total_mb=24576, memory_used_mb=8000),
        ],
    )

    assert selected == [1, 2]


def test_select_cuda_devices_fails_when_no_gpu_matches() -> None:
    with pytest.raises(RuntimeError, match="Could not find 1 CUDA device"):
        select_cuda_devices(
            auto_select_free_devices=True,
            min_free_memory_gb=20,
            gpus=[GPUInfo(index=0, memory_total_mb=24576, memory_used_mb=10000)],
        )
