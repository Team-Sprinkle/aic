from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from aic_gym_gz.run_vlm_trajectory_dataset import _build_scenario, _run_dir


def _args(**overrides):
    values = {
        "task_family": "sfp_to_nic",
        "output_root": "aic_gym_gz/artifacts/inspect_runs",
        "trajectory_index": 1,
        "nic_cards": 2,
        "target_nic_index": 1,
        "target_nic_port": 0,
        "sc_ports": 2,
        "target_sc_index": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_sfp_to_nic_layout_and_target_metadata() -> None:
    args = _args(task_family="sfp_to_nic", nic_cards=2, target_nic_index=1, target_nic_port=0)

    scenario, metadata = _build_scenario(args)

    assert _run_dir(args) == Path("aic_gym_gz/artifacts/inspect_runs/sfp_to_nic/vlm_planner/nic_cards_2/n1")
    task = next(iter(scenario.tasks.values()))
    assert task.target_module_name == "nic_card_mount_1"
    assert task.port_name == "sfp_port_0"
    assert metadata["right_card"] == "nic_card_mount_1"
    assert metadata["right_port_number"] == 0
    assert [rail.present for rail in scenario.task_board.nic_rails.values()] == [True, True, False, False, False]


def test_sc_to_sc_layout_and_target_metadata() -> None:
    args = _args(task_family="sc_to_sc", sc_ports=2, target_sc_index=1)

    scenario, metadata = _build_scenario(args)

    assert _run_dir(args) == Path("aic_gym_gz/artifacts/inspect_runs/sc_to_sc/vlm_planner/sc_ports_2/n1")
    task = next(iter(scenario.tasks.values()))
    assert task.target_module_name == "sc_port_1"
    assert task.port_name == "sc_port_base"
    assert metadata["right_sc_port"] == "sc_port_1"
    assert [rail.present for rail in scenario.task_board.sc_rails.values()] == [True, True]


def test_rejects_target_not_inserted() -> None:
    with pytest.raises(ValueError, match="target-nic-index"):
        _build_scenario(_args(task_family="sfp_to_nic", nic_cards=2, target_nic_index=3))

    with pytest.raises(ValueError, match="target-sc-index"):
        _build_scenario(_args(task_family="sc_to_sc", sc_ports=1, target_sc_index=1))
