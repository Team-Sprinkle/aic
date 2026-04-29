from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generate_trajectory_dataset.py"
)
spec = importlib.util.spec_from_file_location("generate_trajectory_dataset", SCRIPT)
gtd = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(gtd)


def base_request(tmp_path: Path, task_family: str = "sfp_to_nic") -> dict:
    request = {
        "root_dir": str(tmp_path / "outputs"),
        "task_family": task_family,
        "suffix": "unit",
        "generation": {
            "target_accepted_trajectories": 2,
            "max_attempts": 3,
            "policy": "cheatcode",
            "seed": 7,
            "append_if_exists": True,
        },
        "acceptance": {"success_only": True, "min_score": 90.0},
        "scene": {},
    }
    if task_family == "sfp_to_nic":
        request["scene"]["nic_cards"] = {"count": 1}
    else:
        request["scene"]["sc_ports"] = {"count": 2}
    return request


def test_yaml_parsing(tmp_path: Path) -> None:
    path = tmp_path / "request.yaml"
    path.write_text(yaml.safe_dump(base_request(tmp_path)), encoding="utf-8")
    loaded = gtd.load_request(path)
    gtd.validate_request(loaded)
    assert loaded["task_family"] == "sfp_to_nic"


def test_output_directory_derivation(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    out = gtd.derive_output_dir(request)
    assert out == (
        tmp_path
        / "outputs"
        / "sfp_to_nic"
        / "cheatcode"
        / "nic_cards_1"
        / "n2__unit"
    )
    request["scene"]["nic_cards"]["count"] = [1, 2]
    assert "nic_cards_mixed" in str(gtd.derive_output_dir(request))


def test_sample_value_scalar_list_and_minmax() -> None:
    rng = gtd.random.Random(1)
    assert gtd.sample_value(4, None, rng) == 4
    assert gtd.sample_value(["a"], None, rng) == "a"
    assert gtd.sample_value({"min": 2.0, "max": 2.0}, None, rng) == 2.0
    val = gtd.sample_value({"min": 1.0, "max": 3.0}, None, rng)
    assert 1.0 <= val <= 3.0


def test_degree_to_radian_conversion(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["scene"]["board"] = {"yaw_deg": {"min": 180.0, "max": 180.0}}
    trial = next(iter(gtd.generate_trials(request, 1).values()))
    assert trial["scene"]["task_board"]["pose"]["yaw"] == pytest.approx(math.pi, abs=1e-5)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("nic_cards", "yaw_deg", {"min": -11.0, "max": 0.0}),
        ("fixture_mounts", "yaw_deg", {"min": -61.0, "max": 0.0}),
        ("nic_cards", "translation", {"min": -0.03, "max": 0.0}),
        ("sc_ports", "translation", {"min": -0.07, "max": 0.0}),
    ],
)
def test_validation_rejects_out_of_range(
    tmp_path: Path, section: str, field: str, value: dict
) -> None:
    request = base_request(tmp_path)
    request["scene"].setdefault(section, {})
    request["scene"][section][field] = value
    with pytest.raises(ValueError):
        gtd.validate_override_limits(request)


def test_exact_nic_count_behavior(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["scene"]["nic_cards"] = {"count": 3}
    trial = next(iter(gtd.generate_trials(request, 1).values()))
    board = trial["scene"]["task_board"]
    assert sum(1 for rail in gtd.NIC_RAILS if board[rail]["entity_present"]) == 3


def test_exact_sc_count_behavior(tmp_path: Path) -> None:
    request = base_request(tmp_path, task_family="sc_to_sc")
    request["scene"]["sc_ports"] = {"count": 1}
    trial = next(iter(gtd.generate_trials(request, 1).values()))
    board = trial["scene"]["task_board"]
    assert sum(1 for rail in gtd.SC_RAILS if board[rail]["entity_present"]) == 1


def test_dry_run_creates_expected_files(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["suffix"] = "smoke_test"
    request_path = tmp_path / "request.yaml"
    request_path.write_text(yaml.safe_dump(request), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--request-yaml",
            str(request_path),
            "--dry-run",
            "--skip-recording",
            "--target-accepted-override",
            "2",
            "--max-attempts-override",
            "3",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out = (
        tmp_path
        / "outputs"
        / "sfp_to_nic"
        / "cheatcode"
        / "nic_cards_1"
        / "n2__smoke_test"
    )
    assert (out / "request.yaml").exists()
    assert (out / "engine_config.yaml").exists()
    assert (out / "trials" / "trial_000001.yaml").exists()
    assert (out / "generation_summary.json").exists()
