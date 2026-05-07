from __future__ import annotations

import importlib.util
import json
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
TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "config" / "data_generation_templates"
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


def test_packaged_templates_default_to_agent_and_tracked_registry() -> None:
    for name in (
        "sfp_to_nic_minimal.yaml",
        "sfp_to_nic_maximal.yaml",
        "sc_to_sc_minimal.yaml",
        "sc_to_sc_maximal.yaml",
    ):
        request = gtd.load_request(TEMPLATE_DIR / name)
        gtd.validate_request(request)
        assert request["generation"]["policy"] == "agent"
        assert request["generation"]["expert_mode"] == "nominal"
        assert request["generation"]["auto_improve_on_failure"] is True
        assert request["generation"]["write_expert_registry_overlay"] is True

    assert gtd.DEFAULT_EXPERT_SETTING_REGISTRY == (
        Path(__file__).resolve().parents[3]
        / "aic_utils"
        / "lerobot_robot_aic"
        / "config"
        / "expert_setting_registry.json"
    )


@pytest.mark.parametrize(
    ("expert_mode", "flag"),
    [
        ("nominal", "--nominal"),
        ("nominalrecovery", "--nominalrecovery"),
        ("recovery", "--recovery"),
    ],
)
def test_agent_expert_mode_yaml_selects_expert_generator_flag(
    tmp_path: Path, expert_mode: str, flag: str
) -> None:
    request = base_request(tmp_path)
    request["generation"]["policy"] = "agent"
    request["generation"]["expert_mode"] = expert_mode
    engine_path = gtd.write_engine_configs(request, tmp_path, 1)

    cmd = gtd.build_agent_generation_cmd(
        request=request,
        engine_config_path=engine_path,
        output_dir=tmp_path,
        target=1,
        max_attempts=1,
    )

    assert flag in cmd


def test_infers_exact_registry_suffix_from_generated_sfp_trial(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["suffix"] = "batch"
    request["generation"]["max_attempts"] = 1
    request["scene"]["nic_cards"] = {
        "count": 1,
        "rails": ["nic_rail_3"],
        "target_card": 3,
        "target_port": "sfp_port_1",
    }
    engine_path = gtd.write_engine_configs(request, tmp_path, 1)

    assert gtd.infer_registry_suffixes_from_engine_config(engine_path, "sfp_to_nic") == [
        "matrix_sfp2nic_cards1_present3_target3_port1"
    ]


def test_registry_overlay_updates_mode_env(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    overlay_dir = tmp_path / "overlays"
    suffix = "matrix_sfp2nic_cards1_present3_target3_port1"
    registry.write_text(
        json.dumps(
            {
                "settings": {
                    suffix: {
                        "modes": {
                            "nominal": {
                                "status": "unknown_not_logged",
                                "best_score": None,
                                "best_mode_env": None,
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    overlay_dir.mkdir()
    (overlay_dir / "ec2-a.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "aic_expert_registry_overlay/v1",
                "suffix": suffix,
                "mode": "nominal",
                "status": "passed",
                "score": 96.0,
                "mode_env": {"AIC_OFFICIAL_TEACHER_PIN_INSERTION_TARGET": "false"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    request = base_request(tmp_path)
    request["generation"]["policy"] = "agent"
    request["generation"]["expert_setting_registry"] = str(registry)
    request["generation"]["expert_registry_overlay_dir"] = str(overlay_dir)
    request["generation"]["_inferred_expert_registry_suffix"] = suffix

    env = gtd.expert_registry_mode_env(request)

    assert env["AIC_OFFICIAL_TEACHER_PIN_INSERTION_TARGET"] == "false"


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


def test_minimal_sfp_request_defaults_sc_ports_absent(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    trial = next(iter(gtd.generate_trials(request, 1).values()))
    board = trial["scene"]["task_board"]
    assert all(not board[rail]["entity_present"] for rail in gtd.SC_RAILS)


def test_packaged_minimal_templates_cover_official_task_choices() -> None:
    template_dir = Path(__file__).resolve().parents[1] / "config" / "data_generation_templates"

    sfp = gtd.load_request(template_dir / "sfp_to_nic_minimal.yaml")
    gtd.validate_request(sfp)
    assert "target_port" not in sfp["scene"]["nic_cards"]
    ports = set()
    for seed in range(20):
        sfp["generation"]["seed"] = seed
        trial = next(iter(gtd.generate_trials(sfp, 1).values()))
        ports.add(trial["tasks"]["task_1"]["port_name"])
    assert ports == {"sfp_port_0", "sfp_port_1"}

    sc = gtd.load_request(template_dir / "sc_to_sc_minimal.yaml")
    gtd.validate_request(sc)
    assert sc["scene"]["sc_ports"]["count"] == 2
    assert sc["scene"]["nic_cards"]["count"] == [1, 2, 3, 4, 5]


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


def test_agent_policy_dry_run_uses_expert_generator(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["generation"]["policy"] = "agent"
    request["generation"]["expert_mode"] = "nominal"
    request["generation"]["per_trial_timeout_sec"] = 123
    request["suffix"] = "agent_unit"
    request_path = tmp_path / "request_agent.yaml"
    request_path.write_text(yaml.safe_dump(request), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--request-yaml",
            str(request_path),
            "--dry-run",
            "--target-accepted-override",
            "1",
            "--max-attempts-override",
            "2",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "scripts/generate_expert_trajectories.py" in result.stdout
    assert "--nominal" in result.stdout
    assert "--per-trial-timeout-sec 123" in result.stdout
    assert "--planner-recorder-drain-sec 45" in result.stdout
    out = (
        tmp_path
        / "outputs"
        / "sfp_to_nic"
        / "agent"
        / "nic_cards_1"
        / "n1__agent_unit"
    )
    assert (out / "engine_config.yaml").exists()


def test_agent_policy_skips_registry_exhausted_setting(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["generation"]["policy"] = "agent"
    request["generation"]["expert_mode"] = "nominal"
    request["suffix"] = "matrix_sfp2nic_cards1_present0_target0_port0"
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "settings": {
                    request["suffix"]: {
                        "modes": {
                            "nominal": {
                                "status": "skipped_exhausted",
                                "last_reason": "7 attempts exhausted",
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    request["generation"]["expert_setting_registry"] = str(registry)
    request_path = tmp_path / "request_agent_skip.yaml"
    request_path.write_text(yaml.safe_dump(request), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--request-yaml",
            str(request_path),
            "--target-accepted-override",
            "1",
            "--max-attempts-override",
            "2",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Skipped expert dataset generation" in result.stdout
    out = tmp_path / "outputs" / "sfp_to_nic" / "agent" / "nic_cards_1" / f"n1__{request['suffix']}"
    summary = json.loads((out / "generation_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "skipped"
    assert summary["number_attempted"] == 0


def test_agent_policy_uses_passed_registry_env_before_yaml_overrides(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["generation"]["policy"] = "agent"
    request["generation"]["expert_mode"] = "nominal"
    request["generation"]["env"] = {
        "AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_SEC": "0.8",
        "CUSTOM_AGENT_ENV": "yaml",
    }
    request["suffix"] = "matrix_sfp2nic_cards1_present0_target0_port0"
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "settings": {
                    request["suffix"]: {
                        "modes": {
                            "nominal": {
                                "status": "passed",
                                "best_mode_env": {
                                    "AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_SEC": "0.6",
                                    "AIC_OFFICIAL_TEACHER_PIN_INSERTION_TARGET": "false",
                                },
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    request["generation"]["expert_setting_registry"] = str(registry)

    _env, mode_env = gtd.build_agent_generation_env(request)

    assert mode_env["AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_SEC"] == "0.8"
    assert mode_env["AIC_OFFICIAL_TEACHER_PIN_INSERTION_TARGET"] == "false"
    assert mode_env["CUSTOM_AGENT_ENV"] == "yaml"
