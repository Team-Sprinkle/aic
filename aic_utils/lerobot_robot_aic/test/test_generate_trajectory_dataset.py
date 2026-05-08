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


def test_agent_success_only_controls_insertion_event_requirement(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["generation"]["policy"] = "agent"
    request["generation"]["expert_mode"] = "nominal"
    request["acceptance"]["success_only"] = False
    engine_path = gtd.write_engine_configs(request, tmp_path, 1)

    cmd = gtd.build_agent_generation_cmd(
        request=request,
        engine_config_path=engine_path,
        output_dir=tmp_path,
        target=1,
        max_attempts=1,
    )

    assert cmd[cmd.index("--require-insertion-event") + 1] == "false"


def test_agent_success_only_defaults_to_requiring_insertion_event(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["generation"]["policy"] = "agent"
    request["generation"]["expert_mode"] = "nominal"
    request["acceptance"].pop("success_only")
    engine_path = gtd.write_engine_configs(request, tmp_path, 1)

    cmd = gtd.build_agent_generation_cmd(
        request=request,
        engine_config_path=engine_path,
        output_dir=tmp_path,
        target=1,
        max_attempts=1,
    )

    assert cmd[cmd.index("--require-insertion-event") + 1] == "true"


def test_agent_near_gate_acceptance_yaml_uses_min_score_filter_threshold(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["generation"]["policy"] = "agent"
    request["generation"]["expert_mode"] = "nominal"
    request["acceptance"]["min_score"] = 90.0
    request["acceptance"]["stop_near_gate"] = {
        "max_lateral_error_m": 0.003,
        "max_axial_error_m": 0.006,
        "max_force_n": 20.0,
    }

    assert gtd.agent_filter_min_score(request) == 90.0


def test_agent_near_gate_acceptance_yaml_adds_generator_flags(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["generation"]["policy"] = "agent"
    request["generation"]["expert_mode"] = "nominal"
    request["acceptance"]["stop_near_gate"] = {
        "max_lateral_error_m": 0.003,
        "max_axial_error_m": 0.006,
        "max_force_n": 20.0,
    }
    engine_path = gtd.write_engine_configs(request, tmp_path, 1)

    cmd = gtd.build_agent_generation_cmd(
        request=request,
        engine_config_path=engine_path,
        output_dir=tmp_path,
        target=1,
        max_attempts=1,
    )

    assert "--allow-near-gate-acceptance" in cmd
    assert cmd[cmd.index("--near-gate-max-lateral-error-m") + 1] == "0.003"
    assert cmd[cmd.index("--near-gate-max-axial-error-m") + 1] == "0.006"
    assert cmd[cmd.index("--near-gate-max-force-n") + 1] == "20.0"
    assert "--near-gate-max-tcp-speed-mps" not in cmd
    assert "--near-gate-max-force-delta-n" not in cmd
    assert "--near-gate-selection-score" not in cmd
    env, _ = gtd.build_agent_generation_env(request)
    assert env["AIC_OFFICIAL_TEACHER_STOP_AT_NEAR_GATE"] == "true"
    assert env["AIC_OFFICIAL_TEACHER_NEAR_GATE_MAX_LATERAL_ERROR_M"] == "0.003"
    assert env["AIC_OFFICIAL_TEACHER_NEAR_GATE_MAX_AXIAL_ERROR_M"] == "0.006"
    assert env["AIC_OFFICIAL_TEACHER_TRACKING_GATE_MAX_LATERAL_ERROR_M"] == "0.003"


def test_legacy_near_gate_acceptance_yaml_still_adds_generator_flags(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["generation"]["policy"] = "agent"
    request["generation"]["expert_mode"] = "nominal"
    request["acceptance"]["near_gate"] = {
        "max_lateral_error_m": 0.003,
    }
    engine_path = gtd.write_engine_configs(request, tmp_path, 1)

    cmd = gtd.build_agent_generation_cmd(
        request=request,
        engine_config_path=engine_path,
        output_dir=tmp_path,
        target=1,
        max_attempts=1,
    )

    assert "--allow-near-gate-acceptance" in cmd


def test_stop_near_gate_and_legacy_near_gate_are_mutually_exclusive(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["acceptance"]["stop_near_gate"] = {}
    request["acceptance"]["near_gate"] = {}

    with pytest.raises(ValueError, match="stop_near_gate"):
        gtd.validate_request(request)


def test_sc_near_gate_tightens_sc_specific_tracking_gate(tmp_path: Path) -> None:
    request = base_request(tmp_path, task_family="sc_to_sc")
    request["generation"]["policy"] = "agent"
    request["generation"]["expert_mode"] = "nominal"
    request["acceptance"]["stop_near_gate"] = {
        "max_lateral_error_m": 0.004,
    }
    engine_path = gtd.write_engine_configs(request, tmp_path, 1)

    gtd.build_agent_generation_cmd(
        request=request,
        engine_config_path=engine_path,
        output_dir=tmp_path,
        target=1,
        max_attempts=1,
    )
    env, _ = gtd.build_agent_generation_env(request)

    assert env["AIC_OFFICIAL_TEACHER_SC_TRACKING_GATE_MAX_LATERAL_ERROR_M"] == "0.004"


def test_start_near_gate_places_target_gate_at_requested_distance(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["scene"]["start_near_gate"] = {"distance": 0.08}
    gtd.validate_request(request)

    trial = next(iter(gtd.generate_trials(request, 1).values()))
    metadata = trial["generated_metadata"]["start_near_gate"]

    assert metadata["achieved_distance"] == pytest.approx(0.08, abs=2e-6)
    target = gtd._target_gate_position_world(trial)
    reference = metadata["reference_tcp_position"]
    measured = math.dist(target, reference)
    assert measured == pytest.approx(0.08, abs=2e-6)


def test_start_near_gate_places_reference_by_axial_and_lateral_distance(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["scene"]["start_near_gate"] = {
        "axial_distance_m": 0.05,
        "lateral_distance_m": 0.01,
    }
    gtd.validate_request(request)

    trial = next(iter(gtd.generate_trials(request, 1).values()))
    metadata = trial["generated_metadata"]["start_near_gate"]

    assert metadata["achieved_axial_distance_m"] == pytest.approx(0.05, abs=2e-5)
    assert metadata["achieved_lateral_distance_m"] == pytest.approx(0.01, abs=2e-5)
    assert metadata["achieved_distance"] == pytest.approx(math.sqrt(0.05**2 + 0.01**2), abs=2e-5)


def test_start_near_gate_axial_lateral_uses_base_link_frame(tmp_path: Path) -> None:
    request = base_request(tmp_path, task_family="sc_to_sc")
    request["scene"]["start_near_gate"] = {
        "axial_distance_m": 0.035,
        "lateral_distance_m": 0.012,
    }
    gtd.validate_request(request)

    trial = next(iter(gtd.generate_trials(request, 1).values()))
    metadata = trial["generated_metadata"]["start_near_gate"]

    assert metadata["axes"] == "xyz"
    assert metadata["target_gate_position"][2] < 0.40
    assert metadata["achieved_axial_distance_m"] == pytest.approx(0.035, abs=2e-5)
    assert metadata["achieved_lateral_distance_m"] == pytest.approx(0.012, abs=2e-5)


def test_start_near_gate_omitted_keeps_regular_random_board_pose(tmp_path: Path) -> None:
    request = base_request(tmp_path)

    trial = next(iter(gtd.generate_trials(request, 1).values()))

    assert "generated_metadata" not in trial


def test_start_near_gate_requires_distance(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["scene"]["start_near_gate"] = {}

    with pytest.raises(ValueError, match="axial_distance_m"):
        gtd.validate_request(request)


def test_agent_max_planner_attempts_caps_expert_generator_attempts(tmp_path: Path) -> None:
    request = base_request(tmp_path)
    request["generation"]["policy"] = "agent"
    request["generation"]["expert_mode"] = "nominal"
    request["generation"]["max_attempts"] = 12
    request["generation"]["max_planner_attempts"] = 2
    engine_path = gtd.write_engine_configs(request, tmp_path, 2)

    cmd = gtd.build_agent_generation_cmd(
        request=request,
        engine_config_path=engine_path,
        output_dir=tmp_path,
        target=2,
        max_attempts=gtd.effective_max_replay_attempts(request, 12),
    )

    assert gtd.effective_max_replay_attempts(request, 12) == 2
    assert cmd[cmd.index("--max-total-attempts") + 1] == "2"


def test_agent_score_csv_uses_validation_score(tmp_path: Path) -> None:
    dataset_root = tmp_path / "attempt" / "dataset"
    dataset_root.mkdir(parents=True)
    record = {
        "validation": {
            "accepted": True,
            "acceptance_type": "near_gate",
            "score": 10.0,
        }
    }

    score_csv = gtd._agent_score_csv_for_dataset(
        output_dir=tmp_path,
        dataset_root=dataset_root,
        record=record,
        index=1,
    )

    assert "10.0" in score_csv.read_text(encoding="utf-8")


def test_near_gate_overlay_does_not_demote_passed_registry_entry() -> None:
    registry = {
        "settings": {
            "matrix_sc2sc_sc1_present0_target0_nic2": {
                "modes": {
                    "nominal": {
                        "status": "passed",
                        "best_mode_env": {"KEEP": "true"},
                    }
                }
            }
        }
    }

    gtd._merge_registry_overlay_entry(
        registry,
        {
            "suffix": "matrix_sc2sc_sc1_present0_target0_nic2",
            "mode": "nominal",
            "status": "near_gate_passed",
            "mode_env": {"NEAR_GATE": "true"},
            "score": 1.0,
        },
    )

    mode_entry = registry["settings"]["matrix_sc2sc_sc1_present0_target0_nic2"]["modes"]["nominal"]
    assert mode_entry["status"] == "passed"
    assert mode_entry["best_mode_env"] == {"KEEP": "true"}


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


def test_agent_policy_exports_per_suffix_registry_env_for_mixed_batches(tmp_path: Path) -> None:
    request = base_request(tmp_path, task_family="sc_to_sc")
    request["generation"]["policy"] = "agent"
    request["generation"]["expert_mode"] = "nominal"
    request["generation"]["env"] = {"GLOBAL_YAML_OVERRIDE": "yaml"}
    passed_suffix = "matrix_sc2sc_sc1_present0_target0_nic1"
    unknown_suffix = "matrix_sc2sc_sc1_present1_target1_nic1"
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "settings": {
                    passed_suffix: {
                        "modes": {
                            "nominal": {
                                "status": "passed",
                                "best_mode_env": {
                                    "AIC_OFFICIAL_TEACHER_SC_ENABLE_LIVE_Z_REPAIR": "true",
                                },
                            }
                        }
                    },
                    unknown_suffix: {
                        "modes": {
                            "nominal": {
                                "status": "unknown_not_logged",
                                "best_mode_env": {
                                    "SHOULD_NOT_EXPORT": "true",
                                },
                            }
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    request["generation"]["expert_setting_registry"] = str(registry)

    env, _mode_env = gtd.build_agent_generation_env(
        request,
        registry_suffixes=[passed_suffix, unknown_suffix],
    )

    exported = json.loads(env["AIC_EXPERT_REGISTRY_MODE_ENV_BY_SUFFIX"])
    assert env["AIC_EXPERT_TASK_FAMILY"] == "sc_to_sc"
    assert set(exported) == {passed_suffix}
    assert exported[passed_suffix]["AIC_OFFICIAL_TEACHER_SC_ENABLE_LIVE_Z_REPAIR"] == "true"
    assert exported[passed_suffix]["GLOBAL_YAML_OVERRIDE"] == "yaml"


def test_agent_overlay_records_concrete_trial_suffix(tmp_path: Path, monkeypatch) -> None:
    request = base_request(tmp_path, task_family="sc_to_sc")
    request["generation"]["policy"] = "agent"
    request["generation"]["expert_mode"] = "nominal"
    registry_suffix = "matrix_sc2sc_sc1_present0_target0_nic1"
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "settings": {
                    registry_suffix: {
                        "modes": {
                            "nominal": {
                                "status": "passed",
                                "best_mode_env": {
                                    "AIC_OFFICIAL_TEACHER_SC_ENABLE_LIVE_Z_REPAIR": "true",
                                },
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    overlay_dir = tmp_path / "overlays"
    request["generation"]["expert_setting_registry"] = str(registry)
    request["generation"]["expert_registry_overlay_dir"] = str(overlay_dir)
    output_dir = tmp_path / "out"
    trial_config = output_dir / "trials" / "trial_000001.yaml"
    trial_config.parent.mkdir(parents=True)
    trial_config.write_text(
        yaml.safe_dump(
            {
                "trials": {
                    "trial_000001": {
                        "scene": {
                            "task_board": {
                                "sc_rail_0": {"entity_present": True},
                                "nic_rail_1": {"entity_present": True},
                            }
                        },
                        "tasks": {"task_1": {"target_module_name": "sc_port_0"}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    summary_dir = output_dir / "agent_generation"
    summary_dir.mkdir(parents=True)
    (summary_dir / "generation_summary.json").write_text(
        json.dumps(
            {
                "accepted": 1,
                "stopped_reason": "target_reached",
                "records": [
                    {
                        "accepted": True,
                        "validation": {"score": 95.0, "reasons": []},
                        "replay_metrics": {"engine_config": str(trial_config)},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AIC_EXPERT_REGISTRY_OVERLAY_ID", "unit-host")

    path = gtd.write_expert_registry_overlay(
        request=request,
        output_dir=output_dir,
        agent_mode_env={},
    )

    assert path == overlay_dir / "unit-host.jsonl"
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert entries[0]["suffix"] == registry_suffix
    assert entries[0]["status"] == "passed"
    assert entries[0]["mode_env"]["AIC_OFFICIAL_TEACHER_SC_ENABLE_LIVE_Z_REPAIR"] == "true"
