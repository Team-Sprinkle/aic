"""Final reliability accounting must reject changed or incomplete evidence."""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/summarize_frozen_act_evaluation.py"
spec = importlib.util.spec_from_file_location("frozen_act_evaluation", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.fixture
def bundle(tmp_path):
    export = tmp_path / "act_selected_cuda0.pt"
    export.write_bytes(b"frozen model")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    export.with_suffix(".json").write_text(json.dumps({"checkpoint_dir": str(checkpoint)}))
    runtime = {key: None for key in report.RUNTIME_KEYS}
    runtime.update(diagnostic_ground_truth=False, container="test-container", n_action_steps=4,
                   policy_module="aic_example_policies.ros.RunACTTorchScript", act_torchscript=str(export))
    trials = []
    for index in range(2):
        name = f"trial_{index}"
        config = tmp_path / f"{name}.yaml"
        config.write_text(yaml.safe_dump({"trials": {name: {"scene": index}}}))
        score = tmp_path / f"{name}_score.yaml"
        score.write_text(yaml.safe_dump({name: {"tier_1": {"score": 1}, "tier_2": {"score": 10}, "tier_3": {"score": 75}}}))
        summary = tmp_path / f"{name}.json"
        summary.write_text(json.dumps({"evaluation_kind": "policy_rollout", "evaluation_purpose": "final_reliability",
            "runtime_settings": {**runtime, "engine_config_host": str(config)}, "runtime_source_sha256": {},
            "checkpoint": str(checkpoint), "evaluation_complete": True, "policy_ready": True,
            "engine_returncode": 0, "scoring_yaml": str(score)}))
        trials.append({"trial": name, "scene_task_sha256": str(index), "expected_summary": str(summary),
                       "container": "test-container", "config_sha256": report.digest(config)})
    (tmp_path / "frozen_selection.json").write_text(json.dumps({
        "file_sha256": {export.name: report.digest(export)}, "trials": trials, "expected_trials": 2,
        "target_insertions": 2, "settings_inherited_from_development": runtime, "runtime_source_sha256": {}}))
    return tmp_path


def test_success_requires_all_declared_trials(bundle):
    assert report.summarize(bundle)["target_met"]
    (bundle / "trial_1.json").unlink()
    result = report.summarize(bundle)
    assert result["insertions"] == result["completed_trials"] == 1
    assert result["insertion_fraction_of_declared_trials"] == .5
    assert not result["target_met"]


def test_changed_model_rejected(bundle):
    (bundle / "act_selected_cuda0.pt").write_bytes(b"different model")
    with pytest.raises(ValueError, match="Frozen file changed"):
        report.summarize(bundle)


@pytest.mark.parametrize("change", ["privileged", "runtime", "wrong_trial", "incomplete"])
def test_final_evidence_contract(bundle, change):
    path = bundle / "trial_0.json"
    summary = json.loads(path.read_text())
    if change == "privileged":
        summary["runtime_settings"]["diagnostic_ground_truth"] = True
    elif change == "runtime":
        summary["runtime_settings"]["n_action_steps"] = 1
    elif change == "wrong_trial":
        score = Path(summary["scoring_yaml"])
        data = yaml.safe_load(score.read_text())
        data["different_trial"] = data.pop("trial_0")
        score.write_text(yaml.safe_dump(data))
    else:
        summary["engine_returncode"] = 1
    path.write_text(json.dumps(summary))
    if change == "incomplete":
        result = report.summarize(bundle)
        assert result["completed_trials"] == 1 and not result["target_met"]
    else:
        with pytest.raises(ValueError):
            report.summarize(bundle)
