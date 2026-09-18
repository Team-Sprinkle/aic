"""No Docker or ROS: regressions for runtime evaluation bookkeeping."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/evaluate_act_checkpoints_runtime.py"
spec = importlib.util.spec_from_file_location("runtime_evaluator", SCRIPT)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def setup_run(tmp_path):
    config = tmp_path / "aic_engine/config/sample_config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("trials:\n  trial_1: {}\n  trial_2: {}\n")
    run = tmp_path / "run"
    run.mkdir()
    argv = ["--run-dir", str(run), "--workspace-host", str(tmp_path),
            "--command-mode", "delta_pose", "--once-existing"]
    return run, argv


def write_score(path, count=2):
    data = {"total": count}
    for i in range(1, count + 1):
        data[f"trial_{i}"] = {"tier_1": {"score": 1}, "tier_2": {"score": 0}, "tier_3": {"score": 0}}
    path.write_text(yaml.safe_dump(data))


def test_empty_checkpoint_selection_fails_before_runtime(tmp_path, monkeypatch):
    _, argv = setup_run(tmp_path)
    monkeypatch.setattr(runtime, "evaluate_checkpoint", lambda *a: pytest.fail("Runtime must not launch"))
    assert runtime.main(argv) == 2


def test_command_mode_must_be_explicit():
    with pytest.raises(SystemExit) as exc:
        runtime.parse_args(["--run-dir", "."])
    assert exc.value.code == 2


def test_act_deadband_override_changes_evaluation_identity(tmp_path):
    run, argv = setup_run(tmp_path)
    checkpoint = run / "checkpoint.pt"
    checkpoint.write_text("mock")
    default = runtime.parse_args(argv)
    runtime.prepare_args(default)
    exact = runtime.parse_args(argv + ["--translation-deadband", "0", "--rotation-deadband", "0"])
    runtime.prepare_args(exact)
    assert exact.translation_deadband == exact.rotation_deadband == 0
    assert runtime.evaluation_signature(checkpoint, default) != runtime.evaluation_signature(checkpoint, exact)
    with pytest.raises(SystemExit):
        runtime.parse_args(argv + ["--translation-deadband", "-1"])
    bgr = runtime.parse_args(argv + ["--image-channel-order", "bgr"])
    runtime.prepare_args(bgr)
    assert runtime.evaluation_signature(checkpoint, default) != runtime.evaluation_signature(checkpoint, bgr)


def test_absolute_action_export_requires_matching_execution_mode(tmp_path):
    run, argv = setup_run(tmp_path)
    export = run / "actor.pt"
    export.write_text("mock")
    normalizer = run / "normalizer"
    normalizer.mkdir()
    (normalizer / "policy_preprocessor_step_3_normalizer_processor.safetensors").write_text("mock")
    export.with_suffix(".json").write_text(json.dumps({"checkpoint_dir": str(normalizer),
        "chunk_size": 8, "action_representation": "absolute_pose"}))
    args = runtime.parse_args(argv + ["--act-torchscript", str(export)])
    with pytest.raises(ValueError, match="representation"):
        runtime.prepare_args(args)
    args = runtime.parse_args(argv + ["--act-torchscript", str(export), "--command-mode", "absolute_pose"])
    runtime.prepare_args(args)


def test_same_container_cannot_be_evaluated_concurrently(tmp_path, monkeypatch):
    run, argv = setup_run(tmp_path)
    args = runtime.parse_args(argv)
    runtime.prepare_args(args)
    checkpoint = run / "actor.pt"
    checkpoint.write_text("mock")
    def while_locked(checkpoint, args):
        with pytest.raises(RuntimeError, match="Another evaluation"):
            runtime.evaluate_checkpoint(checkpoint, args)
        return {"lock_checked": True}
    monkeypatch.setattr(runtime, "_evaluate_checkpoint_locked", while_locked)
    assert runtime.evaluate_checkpoint(checkpoint, args)["lock_checked"]


def test_changed_runtime_source_invalidates_reuse(tmp_path):
    run, argv = setup_run(tmp_path)
    args = runtime.parse_args(argv)
    runtime.prepare_args(args)
    checkpoint = run / "actor.pt"
    checkpoint.write_text("mock")
    source = tmp_path / "aic_example_policies/aic_example_policies/ros/RunACTTorchScript.py"
    source.parent.mkdir(parents=True)
    source.write_text("before")
    before = runtime.evaluation_signature(checkpoint, args)
    source.write_text("after")
    assert runtime.evaluation_signature(checkpoint, args) != before


def test_privileged_corrective_collection_is_explicit_and_separate(tmp_path):
    run, argv = setup_run(tmp_path)
    with pytest.raises(ValueError, match="Privileged geometry"):
        runtime.prepare_args(runtime.parse_args(argv + ["--diagnostic-ground-truth"]))
    collector = argv + ["--policy-module", "aic_example_policies.ros.CollectCorrectiveCheatCode"]
    with pytest.raises(ValueError, match="Corrective collection requires"):
        runtime.prepare_args(runtime.parse_args(collector))
    args = runtime.parse_args(collector + ["--diagnostic-ground-truth", "--corrective-data-dir", str(run / "data"),
                                          "--checkpoint-glob", "collection_config.json"])
    runtime.prepare_args(args)
    config = run / "collection_config.json"
    config.write_text('{"kind": "expert_collection_configuration"}')
    assert runtime.iter_checkpoints(args) == [config]
    ordinary = runtime.parse_args(argv + ["--checkpoint-glob", "collection_config.json"])
    runtime.prepare_args(ordinary)
    assert runtime.iter_checkpoints(ordinary) == []


@pytest.mark.parametrize("ready,returncode,trials,expected", [
    (True, 0, 2, True), (False, 0, 2, False), (True, 1, 2, False), (True, 0, 1, False),
])
def test_completion_requires_all_trials_and_clean_runtime(tmp_path, ready, returncode, trials, expected):
    score = tmp_path / "scoring.yaml"
    write_score(score, trials)
    summary = {"policy_ready": ready, "engine_returncode": returncode, "scoring_yaml": str(score)}
    # A complete evaluation of an unsuccessful policy is still a valid measurement.
    assert runtime.evaluation_complete(summary, ["trial_1", "trial_2"]) is expected


def test_failed_summary_is_retried_and_failed_runtime_exits_nonzero(tmp_path, monkeypatch):
    run, argv = setup_run(tmp_path)
    checkpoint = run / "checkpoints/175000/pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_text("test checkpoint")
    marker = run / "runtime_eval/175000/eval_summary.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"policy_ready": false}')
    attempts = []
    def fail(path, args):
        attempts.append(path)
        return {"policy_ready": False}
    monkeypatch.setattr(runtime, "evaluate_checkpoint", fail)
    assert runtime.main(argv) == 1
    assert attempts == [checkpoint]


def test_complete_summary_reused_only_with_same_settings(tmp_path, monkeypatch):
    run, argv = setup_run(tmp_path)
    checkpoint = run / "checkpoints/175000/pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_text("test checkpoint")
    args = runtime.parse_args(argv)
    runtime.prepare_args(args)
    marker = run / "runtime_eval/175000/eval_summary.json"
    marker.parent.mkdir(parents=True)
    score = marker.parent / "scoring.yaml"
    write_score(score)
    marker.write_text(json.dumps({"policy_ready": True, "engine_returncode": 0,
                                 "scoring_yaml": str(score),
                                 "evaluation_signature": runtime.evaluation_signature(checkpoint, args)}))
    calls = []
    monkeypatch.setattr(runtime, "evaluate_checkpoint", lambda *a: calls.append(a) or {})
    assert runtime.main(argv) == 0
    assert not calls
    assert runtime.main(argv + ["--n-action-steps", "2"]) == 1
    assert len(calls) == 1


def test_retry_preserves_previous_attempts_and_does_not_use_stale_score(tmp_path, monkeypatch):
    run, argv = setup_run(tmp_path)
    args = runtime.parse_args(argv + ["--eval-attempts", "2", "--retry-delay-sec", "0"])
    runtime.prepare_args(args)
    checkpoint = run / "checkpoint.pt"
    checkpoint.write_text("mock")
    parent = run / "runtime_eval/checkpoint"
    parent.mkdir(parents=True)
    write_score(parent / "scoring.yaml")  # An old successful score is irrelevant.
    previous = parent / "attempt_0001"
    previous.mkdir()
    (previous / "evidence.txt").write_text("preserve me")
    paths = []
    def fake_attempt(checkpoint, args, directory, logs, attempt):
        paths.append(directory)
        directory.mkdir()
        return {"policy_ready": False, "scoring_yaml": None}
    monkeypatch.setattr(runtime, "evaluate_checkpoint_once", fake_attempt)
    result = runtime.evaluate_checkpoint(checkpoint, args)
    assert not runtime.evaluation_complete(result, args.expected_trial_names)
    assert [p.name for p in paths] == ["attempt_0002", "attempt_0003"]
    assert (previous / "evidence.txt").read_text() == "preserve me"


def test_restart_failure_is_recorded_without_starting_a_policy(tmp_path, monkeypatch):
    run, argv = setup_run(tmp_path)
    args = runtime.parse_args(argv)
    runtime.prepare_args(args)
    checkpoint = run / "checkpoint.pt"
    checkpoint.write_text("mock")
    monkeypatch.setattr(runtime, "restart_container", lambda *a: 1)
    monkeypatch.setattr(runtime, "start_long_container_bash", lambda *a: pytest.fail("Must not start after restart failure"))
    result = runtime.evaluate_checkpoint_once(checkpoint, args, run / "attempt", run / "attempt/logs", 1)
    assert result["evaluation_complete"] is False
    assert result["failure_reason"]


def test_cleanup_failure_keeps_attempt_summary(tmp_path, monkeypatch):
    run, argv = setup_run(tmp_path)
    args = runtime.parse_args(argv + ["--sim-wait-sec", "0"])
    runtime.prepare_args(args)
    checkpoint = run / "checkpoint.pt"
    checkpoint.write_text("mock")
    restarts = []
    monkeypatch.setattr(runtime, "restart_container", lambda *a: restarts.append(a) or 0)
    monkeypatch.setattr(runtime, "start_long_container_bash", lambda *a: object())
    monkeypatch.setattr(runtime, "wait_for_policy_ready", lambda *a: False)
    def fail_cleanup(proc):
        raise OSError("mock cleanup failure")
    monkeypatch.setattr(runtime, "stop_process", fail_cleanup)
    result = runtime.evaluate_checkpoint_once(checkpoint, args, run / "attempt", run / "attempt/logs", 1)
    assert len(restarts) == 2
    assert "cleanup" in result["failure_reason"]
    assert (run / "attempt/eval_summary.json").is_file()
    assert result["evaluation_complete"] is False


def test_observation_delta_reference_is_explicit_and_changes_identity(tmp_path):
    run, argv = setup_run(tmp_path)
    checkpoint = run / "checkpoint.pt"
    checkpoint.write_text("mock")
    (run / "policy_preprocessor_step_3_normalizer_processor.safetensors").write_text("mock")
    checkpoint.with_suffix(".json").write_text(json.dumps({"checkpoint_dir": str(run),
        "chunk_size": 8, "action_representation": "delta_pose"}))
    argv += ["--act-torchscript", str(checkpoint)]
    default = runtime.parse_args(argv + ["--policy-module", "aic_example_policies.ros.RunACTTorchScript"])
    observed = runtime.parse_args(argv + ["--policy-module", "aic_example_policies.ros.RunACTTorchScript", "--delta-pose-reference", "observation"])
    runtime.prepare_args(default)
    runtime.prepare_args(observed)
    assert runtime.evaluation_signature(checkpoint, default) != runtime.evaluation_signature(checkpoint, observed)
    invalid = runtime.parse_args(argv + ["--command-mode", "absolute_pose", "--delta-pose-reference", "observation"])
    with pytest.raises(ValueError, match="Observation-referenced"):
        runtime.prepare_args(invalid)


@pytest.mark.parametrize("flag,value", [("--artifact-host-root", "/tmp/artifacts"),
                                        ("--artifact-container-root", "/artifacts")])
def test_artifact_mapping_requires_both_roots(flag, value):
    with pytest.raises(SystemExit):
        runtime.parse_args(["--run-dir", ".", "--command-mode", "delta_pose", flag, value])


def artifact_run(tmp_path):
    workspace = tmp_path / "workspace"
    _, argv = setup_run(workspace)
    artifacts = tmp_path / "nvme"
    run = artifacts / "run"; run.mkdir(parents=True)
    config = artifacts / "scene.yaml"
    config.write_text("trials:\n  trial_1: {}\n  trial_2: {}\n")
    argv += ["--run-dir", str(run), "--workspace-container", "/repo",
             "--artifact-host-root", str(artifacts), "--artifact-container-root", "/artifacts",
             "--engine-config", "/artifacts/scene.yaml"]
    return workspace, artifacts, run, argv


@pytest.mark.parametrize("config_spelling", ["host", "container"])
def test_artifact_mapping_preserves_code_workspace_and_maps_config(tmp_path, config_spelling):
    workspace, artifacts, run, argv = artifact_run(tmp_path)
    if config_spelling == "host":
        argv += ["--engine-config", str(artifacts / "scene.yaml")]
    args = runtime.parse_args(argv); runtime.prepare_args(args)
    assert runtime.host_to_container(workspace / "aic_model", args) == "/repo/aic_model"
    assert runtime.host_to_container(run / "model.pt", args) == "/artifacts/run/model.pt"
    assert args.engine_config == "/artifacts/scene.yaml"
    assert args.engine_config_host == artifacts / "scene.yaml"
    assert args.expected_trial_names == ["trial_1", "trial_2"]
    with pytest.raises(ValueError, match="outside"):
        runtime.host_to_container(tmp_path / "unmounted/model.pt", args)
    with pytest.raises(ValueError, match="outside"):
        runtime.mapped_input_to_host(Path("/unmounted/scene.yaml"), args)
    escaped = artifacts / "escape"; escaped.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        runtime.mapped_input_to_host(Path("/artifacts/escape/secret"), args)
    with pytest.raises(ValueError, match="outside"):
        runtime.host_to_container(escaped / "secret", args)


def test_unmapped_run_and_normalizer_fail_preflight(tmp_path):
    workspace, artifacts, run, argv = artifact_run(tmp_path)
    outside = tmp_path / "outside"; outside.mkdir()
    with pytest.raises(ValueError, match="outside"):
        runtime.prepare_args(runtime.parse_args(argv + ["--run-dir", str(outside)]))
    export = run / "model.pt"; export.write_text("mock")
    export.with_suffix(".json").write_text(json.dumps({"checkpoint_dir": str(outside),
        "chunk_size": 8, "action_representation": "delta_pose"}))
    (outside / "policy_preprocessor_step_3_normalizer_processor.safetensors").write_text("mock")
    with pytest.raises(ValueError, match="outside"):
        runtime.prepare_args(runtime.parse_args(argv + ["--act-torchscript", str(export)]))
    args = runtime.parse_args(argv); runtime.prepare_args(args)
    (run / "escape.pt").symlink_to(outside / "policy_preprocessor_step_3_normalizer_processor.safetensors")
    with pytest.raises(ValueError, match="outside"):
        runtime.iter_checkpoints(runtime.parse_args(argv + ["--checkpoint-glob", "*.pt"]))


@pytest.mark.parametrize("container_root", ["relative/artifacts", "/", "/repo"])
def test_invalid_or_conflicting_artifact_container_root_fails(tmp_path, container_root):
    _, _, _, argv = artifact_run(tmp_path)
    with pytest.raises(ValueError):
        runtime.prepare_args(runtime.parse_args(argv + ["--artifact-container-root", container_root]))


def test_artifact_shell_paths_and_normalizer_are_mapped_without_moving_code(tmp_path, monkeypatch):
    workspace, artifacts, run, argv = artifact_run(tmp_path)
    export = run / "model.pt"; export.write_text("mock")
    (run / "policy_preprocessor_step_3_normalizer_processor.safetensors").write_text("mock")
    export.with_suffix(".json").write_text(json.dumps({"checkpoint_dir": str(run),
        "chunk_size": 8, "action_representation": "delta_pose", "image_channel_order": "rgb"}))
    args = runtime.parse_args(argv + ["--act-torchscript", str(export), "--sim-wait-sec", "0",
                                    "--policy-module", "aic_example_policies.ros.RunACTTorchScript", "--record-rollout"])
    runtime.prepare_args(args)
    scripts = []
    monkeypatch.setattr(runtime, "restart_container", lambda *a: 0)
    monkeypatch.setattr(runtime, "start_long_container_bash", lambda args, script, log: scripts.append(script) or object())
    monkeypatch.setattr(runtime, "stop_process", lambda *a: None)
    monkeypatch.setattr(runtime, "wait_for_policy_ready", lambda *a: True)
    def run_engine(args, script, *extra):
        scripts.append(script)
        write_score(run / "attempt/scoring.yaml")
        return 0, None
    monkeypatch.setattr(runtime, "run_engine_monitoring_sim", run_engine)
    summary = runtime.evaluate_checkpoint_once(export, args, run / "attempt", run / "attempt/logs", 1)
    assert summary["evaluation_complete"]
    assert summary["checkpoint_container"] == "/artifacts/run/model.pt"
    assert summary["eval_dir_container"] == "/artifacts/run/attempt"
    assert all("cd /repo" in script for script in scripts)
    policy, engine = scripts[1:]
    assert "AIC_ACT_NORMALIZER_PATH=/artifacts/run/policy_preprocessor_step_3_normalizer_processor.safetensors" in policy
    assert "AIC_ACT_TORCHSCRIPT=/artifacts/run/model.pt" in policy
    assert "AIC_POLICY_RECORD_DIR=/artifacts/run/attempt/rollout" in policy
    assert "PYTHONPATH=/repo/.pixi/envs/default/lib/python3.12/site-packages:/repo/aic_model:" in policy
    assert "AIC_CHECKOUT_PYTHONPATH=/repo/aic_model:/repo/aic_example_policies" in policy
    assert "AIC_RESULTS_DIR=/artifacts/run/attempt" in engine
    assert "config_file_path:=/artifacts/scene.yaml" in engine
    assert args.image_channel_order == "rgb"
