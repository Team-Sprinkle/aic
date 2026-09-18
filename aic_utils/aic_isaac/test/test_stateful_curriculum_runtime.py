"""CPU-only accounting tests and shell orchestration with a fake simulator process."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "aic_utils/aic_isaac/scripts/stateful_curriculum_runtime.py"
spec = importlib.util.spec_from_file_location("stateful_runtime", SCRIPT)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_terminal_outcomes_are_counted_once_per_episode_including_timeouts():
    tracker = runtime.EpisodeTracker(2)
    def step(number, terminated, truncated, success, reasons):
        return tracker.observe(step=number, terminated=terminated, truncated=truncated, success=success,
                               metadata=[{"episode_id": "a"}, {"episode_id": "b"}], reasons=reasons)
    assert step(1, [False, False], [False, False], [False, False], [[], []]) == []
    first = step(2, [True, False], [False, True], [True, False], [["target_success"], ["time_out"]])
    assert [x["success"] for x in first] == [True, False]
    assert [x["length"] for x in first] == [2, 2]
    # Same config starts another episode; prior success must not carry over.
    second = step(3, [True, False], [False, False], [False, False], [["lateral_bypass_failure"], []])
    assert second[0]["episode_index"] == 1
    assert second[0]["length"] == 1
    assert tracker.completed == 3
    assert tracker.succeeded == 1


def test_simultaneous_failure_overrides_success():
    tracker = runtime.EpisodeTracker(1)
    result = tracker.observe(step=1, terminated=[True], truncated=[False], success=[True],
                             metadata=[{}], reasons=[["target_success", "lateral_bypass_failure"]])
    assert result[0]["success"] is False


@pytest.mark.parametrize("successes,level,failures,expected", [
    (1, 1, 0, (1, 1, "held")), (8, 1, 0, (2, 0, "promoted")),
    (7, 1, 2, (0, 0, "demoted")), (0, 0, 2, (0, 0, "held_at_level_zero")),
])
def test_promotion_uses_episode_fraction_and_configured_demotion(successes, level, failures, expected):
    summary = {"evaluation_complete": True, "completed_episodes": 10, "successful_episodes": successes}
    assert runtime.promotion_decision(summary, level=level, level_count=3, failures=failures,
                                      success_rate=0.8, demote_after=3) == expected


def make_run(path, successes=(True, False)):
    tracker = runtime.EpisodeTracker(1)
    rows = []
    for step, success in enumerate(successes, 1):
        outcomes = tracker.observe(step=step, terminated=[success], truncated=[not success], success=[success],
                                   metadata=[{"episode_id": f"ep{step}"}],
                                   reasons=[["target_success" if success else "time_out"]])
        rows.append({"step": step, "episode_outcomes": outcomes, "completed_episodes_total": tracker.completed,
                     "target_action_guide_collect_blend_effective": 0.0,
                     "insertion_action_guard_applied_fraction": 0.0, "executed_minus_actor_l1_mean": 0.0})
    config = {"args": {"actor_exploration_noise_std": 0, "terminate_on_target_success": True,
        "target_reward_consistency_body": "sfp_module_link", "target_success_axial_threshold": .0005,
        "target_success_lateral_threshold": .0005, "target_success_orientation_threshold": .03,
        "target_success_consistency_axial_threshold": .001, "target_success_consistency_lateral_threshold": .0015}, "result": {
        "stop_reason": "max_completed_episodes", "steps_completed": len(rows), "updates_done": 0,
        "episodes_completed": len(rows)}}
    (path / "train_config.json").write_text(json.dumps(config))
    (path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows, config


def test_summary_requires_complete_episode_coverage(tmp_path):
    make_run(tmp_path)
    summary = runtime.summarize_run(tmp_path, min_episodes=2, evaluation=True, expected_ids=["ep1", "ep2"])
    assert summary["success_rate"] == 0.5
    assert summary["evaluation_complete"] is True
    incomplete = runtime.summarize_run(tmp_path, min_episodes=3, expected_ids=["ep3"])
    assert incomplete["evaluation_complete"] is False
    with pytest.raises(ValueError, match="incomplete"):
        runtime.promotion_decision(incomplete, level=0, level_count=2, failures=0, success_rate=.8, demote_after=3)


@pytest.mark.parametrize("fault", ["empty", "corrupt", "duplicate", "counter", "unknown", "updates", "guide", "legacy", "consistency"])
def test_invalid_evidence_cannot_drive_promotion(tmp_path, fault):
    rows, config = make_run(tmp_path)
    if fault == "empty": rows = []
    if fault == "duplicate": rows[1]["episode_outcomes"] = rows[0]["episode_outcomes"]
    if fault == "counter": rows[1]["completed_episodes_total"] = 2000
    if fault == "unknown": rows[0]["episode_outcomes"][0]["success"] = None
    if fault == "updates": config["result"]["updates_done"] = 1
    if fault == "guide": rows[0]["executed_minus_actor_l1_mean"] = .1
    if fault == "legacy": del rows[0]["episode_outcomes"]
    if fault == "consistency": config["args"]["target_reward_consistency_body"] = "none"
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows) + ("broken\n" if fault == "corrupt" else ""))
    (tmp_path / "train_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        runtime.summarize_run(tmp_path, evaluation=True)


@pytest.mark.parametrize("mode", ["ok", "crash", "incomplete", "no_progress"])
def test_shell_wrapper_with_fake_processes_only(tmp_path, mode):
    # This interpreter intercepts the train.py path and only writes synthetic
    # JSON/text. It never imports Isaac, loads a model, or performs an update.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "aic").symlink_to(REPO, target_is_directory=True)
    mock = tmp_path / "bin"
    mock.mkdir()
    interpreter = mock / "fake_python"
    interpreter.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys, yaml
args = sys.argv[1:]
def arg(key): return args[args.index(key) + 1]
if args[0].endswith("build_stateful_insertion_curriculum.py"):
    cfg = yaml.safe_load(pathlib.Path(arg("--config")).read_text())
    root = pathlib.Path(cfg["output_root"]) / "episodes"
    root.mkdir(parents=True)
    for i in range(cfg["episodes"]):
        (root / f"episode_{i:06d}.yaml").write_text(yaml.safe_dump({"episode_id": f"ep{i}"}))
elif args[0].endswith("serl/train.py"):
    out = pathlib.Path(arg("--output_dir")) / "fake_run"
    out.mkdir(parents=True)
    evaluation = arg("--updates") == "0"
    (out / "argv.json").write_text(json.dumps(args))
    if not evaluation:
        (out / "checkpoint_latest.pt").write_text("synthetic checkpoint, not model weights")
    if os.environ["MOCK_MODE"] == "crash" and not evaluation:
        raise SystemExit(7)
    count = min(2, int(arg("--max_completed_episodes")))
    if evaluation and os.environ["MOCK_MODE"] == "incomplete": count = 1
    rows = []
    for i in range(count):
        rows.append({"step": i+1, "completed_episodes_total": i+1,
          "episode_outcomes": [{"env_index": 0, "episode_index": i, "config_episode_id": f"ep{i}",
            "terminated": False, "truncated": True, "success": False}],
          "target_action_guide_collect_blend_effective": 0.0,
          "insertion_action_guard_applied_fraction": 0.0, "executed_minus_actor_l1_mean": 0.0})
    (out / "metrics.jsonl").write_text("".join(json.dumps(r)+"\\n" for r in rows))
    resolved = {"terminate_on_target_success": "--terminate_on_target_success" in args,
      "target_reward_consistency_body": arg("--target_reward_consistency_body")}
    for key in ("target_success_axial_threshold", "target_success_lateral_threshold",
                "target_success_orientation_threshold", "target_success_consistency_axial_threshold",
                "target_success_consistency_lateral_threshold"):
        resolved[key] = float(arg("--" + key))
    (out / "train_config.json").write_text(json.dumps({"args": resolved, "result": {
       "stop_reason": "max_completed_episodes", "updates_done": 0 if evaluation else 1,
       "steps_completed": count, "episodes_completed": count}}))
else:
    os.execv(sys.executable, [sys.executable, *args])
''')
    interpreter.chmod(0o755)
    for name in ("sleep", "pkill"):
        path = mock / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    if mode == "no_progress":
        date = mock / "date"
        date.write_text(f"#!{sys.executable}\n" + f'''
import pathlib, sys, subprocess
if sys.argv[1:] == ["-u", "+%s"]:
    p = pathlib.Path({str(tmp_path / 'clock')!r})
    value = int(p.read_text()) + 10 if p.exists() else 100
    p.write_text(str(value))
    print(value)
else:
    subprocess.run(["/bin/date", *sys.argv[1:]], check=True)
''')
        date.chmod(0o755)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"episodes": 4, "seed": 1, "start_near_gate": {
        "axial_distance_m": {"initial": .003, "terminal": .04},
        "lateral_distance_m": {"initial": 0, "terminal": .01},
        "orientation_error_rad": {"initial": 0, "terminal": .03}},
        "progression": {"policy": "promotion_gated", "level_count": 2,
                        "eval_every_episodes": 3, "promotion_success_rate": .8,
                        "demotion_on_consecutive_failures": 3}}))
    seed = tmp_path / "seed.pt"
    seed.write_text("synthetic input")
    run = tmp_path / "run"
    env = {k: v for k, v in os.environ.items() if not k.startswith("AIC_STATEFUL_")}
    env.update(PATH=f"{mock}:{env['PATH']}", MOCK_MODE=mode,
               AIC_STATEFUL_WORKSPACE=str(workspace), AIC_STATEFUL_PYTHON=str(interpreter),
               AIC_STATEFUL_CONFIG=str(config), AIC_STATEFUL_RUN_ROOT=str(run),
               AIC_STATEFUL_BASE_CKPT=str(seed), AIC_STATEFUL_ACT_TS=str(seed),
               AIC_STATEFUL_TRAIN_EPISODE_VARIANTS="2", AIC_STATEFUL_EVAL_EPISODE_VARIANTS="2",
               AIC_STATEFUL_NO_PROGRESS_ASSESS_SECONDS="1" if mode == "no_progress" else "0")
    result = subprocess.run(["bash", str(REPO / "tools/train_stateful_promotion_gated_insertion.sh")],
                            env=env, text=True, capture_output=True, timeout=30)
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    if mode == "ok":
        assert result.returncode == 0, result.stderr
        assert events[-1]["episodes_used"] == 4
        assert events[-1]["cycle"] == 2  # Two actual episodes each, not the requested three.
        assert events[-1]["level"] == 0
        assert (run / "train_cycle0001_level000/fake_run/checkpoint_latest.pt").exists()
        args = json.loads((run / "eval_cycle0001_level000/fake_run/argv.json").read_text())
        assert args[args.index("--updates") + 1] == "0"
        assert args[args.index("--target_reward_consistency_body") + 1] == "sfp_module_link"
        assert "--replace_nic_cage_p0_with_aligned_cubes" not in args
    elif mode == "crash":
        assert result.returncode == 7
        assert not any(event["event"] == "train_done" for event in events)
    elif mode == "incomplete":
        assert result.returncode != 0
        assert not any(event["event"] == "eval_done" for event in events)
    else:
        assert result.returncode == 3
        assert events[-1]["event"] == "no_progress_stop"
