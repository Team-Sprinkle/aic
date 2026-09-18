"""Exercise the real CLI loop with fake batches, clocks, and validation results."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_vision_offline_serl.py"


def run_loop(tmp_path, *, timeout=False):
    tree = ast.parse(SCRIPT.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    start = next(i for i, node in enumerate(main.body)
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "step")
    end = next(i for i in range(start, len(main.body)) if isinstance(main.body[i], ast.While))
    loop = ast.Module(body=main.body[start:end + 1], type_ignores=[])
    calls = []
    validations = iter([{"bc_loss": 1.0}, {"bc_loss": 2.0}])
    ticks = iter([0, 61] if timeout else [0])
    trainer = SimpleNamespace(train_step=lambda batch: calls.append(batch) or {}, save_checkpoint=lambda *a, **kw: None)
    namespace = {
        "args": SimpleNamespace(steps=100, max_wall_time_minutes=1 if timeout else 0,
            save_every=0, val_every=1, val_max_batches=1, early_stopping_metric="bc_loss",
            early_stopping_min_delta=0, early_stopping_patience=1),
        "time": SimpleNamespace(monotonic=lambda: next(ticks)),
        "dist": SimpleNamespace(is_initialized=lambda: False),
        "sampler": None, "loader": [0, 1, 2], "trainer": trainer, "val_loader": object(),
        "_is_rank0": lambda: True, "_barrier_if_distributed": lambda: None,
        "_validate": lambda *a, **kw: next(validations), "json": json,
        "metrics_path": tmp_path / "metrics.jsonl", "validation_path": tmp_path / "val.jsonl",
        "run_dir": tmp_path, "train_config": {}, "dataset_summary": {}, "warmstart": {},
    }
    exec(compile(loop, str(SCRIPT), "exec"), namespace)
    return namespace, calls


def test_timeout_records_zero_updates_if_no_batch_was_trained(tmp_path):
    result, calls = run_loop(tmp_path, timeout=True)
    assert result["step"] == 0 and not calls
    assert result["stop_reason"] == "max_wall_time"


def test_early_stopping_records_actual_update_count(tmp_path):
    result, calls = run_loop(tmp_path)
    assert result["step"] == len(calls) == 2
    assert result["stop_reason"] == "early_stopping"
