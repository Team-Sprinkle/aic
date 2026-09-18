"""Exercise the runtime path helper without importing ROS or initializing a model."""
import ast
import os
from pathlib import Path

import pytest


def path_policy():
    source = Path(__file__).resolve().parents[1] / "aic_example_policies/ros/RunACTTorchScript.py"
    tree = ast.parse(source.read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    methods = [node for node in original.body if isinstance(node, ast.FunctionDef)
               and node.name in {"_stats_path", "_required_path"}]
    cls = ast.ClassDef(name="PathPolicy", bases=[], keywords=[], body=methods, decorator_list=[])
    scope = {"Path": Path, "os": os}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), "exec"), scope)
    return scope["PathPolicy"]()


def test_mapped_normalizer_override_and_legacy_metadata_path(tmp_path, monkeypatch):
    policy = path_policy()
    legacy = tmp_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    legacy.write_bytes(b"old stats")
    policy.metadata = {"checkpoint_dir": str(tmp_path)}
    monkeypatch.delenv("AIC_ACT_NORMALIZER_PATH", raising=False)
    assert policy._stats_path() == legacy
    mapped = tmp_path / "mounted_stats.safetensors"; mapped.write_bytes(b"mapped stats")
    monkeypatch.setenv("AIC_ACT_NORMALIZER_PATH", str(mapped))
    policy.metadata = {"checkpoint_dir": "/host/path/unavailable/in/container"}
    assert policy._stats_path() == mapped
    mapped.unlink()
    with pytest.raises(FileNotFoundError, match="AIC_ACT_NORMALIZER_PATH"):
        policy._stats_path()
