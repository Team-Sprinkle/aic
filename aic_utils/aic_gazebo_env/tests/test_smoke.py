"""Smoke tests for package import in the existing workspace."""

import importlib


def test_package_import_smoke() -> None:
    module = importlib.import_module("aic_gazebo_env")

    assert module is not None
    assert hasattr(module, "__all__")


def test_public_api_symbols_are_importable() -> None:
    from aic_gazebo_env import FakeRuntime, GazeboEnv, ResetRequest, StepResponse

    assert GazeboEnv is not None
    assert FakeRuntime is not None
    assert ResetRequest is not None
    assert StepResponse is not None
