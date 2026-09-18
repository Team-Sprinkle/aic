"""Exercise the real actor class on CPU without importing the simulator launcher."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.fixture(scope="module")
def actor_class():
    path = Path(__file__).resolve().parents[1] / "aic_isaaclab/scripts/serl/train.py"
    tree = ast.parse(path.read_text())
    actor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "IsaacACTAdapterActor")
    future = ast.parse("from __future__ import annotations").body[0]
    module = ast.Module(body=[future, actor], type_ignores=[])
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["IsaacACTAdapterActor"]


@pytest.mark.parametrize("width", [1, 2, 3])
def test_actor_rejects_padding_or_truncation_of_state(actor_class, width):
    # Mock only ACT inference/normalization; execute the actual action construction.
    actor = actor_class.__new__(actor_class)
    nn.Module.__init__(actor)
    actor.act_base_device = torch.device("cpu")
    actor.act_normalizer = SimpleNamespace(normalize_state=lambda value: value,
                                           normalize_image=lambda key, value: value,
                                           unnormalize_action=lambda value: value)
    actor.act_base = lambda state, *images: torch.zeros(state.shape[0], 1, 6)
    actor.normalized_state_clip = None
    actor.action_horizon = 1
    actor.action_dim = 6
    actor.adapter_state_dim = 2
    actor.state_encoder = nn.Identity()
    actor.adapter = nn.Linear(8, 6)
    actor.actor_mode = "act_direct"
    actor.adapter_delta_clip = None
    actor.tcp_translation_action_clip = None
    actor.tcp_rotation_action_clip = None
    actor.action_clip = None
    obs = {"state": torch.zeros(1, 2), "actor_state": torch.zeros(1, width),
           "images": {f"observation.images.{name}_camera": torch.zeros(1, 3, 2, 2)
                      for name in ("center", "left", "right")}}
    if width != 2:
        with pytest.raises(ValueError, match="Actor state width"):
            actor.action_components(obs)
    else:
        assert actor.action_components(obs)["final_action"].shape == (1, 6)
