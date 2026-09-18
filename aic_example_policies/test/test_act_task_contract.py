"""Check mixed-task ACT feature layout without requiring ROS or a GPU."""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "aic_utils/lerobot_robot_aic"))
from lerobot_robot_aic.act_state_contract import validate_act_state_contract
from lerobot_robot_aic.runtime_features import AICRuntimeFeatureAssembler
from lerobot_robot_aic.task_encoding import encode_task_vector, task_encoding_schema


def runtime_methods(*names):
    source = ROOT / "aic_example_policies/aic_example_policies/ros/RunACTTorchScript.py"
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef))
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for method in methods:
        method.decorator_list = []
    scope = {"np": np, "Task": object, "Observation": object, "encode_task_vector": encode_task_vector}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), scope)
    return scope


def modern_contract():
    return {"state_shape": [43], "base_state_dim": 42, "include_elapsed_sim_time": True,
            "task_conditioned": True, "task_vector_indices": [32, 42],
            "task_encoding": task_encoding_schema()}


@pytest.mark.parametrize("base_dim,time", [(32, False), (32, True), (42, False), (72, False), (82, False)])
def test_legacy_state_layouts_remain_supported(base_dim, time):
    result = validate_act_state_contract({"state_shape": [base_dim + int(time)],
                                          "base_state_dim": base_dim, "include_elapsed_sim_time": time})
    assert result["base_state_dim"] == base_dim
    assert result["task_vector_indices"] == ([base_dim - 10, base_dim] if base_dim in (42, 82) else None)


def test_task_time_contract_rejects_silent_layout_changes():
    assert validate_act_state_contract(modern_contract()) == modern_contract()
    for key in ("task_conditioned", "task_vector_indices", "task_encoding"):
        contract = modern_contract()
        del contract[key]
        with pytest.raises(ValueError, match="requires explicit"):
            validate_act_state_contract(contract)
    contract = modern_contract()
    contract["task_vector_indices"] = [33, 43]
    with pytest.raises(ValueError, match="before elapsed time"):
        validate_act_state_contract(contract)
    contract = modern_contract()
    contract["task_encoding"]["names"] = ["different_task", *contract["task_encoding"]["names"][1:]]
    with pytest.raises(ValueError, match="canonical"):
        validate_act_state_contract(contract)


def test_task_identity_normalization_does_not_touch_elapsed_time():
    methods = runtime_methods("_apply_task_vector_identity_normalization")
    runner = NS(state_dim=43, task_vector_indices=[32, 42],
                state_mean=torch.full((1, 43), 5.), state_std=torch.full((1, 43), 2.))
    methods["_apply_task_vector_identity_normalization"](runner)
    assert torch.all(runner.state_mean[:, 32:42] == 0)
    assert torch.all(runner.state_std[:, 32:42] == 1)
    assert runner.state_mean[0, 42] == 5 and runner.state_std[0, 42] == 2
    assert runner.state_mean[0, 31] == 5 and runner.state_std[0, 31] == 2


@pytest.mark.parametrize("family,card,port", [("sfp_to_nic", c, p) for c in range(5) for p in range(2)]
                         + [("sc_to_sc", -1, p) for p in range(2)])
def test_runtime_task_mapping_and_43d_state_match_canonical_training(family, card, port):
    methods = runtime_methods("_task_vector", "_parse_index_from_suffix", "_state_vector")
    task = NS(target_module_name=f"nic_card_mount_{card}" if family == "sfp_to_nic" else f"sc_port_{port}",
              port_name=f"sfp_port_{port}" if family == "sfp_to_nic" else "sc_port_base")
    runner = NS(_parse_index_from_suffix=methods["_parse_index_from_suffix"],
                _current_task=task, include_elapsed_sim_time=True, _episode_first_image_time=100.,
                time_clip_sec=40., quaternion_sign="x")
    runner._task_vector = lambda task: methods["_task_vector"](runner, task)
    canonical = encode_task_vector(task_family=family, target_port_index=port, target_card_index=card)
    np.testing.assert_array_equal(runner._task_vector(task), canonical)
    assembler = AICRuntimeFeatureAssembler(42)
    base_state = np.arange(32, dtype=np.float32)
    assembler.assemble_ros = lambda obs: assembler.assemble(base_state)
    runner.feature_assembler = assembler
    obs = NS(center_image=NS(header=NS(stamp=NS(sec=102, nanosec=500_000_000))))
    actual = methods["_state_vector"](runner, obs)
    np.testing.assert_array_equal(actual, np.concatenate([base_state, canonical, [2.5]]))


@pytest.mark.parametrize("module,port", [("nic_card_mount_5", "sfp_port_0"),
                                          ("nic_card_mount_0", "sfp_port_2"), ("sc_port_2", "sc_port_base")])
def test_runtime_task_mapping_rejects_out_of_range_fields(module, port):
    methods = runtime_methods("_task_vector", "_parse_index_from_suffix")
    runner = NS(_parse_index_from_suffix=methods["_parse_index_from_suffix"])
    with pytest.raises(ValueError):
        methods["_task_vector"](runner, NS(target_module_name=module, port_name=port))
