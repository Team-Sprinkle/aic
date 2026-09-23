import importlib.util
import sys
from pathlib import Path
import torch


PATH = Path(__file__).parents[1] / "aic_isaaclab/scripts/serl/build_rpdp_dataset.py"
spec = importlib.util.spec_from_file_location("build_rpdp_dataset", PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def geometry(depth):
    return {
        "body_world_env0": [0, 0, depth],
        "body_orientation_wxyz_by_env": [[1, 0, 0, 0]],
        "entrance_world_env0": [0, 0, 0],
        "target_world_env0": [0, 0, 0.008],
        "target_orientation_wxyz_by_env": [[1, 0, 0, 0]],
        "signed_depth_m_env0": depth,
        "lateral_error_m_env0": 0,
        "orientation_error_rad_env0": 0,
        "success_geometry_by_env": [depth > 0],
    }


def test_terminal_endpoint_uses_retained_pre_reset_observation():
    item = {
        "metadata": {
            "terminated": True,
            "post_step_insertion_geometry": geometry(-0.02),  # auto reset scene
        },
        "terminal_observation": {"insertion_geometry": geometry(0.008)},
    }
    assert module.geometry(item, post=True)["depth"] == 0.008


def test_nonterminal_endpoint_uses_post_step_geometry():
    item = {
        "metadata": {
            "terminated": False,
            "truncated": False,
            "post_step_insertion_geometry": geometry(0.003),
        }
    }
    assert module.geometry(item, post=True)["depth"] == 0.003


def test_observation_phase_uses_force_before_distance():
    assert module.observation_phase(torch.tensor([20.,0.,0.]),torch.tensor([6.,0.,0.])).tolist()==[0.,0.,0.,1.]
    assert module.observation_phase(torch.tensor([4.,0.,0.]),torch.zeros(3)).tolist()==[1.,0.,0.,0.]
    assert module.observation_phase(torch.tensor([1.,0.,0.]),torch.zeros(3)).tolist()==[0.,1.,0.,0.]
    assert module.observation_phase(torch.tensor([.2,0.,0.]),torch.zeros(3)).tolist()==[0.,0.,1.,0.]


def test_recorded_target_action_uses_teacher_not_blended_execution():
    teacher=torch.arange(24,dtype=torch.float32)
    item={"guide_action":teacher,"action":torch.zeros(24),
          "metadata":{"macro_microsteps":4}}
    assert torch.equal(module.recorded_target_action(item),teacher.reshape(4,6))


def test_recorded_target_action_zeros_unexecuted_terminal_padding():
    teacher=torch.ones(24)
    item={"guide_action":teacher,"metadata":{"macro_microsteps":2}}
    result=module.recorded_target_action(item)
    assert torch.equal(result[:2],torch.ones(2,6))
    assert torch.equal(result[2:],torch.zeros(2,6))
