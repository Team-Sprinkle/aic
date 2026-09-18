"""Protect checkpoint arithmetic from dtype, shape and buffer corruption."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/average_act_checkpoints.py"
spec = importlib.util.spec_from_file_location("average_act", SCRIPT)
average = importlib.util.module_from_spec(spec)
spec.loader.exec_module(average)


def test_average_preserves_dtype_buffers_and_sources():
    a = {"weight": torch.tensor([1., 3.]), "counter": torch.tensor(4)}
    b = {"weight": torch.tensor([3., 5.]), "counter": torch.tensor(4)}
    result = average.average_weights([a, b])
    assert torch.equal(result["weight"], torch.tensor([2., 4.]))
    assert result["weight"].dtype == a["weight"].dtype
    assert torch.equal(result["counter"], a["counter"])
    assert torch.equal(a["weight"], torch.tensor([1., 3.]))


@pytest.mark.parametrize("b", [
    {"weight": torch.tensor([1., 2.]), "counter": torch.tensor(5)},
    {"weight": torch.tensor([1.]), "counter": torch.tensor(4)},
    {"weight": torch.tensor([1., float('nan')]), "counter": torch.tensor(4)},
])
def test_incompatible_or_nonfinite_weights_fail(b):
    a = {"weight": torch.tensor([1., 3.]), "counter": torch.tensor(4)}
    with pytest.raises(ValueError):
        average.average_weights([a, b])


def test_average_rejects_mixed_stride_and_preserves_matching_sidecar(tmp_path, monkeypatch):
    config = {"vision_backbone": "resnet18", "replace_final_stride_with_dilation": False}
    stride16 = {"version": 1, "vision_backbone": "resnet18", "feature_layer": "layer4",
                "output_stride": 16, "mode": "resnet18_layer4_stride1"}
    stride32 = {**stride16, "output_stride": 32, "mode": "torchvision"}
    checkpoints = [tmp_path / "run" / "checkpoints" / str(i) / "pretrained_model" for i in (1, 2)]
    for checkpoint, geometry in zip(checkpoints, (stride16, stride32)):
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text(json.dumps(config))
        # Identical action configs and tensor shapes must not hide a sidecar-only mismatch.
        (checkpoint / "aic_action_config.json").write_text("{}")
        (checkpoint / "aic_backbone_config.json").write_text(json.dumps(geometry))
        (checkpoint / "training_step.json").write_text('{"training_config": {}}')
        save_file({"weight": torch.ones(2)}, str(checkpoint / "model.safetensors"))
        save_file({"state.mean": torch.zeros(2)}, str(checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
    output = tmp_path / "average"
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--checkpoints", *map(str, checkpoints), "--output-dir", str(output)])
    with pytest.raises(ValueError, match="backbone geometry differs"):
        average.main()
    assert not output.exists()
    (checkpoints[1] / "aic_backbone_config.json").write_text(json.dumps(stride16))
    average.main()
    destination = output / "checkpoints/000000/pretrained_model"
    assert average.read_backbone_geometry(destination) == stride16
    assert json.loads((destination / "aic_backbone_config.json").read_text()) == stride16
    assert json.loads((output / "model_average.json").read_text())["backbone_geometry"] == stride16
