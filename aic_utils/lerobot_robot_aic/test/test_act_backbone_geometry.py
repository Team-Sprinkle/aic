"""CPU checks for ACT spatial geometry, checkpoint reconstruction and export."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "aic_utils/lerobot_robot_aic"))
from lerobot_robot_aic.act_backbone import (
    apply_backbone_geometry, backbone_geometry, load_act_policy, read_backbone_geometry,
    require_default_backbone_geometry, save_backbone_geometry, validate_backbone_geometry,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_policy():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    config = ACTConfig(device="cpu", input_features={
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(32,)),
        **{f"observation.images.{camera}_camera": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 288))
           for camera in ["center", "left", "right"]}},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        chunk_size=2, n_action_steps=1, dim_model=32, dim_feedforward=64,
        n_encoder_layers=1, n_decoder_layers=1, n_vae_encoder_layers=1,
        pretrained_backbone_weights=None, dropout=0., vision_backbone="resnet18")
    return ACTPolicy(config).eval()


def test_stride16_preserves_every_parameter_and_changes_only_feature_grid():
    torch.manual_seed(411)
    policy = make_policy()
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    count = sum(p.numel() for p in policy.parameters())
    image = torch.randn(1, 3, 256, 288)
    with torch.inference_mode():
        assert policy.model.backbone(image)["feature_map"].shape == (1, 512, 8, 9)
    geometry = backbone_geometry(policy.config, 16)
    apply_backbone_geometry(policy, geometry)
    with torch.inference_mode():
        assert policy.model.backbone(image)["feature_map"].shape == (1, 512, 16, 18)
    assert sum(p.numel() for p in policy.parameters()) == count
    assert before.keys() == policy.state_dict().keys()
    assert all(value.shape == before[key].shape and torch.equal(value, before[key])
               for key, value in policy.state_dict().items())
    block = policy.model.backbone["layer4"][0]
    assert block.conv1.stride == block.downsample[0].stride == (1, 1)
    assert block.conv1.dilation == block.conv2.dilation == (1, 1)
    apply_backbone_geometry(policy, backbone_geometry(policy.config, 32))
    assert block.conv1.stride == block.downsample[0].stride == (2, 2)


def test_saved_checkpoint_and_actual_export_keep_stride_and_predictions(tmp_path, monkeypatch):
    torch.manual_seed(414)
    policy = make_policy()
    geometry = backbone_geometry(policy.config, 16)
    apply_backbone_geometry(policy, geometry)
    checkpoint = tmp_path / "pretrained_model"
    policy.save_pretrained(checkpoint)
    save_backbone_geometry(policy, checkpoint)
    (checkpoint / "aic_action_config.json").write_text(json.dumps({"backbone_geometry": geometry}))
    restored = load_act_policy(checkpoint, local_files_only=True).eval()
    assert read_backbone_geometry(checkpoint) == geometry
    assert restored.model.backbone["layer4"][0].conv1.stride == (1, 1)
    path = ROOT / "aic_utils/lerobot_robot_aic/scripts/export_act_torchscript.py"
    spec = importlib.util.spec_from_file_location("act_geometry_export", path)
    exporter = importlib.util.module_from_spec(spec); spec.loader.exec_module(exporter)
    inputs = (torch.randn(1, 32), *(torch.randn(1, 3, 256, 288) for _ in range(3)))
    with torch.inference_mode():
        expected = exporter.ACTTorchScriptWrapper(policy)(*inputs)
        reconstructed = exporter.ACTTorchScriptWrapper(restored)(*inputs)
    torch.testing.assert_close(reconstructed, expected, atol=0, rtol=0)
    output = tmp_path / "act_stride16.pt"
    monkeypatch.setattr(sys, "argv", [str(path), "--act-checkpoint", str(checkpoint), "--output", str(output), "--device", "cpu"])
    assert exporter.main() == 0
    traced = torch.jit.load(str(output), map_location="cpu").eval()
    with torch.inference_mode():
        actual = traced(*inputs)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert json.loads(output.with_suffix(".json").read_text())["backbone_geometry"] == geometry
    with pytest.raises(ValueError, match="geometry-aware"):
        require_default_backbone_geometry(checkpoint)


def test_legacy_checkpoint_retains_default_stride_and_accepts_default_runner(tmp_path):
    policy = make_policy(); policy.save_pretrained(tmp_path)
    restored = load_act_policy(tmp_path, local_files_only=True)
    assert restored.model.backbone["layer4"][0].conv1.stride == (2, 2)
    assert read_backbone_geometry(tmp_path)["mode"] == "torchvision"
    require_default_backbone_geometry(tmp_path)


def test_invalid_schema_backbone_and_unrecorded_mutation_fail(tmp_path):
    policy = make_policy(); good = backbone_geometry(policy.config, 16)
    with pytest.raises(ValueError, match="geometry"):
        apply_backbone_geometry(policy, {})
    for key, value in [("version", 2), ("output_stride", 8), ("feature_layer", "layer3"),
                       ("vision_backbone", "resnet50"), ("mode", "unknown")]:
        invalid = {**good, key: value}
        with pytest.raises(ValueError, match="geometry"):
            validate_backbone_geometry(invalid, policy.config)
    config = copy.deepcopy(policy.config); config.vision_backbone = "resnet50"
    with pytest.raises(ValueError, match="ResNet18"):
        backbone_geometry(config, 16)
    config.vision_backbone = "resnet_typo"
    with pytest.raises(ValueError, match="ResNet"):
        backbone_geometry(config)
    config.vision_backbone = "resnet18"; config.replace_final_stride_with_dilation = True
    with pytest.raises(ValueError, match="dilation"):
        backbone_geometry(config)
    policy.model.backbone["layer4"][0].conv1.stride = (1, 1)
    with pytest.raises(ValueError, match="strides disagree"):
        save_backbone_geometry(policy, tmp_path)
    (tmp_path / "aic_backbone_config.json").write_text("null")
    with pytest.raises(ValueError, match="geometry"):
        read_backbone_geometry(tmp_path, policy.config)


def test_conflicting_sidecar_and_action_contract_fail_before_model_load(tmp_path):
    policy = make_policy(); policy.save_pretrained(tmp_path)
    save_backbone_geometry(policy, tmp_path)
    (tmp_path / "aic_action_config.json").write_text(json.dumps({"backbone_geometry": backbone_geometry(policy.config, 16)}))
    with pytest.raises(ValueError, match="disagree"):
        load_act_policy(tmp_path, local_files_only=True)
