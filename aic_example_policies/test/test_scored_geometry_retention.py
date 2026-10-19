"""Prevent silent loss of the task's measured physical endpoint during extraction."""
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('scored_extractor_test',ROOT/'scripts/extract_moveit_training_stream.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_selected_sfp_tip_and_body_survive_sc_legacy_filter():
    task=dict(cable_name='cable_0',plug_name='sfp_tip')
    assert module.needed_transform('aic_world','cable_0',task)
    assert module.needed_transform('cable_0','cable_0/sfp_tip_link',task)
    assert not module.needed_transform('cable_0','cable_0/link_12',task)


def test_sc_and_fixed_camera_reference_still_retained():
    task=dict(cable_name='cable_1',plug_name='sc_tip')
    assert module.needed_transform('cable_1','cable_1/sc_tip_link',task)
    assert module.needed_transform('world','tabletop',task)
    assert module.needed_transform('left_camera/sensor_link','left_camera/optical',task)
