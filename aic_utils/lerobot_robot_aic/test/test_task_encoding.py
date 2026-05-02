from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

from lerobot_robot_aic.task_encoding import (  # noqa: E402
    TASK_VECTOR_DIM,
    encode_task_vector,
    encode_task_vector_from_metadata,
    validate_task_vector,
)


def test_task_vector_examples() -> None:
    sfp = encode_task_vector(task_family="sfp_to_nic", target_port_index=1, target_card_index=3)
    sc = encode_task_vector(task_family="sc_to_sc", target_port_index=0, target_card_index=-1)

    assert sfp.tolist() == [1, 0, 0, 1, 0, 0, 0, 1, 0, 1]
    assert sc.tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert len(sfp) == TASK_VECTOR_DIM


def test_task_vector_from_nested_metadata() -> None:
    vector = encode_task_vector_from_metadata(
        {
            "task": {
                "task_family": "sfp_to_nic",
                "target_port_index": 0,
                "target_card_index": 4,
                "target_card_valid": 1,
            }
        }
    )

    assert vector.tolist() == [1, 0, 1, 0, 0, 0, 0, 0, 1, 1]


def test_task_vector_validation_rejects_invalid_masks() -> None:
    with pytest.raises(ValueError, match="target_card one-hot must be all zeros"):
        validate_task_vector([0, 1, 1, 0, 0, 1, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="sfp_to_nic requires target_card_valid"):
        encode_task_vector(
            task_family="sfp_to_nic",
            target_port_index=1,
            target_card_index=3,
            target_card_valid=0,
        )
