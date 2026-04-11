"""Serialization tests for the backend protocol schema."""

from aic_gazebo_env import (
    GetObservationRequest,
    GetObservationResponse,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
)


def test_reset_request_round_trip() -> None:
    message = ResetRequest(seed=123, options={"difficulty": "easy"})

    payload = message.to_dict()
    restored = ResetRequest.from_dict(payload)

    assert payload == {"seed": 123, "options": {"difficulty": "easy"}}
    assert restored == message


def test_reset_response_round_trip() -> None:
    message = ResetResponse(
        observation={"joint_positions": [0.0, 1.0]},
        info={"runtime": "fake"},
    )

    payload = message.to_dict()
    restored = ResetResponse.from_dict(payload)

    assert restored == message
    assert payload["observation"]["joint_positions"] == [0.0, 1.0]


def test_step_request_round_trip() -> None:
    message = StepRequest(action={"command": [0.1, -0.2, 0.3]})

    payload = message.to_dict()
    restored = StepRequest.from_dict(payload)

    assert restored == message
    assert payload == {"action": {"command": [0.1, -0.2, 0.3]}}


def test_step_response_round_trip() -> None:
    message = StepResponse(
        observation={"step_count": 2},
        reward=1.5,
        terminated=False,
        truncated=True,
        info={"backend": "stub"},
    )

    payload = message.to_dict()
    restored = StepResponse.from_dict(payload)

    assert restored == message
    assert payload["reward"] == 1.5
    assert payload["truncated"] is True


def test_get_observation_request_round_trip() -> None:
    message = GetObservationRequest()

    payload = message.to_dict()
    restored = GetObservationRequest.from_dict(payload)

    assert payload == {}
    assert restored == message


def test_get_observation_response_round_trip() -> None:
    message = GetObservationResponse(
        observation={"rgb": "placeholder"},
        info={"timestamp": 42},
    )

    payload = message.to_dict()
    restored = GetObservationResponse.from_dict(payload)

    assert restored == message
    assert payload["info"]["timestamp"] == 42
