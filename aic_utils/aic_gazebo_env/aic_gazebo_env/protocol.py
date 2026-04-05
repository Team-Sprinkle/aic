"""Backend-agnostic request/response schema for environment backends.

These dataclasses define the control protocol between the public Python env API
and a future backend implementation. The protocol is intentionally transport
agnostic. A future Gazebo integration could map each request/response pair to
`gz` transport request/reply services, for example:

- `ResetRequest` -> a reset service carrying seed and options payload
- `StepRequest` -> a step service carrying the action payload
- `GetObservationRequest` -> an observation service returning the latest state

The current milestone only defines schema and serialization helpers. It does
not implement any Gazebo-side plugin or transport binding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ResetRequest:
    """Request to reset a backend environment instance."""

    seed: int | None = None
    options: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        """Serialize the request to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: JsonDict) -> "ResetRequest":
        """Deserialize a request from a plain dict."""
        return cls(
            seed=payload.get("seed"),
            options=dict(payload.get("options", {})),
        )


@dataclass(frozen=True)
class ResetResponse:
    """Response returned after a backend reset."""

    observation: JsonDict
    info: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        """Serialize the response to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: JsonDict) -> "ResetResponse":
        """Deserialize a response from a plain dict."""
        return cls(
            observation=dict(payload["observation"]),
            info=dict(payload.get("info", {})),
        )


@dataclass(frozen=True)
class StepRequest:
    """Request to advance the backend environment by one step."""

    action: JsonDict

    def to_dict(self) -> JsonDict:
        """Serialize the request to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: JsonDict) -> "StepRequest":
        """Deserialize a request from a plain dict."""
        return cls(action=dict(payload["action"]))


@dataclass(frozen=True)
class StepResponse:
    """Response returned after a backend step."""

    observation: JsonDict
    reward: float
    terminated: bool
    truncated: bool
    info: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        """Serialize the response to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: JsonDict) -> "StepResponse":
        """Deserialize a response from a plain dict."""
        return cls(
            observation=dict(payload["observation"]),
            reward=float(payload["reward"]),
            terminated=bool(payload["terminated"]),
            truncated=bool(payload["truncated"]),
            info=dict(payload.get("info", {})),
        )


@dataclass(frozen=True)
class GetObservationRequest:
    """Request to retrieve the latest backend observation."""

    def to_dict(self) -> JsonDict:
        """Serialize the request to a plain dict."""
        return {}

    @classmethod
    def from_dict(cls, payload: JsonDict) -> "GetObservationRequest":
        """Deserialize a request from a plain dict."""
        del payload
        return cls()


@dataclass(frozen=True)
class GetObservationResponse:
    """Response containing the latest backend observation."""

    observation: JsonDict
    info: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        """Serialize the response to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: JsonDict) -> "GetObservationResponse":
        """Deserialize a response from a plain dict."""
        return cls(
            observation=dict(payload["observation"]),
            info=dict(payload.get("info", {})),
        )
