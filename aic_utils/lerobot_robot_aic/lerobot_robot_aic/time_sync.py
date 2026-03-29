from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from numpy.typing import NDArray


Vector = NDArray[np.float64]
EpisodeArrays = dict[str, NDArray]
TorchEpisode = dict[str, torch.Tensor]

ACTION_KEY = "action"
OBSERVATION_STATE_KEY = "observation.state"
TIMESTAMP_KEY = "timestamp"
FRAME_INDEX_KEY = "frame_index"
EPISODE_INDEX_KEY = "episode_index"
INDEX_KEY = "index"
TASK_INDEX_KEY = "task_index"
TASK_KEY = "task"
DEFAULT_IMAGE_KEY_PREFIX = "observation.images."

TCP_POSITION_SLICE = slice(0, 3)
TCP_ORIENTATION_SLICE = slice(3, 7)
TCP_VELOCITY_SLICE = slice(7, 13)
TCP_ERROR_SLICE = slice(13, 19)
JOINT_POSITION_SLICE = slice(19, 26)
WRAPPED_ARM_JOINT_COLUMNS = (0, 1, 2, 3, 4, 5)


@dataclass(frozen=True)
class TimeSyncConfig:
    """Configuration for resampling one episode from one frame rate to another."""

    source_fps: float = 30.0
    target_fps: float = 20.0
    quaternion_keys: tuple[str, ...] = ("observation.tcp_orientation_xyzw",)
    image_keys: tuple[str, ...] = ()
    action_keys: tuple[str, ...] = (ACTION_KEY,)
    action_resampling_mode: str = "interval_average"  # or "zero_order_hold"
    wrapped_angle_columns: dict[str, tuple[int, ...]] = field(
        default_factory=lambda: {"observation.joint_positions": WRAPPED_ARM_JOINT_COLUMNS}
    )


def build_source_timestamps(num_frames: int, fps: float) -> Vector:
    """Return one timestamp per source frame."""

    if num_frames < 2:
        raise ValueError("Need at least two frames to resample an episode.")
    if fps <= 0.0:
        raise ValueError("fps must be positive.")
    return np.arange(num_frames, dtype=np.float64) / fps


def build_target_timestamps(source_timestamps: Vector, target_fps: float) -> Vector:
    """Return output timestamps over the same episode duration."""

    if target_fps <= 0.0:
        raise ValueError("target_fps must be positive.")

    episode_duration = float(source_timestamps[-1])
    target_count = int(np.floor(episode_duration * target_fps)) + 1
    return np.arange(target_count, dtype=np.float64) / target_fps


def interpolate_numeric_series(
    values: NDArray,
    source_timestamps: Vector,
    target_timestamps: Vector,
) -> NDArray:
    """Resample a scalar or vector series with linear interpolation."""

    values_float = np.asarray(values, dtype=np.float64)

    if values_float.ndim == 1:
        return np.interp(target_timestamps, source_timestamps, values_float)

    trailing_shape = values_float.shape[1:]
    flat_values = values_float.reshape(values_float.shape[0], -1)
    flat_output = np.empty((target_timestamps.shape[0], flat_values.shape[1]), dtype=np.float64)

    for column_index in range(flat_values.shape[1]):
        flat_output[:, column_index] = np.interp(
            target_timestamps,
            source_timestamps,
            flat_values[:, column_index],
        )

    return flat_output.reshape((target_timestamps.shape[0],) + trailing_shape)


def average_piecewise_constant_series(
    values: NDArray,
    source_timestamps: Vector,
    target_timestamps: Vector,
) -> NDArray:
    """Average held commands over each target interval."""

    values_float = np.asarray(values, dtype=np.float64)
    squeeze_output = values_float.ndim == 1
    if squeeze_output:
        values_float = values_float[:, None]

    source_count = values_float.shape[0]
    target_count = target_timestamps.shape[0]
    output = np.empty((target_count, values_float.shape[1]), dtype=np.float64)

    source_end_times = np.empty(source_count, dtype=np.float64)
    source_end_times[:-1] = source_timestamps[1:]
    source_end_times[-1] = (
        source_timestamps[-1] + (source_timestamps[-1] - source_timestamps[-2])
        if source_count >= 2
        else source_timestamps[-1]
    )

    for target_index in range(max(target_count - 1, 0)):
        interval_start = target_timestamps[target_index]
        interval_end = target_timestamps[target_index + 1]
        interval_duration = interval_end - interval_start

        accumulated = np.zeros(values_float.shape[1], dtype=np.float64)
        for source_index in range(source_count):
            overlap_start = max(interval_start, source_timestamps[source_index])
            overlap_end = min(interval_end, source_end_times[source_index])
            overlap_duration = overlap_end - overlap_start
            if overlap_duration > 0.0:
                accumulated += overlap_duration * values_float[source_index]

        output[target_index] = (
            values_float[min(target_index, source_count - 1)]
            if interval_duration <= 0.0
            else accumulated / interval_duration
        )

    if target_count == 1:
        nearest_index = int(np.searchsorted(source_timestamps, target_timestamps[0], side="right")) - 1
        output[0] = values_float[min(max(nearest_index, 0), source_count - 1)]
    elif target_count > 1:
        output[-1] = output[-2]

    return output[:, 0] if squeeze_output else output


def sample_piecewise_constant_series(
    values: NDArray,
    source_timestamps: Vector,
    target_timestamps: Vector,
) -> NDArray:
    """Resample commands with zero-order hold."""

    values_array = np.asarray(values)
    source_indices = np.searchsorted(source_timestamps, target_timestamps, side="right") - 1
    source_indices = np.clip(source_indices, 0, len(source_timestamps) - 1)
    return values_array[source_indices]


def wrap_angle_series(values: NDArray, period: float = 2.0 * np.pi) -> NDArray:
    """Wrap angles into the interval [-period/2, period/2)."""

    half_period = period / 2.0
    return ((values + half_period) % period) - half_period


def interpolate_wrapped_angle_series(
    values: NDArray,
    source_timestamps: Vector,
    target_timestamps: Vector,
    period: float = 2.0 * np.pi,
) -> NDArray:
    """Unwrap angles, interpolate them linearly, then wrap them back."""

    values_float = np.asarray(values, dtype=np.float64)
    unwrapped = np.unwrap(values_float, axis=0, period=period)
    interpolated = interpolate_numeric_series(
        unwrapped,
        source_timestamps=source_timestamps,
        target_timestamps=target_timestamps,
    )
    return wrap_angle_series(interpolated, period=period)


def normalize_quaternion(quaternion_xyzw: Vector) -> Vector:
    norm = np.linalg.norm(quaternion_xyzw)
    if norm == 0.0:
        raise ValueError("Quaternion norm is zero.")
    return quaternion_xyzw / norm


def slerp_quaternion(q0_xyzw: Vector, q1_xyzw: Vector, alpha: float) -> Vector:
    """Shortest-path spherical interpolation for xyzw quaternions."""

    q0 = normalize_quaternion(np.asarray(q0_xyzw, dtype=np.float64))
    q1 = normalize_quaternion(np.asarray(q1_xyzw, dtype=np.float64))

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        return normalize_quaternion((1.0 - alpha) * q0 + alpha * q1)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha

    weight_0 = np.sin(theta_0 - theta) / sin_theta_0
    weight_1 = np.sin(theta) / sin_theta_0
    return normalize_quaternion(weight_0 * q0 + weight_1 * q1)


def interpolate_quaternion_series(
    quaternions_xyzw: NDArray,
    source_timestamps: Vector,
    target_timestamps: Vector,
) -> NDArray:
    """Resample an orientation trajectory with quaternion SLERP."""

    quaternions = np.asarray(quaternions_xyzw, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("Expected quaternions with shape [num_frames, 4].")

    output = np.empty((target_timestamps.shape[0], 4), dtype=np.float64)

    for output_index, target_time in enumerate(target_timestamps):
        if target_time <= source_timestamps[0]:
            output[output_index] = normalize_quaternion(quaternions[0])
            continue

        if target_time >= source_timestamps[-1]:
            output[output_index] = normalize_quaternion(quaternions[-1])
            continue

        right_index = int(np.searchsorted(source_timestamps, target_time, side="right"))
        left_index = right_index - 1
        left_time = source_timestamps[left_index]
        right_time = source_timestamps[right_index]
        alpha = float((target_time - left_time) / (right_time - left_time))

        output[output_index] = slerp_quaternion(
            quaternions[left_index],
            quaternions[right_index],
            alpha,
        )

    return output


def choose_nearest_source_indices(
    source_timestamps: Vector,
    target_timestamps: Vector,
) -> NDArray[np.int64]:
    """Pick the closest source frame for each output timestamp."""

    indices = np.empty(target_timestamps.shape[0], dtype=np.int64)

    for output_index, target_time in enumerate(target_timestamps):
        right_index = int(np.searchsorted(source_timestamps, target_time, side="left"))
        if right_index == 0:
            indices[output_index] = 0
        elif right_index >= len(source_timestamps):
            indices[output_index] = len(source_timestamps) - 1
        else:
            left_index = right_index - 1
            left_error = abs(target_time - source_timestamps[left_index])
            right_error = abs(source_timestamps[right_index] - target_time)
            indices[output_index] = left_index if left_error <= right_error else right_index

    return indices


def select_nearest_source_frames(values: Any, source_indices: NDArray[np.int64]) -> NDArray:
    """Return the source frames at the chosen nearest-neighbor indices."""

    return np.asarray(values)[list(source_indices)]


def _validate_episode_lengths(episode: EpisodeArrays) -> int:
    lengths = {key: int(np.asarray(value).shape[0]) for key, value in episode.items()}
    if not lengths:
        raise ValueError("Episode is empty.")

    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"All episode fields must share the same length. Got lengths: {lengths}")

    return unique_lengths.pop()


def downsample_episode_arrays(
    episode: EpisodeArrays,
    config: TimeSyncConfig | None = None,
) -> EpisodeArrays:
    """Downsample one episode of internal arrays to the target rate."""

    config = config or TimeSyncConfig()
    num_frames = _validate_episode_lengths(episode)

    source_timestamps = build_source_timestamps(num_frames, config.source_fps)
    target_timestamps = build_target_timestamps(source_timestamps, config.target_fps)
    image_indices = choose_nearest_source_indices(source_timestamps, target_timestamps)

    output: EpisodeArrays = {}
    quaternion_keys = set(config.quaternion_keys)
    image_keys = set(config.image_keys)
    action_keys = set(config.action_keys)
    wrapped_angle_columns = config.wrapped_angle_columns

    if config.action_resampling_mode not in {"interval_average", "zero_order_hold"}:
        raise ValueError(
            "action_resampling_mode must be 'interval_average' or 'zero_order_hold'."
        )

    for key, values in episode.items():
        if key in quaternion_keys:
            output[key] = interpolate_quaternion_series(values, source_timestamps, target_timestamps)
        elif key in image_keys:
            output[key] = select_nearest_source_frames(values, image_indices)
        elif key in action_keys:
            if config.action_resampling_mode == "interval_average":
                output[key] = average_piecewise_constant_series(values, source_timestamps, target_timestamps)
            else:
                output[key] = sample_piecewise_constant_series(values, source_timestamps, target_timestamps)
        else:
            interpolated = interpolate_numeric_series(values, source_timestamps, target_timestamps)
            angle_columns = wrapped_angle_columns.get(key, ())
            if angle_columns:
                interpolated = np.asarray(interpolated, dtype=np.float64)
                source_values = np.asarray(values, dtype=np.float64)
                for column_index in angle_columns:
                    interpolated[:, column_index] = interpolate_wrapped_angle_series(
                        source_values[:, column_index],
                        source_timestamps=source_timestamps,
                        target_timestamps=target_timestamps,
                    )
            output[key] = interpolated

    return output


def split_observation_state(observation_state: Any) -> EpisodeArrays:
    """Split the AIC observation.state vector into named components."""

    observation_array = np.asarray(observation_state, dtype=np.float64)
    return {
        "observation.tcp_position": observation_array[:, TCP_POSITION_SLICE],
        "observation.tcp_orientation_xyzw": observation_array[:, TCP_ORIENTATION_SLICE],
        "observation.tcp_velocity": observation_array[:, TCP_VELOCITY_SLICE],
        "observation.tcp_error": observation_array[:, TCP_ERROR_SLICE],
        "observation.joint_positions": observation_array[:, JOINT_POSITION_SLICE],
    }


def combine_observation_state(episode_arrays: EpisodeArrays) -> NDArray:
    """Rebuild the original observation.state layout."""

    return np.concatenate(
        [
            episode_arrays["observation.tcp_position"],
            episode_arrays["observation.tcp_orientation_xyzw"],
            episode_arrays["observation.tcp_velocity"],
            episode_arrays["observation.tcp_error"],
            episode_arrays["observation.joint_positions"],
        ],
        axis=1,
    )


def infer_image_keys_from_episode(episode: dict[str, Any]) -> tuple[str, ...]:
    return tuple(key for key in episode if key.startswith(DEFAULT_IMAGE_KEY_PREFIX))


def infer_internal_image_keys(config: TimeSyncConfig, episode: dict[str, Any]) -> tuple[str, ...]:
    return config.image_keys if config.image_keys else infer_image_keys_from_episode(episode)


def clone_config_with_episode_image_keys(
    config: TimeSyncConfig | None,
    episode: dict[str, Any],
) -> TimeSyncConfig:
    base = config or TimeSyncConfig()
    return TimeSyncConfig(
        source_fps=base.source_fps,
        target_fps=base.target_fps,
        quaternion_keys=base.quaternion_keys,
        image_keys=infer_internal_image_keys(base, episode),
        action_keys=base.action_keys,
        action_resampling_mode=base.action_resampling_mode,
        wrapped_angle_columns=base.wrapped_angle_columns,
    )


def infer_torch_dtype(value: Any, default: torch.dtype = torch.float32) -> torch.dtype:
    if isinstance(value, torch.Tensor):
        return value.dtype
    if np.issubdtype(np.asarray(value).dtype, np.integer):
        return torch.int64
    return default


def to_numpy_episode(episode: dict[str, Any]) -> EpisodeArrays:
    return {key: np.asarray(value) for key, value in episode.items()}


def to_torch_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(value))
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def make_lerobot_frame_tensor_columns(
    downsampled_core: EpisodeArrays,
    source_episode: dict[str, Any],
    target_timestamps: Vector,
    image_keys: tuple[str, ...],
    episode_index: int | None = None,
    task_index: int | None = None,
    index_offset: int = 0,
) -> TorchEpisode:
    """Convert internal resampled arrays back into the original LeRobot tensor schema."""

    target_count = int(target_timestamps.shape[0])
    output: TorchEpisode = {
        ACTION_KEY: to_torch_tensor(
            downsampled_core[ACTION_KEY],
            dtype=infer_torch_dtype(source_episode[ACTION_KEY], torch.float32),
        ),
        OBSERVATION_STATE_KEY: to_torch_tensor(
            combine_observation_state(downsampled_core),
            dtype=infer_torch_dtype(source_episode[OBSERVATION_STATE_KEY], torch.float32),
        ),
        TIMESTAMP_KEY: torch.from_numpy(target_timestamps.astype(np.float32, copy=False)),
        FRAME_INDEX_KEY: torch.arange(target_count, dtype=torch.int64),
    }

    for image_key in image_keys:
        output[image_key] = to_torch_tensor(
            downsampled_core[image_key],
            dtype=infer_torch_dtype(source_episode[image_key], torch.float32),
        )

    if episode_index is not None:
        output[EPISODE_INDEX_KEY] = torch.full((target_count,), episode_index, dtype=torch.int64)
    if task_index is not None:
        output[TASK_INDEX_KEY] = torch.full((target_count,), task_index, dtype=torch.int64)

    output[INDEX_KEY] = torch.arange(index_offset, index_offset + target_count, dtype=torch.int64)
    return output


def downsample_lerobot_episode(
    episode: dict[str, Any],
    config: TimeSyncConfig | None = None,
    episode_index: int | None = None,
    task_index: int | None = None,
    index_offset: int = 0,
) -> TorchEpisode:
    """Downsample one episode and return torch tensors in the original LeRobot schema."""

    if ACTION_KEY not in episode or OBSERVATION_STATE_KEY not in episode:
        raise ValueError(
            f"Expected '{ACTION_KEY}' and '{OBSERVATION_STATE_KEY}' in the episode. Got: {sorted(episode)}"
        )

    config_with_images = clone_config_with_episode_image_keys(config, episode)
    image_keys = config_with_images.image_keys

    source_episode = {
        ACTION_KEY: episode[ACTION_KEY],
        OBSERVATION_STATE_KEY: episode[OBSERVATION_STATE_KEY],
        **{key: episode[key] for key in image_keys if key in episode},
    }

    num_frames = int(np.asarray(source_episode[ACTION_KEY]).shape[0])
    source_timestamps = build_source_timestamps(num_frames, config_with_images.source_fps)
    target_timestamps = build_target_timestamps(source_timestamps, config_with_images.target_fps)

    internal_episode: EpisodeArrays = {
        ACTION_KEY: np.asarray(source_episode[ACTION_KEY], dtype=np.float64),
        **split_observation_state(source_episode[OBSERVATION_STATE_KEY]),
        **{key: np.asarray(source_episode[key]) for key in image_keys if key in source_episode},
    }

    internal_config = TimeSyncConfig(
        source_fps=config_with_images.source_fps,
        target_fps=config_with_images.target_fps,
        quaternion_keys=config_with_images.quaternion_keys,
        image_keys=image_keys,
        action_keys=config_with_images.action_keys,
        action_resampling_mode=config_with_images.action_resampling_mode,
        wrapped_angle_columns=config_with_images.wrapped_angle_columns,
    )
    downsampled_core = downsample_episode_arrays(internal_episode, config=internal_config)
    return make_lerobot_frame_tensor_columns(
        downsampled_core=downsampled_core,
        source_episode=source_episode,
        target_timestamps=target_timestamps,
        image_keys=image_keys,
        episode_index=episode_index,
        task_index=task_index,
        index_offset=index_offset,
    )


def load_lerobot_episode_as_torch(
    dataset: Any,
    episode_index: int,
) -> tuple[TorchEpisode, str]:
    """Load one episode from LeRobotDataset as stacked torch tensors plus its task string."""

    episode_meta = dataset.meta.episodes[int(episode_index)]
    start_index = int(episode_meta["dataset_from_index"])
    end_index = int(episode_meta["dataset_to_index"])
    frame_indices = range(start_index, end_index)
    frames = [dataset[idx] for idx in frame_indices]
    if not frames:
        raise ValueError(f"Episode {episode_index} is empty.")

    first_frame = frames[0]
    image_keys = tuple(key for key in first_frame if key.startswith(DEFAULT_IMAGE_KEY_PREFIX))
    stacked_episode: TorchEpisode = {
        ACTION_KEY: torch.stack([frame[ACTION_KEY] for frame in frames]),
        OBSERVATION_STATE_KEY: torch.stack([frame[OBSERVATION_STATE_KEY] for frame in frames]),
        TIMESTAMP_KEY: torch.stack([frame[TIMESTAMP_KEY] for frame in frames]),
        FRAME_INDEX_KEY: torch.stack([frame[FRAME_INDEX_KEY] for frame in frames]),
        EPISODE_INDEX_KEY: torch.stack([frame[EPISODE_INDEX_KEY] for frame in frames]),
        TASK_INDEX_KEY: torch.stack([frame[TASK_INDEX_KEY] for frame in frames]),
        INDEX_KEY: torch.stack([frame[INDEX_KEY] for frame in frames]),
    }
    for image_key in image_keys:
        stacked_episode[image_key] = torch.stack([frame[image_key] for frame in frames])

    return stacked_episode, str(first_frame[TASK_KEY])


def clone_lerobot_features_for_target_fps(source_features: dict[str, Any], target_fps: float) -> dict[str, Any]:
    """Copy LeRobot features while updating the recorded FPS in video feature metadata."""

    features = deepcopy(source_features)
    for feature_spec in features.values():
        if feature_spec.get("dtype") == "video" and "info" in feature_spec:
            feature_spec["info"]["video.fps"] = int(round(target_fps))
    return features


def create_downsampled_lerobot_dataset(
    source_dataset: Any,
    output_repo_id: str,
    root: str | Path | None = None,
    config: TimeSyncConfig | None = None,
    episodes: Iterable[int] | None = None,
    video_backend: str | None = None,
) -> Any:
    """Create a new LeRobotDataset at the target FPS with the same schema as the source dataset."""

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    config = config or TimeSyncConfig()
    features = clone_lerobot_features_for_target_fps(source_dataset.meta.info["features"], config.target_fps)
    output_dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        root=root,
        fps=int(round(config.target_fps)),
        features=features,
        robot_type=source_dataset.meta.info.get("robot_type"),
        use_videos=len(getattr(source_dataset.meta, "video_keys", [])) > 0,
        video_backend=video_backend,
    )

    episode_indices = list(episodes) if episodes is not None else list(range(source_dataset.num_episodes))
    next_index_offset = 0

    for episode_index in episode_indices:
        source_episode, task = load_lerobot_episode_as_torch(source_dataset, episode_index)
        downsampled_episode = downsample_lerobot_episode(
            source_episode,
            config=config,
            episode_index=int(episode_index),
            task_index=int(source_episode[TASK_INDEX_KEY][0].item()),
            index_offset=next_index_offset,
        )
        target_length = int(downsampled_episode[ACTION_KEY].shape[0])
        for frame_index in range(target_length):
            frame: dict[str, Any] = {
                ACTION_KEY: downsampled_episode[ACTION_KEY][frame_index],
                OBSERVATION_STATE_KEY: downsampled_episode[OBSERVATION_STATE_KEY][frame_index],
                TIMESTAMP_KEY: float(downsampled_episode[TIMESTAMP_KEY][frame_index].item()),
                TASK_KEY: task,
            }
            for image_key in infer_image_keys_from_episode(downsampled_episode):
                frame[image_key] = downsampled_episode[image_key][frame_index]
            output_dataset.add_frame(frame)
        output_dataset.save_episode()
        next_index_offset += target_length

    output_dataset.finalize()
    return output_dataset


def downsample_episode(
    episode: dict[str, Any],
    config: TimeSyncConfig | None = None,
) -> TorchEpisode:
    """Public downsampling entry point returning torch tensors in LeRobot schema."""

    return downsample_lerobot_episode(episode, config=config)


def downsample_episode_30hz_to_20hz(
    episode: dict[str, Any],
    action_resampling_mode: str = "interval_average",
    image_keys: Iterable[str] | None = None,
) -> TorchEpisode:
    """Convenience wrapper for the exact conversion requested by the user."""

    inferred_image_keys = tuple(image_keys) if image_keys is not None else infer_image_keys_from_episode(episode)
    config = TimeSyncConfig(
        source_fps=30.0,
        target_fps=20.0,
        image_keys=inferred_image_keys,
        action_resampling_mode=action_resampling_mode,
    )
    return downsample_lerobot_episode(episode, config=config)
