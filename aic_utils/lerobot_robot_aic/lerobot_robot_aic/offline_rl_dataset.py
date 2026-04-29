#!/usr/bin/env python3

#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""Offline RL transition loading for local AIC LeRobot datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .dataset_schema import DatasetSchemaSummary, summarize_dataset_schema

RewardMode = Literal["dataset", "final_success", "zero"]
ObsMode = Literal["lowdim"]


@dataclass(frozen=True)
class TransitionArrays:
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_obs: np.ndarray
    done: np.ndarray
    episode_index: np.ndarray
    frame_index: np.ndarray


@dataclass(frozen=True)
class NormalizationStats:
    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "obs_mean": self.obs_mean.tolist(),
            "obs_std": self.obs_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }


def _read_dataframes(dataset_root: Path) -> pd.DataFrame:
    data_files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No LeRobot parquet data found under {dataset_root / 'data'}")
    frames = [pd.read_parquet(path) for path in data_files]
    df = pd.concat(frames, ignore_index=True)
    required = {"observation.state", "action", "episode_index", "frame_index"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required lowdim columns: {missing}")
    sort_keys = ["episode_index", "frame_index"]
    if "index" in df.columns:
        sort_keys.append("index")
    return df.sort_values(sort_keys)


def _stack_vector_column(series: pd.Series, key: str) -> np.ndarray:
    values = [np.asarray(v, dtype=np.float32).reshape(-1) for v in series]
    if not values:
        raise ValueError(f"Column '{key}' is empty")
    dim = int(values[0].shape[0])
    if dim <= 0:
        raise ValueError(f"Column '{key}' has empty vectors")
    bad = [idx for idx, value in enumerate(values) if int(value.shape[0]) != dim]
    if bad:
        raise ValueError(f"Column '{key}' has inconsistent vector size at row {bad[0]}")
    return np.stack(values, axis=0).astype(np.float32)


def _rewards(df: pd.DataFrame, reward_mode: RewardMode) -> np.ndarray:
    if reward_mode == "dataset":
        if "reward" in df.columns:
            return df["reward"].to_numpy(dtype=np.float32)
        reward_mode = "final_success"
    if reward_mode == "zero":
        return np.zeros(len(df), dtype=np.float32)
    if reward_mode == "final_success":
        rewards = np.zeros(len(df), dtype=np.float32)
        last_indices = df.groupby("episode_index", sort=False).tail(1).index.to_numpy()
        rewards[last_indices] = 1.0
        return rewards
    raise ValueError(f"Unsupported reward mode: {reward_mode}")


def load_lerobot_transitions(
    dataset_root: Path,
    *,
    reward_mode: RewardMode = "dataset",
    obs_mode: ObsMode = "lowdim",
    action_horizon: int = 1,
) -> tuple[TransitionArrays, DatasetSchemaSummary]:
    if obs_mode != "lowdim":
        raise ValueError("Only obs-mode=lowdim is implemented for the offline SERL smoke path.")
    if action_horizon < 1:
        raise ValueError("action_horizon must be >= 1")

    schema = summarize_dataset_schema(dataset_root)
    df = _read_dataframes(dataset_root).reset_index(drop=True)

    obs = _stack_vector_column(df["observation.state"], "observation.state")
    single_step_action = _stack_vector_column(df["action"], "action")
    reward = _rewards(df, reward_mode)
    episode = df["episode_index"].to_numpy(dtype=np.int64)
    frame = df["frame_index"].to_numpy(dtype=np.int64)
    action = _action_chunks(single_step_action, episode, action_horizon)

    next_obs = obs.copy()
    done = np.ones(len(df), dtype=np.float32)
    if len(df) > 1:
        same_episode_next = episode[:-1] == episode[1:]
        next_indices = np.where(same_episode_next)[0]
        next_obs[next_indices] = obs[next_indices + 1]
        done[next_indices] = 0.0

    arrays = TransitionArrays(
        obs=obs,
        action=action,
        reward=reward.astype(np.float32),
        next_obs=next_obs.astype(np.float32),
        done=done,
        episode_index=episode,
        frame_index=frame,
    )
    return arrays, schema


def _action_chunks(action: np.ndarray, episode: np.ndarray, horizon: int) -> np.ndarray:
    if horizon == 1:
        return action
    chunks = np.empty((action.shape[0], action.shape[1] * horizon), dtype=np.float32)
    for idx in range(action.shape[0]):
        episode_id = episode[idx]
        values = []
        last_valid = action[idx]
        for offset in range(horizon):
            src = idx + offset
            if src < action.shape[0] and episode[src] == episode_id:
                last_valid = action[src]
            values.append(last_valid)
        chunks[idx] = np.concatenate(values, axis=0)
    return chunks


class OfflineRLTransitionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        arrays: TransitionArrays,
        *,
        normalize: bool = True,
        eps: float = 1e-6,
    ):
        self.raw = arrays
        self.normalize = normalize
        obs_mean = arrays.obs.mean(axis=0)
        obs_std = arrays.obs.std(axis=0) + eps
        action_mean = arrays.action.mean(axis=0)
        action_std = arrays.action.std(axis=0) + eps
        self.stats = NormalizationStats(obs_mean, obs_std, action_mean, action_std)

        if normalize:
            self.obs = (arrays.obs - obs_mean) / obs_std
            self.next_obs = (arrays.next_obs - obs_mean) / obs_std
            self.action = (arrays.action - action_mean) / action_std
        else:
            self.obs = arrays.obs
            self.next_obs = arrays.next_obs
            self.action = arrays.action

    @property
    def obs_dim(self) -> int:
        return int(self.obs.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.action.shape[1])

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.as_tensor(self.obs[idx], dtype=torch.float32),
            "action": torch.as_tensor(self.action[idx], dtype=torch.float32),
            "reward": torch.as_tensor([self.raw.reward[idx]], dtype=torch.float32),
            "next_obs": torch.as_tensor(self.next_obs[idx], dtype=torch.float32),
            "done": torch.as_tensor([self.raw.done[idx]], dtype=torch.float32),
        }
