"""Teacher replay serialization and comparison."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..env import AicInsertionEnv


@dataclass(frozen=True)
class TeacherReplayArtifact:
    metadata: dict[str, Any]
    trajectory_segments: list[dict[str, Any]]
    probe_results: list[dict[str, Any]]
    planner_candidates: list[dict[str, Any]]
    step_logs: list[dict[str, Any]]
    final_info: dict[str, Any]
    limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "trajectory_segments": self.trajectory_segments,
            "probe_results": self.probe_results,
            "planner_candidates": self.planner_candidates,
            "step_logs": self.step_logs,
            "final_info": self.final_info,
            "limitations": self.limitations,
        }


def save_teacher_replay(artifact: TeacherReplayArtifact, path: Path | str) -> None:
    Path(path).write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_teacher_replay(path: Path | str) -> TeacherReplayArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return TeacherReplayArtifact(
        metadata=payload["metadata"],
        trajectory_segments=payload["trajectory_segments"],
        probe_results=payload.get("probe_results", []),
        planner_candidates=payload.get("planner_candidates", []),
        step_logs=payload.get("step_logs", []),
        final_info=payload.get("final_info", {}),
        limitations=payload.get("limitations", []),
    )


@dataclass
class TeacherReplayRunner:
    env: AicInsertionEnv

    def replay(self, artifact: TeacherReplayArtifact) -> dict[str, Any]:
        seed = artifact.metadata.get("seed")
        trial_id = artifact.metadata.get("trial_id")
        options = {"trial_id": trial_id} if trial_id else {}
        try:
            observation, info = self.env.reset(seed=seed, options=options)
        except KeyError:
            observation, info = self.env.reset(seed=seed)
        records = [
            {
                "sim_tick": int(observation["sim_tick"]),
                "sim_time": float(observation["sim_time"]),
                "distance_to_target": float(observation["plug_to_port_relative"][3]),
                "tcp_pose": np.asarray(observation["tcp_pose"], dtype=np.float64).tolist(),
                "plug_pose": np.asarray(observation["plug_pose"], dtype=np.float64).tolist(),
                "target_port_pose": np.asarray(observation["target_port_pose"], dtype=np.float64).tolist(),
                "plug_to_port_relative": np.asarray(
                    observation["plug_to_port_relative"], dtype=np.float64
                ).tolist(),
                "off_limit_contact": bool(np.asarray(observation["off_limit_contact"]).reshape(-1)[0] > 0.5),
            }
        ]
        final_info = info
        for segment in artifact.trajectory_segments:
            for point in segment.get("points", []):
                action = np.asarray(point["action"], dtype=np.float32)
                observation, reward, terminated, truncated, final_info = self.env.step(action)
                records.append(
                    {
                        "sim_tick": int(observation["sim_tick"]),
                        "sim_time": float(observation["sim_time"]),
                        "distance_to_target": float(observation["plug_to_port_relative"][3]),
                        "tcp_pose": np.asarray(observation["tcp_pose"], dtype=np.float64).tolist(),
                        "plug_pose": np.asarray(observation["plug_pose"], dtype=np.float64).tolist(),
                        "target_port_pose": np.asarray(observation["target_port_pose"], dtype=np.float64).tolist(),
                        "plug_to_port_relative": np.asarray(
                            observation["plug_to_port_relative"], dtype=np.float64
                        ).tolist(),
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "off_limit_contact": bool(
                            np.asarray(observation["off_limit_contact"]).reshape(-1)[0] > 0.5
                        ),
                    }
                )
                if terminated or truncated:
                    return {"records": records, "final_info": final_info}
        return {"records": records, "final_info": final_info}


class TeacherReplayComparator:
    def compare(
        self,
        *,
        original: TeacherReplayArtifact,
        replayed: dict[str, Any],
    ) -> dict[str, Any]:
        original_steps = max(len(original.step_logs), 1)
        replay_steps = max(len(replayed.get("records", [])) - 1, 0)
        original_final = original.step_logs[-1] if original.step_logs else {}
        replay_final = replayed.get("records", [{}])[-1]
        original_final_obs = original_final.get("observation_summary", {})
        replay_final_eval = dict((replayed.get("final_info") or {}).get("final_evaluation") or {})
        return {
            "original_steps": original_steps,
            "replay_steps": replay_steps,
            "step_delta": replay_steps - original_steps,
            "final_sim_time_delta": float(replay_final.get("sim_time", 0.0)) - float(original_final.get("sim_time", 0.0)),
            "distance_to_target_delta": float(replay_final.get("distance_to_target", 0.0))
            - float((original.final_info.get("distance_to_target") or 0.0)),
            "final_tcp_pose_delta": _pose_delta(
                original_final_obs.get("tcp_pose"),
                replay_final.get("tcp_pose"),
            ),
            "final_plug_to_port_relative_delta": _pose_delta(
                original_final_obs.get("plug_to_port_relative"),
                replay_final.get("plug_to_port_relative"),
            ),
            "reward_total_delta": sum(float(record.get("reward", 0.0)) for record in replayed.get("records", []))
            - sum(float(step.get("reward", 0.0)) for step in original.step_logs),
            "replay_gym_final_score": replay_final_eval.get("gym_final_score"),
            "original_gym_final_score": original.metadata.get("final_metrics", {}).get("gym_final_score"),
            "off_limit_contact_delta": sum(bool(record.get("off_limit_contact", False)) for record in replayed.get("records", []))
            - sum(bool(step.get("observation_summary", {}).get("off_limit_contact", False)) for step in original.step_logs),
            "metadata_keys": sorted(original.metadata.keys()),
        }


def _pose_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    left_array = np.asarray(left, dtype=np.float64).reshape(-1)
    right_array = np.asarray(right, dtype=np.float64).reshape(-1)
    count = min(left_array.size, right_array.size)
    if count == 0:
        return None
    return float(np.linalg.norm(left_array[:count] - right_array[:count]))
