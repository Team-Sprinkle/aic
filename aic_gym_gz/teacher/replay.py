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
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
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
        return {
            "original_steps": original_steps,
            "replay_steps": replay_steps,
            "step_delta": replay_steps - original_steps,
            "final_sim_time_delta": float(replay_final.get("sim_time", 0.0)) - float(original_final.get("sim_time", 0.0)),
            "distance_to_target_delta": float(replay_final.get("distance_to_target", 0.0))
            - float((original.final_info.get("distance_to_target") or 0.0)),
        }
