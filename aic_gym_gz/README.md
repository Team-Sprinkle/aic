# aic_gym_gz

`aic_gym_gz` is a standalone Gym-style training path for the AIC cable insertion
challenge. It is intended to be a usable RL substrate with synchronous
`reset()` / `step()` semantics while staying as behaviorally aligned as practical
with the official AIC rollout surface.

It is separate from the official ROS-first evaluation flow. The package reuses:

- the official trial YAML schema from `aic_engine/config/sample_config.yaml`
- the official task board and cable geometry definitions from `aic_description`
- the official scoring shape from `aic_scoring/src/ScoringTier2.cc`

## Quick start

```bash
pixi run python -m unittest discover -s aic_gym_gz/tests
pixi run python -m aic_gym_gz.demo_random_policy --print-every 16
pixi run python -m aic_gym_gz.demo_heuristic_policy
pixi run python -m aic_gym_gz.compare_reward_to_final_score \
  --policies random heuristic toward_target away_from_target oscillate_in_place collide_intentionally \
  --episodes 8
pixi run python -m aic_gym_gz.validate_reward_behavior \
  --output-dir aic_gym_gz/artifacts/validation/reward_behavior
pixi run python -m aic_gym_gz.validate_observation_temporal \
  --output-dir aic_gym_gz/artifacts/validation/observation_temporal
```

## Reward and score labels

Use these labels literally and consistently.

| Name | Computed Where | Purpose | Exact vs Approx |
| --- | --- | --- | --- |
| `rl_step_reward` | `env.step()` via `AicInsertionTask.evaluate_step()` | Dense per-step RL training reward | Local shaped reward, intentionally not equal to final score |
| `gym_final_score` | `task.final_evaluation()` at episode end | Episode-level evaluation and analysis inside `aic_gym_gz` | Local trajectory-level approximation using the strongest available geometry and traces |
| `official_eval_score` | Outside `aic_gym_gz`, via the official toolkit path | Ground-truth official evaluation and leaderboard-facing result | Official source of truth only when actually run |
| `teacher_official_style_score` | Teacher-side systems layered on top of `aic_gym_gz` | Teacher-layer approximation or ranking helper | Teacher-specific approximation, not official |

### `rl_step_reward`

- `rl_step_reward` is the per-step training reward returned by `env.step()`.
- It is dense, local, shaped, and intended for optimization.
- It follows Isaac-Lab-style reward design rather than trying to exactly
  reconstruct the final score trajectory term by term.
- It should not be interpreted as an official score.

### `gym_final_score`

- `gym_final_score` is computed at episode end from the full trajectory stored
  by the task.
- It uses local geometry, timing, wrench/contact traces, and local final-score
  logic.
- It is useful for evaluation, analysis, ablations, and teacher-side ranking.
- It is not the same thing as `official_eval_score`.

### `official_eval_score`

- `official_eval_score` is not computed inside `aic_gym_gz`.
- It requires running the official toolkit / official rollout path.
- If you need leaderboard-relevant or ground-truth numbers, you must use the
  official toolkit path.

## RL reward design

The per-step RL reward follows Isaac-Lab-style shaping with configurable local
terms. The weights live in `aic_gym_gz.reward.AicRlRewardWeights`.

Current reward shaping includes:

- potential-based target progress
- potential-based target-entrance progress
- potential-based insertion-corridor progress
- orientation-alignment progress
- lateral corridor-alignment shaping
- local action magnitude penalty
- local action-delta penalty
- local TCP-velocity-delta penalty
- local force penalty from the current wrench sample
- local off-limit contact penalty
- short-history oscillation penalty
- per-step time penalty
- partial-insertion bonus
- terminal success bonus
- terminal wrong-port penalty

Important:

- The sum of `rl_step_reward` over an episode is not designed to equal
  `gym_final_score`.
- Training should optimize `rl_step_reward`.
- Analysis and model-selection reports may also look at `gym_final_score`.

## Observation schema

The public observation dictionary is stable and explicit.

State-mode keys:

- `step_count`
- `sim_tick`
- `sim_time`
- `joint_positions`
- `joint_velocities`
- `gripper_state`
- `tcp_pose`
- `tcp_velocity`
- `plug_pose`
- `target_port_pose`
- `target_port_entrance_pose`
- `plug_to_port_relative`
- `wrench`
- `wrench_timestamp`
- `off_limit_contact`
- `controller_tcp_pose`
- `controller_reference_tcp_pose`
- `controller_tcp_velocity`
- `controller_tcp_error`
- `controller_reference_joint_state`
- `controller_target_mode`
- `fts_tare_wrench`
- `score_geometry.distance_to_target`
- `score_geometry.distance_threshold`
- `score_geometry.plug_to_port_depth`
- `score_geometry.port_to_entrance_depth`
- `score_geometry.distance_to_entrance`
- `score_geometry.lateral_misalignment`
- `score_geometry.orientation_error`
- `score_geometry.insertion_progress`
- `score_geometry.partial_insertion`

Additional image-mode keys:

- `images["left"]`
- `images["center"]`
- `images["right"]`
- `image_timestamps`
- `camera_info["left"|"center"|"right"]`

### F/T sensor behavior

- The policy receives only the current `wrench` sample for the current step.
- No built-in wrench history is provided in the public observation.
- This is closer to the official participant-facing observation surface than
  providing an implicit history buffer inside the environment.

### Temporal behavior

- One `env.step()` corresponds to one policy observation timestep.
- In the default configuration, the policy sees one observation per held action.
- If `ticks_per_step > 1`, transient contact or force spikes that occur inside
  the held-action window may be missed because only the final sample of the step
  is returned.
- There is currently no built-in within-step max or window summary for F/T.

### Recommended usage for temporal signals

- Build temporal memory at the policy level.
- Use stacking, recurrence, or external history buffers if your policy depends
  on contact transients or settling behavior.
- Treat `wrench` as a current sample, not a trajectory summary.

Example: stack recent wrench samples yourself.

```python
from collections import deque

wrench_history = deque(maxlen=8)

observation, info = env.reset(seed=123)
wrench_history.append(observation["wrench"].copy())

action = env.action_space.sample()
observation, reward, terminated, truncated, step_info = env.step(action)
wrench_history.append(observation["wrench"].copy())

stacked_wrench = list(wrench_history)
```

## Using this environment for RL training

This environment is suitable for gym-style RL training as long as you keep the
current limitations in mind.

Best practices:

- optimize `rl_step_reward`, not `gym_final_score`
- maintain your own observation history if temporal context matters
- start with state-only training before enabling images
- use `reward_terms` and `reward_metrics` from `step_info` to debug shaping
- tune reward weights if your learner exploits weak penalties

Recommended interpretation:

- `rl_step_reward` is for learning
- `gym_final_score` is for local episode analysis
- `official_eval_score` is for final external validation only

Current limitations for RL:

- the mock backend can be overly forgiving
- the local penalties do not necessarily suppress every undesirable behavior
- reward-vs-final-score correlation is only approximate
- F/T is current-sample-only, so transient spikes may be missed

## Agent Teacher Layer (feat/agent-teacher)

`feat/agent-teacher` is an additive teacher stack built on top of the base
environment. It does not redefine the base reward, final-score, geometry, or
observation semantics.

The teacher layer uses:

- temporal history
- probes
- planning through deterministic mock backends and a live OpenAI Responses API backend
- trajectory smoothing
- candidate search
- replay artifacts and dataset export

Teacher-side tooling consumes the base environment exactly as exposed here:

- teacher rollouts optimize and analyze against the base env's `rl_step_reward`
- episode ranking and local evaluation use the base env's `gym_final_score`
- `official_eval_score` remains outside `aic_gym_gz` and outside the teacher
  stack unless the official toolkit path is run separately

Teacher-side extensions remain additive to the base env:

- current observation semantics stay in the base env
- temporal memory is built in `aic_gym_gz.teacher.history.TemporalObservationBuffer`
- planning state now carries explicit controller-state, reference-TCP, camera-info,
  and signal-quality metadata when available
- missing or synthetic wrench/controller/camera-info signals are marked
  explicitly and propagated into ranking, replay, and dataset export metadata

Representative teacher entry points:

```bash
pixi run python -m aic_gym_gz.demo_teacher_rollout \
  --output aic_gym_gz/artifacts/teacher_rollout.json
pixi run python -m aic_gym_gz.demo_teacher_rollout \
  --planner-backend openai \
  --openai-model gpt-5.4-mini \
  --output aic_gym_gz/artifacts/teacher_rollout_openai.json
pixi run python -m aic_gym_gz.demo_teacher_history_context
pixi run python -m aic_gym_gz.replay_teacher_artifact \
  --artifact aic_gym_gz/artifacts/teacher_rollout.json
pixi run python -m aic_gym_gz.compare_teacher_replay \
  --artifact aic_gym_gz/artifacts/teacher_rollout.json
pixi run python -m aic_gym_gz.run_teacher_audit \
  --output aic_gym_gz/artifacts/teacher_audit.json
pixi run python -m aic_gym_gz.run_teacher_search \
  --planner-backend openai \
  --openai-model gpt-5.4-mini \
  --output aic_gym_gz/artifacts/teacher_search.json
pixi run python -m aic_gym_gz.export_teacher_dataset \
  --search-artifact aic_gym_gz/artifacts/teacher_search.json \
  --output-dir aic_gym_gz/artifacts/teacher_dataset \
  --format jsonl
```

## Compatibility with official AIC evaluation

The observation interface is designed to stay close to the official toolkit and
official rollout surface, but it is not identical to running the official
toolchain.

What is intentionally aligned:

- explicit task geometry in observation
- current-sample state exposure rather than built-in history
- controller-state propagation when available
- wrench/contact/image/camera-info propagation when the live dependencies exist

Key differences and caveats:

- the policy must maintain its own temporal memory above the env
- some live fields depend on ROS topics being present
- `gym_final_score` is a local final-score path, not the official toolkit score
- final validation must be done with the official toolkit path
- teacher search/ranking/export are quality-aware but still conservative when
  wrench/controller/camera-info parity is incomplete

State this explicitly:

- `gym_final_score != official_eval_score`
- `official_eval_score` must be treated as the official ground truth

## Example usage

### Step the environment

```python
import numpy as np

from aic_gym_gz.env import make_default_env

env = make_default_env()
observation, info = env.reset(seed=123)

action = np.zeros(6, dtype=np.float32)
observation, reward, terminated, truncated, step_info = env.step(action)

print(step_info["reward_label"], reward)
print(step_info["reward_terms"])
print(step_info["reward_metrics"])

if terminated or truncated:
    print(step_info["final_evaluation"]["gym_final_score"])

env.close()
```

### Interpret reward versus final score

```python
training_reward_total = 0.0

observation, info = env.reset(seed=123)
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, step_info = env.step(action)
    training_reward_total += reward
    done = terminated or truncated

final_report = step_info.get("final_evaluation") or env.task.final_evaluation()

print("training reward total:", training_reward_total)
print("gym final score:", final_report["gym_final_score"])
print("official eval score:", final_report["official_eval_score"])
```

## Runtime parity and checkpoints

Runtime parity audit:

```bash
pixi run python -m aic_gym_gz.audit_runtime \
  --output-json /tmp/aic_gym_runtime_audit.json \
  --output-markdown /tmp/aic_gym_runtime_audit.md
```

Runtime checkpoint export:

```bash
pixi run python -m aic_gym_gz.export_runtime_checkpoint \
  --output /tmp/aic_gym_checkpoint.json
```

Checkpoint status:

- mock backend checkpoint/restore is exact
- live checkpoint export is reset-and-rerun metadata only
- live midpoint restore is still approximate / unavailable

## Validation summary

Current validation status, based on the checked-in validation report and recent
branch checks:

- reward is dense and usable for RL experiments, but imperfect
- reward-vs-score correlation exists, but is inflated in the mock backend
- F/T is current-sample-only, so within-step transients may be missed
- live parity exists in branch artifacts, but was not fully revalidated in every
  environment / shell

The detailed validation write-up is in
[docs/validation_report.md](/home/ubuntu/ws_aic/src/aic/aic_gym_gz/docs/validation_report.md).

## What is real today

- deterministic `reset(seed=...)`
- exact synchronous `step(action)` over a configured number of ticks
- explicit state-only observation schema
- optional image observations with timestamps and `camera_info`
- dense `rl_step_reward` with explicit reward-term breakdown
- separate `gym_final_score` at episode end
- validation scripts for reward behavior, temporal observation behavior, and
  reward-vs-final-score correlation
- checked-in live parity artifacts for official-path comparison
- additive teacher tooling for temporal diagnostics, candidate search, replay,
  and dataset export on top of the base env

## What is still approximate

- the default tested backend in this shell is deterministic and simulator-free
- the live backend is routed through `aic_utils/aic_gazebo_env`, not upstream
  ScenarIO / gym-gz
- image ingestion still uses a ROS sidecar fallback rather than pure Gazebo
  Transport
- live wrench/contact/controller parity depends on ROS topic availability
- `official_eval_score` is not computed inside `aic_gym_gz`
- short-horizon mock rollouts can make `gym_final_score` look less
  discriminative than a real official rollout
- teacher-side candidate ranking is still a local approximation unless the
  official toolkit path is run separately

See also:

- [docs/architecture.md](/home/ubuntu/ws_aic/src/aic/aic_gym_gz/docs/architecture.md)
- [docs/runtime_parity.md](/home/ubuntu/ws_aic/src/aic/aic_gym_gz/docs/runtime_parity.md)
- [docs/rl_reward_design.md](/home/ubuntu/ws_aic/src/aic/aic_gym_gz/docs/rl_reward_design.md)
- [docs/validation_report.md](/home/ubuntu/ws_aic/src/aic/aic_gym_gz/docs/validation_report.md)
- [docs/agent_teacher.md](/home/ubuntu/ws_aic/src/aic/aic_gym_gz/docs/agent_teacher.md)
- [docs/agent_teacher_audit.md](/home/ubuntu/ws_aic/src/aic/aic_gym_gz/docs/agent_teacher_audit.md)
