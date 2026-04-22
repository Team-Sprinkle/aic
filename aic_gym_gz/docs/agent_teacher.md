# Agent Teacher

This document describes the additive teacher stack layered on top of the merged
`feat/gazebo-gym-scn` base environment.

## Layering rules

- The base env remains the source of truth for `rl_step_reward`,
  `gym_final_score`, observation schema, and runtime parity semantics.
- Teacher code builds temporal memory, planning, ranking, replay, and export on
  top of the base env.
- Teacher artifacts never claim `official_eval_score` unless the official
  external path is actually run.

## Teacher planner state parity

`TeacherPlanningState` now carries four distinct views:

- `policy_context`: current-observation-style fields from the base env only.
- `controller_context`: controller-derived fields such as reference TCP pose,
  TCP error, and controller target mode when available.
- `camera_context`: image refs, timestamps, and camera-info summaries when
  available.
- `temporal_context`: teacher-side memory built above the env rather than added
  to the env observation contract.

The planner state also includes `data_quality`, which marks whether each signal
is real, synthetic, or missing. Current tracked signals include:

- `wrench`
- `controller_state`
- `camera_info`
- `target_port_entrance_pose`
- `partial_insertion_depth`
- `tier1_validity`

Missing data is explicit. The teacher layer does not silently treat zero-filled
or unavailable signals as official-quality observations.

## Temporal history wrapper

`TemporalObservationBuffer` is the reusable history helper for teacher planning
and future policy wrappers.

It stores:

- wrench history and timestamps
- action history
- TCP velocity history
- controller-state history when present
- image timestamp history
- image summaries and camera-info snapshots
- signal-quality snapshots

Two views are intentionally separated:

- `current_observation_view()`: official-compatible current observation only
- `teacher_memory_summary()`: additive teacher-side memory/history

This keeps the base env observation contract stable while allowing richer
teacher-side planning.

Example:

```bash
pixi run python -m aic_gym_gz.demo_teacher_history_context
```

## Planner backends

### Mock planner

`aic_gym_gz.planners.mock.DeterministicMockPlannerBackend` remains the default
test and smoke backend.

### OpenAI Responses planner

`aic_gym_gz.planners.openai_backend.OpenAIPlannerBackend` now performs real
Responses API requests with strict JSON Schema output validation.

Implemented behavior:

- runtime API key lookup from `OPENAI_API_KEY`
- strict schema validation before converting model output into `TeacherPlan`
- retry and timeout handling
- per-episode planner call limits
- per-search planner call limits
- optional plan caching keyed by normalized planning input

The backend is conservative about secret handling:

- it reads the key only from the environment
- it does not print or serialize the key
- it strips API-key-like fields out of planner metadata exports

Example rollout:

```bash
pixi run python -m aic_gym_gz.demo_teacher_rollout \
  --planner-backend openai \
  --openai-model gpt-5.4-mini \
  --openai-timeout 20 \
  --openai-max-retries 2 \
  --output aic_gym_gz/artifacts/teacher_rollout_openai.json
```

Example search:

```bash
pixi run python -m aic_gym_gz.run_teacher_search \
  --planner-backend openai \
  --openai-model gpt-5.4-mini \
  --openai-max-calls-per-search 64 \
  --output aic_gym_gz/artifacts/teacher_search_openai.json
```

## Candidate ranking under approximate signals

Teacher search now keeps explicit ranking signals separate:

- `teacher_official_style_score`
- `gym_final_score`
- `rl_step_reward_total`
- signal-quality metadata and penalties

Ranking remains conservative:

- missing or synthetic wrench data reduces ranking trust
- missing controller state reduces ranking trust
- approximate partial-insertion depth is labeled and penalized
- Tier 1 validity remains marked approximate locally

The ranked search artifact preserves full metric breakdowns for all candidates,
not only the selected top candidate.

## Replay and export

Teacher replay and dataset export now preserve:

- scenario metadata
- task metadata
- planner metadata
- selected candidate metrics
- signal-quality flags
- history metadata

JSONL export writes per-step quality flags and history summaries. LeRobot export
uses logged controller TCP error when present and falls back to synthesized
target-vs-observed error only when necessary.

Representative commands:

```bash
pixi run python -m aic_gym_gz.run_teacher_search \
  --output aic_gym_gz/artifacts/teacher_search.json

pixi run python -m aic_gym_gz.export_teacher_official_replay \
  --search-artifact aic_gym_gz/artifacts/teacher_search.json \
  --output aic_gym_gz/artifacts/teacher_selected_replay.json

pixi run python -m aic_gym_gz.export_teacher_dataset \
  --search-artifact aic_gym_gz/artifacts/teacher_search.json \
  --output-dir aic_gym_gz/artifacts/teacher_dataset \
  --format jsonl
```

## Evaluating OpenAI teacher outputs

The evaluation helpers are intentionally analysis-first. They do not change the
base env semantics or infer official scores.

### Rollout evaluation

`aic_gym_gz.evaluate_teacher_rollout` summarizes:

- phase sequence
- planner call count
- candidate count
- segment count
- `rl_step_reward_total`
- `gym_final_score`
- teacher official-style score when available
- signal-quality flags
- outcome / failure mode
- path length, duration, and smoothness
- warnings such as collapse, weak adaptation, or dependence on approximate
  signals

### Search evaluation

`aic_gym_gz.evaluate_teacher_search` compares all ranked candidates and reports:

- top-K ranking summary
- whether quality penalties changed rank order
- candidate diversity / near-duplicate detection
- whether search materially beats the best single planner candidate
- whether the ranking appears dominated by one signal

### Replay evaluation

`aic_gym_gz.evaluate_teacher_replay` replays an artifact and labels the result:

- `faithful`
- `approximately faithful`
- `poor replay match`

The thresholds are local and conservative. They are based on step drift, final
TCP drift, final plug-target drift, reward drift, and local `gym_final_score`
drift.

### Example commands

```bash
pixi run python -m aic_gym_gz.evaluate_teacher_rollout \
  --artifact /tmp/teacher_rollout_openai.json \
  --output-json /tmp/teacher_rollout_eval.json \
  --output-markdown /tmp/teacher_rollout_eval.md

pixi run python -m aic_gym_gz.evaluate_teacher_search \
  --artifact /tmp/teacher_search_openai.json \
  --output-json /tmp/teacher_search_eval.json \
  --output-markdown /tmp/teacher_search_eval.md

pixi run python -m aic_gym_gz.evaluate_teacher_replay \
  --artifact aic_gym_gz/artifacts/teacher_selected_replay.json \
  --output-json /tmp/teacher_replay_eval.json \
  --output-markdown /tmp/teacher_replay_eval.md
```

## Current practical findings

Current OpenAI planning is functional, but evaluation can still reveal:

- planner collapse into one dominant phase
- near-duplicate search candidates
- small gains from search over a single plan
- strong local scores despite non-insertion outcomes

For that reason, treat the teacher stack as useful for local analysis and data
generation, but not yet as a proxy for official evaluation.

## Still approximate relative to official rollout

- live wrench quality still depends on the downstream bridge exposing real data
- controller-state parity is only as good as what the runtime observer receives
- camera-info parity depends on the ROS camera sidecar path
- partial-insertion depth is still a local approximation
- Tier 1 validity is still local-only unless the trajectory is run through
  official `aic_model`
- the base RL reward remains intentionally simpler than official-like teacher
  scoring
