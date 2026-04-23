# Agent Teacher Audit

This audit summarizes the current teacher-layer parity state on top of the
merged base environment.

## Observation parity

| Official observation item | Current teacher status | Status |
| --- | --- | --- |
| Wrist camera images | Live teacher path receives images through the ROS sidecar; mock path still uses placeholders | Approximate |
| Image timestamps | Preserved in current observation view, temporal history, planning state, replay, and export | Matched |
| Wrist wrench | Propagated when runtime publishes timestamped wrench data; otherwise marked synthetic or missing | Approximate |
| Auxiliary within-step force/contact summary | Preserved separately from official-compatible observation and marked real, synthetic, or missing | Teacher-only additive |
| Controller state / reference TCP / TCP error | Included in teacher planning state, rollout logs, replay, and export when runtime exposes it | Approximate |
| CameraInfo | Included in teacher planning state and logs when the sidecar provides it; placeholder or missing cases are flagged | Approximate |
| Current observation vs memory separation | Explicitly split between base env observation and teacher-side temporal history | Matched |

## Ranking and scoring behavior

Teacher search now keeps these signals separate in ranked artifacts:

- `teacher_official_style_score`
- `gym_final_score`
- `rl_step_reward_total`
- signal-quality metadata
- quality-aware ranking penalties
- auxiliary hidden-contact penalties

Conservative handling now implemented:

- missing or synthetic wrench reduces ranking trust
- missing controller state reduces ranking trust
- approximate partial-insertion depth is labeled and penalized
- Tier 1 validity remains explicitly approximate
- repeated hidden transient contacts and quiet-final-sample auxiliary force gaps
  can conservatively penalize ranking without changing official observation
  semantics

## Replay and export behavior

Replay artifacts and dataset exports now preserve:

- scenario metadata
- task metadata
- planner metadata
- final metrics
- selected candidate ranking metrics
- signal-quality flags
- history metadata
- auxiliary summary availability and compact hidden-contact metrics

JSONL export includes per-step data-quality and history summaries. LeRobot
export now prefers logged controller TCP error when available.

## Still approximate

- live wrench fidelity remains limited by runtime bridge availability
- controller-state parity is limited by runtime observer inputs
- camera-info fidelity depends on the ROS sidecar path
- partial insertion depth remains a local approximation
- Tier 1 validity remains approximate unless executed through official
  `aic_model`
- teacher scoring is more official-like than RL reward, but they remain
  intentionally separate
- auxiliary within-step summaries help detect transient contacts under coarse
  stepping, but they remain teacher-only and non-official

## Validation commands

```bash
pixi run python -m aic_gym_gz.run_teacher_audit \
  --output aic_gym_gz/artifacts/teacher_audit.json

pixi run python -m unittest discover -s aic_gym_gz/tests -p 'test_teacher*.py'
```
