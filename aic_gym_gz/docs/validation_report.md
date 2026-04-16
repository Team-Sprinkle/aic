# Gazebo-Gym Validation Report

Date: 2026-04-16
Branch: `feat/gazebo-gym-scn`

This report separates evidence gathered in this turn from pre-existing live
official-parity artifacts already checked into the branch.

## Scope and execution boundary

Freshly executed in this turn:

- `pixi run python -m unittest discover -s aic_gym_gz/tests`
- `pixi run python -m aic_gym_gz.validate_reward_behavior`
- `pixi run python -m aic_gym_gz.validate_observation_temporal`
- `pixi run python -m aic_gym_gz.compare_reward_to_final_score`

Not freshly executable in this shell:

- official ROS/Gazebo rollout capture
- live `aic_gym_gz` attached-runtime replay
- official toolkit scoring

Reason:

- `ros2` is not on `PATH`
- `gz` is not on `PATH`
- `pixi run python -m aic_gym_gz.live_training_smoke ...` fails on missing
  `ros_gz_interfaces`

So the gym-vs-official comparison section below uses checked-in live artifacts:

- `aic_gym_gz/artifacts/deterministic_policy_state/*`
- `aic_gym_gz/artifacts/live_benchmark_state.json`

## 1. Reward sanity

### Result

Partially works.

### What passed

- The reward is dense. In the fresh mock validation, every tested policy had
  `reward_nonzero_fraction = 1.0`.
- Goal-seeking policies score higher than weaker policies:
  - `heuristic`: cumulative `rl_step_reward = 13.901`
  - `toward_target`: cumulative `rl_step_reward = 13.921`
  - `away_from_target`: cumulative `rl_step_reward = 1.878`
  - `oscillate_in_place`: cumulative `rl_step_reward = 7.783`
  - `collide_intentionally`: cumulative `rl_step_reward = -44.344`
- Contact is penalized. The deliberate collision rollout terminated with
  `gym_final_score = -23.0` and a large negative cumulative `rl_step_reward`.
- Large actions are penalized locally. The aggressive target-seeking rollout had
  a negative reward-vs-action correlation of about `-0.346`, and its
  accumulated force penalty was about `-1.824`.

### What only partially passed

- Moving away from the target did reduce reward relative to the target-seeking
  policies, but it still produced positive cumulative reward.
- Oscillation also stayed net positive.

This is visible in the mean reward terms from the fresh validation:

- `away_from_target`
  - `target_progress_reward ≈ +0.0129 / step`
  - `time_penalty = -0.0100 / step`
- `oscillate_in_place`
  - `target_progress_reward ≈ +0.0408 / step`
  - `action_delta_penalty ≈ -0.000254 / step`

In other words, the shaping penalties exist, but in the mock backend they are
too weak to dominate the backend’s built-in plug drift toward the target.

### Key issue

The mock backend is not neutral with respect to anti-progress behavior. Even the
`away_from_target` rollout reduced target distance by about `0.0733 m`, and the
oscillation rollout reduced distance by about `0.2074 m`. That makes the reward
look better than the policy actually is.

## 2. Reward vs final score correlation

### Result

The raw correlation is high, but the signal is weaker than the headline number.

### Fresh measurement

From `aic_gym_gz/artifacts/validation/reward_final_score_correlation.json`:

- overall correlation between total `rl_step_reward` and `gym_final_score`:
  `0.9782340459680854`
- number of episodes: `48`
- `official_eval_score`: always `null` in this run

### Interpretation

This positive correlation is real but overstates alignment quality.

Why:

- for `random`, `heuristic`, `toward_target`, `away_from_target`, and
  `oscillate_in_place`, the mock-side `gym_final_score` was always exactly `1.0`
  at the 128-step cutoff
- only `collide_intentionally` produced a different score: `-23.0`

So the current correlation is mostly separating collision episodes from
non-collision episodes. It is not yet a strong ranking test among different
non-terminal “bad vs decent vs good” trajectories.

### Conclusion

- Positive alignment: yes
- Fine-grained ranking alignment: not yet demonstrated

## 3. Observation correctness

### Public schema

The mock public observation schema includes the expected fields:

- `wrench`
- `wrench_timestamp`
- `tcp_pose`
- `tcp_velocity`
- controller fields:
  - `controller_tcp_pose`
  - `controller_reference_tcp_pose`
  - `controller_tcp_velocity`
  - `controller_tcp_error`
  - `controller_reference_joint_state`
  - `controller_target_mode`
- image fields when enabled:
  - `images[left|center|right]`
  - `image_timestamps`
  - `camera_info[left|center|right]`
- score-critical geometry:
  - `distance_to_target`
  - `distance_to_entrance`
  - `orientation_error`
  - `insertion_progress`
  - `lateral_misalignment`
  - `partial_insertion`

### Matches official expectations

- The field names and shapes needed by RL and teacher-side consumers are present
  in the gym observation contract.
- The score-critical geometry is exposed directly instead of being implicit.

### Missing or approximate

- Live population of wrench, controller state, contact flags, images, image
  timestamps, and `camera_info` still depends on ROS topics.
- In this shell, live ROS/Gazebo paths were not runnable, so only schema-level
  verification and mock-path behavior were validated this turn.

## 4. F/T behavior

### Result

Partially correct.

### Fresh temporal validation

From `aic_gym_gz/artifacts/validation/observation_temporal/observation_temporal_summary.json`:

- `wrench_timestamp` is monotonic for `ticks_per_step = 1, 8, 32`
- `off_limit_contact` becomes observable at:
  - tick `601` for `ticks_per_step = 1`
  - tick `608` for `ticks_per_step = 8`
  - tick `608` for `ticks_per_step = 32`
- `policy_level_observation_is_final_sample_only = true`
- `policy_level_max_or_windowed_ft_summary_present = false`

### What is correct

- Timestamp monotonicity is correct in the mock path.
- Contact flags are propagated at policy level.
- The multi-tick setting clearly changes when contact becomes visible at the
  policy step boundary.

### What is incorrect or incomplete

- The mock “collision then backoff” sequence did not produce a wrench spike:
  `max_wrench_force_l2_norm = 0.0` in all three temporal experiments.
- There is no policy-level within-step max or summary for F/T. Only the final
  sample of each env step is observable.
- This means transient spikes can be lost when `ticks_per_step > 1`.

### Conclusion

- Contact observation: partially correct
- Wrench temporal fidelity: partially correct at best in mock, and not
  validated on live this turn
- Transient preservation across multi-tick steps: not preserved at policy level

## 5. Gym vs official rollout comparison

### Evidence source

This section is based on checked-in live artifacts, not fresh execution in this
turn.

### Deterministic parity artifact

From `aic_gym_gz/artifacts/deterministic_policy_state/summary.json`:

- `passed = true`
- `state_pass = true`
- `score_pass = true`

From `aic_gym_gz/artifacts/deterministic_policy_state/parity_report.json`:

- final success-like classification matched
- state trace errors were small:
  - `plug_x` max abs error: `8.54e-05`
  - `plug_y` max abs error: `8.53e-05`
  - `plug_z` max abs error: `3.69e-05`
- local final-score parity on those traces was exact:
  - `total_score_abs_error = 0.0`

### Live benchmark artifact

From `aic_gym_gz/artifacts/live_benchmark_state.json`:

- final success-like match: `true`
- state parity:
  - `plug_x` max abs error: `5.25e-05`
  - `plug_y` max abs error: `1.35e-04`
  - `plug_z` max abs error: `4.57e-05`
- score parity on the local score path:
  - `total_score_abs_error = 0.0`

### Important limitation

Those score numbers are for the local `gym_final_score` path on both traces.
They are not `official_eval_score`.

### Conclusion

- Similarity level: good, based on the existing deterministic/live parity
  artifacts
- Fresh re-validation in this turn: not possible in the current shell
- Official toolkit score parity: not demonstrated here

## 6. Isaac Lab alignment

### Criteria check

- Progress-based shaping: YES
- Local penalties for action magnitude: YES
- Local penalties for action delta / velocity delta: YES
- Contact penalty: YES
- Terminal success bonus: YES
- Avoids exact per-step reconstruction of final score: YES

### Deviation

- In the current mock dynamics, positive progress terms can dominate the local
  smoothness and anti-oscillation penalties even for visibly undesirable
  behaviors.

So the design matches Isaac Lab philosophy, but the current balance and/or mock
backend dynamics still leave reward-hacking risk.

## 7. Critical gaps

1. Mock dynamics mask anti-progress behavior.
   The `away_from_target` and `oscillate_in_place` policies still reduce target
   distance and stay net positive.

2. Local penalties are present but weak relative to progress shaping.
   Oscillation and action-delta costs are too small to dominate bad local
   behavior in mock rollouts.

3. `gym_final_score` is not very informative on short non-terminal mock
   rollouts.
   Many distinct policies all received `gym_final_score = 1.0`, which makes the
   reward-vs-score correlation look stronger than the ranking problem really is.

4. Policy-level F/T observation is final-sample-only.
   There is no within-step max or summary, so multi-tick transients can be lost.

5. Mock collision does not produce a force spike in the temporal validation.
   Contact and force are not tightly coupled in the mock path, which weakens F/T
   realism for RL debugging.

6. Fresh official-path validation was blocked in this shell.
   Existing checked-in parity artifacts are good evidence, but they are not the
   same as a fresh rerun.

## Minimal high-impact fixes to consider next

1. Make the mock backend less target-seeking under adversarial actions.
   Without that, reward sanity checks overestimate policy quality.

2. Increase or retune anti-oscillation and smoothness penalties.
   The current terms exist, but they are too weak to clearly suppress
   oscillatory behavior.

3. Expose an optional within-step F/T summary.
   A max wrench and/or contact-within-step flag would reduce transient loss when
   `ticks_per_step > 1`.

4. Make short-horizon final scoring more discriminative for audit runs.
   Right now many incomplete non-terminal rollouts collapse to the same
   `gym_final_score`.

5. Re-run the official parity/benchmark scripts in a real ROS/Gazebo container.
   That is required before claiming fresh alignment with the official rollout.

## Bottom-line conclusion

- Ready for RL training:
  - for initial reward-shaping experimentation on the mock backend: partially
    yes
  - for trustworthy RL optimization without additional caveats: no

- Aligned with official rollout:
  - approximately yes, based on checked-in parity artifacts
  - freshly validated in this turn: no

- Label status:
  - `rl_step_reward`: validated as the per-step dense RL reward
  - `gym_final_score`: validated as the local episode-level score path
  - `official_eval_score`: not run in this turn and not produced by the current
    mock validation
