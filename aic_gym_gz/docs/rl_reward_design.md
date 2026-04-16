# RL Reward Design

This document explains the current reward design in `aic_gym_gz` and how it
should be interpreted by RL systems and teacher systems.

## The split: reward versus score

`aic_gym_gz` deliberately separates:

- `rl_step_reward`
- `gym_final_score`
- `official_eval_score`

| Name | Computed Where | Purpose | Exact vs Approx |
| --- | --- | --- | --- |
| `rl_step_reward` | `env.step()` | Dense RL training reward | Local shaped reward, not equal to final score |
| `gym_final_score` | `final_evaluation()` | Local episode-level evaluation and analysis | Local trajectory-level approximation |
| `official_eval_score` | External official toolkit only | Ground-truth official evaluation | Official when actually run |

## Why the split exists

The environment is not trying to make the discounted or undiscounted sum of
step rewards exactly equal the final score. That is intentional.

For RL training, the reward should be:

- dense
- local
- numerically stable
- mostly Markov or short-history-based
- resistant to reward hacking

For analysis and model selection, the environment still needs a trajectory-level
score. That is what `gym_final_score` is for.

## `rl_step_reward`

The current reward follows Isaac-Lab-style shaping:

- target-distance progress potential
- entrance-distance progress potential
- insertion-progress potential
- orientation-alignment progress
- corridor-alignment progress
- local action magnitude penalty
- local action-delta penalty
- local TCP-velocity-delta penalty
- local force penalty from the current wrench sample
- local off-limit contact penalty
- short-history oscillation penalty
- time penalty
- partial-insertion bonus
- terminal success bonus
- terminal wrong-port penalty

All current weights live in `aic_gym_gz.reward.AicRlRewardWeights`.

### Intended interpretation

- This is the reward to use for training.
- It is not an official score.
- It is not an exact per-step decomposition of `gym_final_score`.

### Current limitations

Validation results show that:

- the reward is dense and usable
- the mock backend can make some bad policies look better than they should
- oscillation and anti-progress penalties exist, but are not always dominant in
  the mock path

So the reward design is directionally correct, but not perfectly hardened
against every undesirable behavior yet.

## `gym_final_score`

`gym_final_score` is computed at episode end from the accumulated trace.

It uses:

- timing trace
- TCP path
- plug path
- target-port geometry
- target-port-entrance geometry
- wrench trace when available
- off-limit contact trace when available

### Intended interpretation

- Use it for local evaluation, analysis, debugging, and teacher-side ranking.
- Do not train against it directly as if it were a dense reward.
- Do not call it an official score.

## `official_eval_score`

`official_eval_score` is not computed inside `aic_gym_gz`.

Use the official rollout / official toolkit path when you need:

- ground-truth final evaluation
- official leaderboard-facing numbers
- final pre-submission validation

## Recommended RL usage

Use `aic_gym_gz` for RL with the following assumptions:

- optimize `rl_step_reward`
- use history stacking or recurrence if temporal information matters
- inspect `reward_terms` and `reward_metrics` during training
- treat `gym_final_score` as a local evaluation signal, not a training target

Example:

```python
training_reward_total = 0.0

observation, info = env.reset(seed=123)
done = False
while not done:
    action = policy(observation)
    observation, reward, terminated, truncated, step_info = env.step(action)
    training_reward_total += reward
    done = terminated or truncated

final_report = step_info.get("final_evaluation") or env.task.final_evaluation()

print(training_reward_total)
print(final_report["gym_final_score"])
print(final_report["official_eval_score"])
```

## Recommended teacher usage

Teacher systems should be explicit about which signal they consume:

- `rl_step_reward` for local short-horizon optimization features
- `gym_final_score` for local episode-level ranking
- `official_eval_score` for external ground truth only

Teacher systems should not rename a local approximation to `official`.

## Approximation boundaries

- `official_eval_score` is not computed here.
- Wrench/contact parity depends on live ROS topics being present.
- Jerk uses env-side velocity history, not the official scorer’s full runtime
  path.
- Live checkpoint export remains approximate; mock checkpoint/restore is exact.
- Policy-level F/T is current-sample-only, so multi-tick transients may be
  missed unless the policy keeps history.
