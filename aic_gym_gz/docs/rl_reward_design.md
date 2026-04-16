# RL Reward And Final Scoring

`aic_gym_gz` now treats RL reward and episode scoring as separate products:

- `rl_step_reward`: dense local reward for optimization
- `gym_final_score`: trajectory-level local score for reporting and selection
- `official_eval_score`: external-only score from the official toolkit path

## Why the split exists

The environment is not trying to make the discounted or undiscounted sum of step
rewards exactly equal the final score. That is intentional.

For RL training, the reward should be:

- dense
- local
- numerically stable
- mostly Markov or short-history-based
- resistant to reward hacking

For reporting and model selection, the environment still needs a trajectory-level
summary that resembles the local scoring logic already used for parity work.

## `rl_step_reward`

The RL reward uses Isaac-Lab-style shaping:

- target-distance progress potential
- entrance-distance progress potential
- insertion-progress potential
- orientation-alignment progress
- lateral corridor-alignment progress
- local action magnitude penalty
- local action-delta penalty
- local TCP-velocity-delta penalty
- local force penalty
- off-limit contact penalty
- short-history oscillation penalty
- per-step time penalty
- partial-insertion bonus
- terminal success bonus
- terminal wrong-port penalty

The exact weights live in `aic_gym_gz.reward.AicRlRewardWeights`.

## `gym_final_score`

The final score still uses the accumulated episode trace:

- timing trace
- TCP path
- plug path
- `target_port_entrance_pose`
- wrench trace
- off-limit contact trace

This remains local to `aic_gym_gz`. It should be treated as a useful local score,
not an official result.

## Approximation boundaries

- `official_eval_score` is not computed here.
- Wrench/contact parity depends on live ROS topics being present.
- Jerk uses env-side velocity history, not the official scorer's full runtime path.
- Live checkpoint export remains approximate; mock checkpoint/restore is exact.
