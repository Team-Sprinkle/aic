# YYYY-MM-DD: short experiment name

Status: planned / running / finished / interrupted  
Outcome: pending / success / partial / failed / no_score / infrastructure_failure

## Question and decision rule

- Hypothesis; baseline run/checkpoint; one intended change:
- Criteria for continuing, rejecting, or calling the result inconclusive:
- Wall-time/step budget and no-progress stop rule:

## Reproduction

- Code commit, dirty diff location, seed(s):
- Simulator, image digest/version, GPU, dependency lock/environment export:
- Dataset identity and filtering; starting checkpoint and required sidecars:
- Exact command and resolved config location (including environment overrides):
- Run root on host, container mapping, and backup/artifact-store location:
- Resume lineage, including optimizer/replay state retained or reset:

## Evaluation contract

- Task family, board/card/port, starting distribution, held-out config/seed IDs:
- Observation/state schema, cameras, normalization, privileged inputs:
- Action units, frame, frequency, horizon predicted/executed, clipping:
- Geometry bodies/frames, target depth and tolerances, module consistency:
- Collision modifications; guide, guard, scripted overrides during train/eval:
- Per-episode success definition, actual denominator, hold/termination handling:
- Official Gazebo scoring procedure, or why only a simulator proxy is measured:

## Results and evidence

- Requested versus actual steps, completed episodes, eval trials, and wall time:
- Successful / failed / unscored trials; uncertainty across seeds if measured:
- Geometry and force together at the same steps; reward as a separate metric:
- Actor versus executed actions, loss/update checks, termination reasons:
- Metrics, score files, logs, representative success/failure videos:
- Checkpoint chosen and why; reload/rollout verification:
- Failures, missing artifacts, and deviations from the original plan:

## Interpretation

Observation:

Hypothesis explaining it (mark untested causes):

Decision and next bounded experiment:

Update [the ledger](../EXPERIMENTS.md) and, if needed, [status](../STATUS.md).
