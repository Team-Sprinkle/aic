# 2026-09-17: evaluation and curriculum correctness repairs

Status: implemented and verified with **85 CPU/mocked-process tests**.
No training, simulator rollout, container startup, or model performance
evaluation was run. Branch: `feat/hybrid-train`; base commit: `7534090` plus
the working-tree changes described here. Preceding evidence:
[reentry audit](2026-09-17-reentry-audit.md).

## Purpose

Make failed/incomplete runs visible and prevent invalid success labels or
sample counts from driving future experiments. This changes the reliability of
measurement and progression; it supplies no new evidence of learned insertion.

## Changes

| Source | Repair |
| --- | --- |
| [Gazebo score parser](../../aic_utils/gazebo_rl/gazebo_rl/score_parser.py) | Uses per-trial Tier 3 score 75 for correct insertion, following the local official scorer. Reports completed/scored and successful trial counts separately, rejects nonfinite values, and avoids searching the working directory when no results directory is supplied. Tier 1 is model validity. |
| [Transfer validator](../../aic_utils/gazebo_rl/scripts/serl_transfer_validate.py) | Classifies missing/incomplete scoring as `no_score`, all observed successful trials as `success`, and remaining scored outcomes as `transfer_failure`. An aggregate reward threshold cannot establish insertion. |
| [ACT runtime evaluator](../../scripts/evaluate_act_checkpoints_runtime.py) | Requires explicit command mode, checks input paths/export dependencies, rejects empty checkpoint selections, checks readiness/engine exit/configured trial coverage, retries failed summaries, and preserves numbered attempts. Reuses only complete evaluations with matching settings/file identity. Cleanup failures remain recorded. |
| [Older score-selection pipeline](../../scripts/post_act_eval_and_serl.sh) | Reads the score path from complete evaluation summaries to follow the new attempt layout. This script also launches training and was not executed. |
| [Isaac loop](../../aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py) | Records terminal episode outcomes from termination flags with pre-reset identities. Adds a completed-episode limit and final measured counts. `--updates 0` disables updates without stopping at step one. Actor state widths must match the checkpoint schema. |
| [Accounting helper](../../aic_utils/aic_isaac/scripts/stateful_curriculum_runtime.py) | Validates metrics/final-result consistency, counts distinct terminal episodes, requires complete evaluation coverage and unassisted actions, and implements promotion/demotion from success fractions. It has no Isaac/Torch dependency. |
| [Stateful wrapper](../../tools/train_stateful_promotion_gated_insertion.sh) | Counts actual completed episodes; honors promotion/demotion config; keeps per-level evaluation episodes/seeds fixed; fails on nonzero child exits or incomplete evidence; stops after the configured no-promotion limit; preserves cycle artifacts. |
| [Axial wrapper](../../tools/train_axial_first_inserted_start.sh) | Uses mounted `outputs/experiments/` by default, matching the repaired stateful wrapper. |

## Measurement contract

- Each terminal/truncated environment episode contributes once. Simultaneous
  failure terms override a target-success flag; a simultaneous time limit does
  not override success. If target-success termination is disabled, success is
  unknown and the curriculum summarizer rejects the record.
- Evaluation requires all generated per-level episode IDs to finish, at least
  the requested episode count, zero gradient updates/exploration, and zero
  measured guide/guard/action override. Partial evaluation stops progression.
- The stateful SFP defaults are 0.5 mm axial and lateral error, 0.03 rad
  orientation, and module consistency within 1.0 mm axial / 1.5 mm lateral.
  Module consistency and target-success termination are required. Actual
  criteria and collider flags are copied into each summary's
  `evaluation_contract`; the full arguments remain in `train_config.json`.
- Modified SFP/NIC colliders are opt-in through
  `AIC_STATEFUL_DIAGNOSTIC_COLLIDERS=1`. This is a different physical setting
  and needs a separately labeled experiment.
- Vector steps can exceed an episode budget by up to `num_envs - 1`; reports
  retain the actual count. Timed training segments can finish fewer episodes
  than requested. The total budget counts those completions, not an assumed
  number per segment.
- `AIC_STATEFUL_NO_PROGRESS_ASSESS_SECONDS=7200` is checked after each cycle;
  no promotion within that interval produces `no_progress_stop` and exit 3.
  This is not an exact wall-clock interrupt. Zero disables it.

## Validation

The full command is in [local workflow](../LOCAL_WORKFLOW.md#2-inspect-a-saved-model-without-starting-simulation).
Result: **85 passed in 8.64 seconds**. Local test output:
`outputs/local_checks/evaluation_curriculum_20260917/pytest.txt` (ignored by Git).

Coverage includes:

- Full, partial, wrong-port, missing, nonfinite, and multiple-trial scores.
- Empty selections, failed readiness/restarts/cleanup, incomplete scoring,
  stale score files, retry preservation, and reuse after settings change.
- Vector episode/time-limit accounting, simultaneous failure, promotion and
  demotion thresholds, missing coverage, duplicate/corrupt/legacy metrics,
  disabled consistency, exploration/updates, and action override rejection.
- Four shell-wrapper scenarios using fake child processes: successful
  bookkeeping, crashed process with a leftover checkpoint, incomplete
  evaluation, and the no-progress stop. Fake checkpoints are text fixtures;
  no optimizer or simulation is invoked.
- Actor input dimension validation, existing curriculum generation tests,
  and existing insertion reward geometry checks.

Python compilation, Bash syntax, documentation links/command syntax, and
`git diff --check` also passed. These checks do not establish that the installed
simulator accepts every runtime argument or that reset/contact behavior is sound.

## Compatibility and remaining work

Existing score labels and nominal June counters are historical evidence and
were not rewritten. Legacy metrics cannot be promoted by the new helper because
they lack terminal episode records. Use a fresh run ID with the repaired code;
save the resolved config and local diff.

ACT evaluation callers must pass `--command-mode`; `none` explicitly means an
interface smoke check. Consumers must follow `eval_summary.json.scoring_yaml`
into the selected attempt. Use a fresh evaluation subdirectory after policy
code, simulator, or container changes: these are not covered by the reuse
signature. A complete evaluation can correctly report zero insertions.

No architecture replacement, dataset regeneration, checkpoint conversion, or
physics retuning was attempted. Next steps are live terminal-accounting checks,
reset/action probes, and a scored ACT baseline. The direct visual actor/DINOv2
direction remains a separate experiment after those checks; see
[current status](../STATUS.md).

Subsequent work, after this repair pass, is recorded separately in
[direct visual model checks](2026-09-17-direct-visual-policy.md).
