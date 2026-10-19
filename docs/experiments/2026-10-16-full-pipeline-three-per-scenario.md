> **Scope revision:** The user reduced the scenario sweep to **one episode per group (62 total)**. The separate complete sample/eval YAML runs remain three repetitions each. The current rollout finishes before the scheduler switches; no models, policies, rules or scenes change. Original three-per-group plan and manifest remain archived in the artifact folder.

<!-- full-insertion-evaluation-live:start -->
## Full insertion evaluation continuation

- Stage: `scenario`. Artifacts: `/home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1`.
- Frozen rule transport → requested-port coarse/native refinement → shared 150 mm BC insertion actor.
- Planned: 62 scenario rollouts (62 groups × one), plus three complete sample YAML runs and three complete released-evaluation YAML runs.
- Hierarchical insertion recovery is rule-based; no new RL or training. Released evaluation is scoring only; reserved final scenes stay sealed.
- Counts and scores: see `progress.json`, `results_by_scenario.json` and `configuration_scores.json` in the artifact folder.
<!-- full-insertion-evaluation-live:end -->

# Full transport-to-insertion evaluation: one episode per scenario

## Approved scope

Finish the active frozen 130-episode transport/acquisition evaluation and its trace analysis first. Do not interrupt or alter it. Then measure **full autonomous rollouts**, including the trained insertion actor, across all declared sample-derived task groups. Separately obtain the engine's scores from the exact complete `sample_config.yaml` and released qualification `eval_config.yaml`.

## Runtime

The new `RunSampleThenSharedInsertion` preserves the older insertion wrapper. It connects the newly tested sample-derived `RunSampleTransportRecovery` to the existing shared SC/SFP BC actor. It uses the same requested-port coarse/native perception during fresh acquisition, including after large insertion retreats. The actor owns the full approximately 150 mm approach, alignment and insertion. Hierarchical measured-path backtracking and coherent lateral retries are a **rule supervisor, not RL**. No privileged geometry or contact identity enters decisions.

The prior transport tests ended at acquisition and never invoked insertion. Their results are not full-pipeline success. Connecting these separately tested components requires an integration check.

## Counts and splits

- SC: NIC obstacle count 0–5 × two requested SC ports = **12 groups**, one episode each: **12**.
- SFP: NIC count 1–5 × five requested NIC identities × two requested SFP ports = **50 groups**, one episode each: **50**. Requested NIC is always present.
- Total: **62 groups / 62 full rollouts**, using a fresh sample-derived randomization seed 2026101691. Board/grasp/rail variations follow the independently declared sample-derived envelope, not fitted official-scene coordinates.
- Four separate integration rollouts compare both completed BC checkpoints on one SC and one SFP development configuration, with recovery off. They are not added to the 62 denominator.
- Three complete sample YAML runs and three complete released-evaluation YAML runs, retaining all three trials in their original order: **six configuration launches / 18 task episodes**. These are not sums of separately reset trial scores.
- Reserved internal final scenes stay sealed. The released evaluation YAML is already exposed and is used only for scoring after model/rule freeze.

Only one episode per group is a coverage screen; it does not establish high population reliability. The selected actor is fixed **before autonomous outcomes** from the lower mean over families of held-out first-target p95 mm plus orientation p95 degrees. Both training summaries and all checkpoints are retained. This heuristic is not a claim of the best possible actor. The integration comparison does not tune the official evaluation pipeline.

## Metrics and stopping

For each isolated scenario report actor execution, requested-port lock errors, official tier scores, causal physical depth/lateral/orientation, no-harm scoring, phase-specific p50/p95/p99 latency, backtracks and reacquisitions. Strict physical insertion remains SC >=15.5 mm or SFP >=44 mm, lateral <=0.5 mm, orientation <=1 degree, official tier 3 =75, and no scored harm. The stricter qualified flag also checks valid fresh locks, initial public-force reference, latency and no runtime geometry input.

Ordinary transport, perception or insertion failures stay in the denominator, and the requested sweep continues. Runtime/causal-scoring/infrastructure errors stop with preserved artifacts for diagnosis. Do not silently skip a partial run or count it as a model-performance failure. Do not alter checkpoints, thresholds or rules mid-sweep. No routine collection, retraining, RL or automatic promotion follows.

The full YAML runs report the original engine `total`, per-trial tier 1/2/3 scores and categories, plus mean/min/max totals across three repetitions. These full-configuration scores are separate from the stricter geometry audit, which currently operates on isolated single-task runs. They are local instrumented Gazebo results, not certified cloud leaderboard scores. Original YAML bytes, home posture, task time limits and trial ordering are preserved. Only the whole-container watchdog increases from 450 to 1,100 seconds to accommodate three unchanged 180-second tasks; it does not extend individual task limits.

## Commands, artifacts and budget

```bash
.pixi/envs/default/bin/python -u scripts/run_full_sample_insertion_evaluation.py \
  --transport-root /home/nas/chmin/aic_archives/2026-10-16_sample_adaptive_transport_evaluation_v3 \
  --output /home/nas/chmin/aic_archives/2026-10-16_full_sample_insertion_evaluation_v1 \
  --gpu 5
```

The coordinator waits without a GPU until the current transport summary and decision exist, writes `transport_analysis.{json,md}`, freezes sources/models/manifests, and runs sequentially on GPU5 with rootless Docker, cooperative GPU locks and isolated ROS domain97. It does not interrupt existing jobs. Estimated budget **5–10 hours**, dependent on startup, actual task durations and video encoding. All bulk artifacts are on NAS.

Key artifacts: `state.json`, `execution_error.json` if blocked, `plan.json`, `frozen/manifest.json`, `frozen/selection.json`, `current_job.json`, `progress.json`, `results_by_scenario.json`, `configuration_scores.json`, `results.md`, `decision.json`, per-run commands/container environment/raw bags/scoring/terminal RGB. Save smooth five-view 20fps videos for integration, at least one per scenario, and all failed isolated rollouts. Full multi-trial bags and terminal RGB are preserved without applying the single-trial video cutter across resets.

## Interim full-rollout outcome

Snapshot 2026-10-16T23:31:26.357442+00:00: **0/22 qualified full insertions** (SC0/12; SFP0/10), plus four failed integration insertions. Seven SC scenario runs have scored harm. Recovery is enabled but no complete retreat/retry has yet succeeded; an inspected SC retreat aborts on rising force. SFP usually remains far from the opening. The cause remains unestablished; preserve the frozen comparison and diagnose insertion/control before new training. Full YAML scoring is still pending. See [easy status explanation](../answers/2026-10-16-full-rollout-status.md).

## October17 scoring stop



The next SFP rollout completed, but its independent scoring check found that the raw-bag and extracted-stream methods disagree about terminal lateral offset by **1.437 mm** (axial difference0.054mm). The consistency limit is1mm, so the coordinator stopped rather than report an untrusted result. This episode ended during transport, before the insertion actor ran. It is not included in the40 scored episodes. The cause of the discrepancy has not been established; no threshold was relaxed and no model was changed. The coordinator is no longer running.

The separate complete sample/eval YAML scoring runs have **not started**. Their full-pipeline scores are unavailable.

Next: resolve the causal terminal-geometry mismatch, preserve the failed audit, and resume the remaining frozen evaluations. Separately diagnose the insertion/control failure from retained traces and videos before new training or RL. Transport-only results remain121/130 qualified handoffs; they do not establish insertion success. Reserved final scenes remain sealed.

