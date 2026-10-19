# Was backoff enabled in the released official evaluation?

**Yes, but only the SC transport trial actually backtracked.** The frozen run selected `AIC_SHARED_RECOVERY_MODE=local`. Its rule-based transport also has a measured-path route retreat and alternate-corridor retry. It did **not** select the broader `hierarchical` insertion-recovery mode, and no RL policy controlled the backoff.

| Trial | What happened |
| --- | --- |
| SFP trial 1 | Inserted without retreat. |
| SFP trial 2 | The immediate force guard stopped transport near an intervening NIC card. No retreat event was recorded. |
| SC trial 3 | A **visual-acquisition timeout**, rather than a confirmed transport blockage, triggered a measured-path retreat. It reversed through 322 recorded waypoints, reached the earlier checkpoint at simulation time 130.9 s, then started an alternate-corridor route. The controller's wall-clock budget ended the retry before port acquisition. The trace's `force_or_measured_blockage` reason was misleading. |

The trace records one `transport_route_backtrack_requested` and one `transport_measured_route_retreat_complete`. It records **zero** `measured_retreat_complete` and **zero** `lateral_retry_complete` insertion-recovery events, because the failed trials never reached insertion. This evaluation shows that the transport backtrack executed; it does not show a successful hierarchical recovery or learned lateral exploration.

Source: the frozen official run's `command.json`, `rollout/runtime_trace.jsonl`, and [full evaluation report](../experiments/2026-10-18-frozen-V56-official-evaluation.md) under `/home/nas/chmin/aic_archives/2026-10-18_V56_official_qualification_evaluation_v1/`.
