# Full rollout evaluation status

The one-episode-per-scenario schedule is active. **22/62 full rollouts are scored: no successful full insertions so far.** Four separate integration rollouts also completed without a qualified insertion.

| Task | Completed scenarios | Insertion actor ran | Qualified full insertions | Episodes with scored harm |
| --- | --- | --- | --- | --- |
| SC-to-SC | 12/12 | 10 | 0 | 7 |
| SFP-to-NIC | 10/50 | 10 | 0 | 0 |

SC often develops large lateral/orientation errors during the insertion stage; some cross the opening plane while substantially misaligned, which is not insertion. Two SC episodes do not reach actor execution. SFP generally makes insufficient approach progress and stops well before the opening. These observations locate the performance gap but do not establish whether the cause is learned control, training coverage, or runtime integration.

Hierarchical recovery is enabled, but no completed retreat-and-lateral-retry has been recorded in these scenario results. This does not mean it never attempted recovery: one inspected SC run aborts its retreat because force rises. We cannot claim a recovery benefit.

The preceding transport-only batch is complete: **121/130 qualified handoffs**. Sample-derived: SC22/24 and SFP97/100; exposed official scenes: SC0/2 and SFP2/4. These are handoffs, not insertions. No wrong-port acceptance or scored harm occurred in that transport-only batch. Thus earlier development successes do not establish reliability on the exposed official layouts.

The process is still running; at this check it is encoding the latest rollout video before advancing. The three complete sample YAML runs and three complete released-evaluation YAML runs have **not started**, so their full-pipeline scores are not yet available. Models/rules remain frozen, no training or RL is running, and reserved final scenes remain sealed. Finish the requested evaluation and use the retained traces/videos for a bounded insertion-stage diagnosis before further training.

Snapshot: 2026-10-16T23:31:26.357442+00:00
