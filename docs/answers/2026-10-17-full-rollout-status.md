# Full rollout status — October 17

**The evaluation stopped after 40/62 scored scenario rollouts. No full insertion has succeeded.**

| Task | Scenarios scored | Full insertions | Actor ran | Episodes with scored contact/force harm |
| --- | --- | --- | --- | --- |
| SC-to-SC | 12/12 | 0 | 10 | 7 |
| SFP-to-NIC | 28/50 | 0 | 27 | 0 |

The four separate integration rollouts also produced no full insertion. SC shows substantial lateral/orientation drift during learned insertion; SFP generally makes insufficient approach progress. No completed backtrack-and-lateral retry is recorded, so there is no demonstrated hierarchical recovery benefit in this batch.

## Why it stopped

The next SFP rollout completed, but its independent scoring check found that the raw-bag and extracted-stream methods disagree about terminal lateral offset by **1.437 mm** (axial difference0.054mm). The consistency limit is1mm, so the coordinator stopped rather than report an untrusted result. This episode ended during transport, before the insertion actor ran. It is not included in the40 scored episodes. The cause of the discrepancy has not been established; no threshold was relaxed and no model was changed. The coordinator is no longer running.

The separate complete sample/eval YAML scoring runs have **not started**. Their full-pipeline scores are unavailable.

Next: resolve the causal terminal-geometry mismatch, preserve the failed audit, and resume the remaining frozen evaluations. Separately diagnose the insertion/control failure from retained traces and videos before new training or RL. Transport-only results remain121/130 qualified handoffs; they do not establish insertion success. Reserved final scenes remain sealed.
