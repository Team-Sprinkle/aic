# Status — 2026-10-16 10:31 UTC

## Training is finished; reliable insertion is not yet demonstrated

- Both shared SC/SFP insertion actors completed **30,000 training updates each** on **67,039 examples from 500 usable episodes** (421 training / 79 validation). One started with ImageNet vision weights; the other reused visual weights from the transport model while learning fresh action heads.
- Both stopped at the update limit. A subsequent convergence check found no meaningful late validation improvement, so it did **not** start additional training. This supports stopping this fit; it does not prove the policy is good enough.

## Transport and perception results

| Test | Result | Meaning |
|---|---|---|
| Broad SC test, earlier perception configuration | **44/60 qualified** | Primary board: 44/50. Alternate board: 0/10. Coverage is still incomplete. |
| Corrected SC/SFP configuration | **9/10 SC; 9/10 SFP qualified** | SC missed once because inference p95 was 321 ms. SFP stalled before reaching the acquisition position once. The combined prerequisite gate failed. |
| New SC refiner repair | **20/20 acquired the correct port; 18/20 met all requirements** | Two episodes exceeded the 300 ms inference limit. This separate diagnostic gate passed, but does not establish alternate-board coverage or insertion success. |

These are **transport plus visual port-acquisition tests, not insertion tests**. No wrong-port acceptance or scored harmful contact was observed in these tests. The failed SFP acquisition made no acceptance decision.

## Insertion and backtracking

The planned completed-model rollout comparison **did not run**: it correctly stopped when the corrected transport/perception prerequisite failed. Therefore, there is no completed-model full-insertion success rate yet.

Earlier integration tests used an intermediate checkpoint, one SC and one SFP episode. Neither fully inserted. Hierarchical backtracking and fresh visual reacquisition are implemented, but their benefit with the newly completed actors remains unmeasured. **No new RL training has run.**

## Next work

1. Diagnose the specific five-card SFP transport stall and the alternate-board SC failures using their saved traces and videos.
2. Address the perception latency spikes and validate the repaired configuration with a bounded comparison.
3. Once that prerequisite passes, run the matched completed-actor insertion comparison, then compare no backtracking, local backtracking, and hierarchical backtracking if the actor meets its insertion gate.

No new dataset sweep or retraining is justified yet. Reserved final scenes remain sealed. No continuation training/evaluation job is currently running; only the previously retained legacy GPU process remains (about 1.35 GB).

Artifacts: `/home/nas/chmin/aic_archives/2026-10-15_rule_perception_hierarchical_v1/`.
