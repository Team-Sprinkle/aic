# Connector orientation, new training and SFP oscillation

## 1. Yes, grasp angle can differ between episodes

We know the robot TCP's orientation. To work out the connector's actual orientation, we also need its angle relative to that TCP.

That relative angle changes between episodes. In the recordings audited here, it stays almost constant within an episode. The plan is therefore to estimate it from clear camera views, stabilize the estimate, and retain it during insertion. Reacquire after a substantial retreat; do not replace it with an arbitrary noisy estimate every frame.

The “tip orientation” means the orientation of the connector's mating frame—the axes that must line up with the opening.

## 2. I have trained image-based orientation estimators

“Validate” means testing on complete episodes withheld from training. I also trained new models explicitly to estimate connector orientation from three camera images.

| Task | Training / held-out episodes | Previous angular error p95 | New angular error p95 |
| --- | --- | --- | --- |
| SC-to-SC | 83 / 21 | 1.16° | **0.89°** |
| SFP-to-NIC | 307 / 47 | 2.29° | **0.49°** |

The corpora contain 1,248 SC and 6,115 SFP image triplets. Each triplet has three camera images. These selected fits reached their validation stopping rule; they were not stopped because a wall-time budget expired.

For SC, I retained the accurate existing visual tip-position estimator and trained orientation separately with image augmentation. The first joint SC fit overfit and is preserved as a failed comparison. For SFP, the new full-pose fit improved orientation substantially, but the existing position estimator is still more accurate in its approved acquisition region, so I retain that position component too.

P95 is not a guarantee for every frame. Prediction/ground-truth overlays and the error tails are saved. Simulator geometry supplies training/evaluation labels only; the estimators receive images.

## 3. What I found about SFP back-and-forth motion

Your history and coverage concerns are reasonable. The evidence now narrows the possibilities:

- **The actor can oscillate without hardcoded backtracking.** The latest seven SFP comparisons disabled recovery and recorded zero backoff events.
- **The training data is not forward-only.** About 15% of near-port training examples command more than 1 mm backward. This does not prove that those examples are sufficiently distinguishable from forward examples.
- **Some teacher decisions depend on hidden controller state.** In one traced example, the teacher approaches 25 mm, settles slightly backward, then begins insertion. Its internal phase and accumulated controller bias affect the command, but are not explicit actor inputs.
- **Current insertion history is limited:** four robot-state/relative-pose samples spanning 1.2 seconds, plus current images. It does not include historical images. Transport history now survives handoff, but a longer ambiguous settling sequence may still exceed that context.

Thus, increasing history could help, but it may not fully solve the problem. We must also distinguish settling, insertion and recovery, and account for previous commands/controller behavior. This is a plausible mechanism, not a proven explanation for every failure.

## 4. What is running next

I prepared a separate full-connector-pose dataset using **60,671 matched observations from 499 source episodes**. It preserves the original data and recorded TCP commands. The new representation explicitly includes connector orientation and converts complete predicted connector poses back into TCP commands.

One controlled H4 BC fit is running on **GPU 4**, with the same architecture/data/history opportunity as the previous H4 control. Its stopping rule distinguishes a validation plateau from the 50,000-update/six-hour safety cap. This isolates the full-pose representation change before another architecture sweep.

**No insertion improvement is established yet.** The prior 14-rollout comparison had zero strict full insertions. Causal orientation-window checks and full-pose runtime integration must pass before testing the new actor. If geometry improves but autonomous alignment does not, the next targeted repair is phase/command context, including a longer causal history—not routine collection or another broad retrain.

Detailed record: [orientation and oscillation experiment](../experiments/2026-10-17-connector-orientation-and-sfp-oscillation.md).

Artifacts: `/home/nas/chmin/aic_archives/2026-10-17_connector_orientation_and_sfp_oscillation_v1/`. See `continuation_state.json`, `sc_review/`, `sfp_review/`, and `oscillation_audit/`.
