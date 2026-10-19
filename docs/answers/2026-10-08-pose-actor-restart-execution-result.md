# Pose-conditioned actor restart: execution result

I tested the training-method hypothesis without opening any new scene. All nine 15-second actor-to-teacher episodes had a valid same-reset, subsecond join from the actor's last observation to the first recorded teacher action. The predicted port pose was accepted at that first correction in only **3/6 training** and **1/3 development** episodes. This means the actor often has no pose input exactly when a correction starts.

A 500-step small-set fit learned the 112 accepted-pose training actions (0.60 mm first-action translation error p95), but removing pose changed its held-out actions by only 0.025 mm p95. Resetting just the action head in that fit did not help: held-out error rose from 2.56 to 7.39 mm p95.

I then ran a matched 2,000-update comparison on the existing native-image SFP and SC data. Both arms got stronger corrective sampling and no old-actor preservation loss; one retained the old action head and one reset it. Late development action error was **2.83 mm p95 warm versus 3.69 mm reset**. Removing pose changed their actions by only **0.044 and 0.038 mm p95**, respectively. Resetting also worsened ordinary SFP and SC. The late development episodes were used to choose checkpoints, so these figures are a diagnostic comparison, not an untouched generalization score.

**Decision:** a full actor-from-scratch run is not justified by this test. The smaller reset lost useful behavior and still ignored pose. The next bounded design should train an explicit target-relative route-error, waypoint or short progress objective and use it to decode a complete TCP-body action chunk. It should also improve predicted-pose coverage at the first actor-visited correction and reserve a fresh grouped development split for evaluation. Do not open the prepared autonomous starts until that offline gate passes. Handoff, insertion, RL and final scenes remain closed.

Commands, artifacts, checkpoints and limitations: [experiment record](../experiments/2026-10-08-sfp-pose-actor-training-method.md), [machine decision](../../outputs/experiments/2026-10-08_sfp_pose_actor_training_diagnostic_v1/decision.json).
