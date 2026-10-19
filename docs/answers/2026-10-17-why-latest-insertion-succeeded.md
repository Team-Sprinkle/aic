# Why the latest SFP insertion succeeded

**Starting close helped, but it does not explain the difference by itself: the earlier failed comparisons also handed control to the insertion actor about 25 mm before the opening.** The complete rollout still included rule-based transport and a visual/proprioceptive approach; we did not simply spawn the robot next to the port.

## What changed

The main change was teaching the system to reason about **the actual connector's position and orientation**, rather than combining the connector-tip position with the gripper/TCP orientation.

1. **Estimate how the connector is held.** We trained an image-based estimator for the connector's rotation relative to the gripper. A connector can be held slightly crooked even when the gripper orientation is known exactly.
2. **Retrain the insertion actor using that information.** Its inputs and training targets now describe the connector's full position and orientation relative to the port. Previously, the actor had to implicitly learn the difference between gripper orientation and connector orientation.
3. **Convert commands using the full grasp transform.** At runtime, the desired connector pose is converted into the corresponding gripper command using the estimated connector-to-gripper relationship. Three consistent image estimates are combined and then held fixed during the attempt, avoiding a reference frame that changes every frame.

These changes belong together: estimating connector rotation would have limited value if the actor and command conversion did not use it consistently.

## What the results actually show

An important qualification: **an older policy had already reached full axial depth, but failed our orientation requirement.** The newest result is not the first time a policy pushed the plug that far.

| SFP rollout | Depth | Lateral error | Orientation error | Passes our full geometric gate? |
| --- | ---: | ---: | ---: | --- |
| Earlier visual-warm policy, also starting around 25 mm away | 46.78 mm | 0.26 mm | 2.55° | No: orientation |
| New full-connector policy, repeat 1 | 46.75 mm | 0.48 mm | 0.46° | Yes |
| New full-connector policy, repeat 2 | 37.43 mm | 0.25 mm | 0.53° | No: insufficient depth |

The unchanged SFP gate requires depth ≥44 mm, lateral error ≤0.5 mm, orientation error ≤1°, and no scored harm. Separately, compared with the previous H4 actor, early SFP command jumps fell from about 15.4 mm p95 to 4.4/3.2 mm. Abrupt motion still remains.

## How confident are we about the cause?

**Better handling of connector orientation is the strongest explanation, but it is not yet proven by an isolated ablation.** We changed the orientation estimator, pose representation, and command conversion together, then retrained. The matched H4 comparison used the same 60,671 data anchors and episode split, architecture, 1.2-second robot history, and four-step prediction/execution. Both fits stopped on validation plateaus; their update counts differed.

There was **no RL, backtracking, larger actor, or longer history** in these four tests. SFP succeeded in only **1 of 2 repeats**, and SC succeeded in **0 of 2**. So this is an encouraging orientation/alignment improvement, not reliable insertion yet—and it does not establish that the actor can handle the entire 150 mm approach.
