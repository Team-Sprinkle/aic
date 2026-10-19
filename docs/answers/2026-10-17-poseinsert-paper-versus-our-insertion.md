# PoseInsert versus our current insertion method

## 1. How the paper works

The paper is **PoseInsert**. It proposes a pose-only version, **PoseDP**, and a version with additional visual information, **RPDP**.

Its basic idea is:

> Describe where the connector is relative to the socket, then learn how that relationship should change to complete insertion.

The process is:

1. Estimate the connector's position **and orientation** and the socket's position **and orientation**.
2. Express the connector pose using the socket's axes. For example: “20 mm in front, 2 mm to the left, tilted by 3 degrees.”
3. A diffusion policy predicts a short sequence of future connector poses in that socket frame.
4. Use the estimated relationship between gripper and connector to turn those predictions into robot commands.
5. Execute part of the sequence, observe again, and replan.

RPDP additionally compares current and goal RGBD crops. A learned gate adds visual features to the pose features. **This combines features; it does not require a delta-action adapter.** [PoseInsert paper](https://arxiv.org/html/2505.09424v2)

## 2. How our method works

Our current system is **PoseInsert-inspired**, rather than a complete implementation of RPDP.

```text
Rule-based transport
        |
        v
Coarse model identifies the requested port region
        |
        v
Refiner estimates the port pose; grasp estimator estimates tip offset
        |
        v
Accept a stable estimate and hold that port frame fixed
        |
        v
Explicit visual/proprioceptive approach brings the tip to about 25 mm
        |
        v
Learned insertion actor predicts a short target sequence
        |
        v
Convert targets to robot commands, execute a prefix, and replan
```

The 25 mm approach is the newer diagnostic runtime. Older full-rollout evaluations handed over at approximately 150 mm. Successful handoff is not successful insertion.

The insertion actor receives:

- Current images from all three cameras.
- Robot state and public task code, including which target is requested.
- Current TCP pose expressed in the accepted port frame.
- An estimated offset from the TCP to the connector tip.
- A short history of robot/relative-pose observations. The repaired runtime preserves this history through transport; it does not add historical images to this actor.

It predicts **complete target tip positions and TCP orientations in the port frame**. Geometry converts these into TCP commands. These are not corrections added to a separate BC action.

The current insertion network is a **deterministic transformer, not diffusion**. The newer comparison changes prediction horizon, executed prefix and continuity training; those changes do not make it RPDP.

The accepted port pose stays fixed during an insertion attempt. Robot motion still changes the computed relative pose. We therefore are not repeatedly changing the reference frame from noisy refiner predictions at every action.

## 3. The most important representation gap

**We explicitly estimate the tip's position, but do not yet supply an approved estimate of the connector's full orientation relative to the gripper.**

Imagine holding a pen:

- Knowing your hand's orientation tells you which way your hand faces.
- It tells you which way the pen faces only if you also know exactly how the pen sits in your hand.

Our actor explicitly knows the equivalent of the hand orientation and the tip offset. If the connector is grasped at a different angle, it must infer the remaining angular relationship implicitly from images and its training experience.

In symbols:

\[
R_{\text{connector in port}}
=R_{\text{TCP in port}}\,R_{\text{connector in TCP}}.
\]

The last rotation is the missing approved input. Our grasp estimators already produce diagnostic rotation estimates, but those have not been approved for autonomous control. We must validate them before using them.

**An accurate port pose alone does not establish that the connector is correctly aligned.**

## 4. Other differences and why they exist

| Component | Our current implementation | Practical implication |
| --- | --- | --- |
| Action generator | Deterministic transformer | Does not reproduce the paper's diffusion policy. |
| Pose representation | Tip position plus TCP orientation | Full connector orientation is partly implicit. |
| Visual conditioning | Three current RGB images | No matched current/goal RGBD branch or pose-primary gated fusion. |
| Training targets | Recorded expert command targets, converted into the port frame | Must preserve controller semantics; measured future motion is not interchangeable with a command under compliance. |
| Recovery | Separate hardcoded backtracking logic | Its integration must be tested independently; it is disabled in the current controlled near-port actor comparison. |

These differences accumulated as the project evolved. Earlier small development tests favored a direct BC actor over the diffusion implementation available then. **That was not a fair basis for concluding that diffusion is generally worse.** The present implementation also lacks the matched goal-RGBD interface and validated full grasp rotation needed for a closer reproduction.

There are additional engineering problems beyond architecture: earlier traces showed large changes between successive predicted chunks, and the old runtime discarded observation history at handoff. The conversion audit found no gross numerical error in the tested frame transformations, but correct transforms do not make inaccurate predictions correct.

## 5. Can we fix the gap?

**Yes, the implementation gaps are fixable. We do not yet know which repair will produce reliable insertion.**

The sensible order is:

1. **Validate full connector orientation.** Audit the existing grasp-rotation estimates on held-out episodes using simulator geometry only as evaluation/training labels. Measure errors by task and occlusion.
2. **If reliable, make the representation fully object-relative.** Supply connector position and orientation relative to the port, predict future connector poses, and convert them back using the full estimated grasp transform. Rebuild labels consistently from recorded expert commands.
3. **Make geometry an explicit primary input.** Compare a separate pose branch with visual assistance against the current fusion. A feature gate would not reintroduce the rejected delta-action adapter.
4. **Finish the controlled history/horizon/continuity comparison.** Use physical insertion and measured motion, not merely lower training loss, to judge improvement.
5. **Then consider a matched diffusion comparison.** Keep data, representation, starts and evaluation criteria fixed so that architecture is the actual variable.

We should not replace all components simultaneously. That would make it difficult to tell which change helped.

The missing connector orientation is a plausible fine-alignment problem. **It has not been proven to explain the large command jumps or the earlier failure to approach from 150 mm.** Contact control, training coverage and recovery can still fail even after the geometric representation is repaired.

## Conclusion

Our method shares the useful idea of expressing insertion relative to the port. It currently stops short of the paper's full connector-relative representation and policy design. The clearest representation repair is to validate and explicitly use the connector's orientation, while separately addressing unstable replanning. Neither repair should be presented as a guaranteed insertion solution before rollout evidence supports it.

Implementation and experiment details: [history/horizon continuation](../experiments/2026-10-17-history-horizon-poseinsert-continuation.md).
