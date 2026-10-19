# Overall recommendation after the pose-actor reset test

## The current bottleneck

The project needs to move a plug around cards, arrive near the **requested** port, identify that port precisely, insert, and recover from contact. We have useful pieces, but have not demonstrated this whole sequence autonomously.

The strongest current bottleneck is **closed-loop transport**. Actors trained on successful teacher routes can predict individual teacher actions within a few millimeters, yet the latest independent 2/3/5-card Gazebo comparison reached the safe approach region on **0/3 starts for both actors**. Small errors compound after the robot leaves teacher states. This is the type of state-distribution problem studied by [DAgger](https://arxiv.org/abs/1011.0686); our teacher takeovers are relevant, but the existing late takeovers did not supply enough usable visual correction supervision at their first decision.

Perception is promising but not a complete solution. Frozen requested-port perception accepted the correct port on 3/3 saved off-route actor prefixes, with 3.10 mm accepted 3D p95 on those sampled observations. That is useful for a coarse transport cue, while insertion needs a tighter local estimate. At the first recorded late teacher correction, this pose was accepted on only 3/6 training and 1/3 development episodes. Two full-data actor fits and the latest partial-restart test showed the actor barely changes its actions when pose is removed. Resetting the action head worsened performance. The correct port position also does **not** specify a collision-free path around cards or a trailing cable.

The near-port insertion work has separate promise, including 6/6 short local seats in a development test **with a privileged startup tip-restoration step**. That is an 8 mm local seating result, not 40+ mm autonomous insertion from the normal episode start. The SC lateral-bypass teacher generated useful full-insertion examples, but its privileged route success is not an autonomous SC transport result. Recovery and RL should be evaluated after the normal transport-to-insertion chain works.

## The system I would aim for

```text
task code + three cameras + measured robot state + distance-sampled history
                              |
                              v
             TRANSPORT: choose a short safe route waypoint
             and execute/replan, not just copy the next teacher action
                              |
                    proposes a near-port stop
                              v
             VISUAL CHECK: is this the requested port,
             visible and safely positioned for local control?
                              |
                    accepted selected-port crop/pose
                              v
             INSERTION: local port-frame policy using refined
             pose, images, robot state and force
                              |
                contact or blocked progress
                              v
             RECOVERY: measured-path backtrack, then a
             controlled alternative approach; learn search later
```

This retains the agreed two-actor split: one transport actor across normal transport and transport recovery, then a separate local insertion actor. The transport actor should consume the public requested card/port code and may use a frozen **predicted** port cue when one is accepted. The local actor uses a refined pose expressed relative to the selected port. Neither actor receives true simulator port geometry at deployment. The stop/check logic remains a separate safety gate. The hierarchical backtrack can remain explicit for blocked motion; RL can later learn which alternative approach works after backtracking.

## Change the transport learning problem

I recommend **one bounded route-waypoint experiment**, rather than another full-action BC run with a pose vector appended near the output. The transport model should predict a short, obstacle-aware waypoint or route corridor in the current TCP/body frame, plus route progress. A small controller or waypoint-conditioned complete-action decoder then follows that prediction and replans from new observations. The waypoint becomes an explicit control bottleneck: the actor cannot quietly ignore the route target. It is **not** an action correction added to an old policy.

This differs from the earlier frozen one-second motion-goal test: that probe was trained only for the first 0–1.6 seconds, its later error grew sharply, and the goal was simply added to an action network (then gated off after 10 mm of progress). The new route target must be learned across the relevant transport phases, remain observable as a policy output, and actually determine the commanded path. An offline waypoint metric alone is still insufficient; the closed-loop arrival test is decisive.

Use the existing successful SFP teacher routes to label the next safe intermediate target, preferably a waypoint that captures the chosen side of the card row rather than simply the final port position. Start by auditing the existing 5–8 second actor-to-teacher takeovers for first-correction image/pose coverage and causal action labels. If they are insufficient, collect only targeted *early-departure* takeovers with a fixed stopping rule, where the requested port or card route is still observable. Split by complete reset configuration and requested target. Compare a route-waypoint actor to a same-data action-only control. A predicted target-port pose can be an input or a geometric consistency target, but the waypoint must be supervised directly from the teacher route and tested for use.

The offline tests should ask: does the waypoint pick the correct side of the row, is it geometrically reachable without card contact, does its error remain within the measured route clearance, and does changing the requested card/port change the chosen route appropriately? Then test complete closed-loop transport on **three new nonfinal starts**; require at least **2/3 safe arrivals**, no worse contact, and complete inference p95 below 300 ms before expanding to a larger grouped development set. The three-start gate is only a pilot, not a reliability claim. Keep reserved final scenes sealed. If a visually conditioned waypoint still fails despite correct offline route labels, inspect the first divergence in the rollout; do not keep scaling the same data or switch to a larger model by default.

This is consistent with the reason [DAgger](https://arxiv.org/abs/1011.0686) labels states visited by the learner. Replanning after executing only part of a predicted horizon is also a standard closed-loop idea in [Diffusion Policy](https://arxiv.org/abs/2303.04137), but our next test does not need diffusion. For the later precision stage, [PoseInsert](https://arxiv.org/abs/2505.09424) motivates the port-relative pose/action representation; that does not solve the earlier obstacle route by itself.

## What I would defer

- **Full actor training from scratch:** the smaller action-head reset already lost ordinary behavior and still ignored pose.
- **Another generic data-scale or hyperparameter sweep:** current evidence points to route supervision, accepted-pose coverage at correction, and closed-loop state shift.
- **Automatic handoff and insertion:** first show the transport actor repeatedly reaches the safe region and that the visual check rejects wrong ports on its own views.
- **RL and world-model branches:** later use RL for local contact/search and cable interactions once the supervised route and normal insertion path work. The existing world-dynamics fidelity does not justify relying on imagined cable outcomes now.

The immediate question is therefore **“Can a visually conditioned actor choose and follow the correct intermediate route?”**, not “Can a larger actor imitate the next command more accurately?”

Local evidence: [latest training-method diagnostic](../experiments/2026-10-08-sfp-pose-actor-training-method.md), [transport and perception record](../experiments/2026-10-07-sfp-early-branch-support.md), and [project overview](../robotic_cable_insertion_project.md).
