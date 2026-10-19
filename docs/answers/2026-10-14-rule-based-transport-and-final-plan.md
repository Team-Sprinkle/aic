# Rule-based transport and the consolidated plan

Yes—**rule-based transport is worth testing**, particularly for SC-to-SC. I added it as an early branch in the finalized high-level plan:

**[docs/TRANSPORT_HANDOFF_IMPROVEMENT_PLAN.md](/data1/chmin/yj/ws_aic/src/aic/docs/TRANSPORT_HANDOFF_IMPROVEMENT_PLAN.md)**

The earlier noisy-distance result does not rule this out. That experiment tested whether a nominal port coordinate could decide when to stop. It did not test whether a conservative waypoint route could bring the robot into a useful viewing position.

My recommended version is:

1. Follow the familiar **back → left-and-down → forward beside the cards → right → tilt → return** sequence.
2. Advance when measured position and orientation show that each waypoint is reached—not when a fixed timer expires.
3. Use conservative clearances, with rough visual adjustment if startup/layout variation requires it.
4. Hold near the expected target area, then let selected-port perception authorize handoff to insertion.

There is one important limitation: our successful teachers adjusted their waypoints using true board/port geometry. They are not evidence that one fixed trajectory works autonomously. Across the 83 training plans, stage waypoint positions vary by roughly **5–6 cm at p95**, so averaging them with small margins is not justified.

The plan compares a **conservative common corridor**, an **observation-adapted waypoint template**, and—only if needed—a small route library or hybrid prefix. It puts a bounded Gazebo feasibility test before further broad actor training, while retaining the terminal-label repairs and learned-actor comparison.

If the rule-based route wins, we can replace learned transport for the supported cases and concentrate learning on perception, insertion and recovery. **No new training or rollouts were launched.** Main status and handoff documents now point to the consolidated plan.
