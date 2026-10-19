# Future work

The demonstration phase is complete. This is the **single current improvement plan**; the old standalone ACT/world-model, actor, handoff and recovery roadmaps are retired. Detailed historical experiments remain evidence, not queued work.

## 1. Repair the four measured reliability gaps

The fixed released-config batch inserted 53/57 trials. Diagnose the initial quiet-force-reference timeout, SFP crop/acquisition timeout, SC blockage before marker acquisition and SC missing-marker timeout separately. Add causal force/timestamp diagnostics, a bounded observation maneuver and safe marker acquisition before committing to a large sideways move. Compare each repair against the frozen controller on complete episode groups, including failures; do not relax contact or pose gates just to improve the score.

## 2. Compare local and hierarchical recovery fairly

Retain measured TCP position/orientation checkpoints and the actual issued command references. Start with a short retreat and exploration perpendicular to recent motion; repeated blockage can escalate to an earlier checkpoint and fresh port/plug perception. Use identical starts and budgets for no recovery, local recovery and hierarchical recovery. Report full insertion, force, collisions and latency. Hierarchical benefit is a hypothesis; the final recordings used local mode.

## 3. Learn recovery with SERL-style online RL

Use one actor for ordinary approach and recovery across SFP and SC. Warm it with verified commands from both families; retain the current rule transport as a baseline until the learned actor qualifies. Train probabilistic **complete actions/targets**, rather than a small residual on a frozen BC action. Keep rule-driven backtracking explicitly labeled and exclude those commands from actor imitation. Use correlated exploration during lateral retry.

A bounded offline critic warm-up may be useful if successful and failed transitions have trustworthy measured rewards and causal endpoints; offline RL cannot create missing backoff or detour experience. Then collect short online failure/retry episodes, with replay and BC anchoring to limit regression. Use observed insertion and force/progress outcomes, not an unobserved terminal proxy. Compare policy improvement with the same rule supervisor enabled in both arms.

## 4. Establish generalization and simulator transfer

Generate independent board/card/port/grasp scene groups before freezing the next comparison. Include early transport blockage, port-rim misalignment, gripper/card contact and reproducible cable/card interference. Keep the reserved internal final split sealed. Isaac can supply parallel recovery practice only for mechanically validated failures; Gazebo remains the final runtime. Any Isaac-to-Gazebo BC/RL adaptation must be tested on Gazebo with the same observation contract, conservative force behavior and complete inference p95 below 300 ms. Use rootless Docker, cooperative locks and at most four project GPUs.

[Measured gaps and experiment history](EXPERIMENTS.md) · [Detailed completion record](https://github.com/Team-Sprinkle/aic/blob/82cae80/docs/answers/2026-10-19-final-project-completion-notes.md)
