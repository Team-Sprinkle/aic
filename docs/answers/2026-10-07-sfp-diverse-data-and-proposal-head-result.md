# SFP data-diversity and proposal-head result

I tested the idea that the transport actor lacked varied, correctly labeled
routes. The previous 326 SFP training episodes were heavily skewed toward
card 0/port 1 (235 episodes), with several sparse one- and two-card request
combinations. I selected new **development** scene configurations across
card counts, requested cards, ports and reset geometry. I used their scene
YAML only, then generated fresh CheatCode trajectories in rootless Gazebo;
old agent/VLM action arrays were not used.

The fixed first batch produced **132 full insertions in 137 attempts**.
Five partial insertions were excluded. A bounded one-scene-per-affected-cell
retry produced four additional full insertions and one no-insertion. One
Gazebo spawn crash was preserved and rerun on the **same** scene, which then
fully inserted. All 136 full-scoring episodes passed the causal camera,
measured-state and recorded-TCP-command audit, adding **35,383 TCP-body
transport chunks**. I held out 17 complete new scene groups for development.
The merged SFP dataset has **445 training episodes and 27 development
episodes**. One sparse cell, one-card/card-4/port-1, remains one example
short after its allowed retry failed. The original expert dataset is intact.

I trained two matched temporal transport actors for 3,000 updates from the
same checkpoint. The control used the old training episodes; the other also
used the new verified training episodes. On the 17 new held-out scenes,
first-action position error p95 fell from **1.856 to 1.668 mm** (10.09%). On
the ten older SFP development scenes it fell from **2.033 to 1.676 mm**;
the SC subset rose from **1.899 to 2.032 mm** (7.04%, within the declared
10% regression limit). These are imitation metrics, not autonomous
insertion scores.

The important route-choice test failed. When I kept the *same images, robot
state and history* and changed only the requested card/port code, the
first-action spread rose merely **0.572→0.620 mm (1.08×)**; the gate was
2×. The new scenes added visual variety but no same-reset alternate-request
action pairs. This explains why ordinary action error improved while the
actor still showed little evidence that it chooses a different route for a
different requested port. I therefore did **not** run a live promotion
rollout or claim improved autonomous full insertion.

I also trained the requested **pause-proposal probability head**, leaving
the actor's action weights frozen. Its selected checkpoint proposed in time
on **26/27 grouped teacher development routes** with no false-positive
teacher rows at the chosen threshold. The earlier head scored the same
26/27 on those scenes. On three saved autonomous routes, the new head's
first proposal was correctly inside the broad pause region twice and
**false once**. It was also overconfident after passing the window. Its
sigmoid score is not calibrated; it remains **shadow-only** and cannot hand
control to perception or insertion.

The next experiment should supply a causal requested-goal signal: shared
reset/observation examples where two requested ports require materially
different expert next actions, or a better observation-based selected-goal
representation if such action pairs cannot be generated. Separately, train
the proposer with actor-route states **after** and **outside** the pause
window, then test on new grouped off-route starts. The transport, visual
identity, latency and safety gates must pass before autonomous insertion or
RL; reserved final scenes remain sealed.

A bounded version is at most **20 new nonfinal resets and 40 teacher route
queries on one Gazebo GPU**, with both requested tasks queried from the same
measured observation at the route branch. Proceed to one matched actor fit
only if this yields at least **30 causal pairs whose next TCP-body targets
differ by ≥5 mm**. Stop this data branch otherwise and work on the
observation-based selected-goal representation. The paired fit must at least
double same-image request sensitivity without >10% old/new action-error
regression before at most three matched autonomous starts. Keep the proposer
repair separate, using new actor-route post-window negatives and a held-out
off-route scene split.

Artifacts: [full experiment record](../experiments/2026-10-06-sfp-diverse-transport-and-proposal.md),
[machine decision](../../outputs/experiments/2026-10-06_sfp_diverse_transport/decision_summary.json),
and [dataset map](../DATASETS.md).
