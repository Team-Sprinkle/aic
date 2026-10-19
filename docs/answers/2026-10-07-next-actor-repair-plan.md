# Next actor repair plan

**Execution update, October 7:** Steps 1–4 were completed. The later-state
teacher gate passed 6/6 and the matched offline correction gate passed, but
the fresh autonomous actor reached the safe buffer only 1/3 times, below the
2/3 live gate. Therefore step 5 is **not authorized by this plan**: no
automatic handoff, insertion or RL was run. The later-state correction
recipe is stopped; see the [final record and distinct next experiment](../experiments/2026-10-07-sfp-later-state-causal-repair.md#fresh-autonomous-live-result-gate-failed).

The current blocker is **autonomous SFP transport**: on the latest three
fresh starts, both the revised actor and its control reached the safe area
for perception handoff 0/3 times. The stall-filtered training comparison
did not improve held-out metrics. The selected-port coarse crop passed its
offline nearby-port check, but has not been validated as a live handoff.

## 1. Find the first *causal* route error

Use the saved 2-, 3-, and 5-card actor/teacher rollouts. Compare the first
several actor decisions against the teacher at **matched robot pose and route
progress**, rather than just at equal clock time. For each decision inspect
all three actual camera images, task code, previous actions, teacher and actor
TCP-body targets, measured TCP response, and contact/force. Check image age,
history construction and command conversion at runtime against training.

This decides whether the actor **chooses the wrong direction while motion is
still free**, or whether it chooses a reasonable target that the controller
cannot execute. If the latter, diagnose the specific collision or control
limit before collecting more policy labels. Budget: the three archived scenes,
read-only plus at most two replay controls; no new model fit.

## 2. Test recovery from the *later* actor-visited deviation

If step 1 shows a policy route error, take at most six nonfinal **training**
resets across 2–5 cards. Run the frozen actor to the observed **simulation-time** 8–12 s
deviation, then let a privileged, collision-checked teacher attempt a
continuation. Retain the real images, measured state and recorded full
TCP-body targets only after takeover. Record whether the teacher returns to a
safe route and fully inserts, and whether contact occurs. Simulator geometry
is a teacher label and scoring tool only.

**Gate:** at least four of six later-state continuations must be contact-safe
and fully insert. This is deliberately different from the already failed
October 3 experiment, which collected ten successful continuations after
only a two-second actor prefix yet still produced 0/3 autonomous arrivals.
If the later-state teacher cannot recover, stop data collection and repair
the route/controller or design a safe recovery teacher first.

## 3. Train one matched, full-action correction comparison

Only if step 2 passes, train one actor with the new actor-visited examples
and one equal-budget control without them. Keep the same architecture,
initialization, complete-episode split, request-pair supervision, seed and
update count (cap **2,000 updates**). Keep ordinary successful SFP and SC
examples so the one actor still handles normal transport. Do not predict a
delta on top of the old actor or give true port geometry to the policy.

First check held-out full-action error by task/card count and whether the
actor commands the safe correction on withheld actor-visited states. Stop
before live rollout if any grouped old/new SFP or SC action p95 worsens by
>10%, or if the correction direction is not reliable on withheld states.

## 4. Test autonomous transport on fresh development starts

If the offline gate passes, run at most three fresh nonfinal starts spanning
2, 3 and 5 cards, matched against the frozen control. Require **at least
2/3 safe buffered arrivals**, no worse official contact score, and full
actor inference p95 below **300 ms**. Save videos, measured routes, forces,
and first-divergence reports. Do not open the reserved final configurations.

If this live gate fails, stop this BC correction recipe. Use the step-1
diagnosis to choose one distinct route-progress or goal-representation
experiment; do not simply increase the number of similar teacher episodes.
Earlier discrete route/FiLM and short-prefix correction attempts already
failed their gates, so any new representation must address a specific
measured error those attempts missed.

## 5. Connect handoff only after transport passes

Run the pause-probability head and original-image selected-port crop in
shadow on the successful autonomous approaches. The current pause head is
tied to an older actor's features, so it must be refit or checked against the
selected actor. Require no accepted wrong port, reliable timely proposal,
and complete pipeline p95 under 300 ms before automatic handoff. Then test
the near-port insertion actor. RL remains deferred until this supervised
transport/handoff path is reliable enough to provide a safe warm start.

Existing evidence: [actor/coarse live gate](../experiments/2026-10-07-actor-proposer-coarse-continuation.md),
[stall-data test](../experiments/2026-10-07-sfp-stall-data-audit-and-cleaned-fit.md),
and [earlier failed short-prefix correction](../experiments/2026-10-03-sfp-balanced-arrival-onpolicy-check.md).
