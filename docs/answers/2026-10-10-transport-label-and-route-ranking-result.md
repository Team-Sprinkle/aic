# Transport continuation result

I ran the two bounded checks we agreed on. **I did not train a new transport actor**, because the proposed supervision and route-comparison gates did not both pass. The existing actor and reserved final scenes were untouched.

## Stage-aware teacher labels

The earlier actor-to-CheatCode demonstrations were real. I reanalyzed their **measured TCP motion** rather than treating the first teacher command or a fixed one-second waypoint as the correction. In the existing cache, all **16 training and six held-out development episodes** had a short coherent motion segment; an independent check against later motion agreed in **5/6** development episodes. I also audited the separate October 8 three-run pilot, retaining its partial-insertion failure.

I ran **two new, distinct four-card training takeovers** in rootless Gazebo with cable-contact recording. Both fully inserted. Together with two successful pilot runs, that gives **20 distinct training and six held-out development** resets with short motion segments. These counts satisfy the numeric support target, but they do **not** make the actor-boundary label correct: in **six of the original 22** episodes, the teacher had moved more than 5 mm before the identifiable segment began. Attaching that later segment to the earlier actor image/state would be a silent causal mismatch.

The safety check also mattered. One newly successful teacher continuation reached **55.52 N sampled cable-link/NIC-card point normal force** later in the route, although the official challenge score recorded full insertion and no prohibited contact. Its short initial label segment stayed below the 10 N engineering threshold, but the *whole route* cannot be called a cable-safe expert continuation under the frozen gate. The other new continuation peaked at 2.00 N after takeover. Thus a short label can look good while the route later creates a cable problem.

## Alternative short-route ranking

I first checked whether prior Gazebo routes actually began from the same state. Five runs used the same evaluation configuration and request, but their measured actor-takeover TCP positions differed **11.83–31.07 mm** pairwise. The high-side route passed once in three starts. Because the robot—and likely cable—started differently, those outcomes cannot fairly tell a route ranker which *route* was better at a fixed state. The repository has no existing exact complete cable-state restore path. Training a route selector on these five examples would overstate the evidence.

I [reviewed the takeover camera gallery](../../outputs/experiments/2026-10-10_sfp_stage_aware_label_gate_v1/route_takeover_camera_gallery.jpg) as an observability check. A diagnostic overhead camera shows the broader orange cable loop, while the three wrist cameras supplied to the actor mainly show the port area and a short cable segment. This does not prove cable risk is invisible—history and force might help—but it gives a concrete reason to test observability before training a cable-risk head.

## Decision and next bounded repair

**Stop this actor fit and the proposed matched-state ranker fit.** The immediate engineering test is whether one nonfinal Gazebo actor takeover can be saved and reproduced with its robot joints, cable segment poses/velocities and contact state. Compare two no-action restores before testing alternate routes. Budget one GPU and up to two hours; require TCP within 2 mm, orientation within 1°, proximal cable-link RMS within 5 mm and first-second wrist-force p95 difference within 2 N. These are proposed tolerances, not achieved results. If exact restoration is infeasible, change the learning question to predicting route risk *conditioned on the actual observed cable/robot state*, using independent whole-reset tests rather than pretending the alternatives started identically. Any subsequent actor training must represent the complete takeover-to-motion-onset sequence and use contact-instrumented safety labels. Automatic handoff, insertion and RL stay off.

The [experiment record](../experiments/2026-10-10-sfp-stage-aware-transport-label-gate.md) contains the commands, data provenance, thresholds and failure analysis. The [machine-readable decision](../../outputs/experiments/2026-10-10_sfp_stage_aware_label_gate_v1/decision.json) includes every new contact peak and all ten pairwise route-start comparisons. The [current status](../STATUS.md) and [experiment ledger](../EXPERIMENTS.md) are updated.
