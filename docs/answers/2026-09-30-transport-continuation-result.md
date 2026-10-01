# September 30 continuation result

The accepted **teacher transport** gate still stands. A new matched-world, Cartesian SC port-0 teacher fully inserted (official Tier 3 score 75), so the current v9 BC set contains two fully inserted SC demonstrations, one per selected port, in the same five-card world. Failed insertion actions were excluded. The verified SFP source remains included.

The targeted balanced-port transport BC run finished 1,500 updates. On the same two already-open autonomous development starts, it **did not improve insertion or handoff**:

| Frozen transport actor | Port 0 / two cards: final standoff error | Port 1 / three cards: final standoff error | Full insertion |
| --- | ---: | ---: | ---: |
| Earlier controller-error-masked actor | 76 mm | 74 mm | 0/2 |
| New paired-port balanced actor | 103 mm | 603 mm | 0/2 |

These errors are distance from the recorded transport standoff, not perception error or insertion depth. Both new rollouts remained far outside the 12 mm lateral / 20 mm axial / 3° handoff envelope. Inference p95 was 140/128 ms, below the 300 ms limit; the occasional cold-start latency spike is recorded separately. Both rollouts were autonomous: only ordinary camera images, robot state and task code reached the actor. Final scenes remained sealed.

The bounded trace and video review points to an **early route-command failure**, not a cable snag. The port-0 actor asks for the wrong lateral direction near the start; the port-1 actor and robot move far from the selected opening within seconds. Force stays near its gravity-loaded baseline, and the wide-camera videos do not show a sustained cable blockage. A same-image/state selected-port swap changes the command only slightly at the problematic states. The precise cause of that weak target-conditioned steering remains open; visibility labels are not yet certified.

The predeclared stopping rule therefore applies: stop routine feed-forward transport BC expansion. Across ten candidate rollouts on the same two development starts, none fully inserted, and the learned transport handoff has not passed. The local actor, temporal handoff and RL are not promoted.

**Next bounded experiment:** using existing grouped episodes and saved observations only, test whether an observation-only selected-port route subgoal and route-progress signal distinguish the early wrong-direction states. Budget: at most two CPU hours, no new data or GPU training. On held-out rows with at least 5 mm teacher lateral travel, require at least 80% direction accuracy and a 20 percentage-point gain over the frozen v9 actor, plus correct direction during the first two seconds of both saved starts. If this passes, allow one distinct temporal goal-conditioned actor, at most 1,500 updates on one GPU and two matched existing development starts. Continue only if both meet the measured handoff envelope with acceptable force and p95 inference below 300 ms. If the offline check fails, revise the observation/history or acquire a separately justified matched coverage set. Broad RL and final-scene tests remain closed.

Evidence: [experiment record](../experiments/2026-09-30-transport-to-two-actor-continuation.md#paired-port-bc-result-and-bounded-failure-diagnosis), [machine status](../experiments/2026-09-30-transport-continuation-status.json), [artifact map](../../outputs_README.md). Five-view 20 fps recordings: [port 0](../../outputs/experiments/2026-09-30_transport_continuation/autonomous_transport/paired_ports_port0_two/video/all_views_20fps.mp4), [port 1](../../outputs/experiments/2026-09-30_transport_continuation/autonomous_transport/paired_ports_port1_three/video/all_views_20fps.mp4), and [successful teacher](../../outputs/experiments/2026-09-30_transport_continuation/cartesian_matched_teachers/port0_five_paired/video/all_views_20fps.mp4).
