# Are we headed toward a reliable handoff?

**Execution update, October 7:** A bounded local-slot model and a shared-3D repair both failed the requested-port identity/accuracy gates; a matched privileged target-cue head did not materially improve teacher-action prediction. The proposed board-origin/slot-order *temporal* representation remains untested, and an independent review of the complete public observation remains outstanding. [Execution record](../experiments/2026-10-07-sfp-slot-feature-port-cue.md).

## Short answer

The **two-actor idea is still sound**: a transport controller gets the plug into a safe approach area, and a precise visual controller aligns and inserts it. The current *way we connect them* is unlikely to reach high reliability by simply training the same models longer. I would rebuild the **target-identification and transport interface**, while retaining the useful pose refiner and the conservative verification principle. This is a diagnosis of the present evidence, not proof that a rebuilt system will work.

## Where we are

The proposed deployment sequence is transport → coarse requested-port check/stop → original-image pose refinement → insertion. It has not demonstrated a complete autonomous handoff and insertion across the task settings.

| Part | Best relevant evidence | What it means |
|---|---|---|
| Transport to requested SFP port | 0/3 safe buffered arrivals on new sparse-card starts; matched teachers inserted | The actor can move, but these routes did not end in the right approach corridor. Three starts are diagnostic, not a success-rate estimate. |
| Requested-port identity | 3/10 early on a harder ten-request replay; a permissive multi-camera matcher accepted eight wrong neighboring ports on ten fresh groups | The system can find *a* plausible opening more often than it can establish that it is the requested numbered opening. |
| Precise pose after acceptance | 2.71 mm accepted 3D p95 with zero wrong selections on one fresh ten-group comparison, but only 6/10 early coverage on another ten-group set | Refinement is useful after a valid candidate; conditional accuracy does not solve candidate identity or availability. It is not yet the 0.5 mm insertion guarantee. |
| SC shadow handoff | 4/4 new development starts accepted a visually checked transport pause, without insertion; known hard identity/posture cases remain | The staged mechanism is feasible in some SC scenes, but it is not evidence of general full-task success. |

The recent 320 px crop gave clean two-camera separation in 6/10 development groups, below the planned 9/10. The first human RGB review omitted measured TCP state, so its abstentions do not settle what the *complete* observation can identify. A corrected 30-panel packet is ready but has no independent score. Nominal world coordinates are too inaccurate to substitute for visual identification: the measured development median port error was 44.6 mm.

## What looks architectural

1. **Target identity arrives too late for transport.** The current actor consumes pooled camera features, measured state, history and a task code, but no trustworthy explicit observation-derived requested-card location. On the same image, changing the SFP request changed its first action target by only 0.554 mm median, while paired teachers took noticeably different routes. The actor must infer both *which object matters* and *how to move* inside one action loss. This makes wrong-route behavior unsurprising. It is an architectural and supervision mismatch, not proof the transformer is too small.
2. **The current pipeline creates a dependency loop.** We wait for transport to approach before asking perception to identify the nearby target, yet transport may need target identity to choose the approach route. A coarse check must run when the board is *most identifiable*, which may be earlier than the final stopping point. Its result can guide transport; the precise refiner can still wait until near the port.
3. **Cross-view geometry verifies a 3D point, not its name.** Two or three cameras can agree on the wrong neighboring opening; the eight fresh-group false accepts demonstrate this. Temporal persistence also cannot fix a stable wrong identity. A smaller crop improved neighbor exclusion but lost or still mixed targets in four groups. We need an explicit board-origin/slot association with an abstain state, rather than more confidence in whichever heatmap peak triangulates best.
4. **A stop probability learned from teacher timing is not enough.** The SFP frozen stop head proposed on only 2/10 held-out teacher episodes at its strict threshold; relaxing it caused early events. A coordinate-region veto is broad, and a prior diagnostic stopped 162 mm from the requested SFP opening. The stop should mean a fresh, observation-supported *safe approach to the correct target*, including TCP/plug pose and orientation, not merely a high learned phase probability.

There are also data and evaluation gaps: SFP training remains skewed toward card 0/port 1 (235/326 in the expanded transport set); ordinary successful teacher trajectories cover little autonomous off-route behavior; the important identity results have only about ten grouped scenes per check. These gaps matter, but the matched paired-data and final-head experiments already show that more of the same action BC did not restore target-sensitive live routing.

## Recommended redesign

Use a **shared, task-conditioned scene representation** before and during transport:

```text
three cameras + sparse distance-based history + measured TCP + task request
                         |
                         v
         board/row map: board origin, slot order, occupancy,
         per-slot candidate ports, visible/unknown evidence
                         |
            requested slot/port query
                         |
             +-----------+-----------+
             |                       |
      target cue for transport    safe-to-inspect proposal
             |                       |
      target-aware route         fresh high-resolution
      and generic safe motion    selected-port refiner
             |                       |
             +-----------+-----------+
                         |
              measured safe handoff check
                         |
                insertion actor + recovery
```

The board map is **not** a requirement to predict every precise port pose all the time. It establishes the row origin and slot order when visible, tracks those identities through travel, and reports *unknown* if the origin/target is occluded. The task query selects one slot; the local refiner then estimates only that opening's accurate pose from original-resolution images. Use the same target cue for transport and stopping so they cannot silently disagree about which card is requested. Measured TCP helps relate observations across time; fixed camera calibration and CAD spacing are allowed, but per-scene simulator port geometry remains training/scoring data only.

If the board origin is not visible at startup, the first transport stage should go to a **generic safe inspection posture** that exposes the row, then choose a target-specific route. This is not a fixed nominal world-port stop. If no safe camera posture reveals a slot reference, the current observation set may be insufficient for reliable numbered-card selection; we would need an earlier view, a changed camera, or a route that remembers the board origin from a visible moment. We should establish this empirically with the corrected review rather than declare the task impossible from the partial RGB audit.

The insertion refiner and conservative acceptance logic are worth retaining. They should accept only after the board map confirms the requested identity, the refiner gives a consistent pose, and measured plug/TCP position and orientation are inside a safe handoff envelope. A rejection resumes observation/transport instead of guessing. The full composed inference path must stay below 300 ms p95.

## Decisive next tests, in order

1. **Resolve observability using complete public input.** Independently annotate the frozen 30-panel packet, including measured TCP and task request, without opening its sealed target mapping. Separate “requested slot clearly identifiable,” “only generic row visible,” and “ambiguous/occluded.” Score by complete reset and phase. This tells us whether the proposed board map has information to learn and when to run it.
2. **Build one explicit board/slot association candidate on existing grouped data.** Compare it with the frozen selected-port model on hard same-board alternate requests and wrong-neighbor cases. Use geometry only for labels and post-run scoring. Require zero accepted wrong identities and at least 9/10 early usable proposals on new nonfinal groups before it drives the robot. Count abstentions and full latency. If identity remains impossible in the current views, stop fitting crop variants and test one safe earlier inspection posture instead.
3. **Test whether explicit target information repairs transport.** First use the *predicted* board-map target cue in a matched actor comparison with identical data and update budget. Check whether same-board alternate requests cause the correct distinct early route and whether ≥2/3 new development starts reach the 10 mm buffered region without added contact. A privileged true-target cue may be used only as a **diagnostic upper bound**, never for autonomous promotion. If even that upper bound does not repair transport, investigate route/action supervision and off-route data rather than perception.
4. **Then connect stop, refiner and insertion.** Evaluate the whole chain on new episode-grouped development starts: correct requested identity, safe arrival, explicit orientation, no harmful contact, complete p95 latency <300 ms, and full insertion. Keep the reserved final scenes sealed until these gates pass. RL should follow a reliable supervised handoff, not compensate for an unknown target identity.

**My recommendation:** change the interface now, not the entire project. The likely high-value rebuild is a board-structured target cue shared by transport and handoff, plus an early observation step when the target row is not identifiable. Retain the precise refiner and two-actor division. The strongest current warning is that a model can be confidently and geometrically *consistent about the wrong card*; no amount of pose refinement after that will make the handoff correct.

## Evidence

- [Transport versus coarse bottleneck](2026-10-06-transport-vs-coarse-bottleneck.md)
- [Matched route and handoff continuation](../experiments/2026-10-05-transport-route-handoff-continuation.md)
- [Spatial target prerequisite](../experiments/2026-10-05-spatial-transport-prerequisite.md)
- [Joint-camera wrong-card cases](../experiments/2026-10-06-sfp-joint-camera-candidates.md)
- [Corrected review and crop gate](../experiments/2026-10-07-sfp-observation-complete-review-and-tight-crop.md)
- [SC shadow handoff](../experiments/2026-10-03-live-sc-shadow-handoff.md)
