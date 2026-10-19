# Coarse perception, history, and handoff: current status

Yes, the recent work focused on **coarse perception for SFP-to-NIC**: the part that should find a small image region containing the **requested** port so a precise refiner can estimate its pose. The newer requested-port model was an **experiment**, not a replacement that passed the safety gate. The older all-candidate coarse model is still the comparison baseline; it looks at one instant and outputs possible locations for all ports.

```text
Task code + cameras + robot state
          |
          v
Transport actor ---> proposes “pause and inspect”
                         |
                         v
Coarse perception ---> requested-port image crop, or “invalid”
                         |
                         v
Fine refiner + verifier ---> precise pose and safe handoff decision
                         |
                         v
Insertion actor
```

**Where history is used:** The transport actor has a longer observation history. The experimental requested-port coarse model used the current three camera views, views about **1.5, 3, and 6 seconds earlier**, measured TCP motion, and the requested card/port code. It predicted only that requested port's crop. This model is **not connected** to the refiner or deployed handoff. The older all-candidate coarse model itself is **single-frame**. History helped retain some later estimates after an identity was found, but it did not reliably establish the right identity in the first place.

**The gaps are at two different stages, and both matter.**

1. **Transport / when to inspect:** On three frozen autonomous SFP development starts, a corrective transport actor reached **0/3** requested-port buffered regions. That is a route problem *before* a camera handoff can succeed. The current SFP arrival decision has not been validated as a reliable trigger. A teacher frame centered on the target does not prove autonomous transport will put the target there.
2. **Coarse perception / which port is it:** Even on privileged-teacher views near the target, the requested-port history model often chose a neighboring card or port. Its best inspected stress replay had **173 wrong-port centers among 572 raw crop proposals**. A strict three-camera geometry check rejected the wrong centers in that inspected replay but left an early usable crop in only **3/10** episodes, versus the **9/10** target. Therefore it cannot safely authorize the fine refiner or handoff. Wider images helped show more cards, but in a ten-scene blind audit the exact numbered slot map was clear in only **4/10**, and **0/5** sparse-card scenes. Empty slots and clipped board ends are a major identity ambiguity.

The training score called “in frame” only means the simulator opening projects inside image bounds; it does **not** prove that the requested opening is visibly identifiable through occlusion. Thus neither that score nor agreement between cameras should be treated as a calibrated handoff confidence. The precise refiner has encouraging **SC clear-view** position results once given the correct crop, but it cannot repair a crop from the wrong port, and the complete SFP handoff has not passed its gate.

So the short answer is: **yes, we are improving coarse perception, and yes, we tested history; both requested-port identification and the preceding SFP transport/stop decision remain unresolved.** We have not enabled the transport-to-insertion switch. The next useful check is to inspect actual autonomous transport pauses: first ask whether the requested card is truly in the local camera neighborhood, then whether an observation-only coarse verifier selects that card and rejects its neighbors. If the route misses the neighborhood, repair transport first.

Evidence: [requested-port history experiment](../experiments/2026-10-04-requested-history-crop.md), [blind board-slot audit](../experiments/2026-10-04-sfp-board-slot-reference.md), [autonomous SFP transport check](../experiments/2026-10-03-sfp-balanced-arrival-onpolicy-check.md), and [handoff design](../../obsolete/docs/2026-10-19-retired-plans/SELECTED_PORT_HANDOFF_PLAN.md).
