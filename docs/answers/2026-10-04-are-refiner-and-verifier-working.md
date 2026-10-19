# Are the refiner and final camera check working?

**Not reliably enough for automatic handoff.** We have promising *component* results, especially SC position in clear views, but the complete transport → camera check → insertion chain has not passed its safety and coverage gates.

| Part | What works | What is still wrong or unproven |
| --- | --- | --- |
| **Original-image crop refiner** | When it is given a correct, clear SC crop, three held-out development resets gave near-port lateral position p95 errors of about **0.15–0.23 mm**. On the SFP ten-scene camera replay, the requested opening lay inside all three proposed crops in **335/346** proposals. | On that SFP replay, the refiner still **accepted the other port on the same card 11 times**. A crop containing the correct opening does not guarantee that the model selects it. SC clear-view results do not prove accuracy through cable/gripper occlusion, on new transport-produced views, or for every orientation. The refiner mainly supplies image keypoints/position; orientation has separate estimation and checks. |
| **Final camera check / verifier** | It can combine three camera predictions into an estimated 3D opening and reject some disagreements, bad posture, or force. A conservative selected-only plus history diagnostic had **zero wrong accepts on ten reused SFP teacher episodes**. | That SFP diagnostic found an early identity in only **3/10** episodes. Other SFP heads can agree on a neighboring port. An SC visual stop once looked excellent in position (**1.02 mm**) but was **6.71°** misoriented after a wrong route. Hard occlusion, wrong-port requests, and complete live camera-to-first-insertion-command latency have not passed independent grouped tests. The automatic switch remains off. |

So the work is not just “fix transport and coarse perception.” We also need to:

1. **Repair SFP local port identity.** Predict or check both ports of the requested card together so the refiner cannot silently swap port 0 and port 1. The paired-port experiment is drafted, not yet trained.
2. **Certify visibility and orientation.** “Projects inside the image” is not the same as “the requested opening is identifiable.” The full verifier must reject uncertain/occluded views and unsafe tool orientation or contact.
3. **Test the composed handoff on actor-generated views.** Good teacher-camera results may fail after transport deviates. Measure correct pause, valid/invalid decision, wrong-port acceptance, full live p95 below **300 ms**, and then insertion from actual accepted handoffs. Keep reserved final scenes sealed.

In short: **SC clear-view position refinement is promising; SFP port identity and the complete verifier are not solved.** The refiner and verifier should remain separate concepts: the refiner estimates, while the verifier decides whether those estimates and the robot state are trustworthy enough to hand over.

Sources: [SC clear-view position and limits](../experiments/2026-10-02-selected-port-handoff-execution.md), [SFP fresh identity failure](../experiments/2026-10-04-sfp-fresh-anchor-shadow.md), [SFP context/history audit](../experiments/2026-10-04-sfp-context-sufficiency-audit.md), [SC stopping comparison](../experiments/2026-10-02-transport-stop-comparison.md), [handoff plan](../SELECTED_PORT_HANDOFF_PLAN.md).
