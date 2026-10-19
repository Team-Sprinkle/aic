# How good is the coarse camera check?

**It is not reliable enough to decide an automatic handoff.** There is currently no single deployed coarse model that both finds the requested port and safely says “pause now.” We tested those abilities in separate offline or shadow components.

The coarse **port-finding** component did well on the three recent one/two-card autonomous routes: two frozen RGB locators each found the requested port early in **3/3**, with no accepted wrong port on those routes. But on the harder ten-request replay, the selected-port locator found an early anchor in only **3/10** episodes. It found none for the six requests for cards 1–3. This is mainly a coverage and physical-slot identity problem; the easy 3/3 result does not establish a general camera check.

The experimental **proposal verifier** (the part most like a camera check) was trained to accept or reject a proposed target crop. With a threshold chosen on ten grouped development episodes, it accepted **63/136** proposals and accepted **zero wrong** ones there. When the same frozen threshold was applied to ten previously inspected harder scenes, it accepted **248/581** proposals, including **28 wrong accepted**. That is too many wrong-port approvals. It did **not** learn true visual visibility or whether the robot was at a safe stopping pose. Its scores are therefore not permission to insert.

The separate learned **transport stop** head was timely on only **2/10** held-out SFP teacher episodes at its strict three-decision rule and missed eight. Lowering the threshold created early stops. It is disabled. A broad base-coordinate region can veto far-away stop proposals but cannot identify the port or certify safe alignment.

For **SC-to-SC**, a narrower spatial identity test was more promising: an all-three-camera check rejected 109/109 counterfactual requests for a physically absent port while accepting 96/100 real requests projected into all views on four new development reset groups. That is a provisional identity subtest, not a validated stop or full handoff. Hard occlusion, wrong posture, valid-arrival recall and complete live latency are still open.

In plain words: the cameras can often recognize a nearby socket, especially in simple scenes, but the current check can still approve a neighboring socket in harder SFP scenes or miss the correct one. It also cannot yet reliably tell **when** the robot should stop. That is why the coarse check is not connected to automatic insertion.

Evidence: [coarse experiment](../experiments/2026-10-04-coarse-shared-history-execution.md), [spatial replay](../experiments/2026-10-05-spatial-transport-prerequisite.md), [stop-head comparison](../experiments/2026-10-05-transport-route-handoff-continuation.md), and [SC identity test](../experiments/2026-10-02-sc-absent-port-identity.md).
