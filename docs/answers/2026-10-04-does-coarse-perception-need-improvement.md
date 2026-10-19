# Does coarse perception need to improve for handoff?

**Yes, especially for SFP-to-NIC, but it is one of several handoff requirements.** A useful coarse check should answer: “Am I near the **requested numbered card and port**, with enough evidence to pause for a precise inspection?” It needs an approximate target location and a reliable ability to reject a wrong or unidentifiable target. It does **not** need submillimeter precision; the later original-image refinement handles that.

Our current SFP evidence shows the coarse identity problem: the conservative selected-port model found an early anchor in only **3/10** reused development episodes. Combining four heads increased early coverage to **7/10** but produced **46 accepted wrong-model predictions**. Carrying a conservative estimate through history added later accepted frames but found no additional episodes. Thus the current coarse check cannot safely authorize or reliably trigger handoff.

Improving it will not, by itself, finish the handoff. **Transport** must actually bring the robot to a safe view of the requested destination; recent autonomous SFP transport tests missed that neighborhood. After a coarse pause, the **final verifier** must confirm the requested port's precise pose, orientation, visibility, tool posture, and force. A coarse “yes” means *pause and inspect*, never *start insertion*.

The highest-value coarse repair is to use the **whole board view** to identify numbered slots, including empty ones, while retaining a local crop for the requested port. Keep a task-specific history only after a trustworthy visual identification, and return “unknown” when the image does not resolve the slot. Test it on complete new development episodes, including wrong-card and wrong-port stops and actor-generated views. Keep final scenes sealed.

There is no fundamental requirement for two separate neural networks. If a precise task-selected model can run often enough within the **300 ms complete inference budget**, it could also provide the coarse pause signal. The important distinction is **two decisions with different evidence thresholds**: whether to hold for inspection, and whether the verified pose is safe enough for insertion.

Related: [handoff design](../../obsolete/docs/2026-10-19-retired-plans/SELECTED_PORT_HANDOFF_PLAN.md), [SFP fresh identity test](../experiments/2026-10-04-sfp-fresh-anchor-shadow.md), [context/history audit](../experiments/2026-10-04-sfp-context-sufficiency-audit.md).
