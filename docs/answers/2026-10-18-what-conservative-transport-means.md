# What “more conservative transport” means here

**Partly, but it does not mean increasing every offset.** The SFP failure involved cable contact with the card between the starting plug and the requested card. The route should keep the *plug, gripper and trailing cable* on a safe side of that obstacle. More vertical clearance for the TCP alone can leave the cable draped across a card; a larger fixed detour can add time and still be wrong when board pose and card occupancy change.

I would test an observation-checked route: identify the selected card and intervening occupied rails, choose a corridor around the card row with margin for the cable, move to an intermediate checkpoint, and check force plus measured plug/TCP progress before proceeding. Keep the route within the robot's workspace and the task deadline. The selected target's identity comes from the task code; obstacle and target location used at runtime must come from permitted observations and published geometry bounds, never the released evaluation's exact scene coordinates.

The two failure classes need different repairs:

- **SFP trial2:** prevent the cable/card contact during transport. The immediate high-force stop happened before backtracking was called. A safe retreat after that guard is a separate control change and must stop if unloading increases force.
- **SC trial3:** the controller chose an alternate corridor after a **visual-acquisition timeout was wrongly classified as blockage**. Its return over322recorded waypoints took about45simulation seconds. The first fix is to keep visual-acquisition failure separate from physical obstruction; route retreat should require measured blockage. Simply moving the original route farther away will not fix the timeout.

After a route reaches the requested port consistently on varied new development layouts, train short contact-recovery episodes for rim misalignment. For card snags, retain route and cable history in each training example; the correct detour may be undecidable from a contact snapshot alone. Do not tune these changes to the now-exposed official `eval_config.yaml` trials.

[Official run and contacts](../experiments/2026-10-18-frozen-V56-official-evaluation.md).
