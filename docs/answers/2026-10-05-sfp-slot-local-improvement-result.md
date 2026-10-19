# SFP perception improvement result

I replaced the board model's whole-image occupied-slot head with a small head that reads features around **each of the five rail positions predicted from RGB**. The model still uses all three wrist cameras. Simulator board geometry and occupied-slot truth supplied training and scoring labels only; no simulator geometry or geometry-selected crop enters inference.

This made a large difference on the occupied/empty question:

| Complete reset groups | Old pooled head | New slot-local head |
| --- | ---: | ---: |
| Ten grouped development scenes | 5/10 exact five-slot maps | **10/10** |
| Ten further disjoint nonfinal scenes | 5/10 | **10/10** |
| Three saved sparse-card autonomous routes | 3/3 | **3/3** |

The new head made **zero confident false occupied-slot calls** on those grouped tests. It completed the planned 2,000-update, one-GPU run; the best checkpoint was update 1,200, selected by the fixed development-loss rule. Its development rail-position p90 was **25.7 px** against projected CAD labels, slightly better than the old 26.4 px. I checked original-size annotated examples; predictions generally fall near the projected rail grooves. Standalone three-camera JPEG decoding, resizing and model inference was **83.8 ms p95** over 200 repeats. This excludes the transport actor, precise port refiner, ROS and controller, so it is not a full deployment latency number.

I also tested whether these learned rails could verify the **requested port** from an existing frozen port detector. A direct image-coordinate rule failed because a card's port mouth and its rail anchor are at different depths, causing perspective offset. A causal 3D rule rejected wrong candidates, but obtained a trustworthy two-rail reference in only **2/10** scenes and an early correct candidate in only **1/10**. Most frames failed its multi-camera reprojection limit. **The model is now better at telling which card slots are occupied; it does not yet safely identify the requested port or decide when transport should stop.** Automatic handoff, insertion and RL remain disabled; final scenes were not opened.

The next targeted repair is multi-view board-reference consistency and persistent requested-slot/port correspondence through changing or occluded views, followed by new nonfinal autonomous starts with zero wrong accepts and useful early coverage. The [experiment record](../experiments/2026-10-05-sfp-slot-local-occupancy-ablation.md) includes exact commands, checkpoints, per-scene results, annotated images and the failed downstream checks. The [machine summary](../experiments/2026-10-05-sfp-slot-local-occupancy-ablation.summary.json), [status](../STATUS.md), [experiment ledger](../EXPERIMENTS.md) and [handoff plan](../../obsolete/docs/2026-10-19-retired-plans/SELECTED_PORT_HANDOFF_PLAN.md) are updated.
