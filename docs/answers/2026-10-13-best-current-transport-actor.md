# Which transport actor is best so far?

**There is no overall winner ready for deployment.** In the matched six-scene autonomous Gazebo comparison, the original 4/4 actor, 8/4 actor, and 12/6 actor each reached **0/6 safe approach buffers and 0/6 full insertions**.

**12/6 is the most promising SFP-to-NIC research checkpoint.** It came closer to the selected SFP port in all three matched scenes than the original 4/4 actor. Closest lateral offsets in the axial approach band were 19.07, 43.20, and 11.52 mm for 12/6, versus 30.65, 54.36, and 56.74 mm for 4/4. Its median across these three runs was 19.07 mm versus 54.36 mm. The narrowest SFP miss was still outside the current 10 mm buffer; none inserted.

**12/6 is not best for SC-to-SC on the tested scenes.** It reached the axial approach band in 0/3 SC runs. The original 4/4 actor reached that band in 1/3, although it was 214.45 mm laterally away, so neither had a useful SC arrival. The 8/4 actor had no meaningful live advantage: 0/3 approach-band visits for SC and SFP misses of 47.03, 57.35, and 50.93 mm.

Offline teacher-state action error favors 12/6, but that cannot override the failed autonomous route and handoff gate. The original 4/4 checkpoint was trained for 200 ms decisions, so its offline comparison on the 12/6 model's 300 ms rows also has a history-cadence mismatch; the live test used each actor's intended cadence.

**Practical choice:** keep the original 4/4 model as a control, use 12/6 as the SFP candidate for the next targeted route/handoff experiment, and do not promote 8/4 or declare a shared SFP/SC winner. SC needs a route-progress repair and new autonomous evidence. The 12/6 inference delay was allowed for this experiment but remains an implementation issue for eventual deployment. The small six-scene sample does not establish reliable generalization.

[Full horizon comparison](../experiments/2026-10-13-route-action-horizon.md) · [per-scene machine summary](../../outputs/experiments/2026-10-13_route_horizon_comparison_v1/summary.json).
