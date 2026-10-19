# Transport misses and the visual handoff tolerance

The October 13 matched test used three SFP-to-NIC and three SC-to-SC nonfinal scenes per actor. The distance below is the **smallest lateral offset from the requested opening while the plug was in the specified axial approach band**. It is scored afterward using Gazebo geometry; the autonomous actor did not receive that geometry. A smaller value is better. These are per-episode best points, not final positions, continuous residence in the band, or insertion success.

| Actor | SFP best lateral misses, mm | SFP p50, mm | SFP p90, mm | SC approach-band visits |
| --- | --- | ---: | ---: | ---: |
| Original 4 predicted / 4 executed | 30.65, 54.36, 56.74 | 54.36 | 56.26 | 1/3; that visit was 214.45 mm lateral |
| 8 predicted / 4 executed | 47.03, 57.35, 50.93 | 50.93 | 56.06 | 0/3 |
| 12 predicted / 6 executed | 19.07, 43.20, 11.52 | **19.07** | **38.38** | 0/3 |

The p90 uses linear interpolation among only **three** SFP episodes. It is descriptive, not a statistically reliable tail estimate. For SC, the 8/4 and 12/6 models never reached the axial band on these scenes. Their lateral miss *in that band* is undefined, so a p50 or p90 would hide the dominant failure: the route did not arrive. All three models reached zero of six safe buffers and zero full insertions. [Experiment and exact scene summaries](../experiments/2026-10-13-route-action-horizon.md) · [machine summary](../../outputs/experiments/2026-10-13_route_horizon_comparison_v1/summary.json).

## What can be handed to the insertion actor?

There are two distinct distances:

1. **Distance at which coarse vision can recognize the requested port and propose inspection.** Existing selected-port models have sometimes found it earlier than the transport buffer. For example, a native SC crop diagnostic worked on clear views roughly 48–176 mm from the opening, and a later SFP selected-port crop test gave 18/18 early anchors on grouped 3–5-card stress scenes. These results show possible *detection*, not a validated safe insertion start. The model may abstain under occlusion or accept the wrong neighboring port; identity and visibility must be verified.
2. **Pose from which the current insertion actor can safely take over.** The present development transport gate is an axial standoff about **70–100 mm before the opening**, lateral offset **at most 10 mm**, and roughly **3°** orientation error, with a safe route and acceptable force. The opening pose must come from observation-only perception at runtime, and the true geometry is used only to score the gate afterward. This is already an engineering acceptance envelope, **not a demonstrated capture range** of the insertion actor. Earlier local-actor tests failed even from much cleaner starts, for example about 36 mm before the opening and 1.45 mm lateral with accurate RGB pose. Thus even 10 mm transport tolerance cannot presently be claimed sufficient for full insertion.

Consequently, the **12/6 median SFP miss of 19 mm** is enough to justify a *pause and visual inspection* if the route and view are safe, but not an automatic switch to insertion. Its approximately **38 mm p90** and the 43 mm worst observed miss are far outside the present 10 mm handoff gate. A coarse estimator could tell a future controller how far and in which direction to correct, provided it identifies the right port; it cannot itself eliminate that displacement. The current insertion actor has not been shown to recover it. SC needs a route repair before a visual handoff tolerance is meaningful, because 12/6 did not enter the approach band in any tested SC scene.

The practical next measurement is a **capture-range experiment**: start the frozen local controller from checked, collision-free standoffs with lateral errors stepped through, for example, 2, 5, 10, 20, and 40 mm, with both correct and wrong-port visual cases. Score autonomous insertion and force/contact by family, orientation, and distance. Keep the coarse inspection decision separate from the final insertion switch. Until such a test passes, do not raise the 10 mm switch threshold to accommodate the transport actor's observed miss distribution.
