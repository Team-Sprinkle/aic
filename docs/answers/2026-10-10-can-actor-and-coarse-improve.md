# Can the actor and coarse model improve further?

**Yes. I do not think either has reached an inherent performance ceiling.** I think *repeating the same training recipe on more ordinary demonstrations* is unlikely to make a large difference. That is especially true for the actor.

## Transport actor: substantial room to improve

The actor is still the main bottleneck. Its supervised action error improved in several experiments, but independent autonomous transport still failed the safe-arrival gate (0/3 for both arms in the matched SFP actor-prefix test; four SC BC variants gave 0/16 stable 10 mm handoffs). This gap means a low error on teacher frames does not tell us the actor will recover when its own earlier command puts it on a different route.

The actor has already trained on substantial verified data, including an SFP experiment with 445 training episodes and SC training with 83 cleaned successful routes. That rules out the simple explanation that we tried only a handful of episodes. It does **not** rule out a much better actor. The current dataset has relatively few *safe, useful corrections from actor-visited wrong-route states*. The task request also had a weak effect on some actor outputs, and merely appending the estimated port offset did not make the actor use it. Those are training-target and control-design problems, not proof that vision or a temporal policy lacks capacity.

The promising next change is to obtain **measured, cable-safe route targets starting from actual actor states**, label a meaningful stage or progress target after controller settling, and train a full-action policy against that target. Compare it with the existing actor on new reset-grouped starts. If no sufficiently safe corrective labels can be made, stop this supervised route branch and reconsider the planner/state representation. Simply adding epochs or more ordinary successful teacher routes is not the same experiment.

## Coarse perception: good progress, but only in tested conditions

The current SFP board-to-requested-crop pipeline already improved early requested-port anchors from 15/18 to 18/18 on its fixed dense development stress set, with zero accepted wrong ports there. SC's spatial requested-port head also greatly improved an absent-port stress test. So the coarse stage **has improved clearly** and, when it receives a suitable view, is often good enough to locate a crop for the precise refiner. This does not mean it is universally reliable: occlusion, one-port SC scenes, confusing neighboring ports and off-route camera views remain incompletely tested. The current “in frame” training label is a geometric projection, not a checked statement that the port is visually identifiable.

I would preserve these weights and test them on new, complete reset groups and real actor-arrival views before another broad fit. If failures cluster in occluded or wrong-request views, the useful improvement is a targeted identity/visibility dataset with reviewed labels, possibly a safer view or a better spatial/temporal check. More projected labels of the same kind may leave the same ambiguity. The SFP 91-scene native-crop refit was already a meaningful scale-up; the SC one-port addition was much smaller and has a stronger case for targeted expansion.

## Relative priority

**Actor: high potential improvement, but needs a different source of corrective supervision or a better route objective. Coarse: already strong on selected clear-view tests; likely incremental gains there, with potentially important gains on hard identity and occlusion cases.** Improving the coarse model alone will not fix the robot's current failure to reach the requested-port corridor. Conversely, once the actor arrives consistently, coarse mistakes will matter much more and must be tested before automatic handoff.

No current result establishes reliable full-start autonomous insertion. The [model wrap-up](2026-10-09-actor-coarse-refiner-wrap-up.md), [latest route-label gate](../experiments/2026-10-09-sfp-cable-side-route-repair.md), and [handoff plan](../SELECTED_PORT_HANDOFF_PLAN.md) contain the supporting counts and stopping rules.
