# Short route-waypoint experiment: result

I tested the proposed SFP transport redesign on saved development data. The
model predicts a short TCP-frame route waypoint from frozen camera features,
robot state, history and the requested task code. A capped controller would
move toward that waypoint, so the prediction directly determines the route.

It learned ordinary expert motion reasonably well: the boundary-weighted
model had **13.51 mm p95** waypoint error across 41 held-out teacher episodes.
But at the states where the existing actor had drifted and the expert took
over, its proposed direction matched the expert's subsequent motion on
**0 of 6** episode groups. A target based on the next 20 mm of measured path
also failed, with **27.37 mm p95** error on its four eligible development
boundaries. Those samples are too few for a reliability claim, but they are
enough to reject this controller at the present safety gate.

I therefore did **not** run it in Gazebo or promote it. Autonomous transport
is still at the previous independent **0/3 safe arrivals for both actors**;
this work has not improved full insertion. I preserved both trained variants,
the failed comparison, the six direction vectors, exact commands and logs.
The final scenes remain sealed.

The likely gap is labels at states the actor actually visits. Some expert
takeovers barely moved during their first second, so a fixed one-second
future-position label can describe controller settling rather than the
intended recovery route. The next bounded test should query or run the expert
from actor-visited states, verify its proposed direction with short nonfinal
takeovers, and require at least 20 distinct training reset states plus six
held-out episode groups. At least five of those six held-out directions must
agree with safe measured recovery before another actor fit. If that fails,
revisit explicit route planning or a safer recovery controller rather than
collecting more routine teacher routes.

Full methods, metrics and artifact paths: [experiment record](../experiments/2026-10-08-sfp-short-route-waypoint-feasibility.md).
