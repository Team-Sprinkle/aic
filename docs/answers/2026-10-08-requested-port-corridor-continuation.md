# Requested-port SFP corridor continuation

I built and tested an SFP route planner on six saved, nonfinal states reached
after five seconds of autonomous transport. It uses each scene's requested
card and port as an **offline planning goal** and plans three alternatives:
direct travel, lateral alignment before approach, and retreat before
alignment. All **6/6** states produced rigid MoveIt paths to a TCP position
85 mm outside the requested opening. This removes the previous dependence
on a future expert joint state as the planning goal.

I then checked whether the measured cable shape at takeover could travel
with each planned path without entering the NIC collision solids. This is a
conservative screening calculation, not a physical cable simulation. The
best route cleared all 20 cable links on only **3/6** states; the five links
nearest the robot cleared on **4/6**. I also checked the cable in the saved
teacher runs. It crossed the same conservative bound in all six, including
five runs that fully inserted. The screen therefore cannot turn these paths
into reliable safe-route labels. The previously set **at least 5/6 safe
routes** gate remains open.

I did not replay these candidates in Gazebo, train another actor, or open
reserved final scenes. The independent autonomous transport result is still
**0/3 safe arrivals**; no improved insertion rate is claimed. Planning and
auditing used no GPU or new Docker container.

The [experiment record](../experiments/2026-10-08-sfp-requested-port-corridor-audit.md)
has the exact commands, per-scene metrics, limitations and next bounded
repair. The [inspected comparison plot](</home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/cable_proxy_comparison.png>)
shows all three route variants. The exact paths, measured cable traces and
machine-readable decision are under
`/home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/`.

Next, use the existing bags to test a cable-aware outward detour and a
contact screen calibrated against measured safe and unsafe motion. Limit it
to these six nonfinal states and one revised candidate set. Continue to at
most three matched Gazebo replays only if at least five routes have an
explainable safety case; otherwise revisit the transport control design.
