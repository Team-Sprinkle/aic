# Actual Gazebo recording of a MoveIt route

[Watch the five-camera Gazebo video](</home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/gazebo_case5_live_replan_v1/video/all_views_20fps.mp4>).

For a closer look at the cable, use the [overhead](</home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/gazebo_case5_live_replan_v1/video/overhead_20fps.mp4>)
or [side](</home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/gazebo_case5_live_replan_v1/video/side_20fps.mp4>)
video. These are **actual Gazebo camera recordings**, at 20 fps, of the
robot and cable moving. They replace the earlier offline path animation for
this purpose.

The frozen visual actor first moved for five seconds in a five-card SFP
scene. MoveIt then planned from that run's **measured** robot state toward
the requested card 0 / port 0, and Gazebo executed all 57 planned joint
waypoints. The measured TCP ended **3.76 mm** from the planned point 85 mm
in front of the port. The official scorer reported **no contact** and **no
insertion**, because this route intentionally stopped before insertion.

An initial attempt to reuse the earlier saved path aborted before moving
along it: the new run began the replay 23.96 mm from the saved path's start.
Replanning from the live measured state solved that mismatch. This is a
**privileged planner diagnostic**, not an autonomous actor success or proof
that the route is reliable across settings. The [experiment record](../experiments/2026-10-08-sfp-requested-port-corridor-audit.md#requested-gazebo-replay-actual-robot-and-cable-motion)
contains the command, force and tracking measurements, official score, and
artifact paths.
