# Older-route failure video and descent height

Yes. I rendered the matched older-route failure as a [five-view 20 fps video](../../outputs/experiments/2026-09-30_moveit_tilt_before_return/port1_old_tilt_matched_failure_video/all_views_20fps.mp4). At about **39 seconds** in the video, the plug is close to SC port 1 but force reaches **3.50 N** while the insertion command is at a **+6.48 mm Z offset**. The attempted retreat then fails to release force. The trace supports a near-opening blockage; the wide video does not resolve the exact contact surface, so I cannot say it hit a particular card edge. This is the second matched old-plan repeat; the first fully inserted.

I checked the saved MoveIt waypoints and scored plug-tip heights. I **did not reduce the low traverse height while moving the tilt**. Both the earlier low-cross port-1 route and the new tilt-before-return route use **TCP Z = 0.1800 m** for the behind-row shift, beside-row traverse, and beyond-row sweep. Both descend from a 0.2490 m clear-side waypoint to 0.1800 m, a planned **69 mm** drop. Their measured plug-tip positions relative to the highest card top were almost identical early in the route:

| Trial | At 10 s | At 20 s | First time below card top |
| --- | ---: | ---: | ---: |
| Earlier low-cross port-1 route | 21.7 mm below | 32.1 mm below | 5.81 s |
| New explicit-tilt route | 21.1 mm below | 31.1 mm below | 5.59 s |

The much older **high** `strict_return` route made a larger-looking downward move, from TCP Z 0.3380 to 0.2069 m, or about **131 mm**. It started **89 mm higher** and traversed **26.9 mm higher** than the current low route. The visible drop became shorter because the revised route began lower, not because it stopped higher. The new run also pauses after the sweep and tilt, which changes the video's timing.

There is one rise that may be what you noticed: **after** the low traverse and tilt, the return toward the port moves the TCP from Z 0.1800 to 0.2069 m, about **26.9 mm upward**, while moving horizontally toward the port. I retained that standoff because the prior insertion controller expects it; the later insertion motion descends. Despite the TCP rise, the measured plug tip stayed roughly **33–39 mm below** the highest card top around the tilt. This does not establish clearance for the entire flexible cable, but the inspected route progressed and the latest explicit-tilt teacher inserted in 2/2 repeats. I would not lower the route further based only on the apparent amount of motion in the video; a deeper variant should have a specific measured snag or clearance failure to repair.

The [experiment record](../experiments/2026-09-30-sc-tilt-before-return.md) now includes the height comparison, the failure-video link, and its hash. This answer is saved in [2026-09-30-older-route-failure-and-descent.md](2026-09-30-older-route-failure-and-descent.md).
