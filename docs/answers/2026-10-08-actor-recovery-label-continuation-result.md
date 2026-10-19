# Actor-state route-label continuation: result

I continued the SFP waypoint experiment with a bounded check at states the
transport actor actually visited. A one-shot CheatCode query could not supply
a route waypoint: its first command starts an interpolation from the current
pose, so it holds the robot. I therefore ran **three train-split, nonfinal
Gazebo episodes** with the frozen actor for five seconds, followed by the
privileged expert, and measured the TCP path after takeover.

The expert fully inserted on **two of three** episodes. Only **one of three**
met the preset waypoint-label rule: after 20 mm of TCP travel, the endpoint
had to be at least 10 mm from the takeover pose. One successful episode
traveled 20 mm but ended only 9.17 mm away; the failed insertion case ended
7.66 mm away. The pilot needed two of three episodes to both insert safely
and produce a valid label, so it **failed**. The official score, rather than
CheatCode's return value, determined insertion success.

I also checked a different label without collecting more data: take the
first point 10 mm from the takeover pose. All existing 16 training and six
held-out boundary episodes reached that distance within four seconds. Yet
only **11/16 training and 4/6 held-out** paths were direct enough under a
1.5× travel-to-displacement screen. A straight command to the endpoint
could skip the expert's detour. I did not train or deploy an actor from these
labels. The earlier independent autonomous result is still **0/3 safe
transport arrivals for both actors**; no final scenes were opened.

The next distinct repair is a route planner or recovery controller that can
propose a **collision-checked path from the actor's actual state**, then pass
an offline six-scene safety check before any new collection or actor fit. The
repo has MoveItPy collision-scene machinery and an SC-specific planner, but
no ready SFP actor-state planner. Rigid collision checks would still need a
later Gazebo cable/force test.

I ran that bounded MoveIt smoke test on three saved development and three
train actor states. MoveIt found rigid paths to **5/6** available future
expert states, with exact measured-start FK and 124–233 scene collision
objects. On **two of three development scenes**, however, the nearly straight
plan skipped a long detour the expert actually took. The planner received the
expert's future joint state and did not model the cable or choose the
requested port. **The safe-route-label gate remains failed.** No new actor
was trained and no Gazebo replay was started. The next repair needs an
SFP-specific planner that selects a complete requested-port corridor and
checks cable clearance before limited simulator execution.

See the [full experiment record](../experiments/2026-10-08-sfp-actor-recovery-label-feasibility.md),
including exact commands, scores and the [TCP path plot](</home/nas/chmin/aic_archives/2026-10-08_sfp_actor_recovery_label_pilot_v1/takeover_tcp_paths.png>).
The [MoveIt comparison](../experiments/2026-10-08-sfp-actor-state-moveit-feasibility.md)
includes its [measured-versus-planned path plot](</home/nas/chmin/aic_archives/2026-10-08_sfp_actor_state_moveit_six_v5_paths/measured_vs_moveit_paths.png>).
