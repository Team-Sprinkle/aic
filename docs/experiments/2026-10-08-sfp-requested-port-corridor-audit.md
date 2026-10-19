# SFP requested-port corridor from actor takeover states

## Decision

**Scope:** This was an SFP actor-state *route-label feasibility* check,
carried forward from the failed SFP autonomous transport branch. It was
not intended to reproduce or solve the multi-card SC-to-SC cable snag.
Normal privileged SFP teachers generally traversed this region without a
demonstrated cable snag; the later one-scene Gazebo replay below adds no SC
evidence. The SC MoveIt/lateral-bypass experiments are recorded separately
in [the SC pilot](2026-09-29-moveit-sc-route-pilot.md) and
[SC recovery plan](../SC_CABLE_SNAG_RECOVERY_PLAN.md).

**The arm-planning hypothesis works, but the safe route-label gate remains
open.** From six previously saved, nonfinal, five-second SFP actor takeovers,
MoveIt planned three rigid collision-checked variants to the *requested*
card and port on **6/6** distinct reset groups. Each target was a TCP pose
85 mm axially outside that port opening. The start FK agreed with the
recorded TCP to numerical precision. Unlike the preceding smoke test, the
goal was derived from the requested port, **not a future expert joint state**.

The arm success does not establish a safe cable route. An offline screen
carried the measured takeover cable shape rigidly along each planned TCP
path and computed a conservative lower bound on clearance to the exact
expanded Gazebo NIC collision solids. The best route per group stayed above
zero for all 20 cable links on only **3/6** groups; the proximal five links
stayed above zero on **4/6**. Retreat before alignment improved some proximal
paths, but case 2 still entered the NIC-bound screen, and case 3 began with
a negative bound. The required ≥5/6 *safe route* gate did not pass. There
was no Gazebo execution in the offline selection phase. A later **single-scene
diagnostic replay**, requested by the user, is documented below; it did not
change that gate. No new actor training or autonomous insertion occurred.
Final scenes remain sealed.

The cable screen is intentionally a rejection/ranking proxy: holding a
flexible cable rigidly to the TCP overpredicts where its free links travel;
its sampled cylinder bound also subtracts cable radius and half the sample
pitch. The measured teacher cable later crossed the same bound in **all six**
saved runs, five of which scored full insertion (tier 3 = 75). Thus a
negative lower bound is **not proof of actual contact or snag**. Conversely,
a positive lower bound at sampled poses does not prove dynamic cable safety.
The one teacher partial insertion was case 4, whose three planned routes all
had >44 mm proxy clearance; this also shows that this cable screen cannot
predict insertion success by itself.

## Exact comparison

The first three groups are episode-grouped development scenes with two,
three, and five cards. The last three are new train-split takeovers with
two, three, and five cards. All six are reused scored bags; no reset was
opened here. For each, the task selected one card and one of its two ports.
The three routes used the measured takeover joints and exact randomized
Gazebo collision objects:

1. Direct to the buffered requested-port TCP target.
2. Align laterally farther from the opening, then approach the buffer.
3. Retreat about 40 mm, align 220 mm from the opening, then approach.

All three preserve the measured TCP orientation and end at the same
position target. They are **transport candidates**, not insertion-ready
orientations or complete insertions. The port geometry is a privileged
**offline planning label** from the randomized scene and a previously
validated SFP opening calibration. It was not passed to the visual actor.

| Saved case | Requested card/port | Best proximal bound | Best all-link bound | Comment |
| --- | --- | ---: | ---: | --- |
| Development 0 | 3 / 1 | +8.6 mm | −5.0 mm | Retreat protects proximal links; distal rigid-carry path overlaps a NIC bound. |
| Development 1 | 2 / 0 | +50.5 mm | +50.5 mm | All three candidates clear the static screen. |
| Development 2 | 3 / 1 | −5.0 mm | −5.0 mm | All three variants fail the screen; distal cable already has −2.6 mm at takeover. |
| Train 3 | 4 / 1 | −3.2 mm | −3.2 mm | Begins below the screen; retreat avoids making proximal clearance worse. |
| Train 4 | 0 / 0 | +46.3 mm | +46.3 mm | Clear static screen; teacher only partially inserted. |
| Train 5 | 0 / 0 | +45.7 mm | +45.7 mm | Clear static screen. |

These are **lower-bound proxy values**, not measured penetration depths.
The [inspected comparison plot](</home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/cable_proxy_comparison.png>)
shows all route variants. The case-specific traces and responsible cable
link / NIC collision solid are saved in `cable_proxy.json`.

## Video of a saved planned path

The [case 2 retreat/align/approach video](</home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/case2_retreat_offline_h264.mp4>)
animates the complete saved MoveIt TCP trajectory at 20 fps. This is the
five-card development scene targeting card 3, port 1. The left camera view
is a **frozen frame at actor takeover**; the moving dot in the two plots is
the *planned* TCP position relative to the requested opening. Total planned
TCP travel is 206.35 mm, ending 85 mm axially before the opening. It is
**not a Gazebo execution or a camera recording of moving hardware/cable**.
The video is H.264/yuv420p for ordinary Ubuntu playback. The renderer is
`scripts/render_sfp_moveit_corridor_video.py`; the raw intermediate and an
inspected midpoint PNG are retained beside the H.264 file. To regenerate
under new filenames:

```bash
.pixi/envs/default/bin/python scripts/render_sfp_moveit_corridor_video.py \
  --plan /home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/case2.json \
  --variant retreat_align_approach --fps 20 --output /tmp/case2_moveit_raw.mp4
ffmpeg -i /tmp/case2_moveit_raw.mp4 -c:v libx264 -pix_fmt yuv420p \
  -movflags +faststart -crf 22 /tmp/case2_moveit_h264.mp4
```

## Requested Gazebo replay: actual robot and cable motion

After the offline audit, the user requested a **real Gazebo video**. One
previously used, nonfinal five-card SFP scene (case 5, card 0 / port 0) was
replayed with the frozen visual transport actor for five seconds. The first
attempt to execute the *saved* plan aborted before any MoveIt command: the
new run's takeover pose was 23.96 mm / 0.070 rad from the saved plan start.
The actor code and checkpoint hashes matched; the simulated prefix differed
across runs. The guard worked, and that attempt's tier 3 score was zero.

For the second run, MoveIt planned the same retreat/align/approach corridor
**from the live measured Gazebo takeover state**, using the exact randomized
scene and requested-port geometry. This was privileged diagnostic planning,
not actor inference. The planner applied 233 collision objects and produced
a 188.46 mm TCP path with 57 commanded waypoints. The Gazebo robot executed
all 57, ending **3.76 mm from the planned 85 mm buffered TCP goal**. Its
measured port-frame TCP position was approximately
`[+3.75, +0.04, −84.71] mm`; that is a transport approach point, not an
insertion pose. The official scorer reported **no contact detected** and
**tier 3 = 0**, as expected because no insertion was attempted. Relative
force change peaked at 1.05 N. Intermediate waypoint tracking lag was
substantial (27.66 mm p95, 28.92 mm max), so smooth target tracking is not
established even though the final goal was reached closely.

The [actual five-camera Gazebo recording](</home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/gazebo_case5_live_replan_v1/video/all_views_20fps.mp4>)
shows the robot and cable moving at 20 fps; separate
[overhead](</home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/gazebo_case5_live_replan_v1/video/overhead_20fps.mp4>)
and [side](</home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/gazebo_case5_live_replan_v1/video/side_20fps.mp4>)
views are also saved. I inspected frames at 6 s and 17 s and verified the
videos are H.264, 20 fps, 18.9 s long. Exact commands, scene, scoring bag,
camera frames, both planner and policy logs, waypoint-by-waypoint measured
TCP/force, and [machine summary](</home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/gazebo_case5_live_replan_v1/gazebo_replay_summary.json>)
are in `.../gazebo_case5_live_replan_v1/`. The aborted first attempt is
preserved separately in `.../gazebo_case5_retreat_v1/`.

```bash
bash scripts/run_sfp_moveit_gazebo_recording.sh \
  /home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/case5.json \
  /home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/gazebo_case5_live_replan_v1 \
  3
.pixi/envs/default/bin/python artifacts/prod_cheatcode_audit/render_smooth_wide.py \
  /home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/gazebo_case5_live_replan_v1 \
  /home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/gazebo_case5_live_replan_v1/results/bag_trial_000288_20261008_153012_918 \
  /home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/gazebo_case5_live_replan_v1/video
```

Use new output names on rerun. The replay used one rootless container and
GPU 3; the older Isaac job on GPU 0 was not interrupted. **This one
privileged transport success does not satisfy the six-scene safe-route gate,
an autonomous transport gate, or insertion.**

## Reproduction and artifacts

The archive is
`/home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/`.
It holds six complete joint/TCP path JSON files, commands and MoveIt logs,
`summary.json`, `cable_proxy.json`, `measured_teacher_cable_clearance.json`,
`decision.json`, and `cable_proxy_comparison.png`. Earlier v1/v2 corridor
archives are retained; v3 added planned TCP quaternions needed for the cable
screen. Source scripts:
`scripts/plan_sfp_actor_state_corridor.py`,
`scripts/run_sfp_requested_corridor_six.sh`, and
`scripts/audit_sfp_requested_corridor_cable.py`. The measured-teacher
control is `scripts/audit_sfp_measured_teacher_cable.py`.

```bash
bash scripts/run_sfp_requested_corridor_six.sh \
  /home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat

source .pixi/envs/default/setup.bash
PYTHONPATH=/var/tmp/chmin_aic_mcap_reader_py312:$PWD/aic_teacher_official:$PWD/scripts:${PYTHONPATH:-} \
  .pixi/envs/default/bin/python scripts/audit_sfp_requested_corridor_cable.py \
  --plans /home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat \
  --output /home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/cable_proxy.json

PYTHONPATH=/var/tmp/chmin_aic_mcap_reader_py312:$PWD/aic_teacher_official:$PWD/scripts:${PYTHONPATH:-} \
  .pixi/envs/default/bin/python scripts/audit_sfp_measured_teacher_cable.py \
  --plans /home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat \
  --output /home/nas/chmin/aic_archives/2026-10-08_sfp_requested_corridor_v3_with_quat/measured_teacher_cable_clearance.json
```

The outputs are immutable: choose new directories/files to rerun. The pure
Python MCAP reader is installed at the listed temporary path on this host;
the script also needs the repository's ROS message environment. No GPU or
Docker container was used for offline planning or screening.

## Next bounded repair

Do not fit a route actor from these paths or call any route safe. The next
distinct hypothesis is a **cable-aware corridor**: use the measured cable
shape and task scene only as offline planning labels to choose an outward
detour, and evaluate proximal and full-cable clearance *relative to the
measured start* with an explicit contact budget. First verify that the
screen distinguishes safe and unsafe **measured** routes on the existing
bags; the present rigid-carry screen does not. Budget six reused nonfinal
states and at most one revised set of corridor candidates, with no new
training. Stop if fewer than five groups have an explainable, checked
candidate. Only then replay at most three matched actor-prefix starts in
Gazebo, with measured cable motion, contact and force. Preserve the 0/3
independent autonomous transport result and keep automatic handoff,
insertion and RL gated until an actual route and policy pass their own gates.
