# SC lateral bypass and visibility-gated policy design

Date: September 25, 2026. These are **privileged Gazebo teacher/route tests**,
not autonomous actor successes, production scores, or BC-admitted episodes.
The reserved final scenes remained sealed. No policy training or RL was run.

## Why the old teacher route is inadequate for crowded SC scenes

The prior SC collector either followed stock CheatCode's roughly straight
approach or inserted an 8 cm pre-route **vertical lift**. Full insertion alone
does not show that the cable took a desirable path. On a five-card scene,
historical across-card videos show the cable draping over the card stack;
an older outside-left diagnostic was sometimes better. The [previous
route probe](2026-09-23-fixed-five-card-route-probe.md) and
[historical bypass template](../expert_matrix_template_fixes.md#sc-to-sc-nic-bypass-full-insertion-template)
motivated this targeted repair. Existing over-card SC clips should not be
treated as route-quality demonstrations merely because a teacher inserted.

The new route uses simulator card and port TF **only inside the teacher
collector**. It moves away from the cards on the starting side if needed,
goes behind the row, shifts laterally to an outside-left lane, descends
there, travels alongside the row, sweeps toward the selected port only
after passing the last card, and then lets the precise CheatCode insertion
controller finish. The waypoint stage is captured in the same synchronized
three-camera image/teacher-target log as insertion. At autonomous test time,
the actor has no card or port TF. The new route is opt-in through
`--corrective-sc-bypass-side left`; stock collection stays the default.

```text
start on one side of cards
  -> move farther away on that side if clearance is small
  -> get behind the first card
  -> cross behind the row to an outside lane
  -> move forward BESIDE the row
  -> sweep toward the port AFTER the last card
  -> align and insert
```

## Bounded results and recordings

| Run | Official result | Route and cable evidence | Recording |
| --- | --- | --- | --- |
| Five cards, SC port 0, first pilot | No attempt; teacher safety gate rejected a 34 mm starting x gap before executing the route | Retained as a failed route-plan attempt, not expert data | [Pilot score](../../outputs/experiments/2026-09-25_sc_lateral_bypass/pilot_port1/scoring.yaml) |
| Five cards, SC port 1, 85 mm outside-left lane | **Full insertion, Tier 3 = 75**; no scored off-limit contact or force penalty | Measured TCP x stayed 78 mm outside the nearest card-center x during the row crossing. Sampled cable-segment center came within **2.11 mm** of a main PCB box, too close to certify as a clean cable bypass. Do not admit for BC yet. | [Three wrist views](../../outputs/experiments/2026-09-25_sc_lateral_bypass/pilot_fivecard_port1_wrist_3view.mp4), [cable plot](../../outputs/experiments/2026-09-25_sc_lateral_bypass/pilot_fivecard_port1_cable_geometry.png) |
| Same five-card port-0 scene, adaptive starting-side egress | **Full insertion, Tier 3 = 75**; no scored off-limit contact or force penalty | Starting TCP first moved farther right, then crossed behind the row to the left lane. During row crossing, measured TCP x stayed about 82 mm outside the nearest card-center x. Minimum sampled cable-center/main-PCB gap **11.50 mm**. | [Three wrist views](../../outputs/experiments/2026-09-25_sc_lateral_bypass/pilot_fivecard_port0_egress_wrist_3view.mp4), [cable plot](../../outputs/experiments/2026-09-25_sc_lateral_bypass/pilot_port0_egress/cable_overhead_side_force.png) |
| Fixed five-card seed 51500, behind-first outside-left diagnostic | **Full insertion, Tier 3 = 75**; no scored off-limit contact or force penalty | Static overhead and side cameras visibly show the cable loop routing beside the cards. Sampled cable-center/main-PCB gap **13.81 mm** minimum. | [All five views at synchronized 20 Hz](../../outputs/experiments/2026-09-25_sc_lateral_bypass/wide_51500/all_views_20fps.mp4), [overhead](../../outputs/experiments/2026-09-25_sc_lateral_bypass/wide_51500/overhead_20fps.mp4), [side](../../outputs/experiments/2026-09-25_sc_lateral_bypass/wide_51500/side_20fps.mp4), [cable plot](../../outputs/experiments/2026-09-25_sc_lateral_bypass/wide_51500/cable_overhead_side_force.png) |
| Same five-card port-1 scene, lane widened 85→110 mm | **No insertion, Tier 3 = 22.02**, final plug-port distance 40 mm; no scored penalty, but force reached 41.32 N for 0.88 s | The minimum sampled cable-center/main-PCB gap improved only **2.11→3.25 mm** and remained below the 5 mm route-quality gate. The longer route did not preserve insertion; stop this simple lane-margin sweep. | [Failed three-view video](../../outputs/experiments/2026-09-25_sc_lateral_bypass/pilot_fivecard_port1_110mm_lane_failed_wrist_3view.mp4), [cable plot](../../outputs/experiments/2026-09-25_sc_lateral_bypass/pilot_port1_110mm_lane/cable_overhead_side_force.png) |

The full-scene video has **64.55 seconds** of actual simulation-time coverage
at 20 fps and H.264/yuv420p encoding; its fixed wide cameras are diagnostic
views only. The collector wrist videos use frame-order 20 fps playback of
recorded 288×256 images; simulation timestamps contain gaps, so those
playback durations are compressed. We visually inspected the overhead and
side contact sheets and sampled both collector videos. The cable-gap metric
measures link centers to simplified main PCB boxes, omitting cable radius,
other card features, and named physics contacts. It supports the lateral
route interpretation but is not proof of zero contact with every surface.

The collector commands are saved as `run.json` under the linked
`outputs/experiments/2026-09-25_sc_lateral_bypass/pilot_*` folders and in the
local `/var/tmp/chmin_aic_20260918_act/sc_fivecard_lateral_bypass*` run roots.
The wide run used `AIC_AUDIT_SMOOTH_CAPTURE=1` with
`scripts/run_recovery_gazebo_diagnostic.sh` and the
`outside_left_behind_first` variant. Its camera renderer is
`artifacts/prod_cheatcode_audit/render_smooth_wide.py`. The scored-bag
geometry extractor is `artifacts/prod_cheatcode_audit/analyze_cable_route_bag.py`.
The [pilot bags](2026-09-25-sc-fivecard-lateral-bypass-pilot-mcap-archive.json),
[port-0 bag](2026-09-25-sc-fivecard-lateral-bypass-port0-mcap-archive.json),
[wide bag](2026-09-25-sc-lateral-bypass-wide51500-mcap-archive.json),
and [failed wider-lane bag](2026-09-25-sc-fivecard-lateral-bypass-port1-110mm-mcap-archive.json)
were compressed to NAS and verified by restoration hash before local MCAP
removal. Failed runs remain represented in their score and archive.

These few scenes demonstrate a feasible lateral teacher route, not a
general planner for every card pose, cable shape, or SC target. Before BC
admission, each candidate needs the existing causal command/image, physical
grasp, scene-split, force, and route-clearance checks. This targeted route
test does **not** reopen the stalled step-5 autonomous BC gate. The 85 mm
port-1 episode is scored full but fails the conservative cable-clearance
gate; the 110 mm rerun fails both clearance and insertion. Neither belongs
in the new route-valid BC manifest. The port-0 episode is a promising
candidate, subject to the remaining causal and grasp audits.

## What the earlier two-stage PoseInsert result actually showed

The earlier [AIC-RPDP result](2026-09-22-rpdp-dppo.md) **did** run an
observation-only RGB pose estimator first and feed its estimate to a
port-relative actor. It achieved **14/14 local 8 mm seating events** on
selected Isaac SFP-to-NIC near-port development starts, then **4/5** on a
further same-scene cable-template check. It had no teacher or true pose at
evaluation. This is promising for local alignment/insertion, but it does
**not** establish full-depth (~40+ mm) official Gazebo insertion, SC-to-SC
insertion, transport from a normal start, or robustness when the target is
outside every camera. The later Gazebo shared **body-target** actor is a
different architecture: its predicted pose is auxiliary and does not feed
the action head.

## Options for a visibility-aware shared task solution

All options pass the normal 10D task code to target selection: two task-family
bits, two selected-port bits, five SFP target-card bits, and a card-valid
bit. SC has no target NIC card; obstacle count must be perceived from RGB.
Simulator TF can supervise training labels but never select runtime crops,
poses, or actions.

1. **Two actors with an explicit handoff (recommended first test).** A
   transport actor consumes full RGB/state/task and predicts complete
   TCP-body targets, trained on route-valid teacher paths such as the lateral
   bypass. A separate local actor consumes a task-conditioned, image-based
   relative pose estimate and predicts complete port-frame trajectories,
   following the successful local AIC-RPDP pattern. A visibility/confidence
   model reports whether the *selected* opening is visible and whether the
   3D estimate is trustworthy. Switch only after several consistent frames
   in a validated near-port region; fall back if confidence drops. The
   distinct action frames stay explicit, with a measured-TCP transform at
   command execution. Train and test the handoff on Gazebo SC and SFP before
   claiming one full policy. This is easier to diagnose, but uses two actor
   heads and requires careful phase/switch data.
2. **One shared image encoder with transport and insertion heads.** The
   visibility/phase model selects the body-frame transport head or
   port-frame insertion head. This shares computation and may help the
   300 ms latency budget, but incorrect gating can cause discontinuous
   commands. Use hysteresis, commanded-target continuity checks, and matched
   per-phase tests before live use.
3. **One actor with pose and visibility features, always outputting body-frame
   targets.** It avoids a frame switch and can use all data jointly, but the
   earlier generic BC largely ignored pose information, and the current
   body-target actor still fails autonomous SC full starts. It is a useful
   matched baseline, not the favored architecture on present evidence.

The visibility signal should be **per camera plus uncertainty**, not simply
"some blue port pixel was predicted." In six of eight yaw-edge Gazebo SC
starts, the selected opening was outside both side-camera images at frame
zero. If no view supports a calibrated pose, the transport actor must move
to a safe viewpoint using only ordinary observations and task encoding.
The handoff gate must be evaluated on new episode-grouped development starts;
the current near-port pose estimates still have scene outliers beyond the
0.5 mm insertion corridor. Complete live inference, including pose and
both actor paths, must remain below 300 ms p95.

RL follows only after the supervised route, visibility handoff, and local
insertion are working autonomously. The successful Isaac local controller is
a useful warm start for that stage, but sparse full-start reward does not
repair an unseen target or a bad transport route by itself.
