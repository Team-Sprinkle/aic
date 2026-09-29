# SC five-card behind/side/return route sensitivity

Date: September 26, 2026. Status: controlled Gazebo teacher diagnostics;
the fixed-scene low-cross and forward-margin sweeps are complete. These are privileged teacher
routes, not autonomous actor results.
No BC/RL training or reserved final-scene evaluation is involved.

## Hypothesis and method

The September 25 behind-first route could still let the orange cable cross
or brush the card stack. A full-insertion Tier 3 score alone does not certify
the cable route. Test the user's proposed longer path: move away on the
starting side, cross farther **behind** the five-card row, follow an
outside-left lane farther **forward** than the last card, sweep right toward
the port, return backwards to its y coordinate, then hand off to CheatCode
for final alignment/descent/insertion. Vary one waypoint margin by ±10 mm
while holding the same Gazebo scene and all other margins fixed. Audit both
physical insertion and the cable path.

The test tool is `artifacts/prod_cheatcode_audit/CableRouteDiagnostic.py`,
invoked by `scripts/run_recovery_gazebo_diagnostic.sh` in rootless Docker on
GPU 1. It is **not MoveIt/OMPL or a collision-aware motion planner**. The
teacher reads exact card, TCP, and selected-port TF, constructs Cartesian
waypoints, interpolates each segment with a fifth-order smoothstep at 20 Hz,
holds the starting TCP orientation during routing, and sends the resulting
target through the official Gazebo controller. Stock CheatCode subsequently
aligns and inserts. Privileged TF is used only to create these diagnostic
teacher commands. The fixed overhead/side cameras are diagnostic only.
For example, after preparing the fixed scene's `eval_config.yaml` and
`world_audit.sdf`, the nominal low-cross diagnostic was launched as:

```bash
DOCKER_HOST=unix:///run/user/1008/docker.sock \
AIC_AUDIT_SMOOTH_CAPTURE=1 \
bash scripts/run_recovery_gazebo_diagnostic.sh \
  /var/tmp/chmin_aic_20260918_act/sc_low_cross_left_20260926 \
  outside_left_low_cross_return
```

The one-axis variants set exactly one of
`AIC_CABLE_ROUTE_BEHIND_EXTRA_M`, `AIC_CABLE_ROUTE_LANE_EXTRA_M`, or
`AIC_CABLE_ROUTE_FORWARD_EXTRA_M` to `0.01` or `-0.01`. Exact configs,
bags, camera frames, and verified NAS copies are indexed in the
[raw archive manifest](2026-09-26-sc-low-cross-raw-archive.json).
The [machine summary](2026-09-26-sc-far-return-summary.json) lists all
scores, sampled geometry, and video paths.

The first route (`outside_left_far_return`) uses 60 mm behind the first card
center, 90 mm outside the most-left card center, and 105 mm beyond the last
card center. The September 25 route used 45/80/55 mm respectively. A
second corrective route (`outside_left_clear_return`) uses 125/90/125 mm.
The first route is retained as a controlled baseline even if it inserts,
because the side video may still show the cable above the cards early in
the crossing. The 125 mm behind offset addresses the modeled main PCB's
roughly 72.5 mm half-length along its local card axis, with room for cable
and pose variation; this is a hypothesis, not a collision guarantee.

`/var/tmp/chmin_aic_20260918_act/sc_far_return_sweep_20260926.sh`
records the six one-axis ±10 mm runs on a fixed all-left card scene. All
five NIC rails use their allowed minimum translation in that scene; the
measured card centers shifted to more-negative base x. An earlier all-maximum
rail run was accidentally a **right** shift, because the sign in base x was
opposite the initial assumption. Its result is retained as a control.
Do not infer direction from the rail parameter alone; each run saves
`route_plan.json` with measured card and port positions.

## Acceptance criteria

For a candidate route, require official full insertion; no scored off-limit
contact or sustained high-force penalty; sampled cable-link centers clear
of the simplified main PCB bodies by a margin that accounts for cable
radius; and overhead/side video without the cable draped over card tops
through transport and insertion. The geometry script
`artifacts/prod_cheatcode_audit/analyze_cable_route_bag.py` samples scored
TF every 0.5 seconds and computes center-to-main-PCB distance. That metric
omits cable radius, card protrusions, and named cable/card contacts, so it
cannot replace video or a full collider/contact audit. A 5 mm center gap is
only a conservative preliminary screen, not proof of the user's stronger
no-over-card condition. The fixed-scene ±10 mm sweep tests local sensitivity;
it cannot establish reliability across all SC ports, starts, card counts,
and cable shapes.

## Preliminary observations

- Fixed seed 51500, original five-card positions: `outside_left_far_return`
  scored Tier 3 = 75 and had a 26.54 mm minimum sampled cable-center/main-PCB
  gap versus 13.81 mm for the September 25 route. The synchronized
  [five-view recording](../../outputs/experiments/2026-09-26_sc_far_return/all_views_20fps.mp4)
  and [cable plot](../../outputs/experiments/2026-09-26_sc_far_return/cable_overhead_side_force.png)
  are preserved.
- All-maximum card-rail control shifted card centers **right**, not left.
  It also scored Tier 3 = 75, with a 24.68 mm sampled minimum gap.
- All-minimum card-rail hard case shifted all five card centers left. The
  initial route still scored Tier 3 = 75, but the gap fell to 10.96 mm and
  peak wrist force reached 37.43 N. The
  [five-view hard-case recording](../../outputs/experiments/2026-09-26_sc_far_return/hard_left/all_views_20fps.mp4)
  shows the cable near/above the card tops during early crossing. This is
  **not accepted as a clean cable route** merely because it inserted.
- The first +10 mm behind-row perturbation also inserted but narrowed the
  sampled gap to 5.44 mm, another warning about cable sensitivity.

The completed first-route sensitivity sweep used the **same all-left card
scene** for each run. `Tier 3=75` means full insertion; the other two scores
are partial insertions of about 10 mm, not full seating. `Gap` is the
minimum sampled cable-link-center to simplified main PCB distance.

| Waypoint change | Tier 3 | Gap (mm) | Peak wrist force (N) |
| --- | ---: | ---: | ---: |
| Baseline far-return | 75 | 10.96 | 37.43 |
| Behind +10 mm | 75 | 5.44 | 21.70 |
| Behind −10 mm | 75 | 8.00 | 37.98 |
| Outside lane +10 mm | 75 | 9.57 | 28.48 |
| Outside lane −10 mm | 39.61 | 3.80 | 30.30 |
| Forward +10 mm | 75 | 5.24 | 21.96 |
| Forward −10 mm | 39.67 | 9.97 | 21.95 |

This first route fails the user's local-robustness criterion: **5/7 full
insertions**, two partial insertions, and several small cable gaps. The
scorer's "No contact detected" category is for scored off-limit robot
contacts and does not certify absence of cable/card contact.
All six 20 Hz five-view variation recordings, exact waypoint plans, scores,
and cable plots are under
[`far_return_variations/`](../../outputs/experiments/2026-09-26_sc_far_return/far_return_variations/).
In particular, review the [inward-lane partial insertion](../../outputs/experiments/2026-09-26_sc_far_return/far_return_variations/lane_minus10/all_views_20fps.mp4)
and [short-forward partial insertion](../../outputs/experiments/2026-09-26_sc_far_return/far_return_variations/forward_minus10/all_views_20fps.mp4).

Increasing the behind-row offset alone to 125 mm (`outside_left_clear_return`)
still gave Tier 3=75 but only **5.80 mm** minimum sampled gap. The
[overhead/side video](../../outputs/experiments/2026-09-26_sc_far_return/clear_behind_high_cross/all_views_20fps.mp4)
still shows the cable near the card tops. The subsequent targeted
`outside_left_low_cross_return` lowers the TCP **while behind the row**
before shifting laterally. Its all-left baseline scored Tier 3=75 with
**16.51 mm** minimum sampled gap and 22.37 N peak wrist force; the
[20 Hz five-view video](../../outputs/experiments/2026-09-26_sc_far_return/low_cross_candidate/all_views_20fps.mp4)
and [eight-phase contact sheet](../../outputs/experiments/2026-09-26_sc_far_return/low_cross_candidate/crossing_contact_sheet.jpg)
show a more promising cable path. The gripper occludes part of the cable
in these views, so this is a **candidate**, not proof that the cable never
passes over every card feature.

The six one-axis ±10 mm low-cross runs completed on the same all-left
scene. This improved full insertion from **5/7 to 6/7**, and every sampled
minimum gap exceeded 12 mm. The shorter forward traverse still failed
to fully insert, in both route families, despite no scored force penalty.
That failure has no diagnosed cable/card contact.

| Low-cross waypoint change | Tier 3 | Gap (mm) | Peak wrist force (N) |
| --- | ---: | ---: | ---: |
| Baseline low-cross | 75 | 16.51 | 22.37 |
| Behind +10 mm | 75 | 16.86 | 21.85 |
| Behind −10 mm | 75 | 12.04 | 21.96 |
| Outside lane +10 mm | 75 | 15.03 | 21.71 |
| Outside lane −10 mm | 75 | 16.10 | 21.75 |
| Forward +10 mm | 75 | 13.62 | 21.67 |
| Forward −10 mm | 39.67 | 14.86 | 23.40 |

The candidate therefore **does not pass** the requested all-variation
insertion criterion. One targeted margin revision is preregistered: raise
the nominal beyond-last-card distance from 125 to 145 mm and test 135,
145, and 155 mm with the low crossing unchanged. If any of those three
fails full insertion or has a sampled gap below 10 mm, stop this route
branch. If all three pass, run the other four one-axis ±10 mm tests at the
145 mm nominal forward distance. Even a 7/7 fixed-scene result would be
only local route evidence; test both SC ports and new scene groups before
BC admission or generality claims.

That 145 mm test **failed its preregistered gate**. The 145 mm run inserted;
the 135 and 155 mm runs reached only about 10 mm insertion. Their minimum
sampled cable-center/main-PCB gaps were **13.02, 13.61, and 14.74 mm**
respectively, with peak wrist forces about 21.7–21.9 N. Thus the partial
insertions are not diagnosed cable/card contact. The older successful
135 mm low-cross run and the new partial 135 mm run had byte-identical
`eval_config.yaml`, `world_audit.sdf`, and diagnostic code; their planned
waypoints matched exactly except about 0.01 mm start-TCP jitter. This is
evidence of sensitivity in the physical simulation/insertion outcome,
not a clean geometric margin threshold. The four other 145 mm perturbations
were **not run**, per the stopping rule.

The actual `CollectCorrectiveCheatCode` teacher was then tested once on
the same all-left scene at 135 mm forward distance, with the new opt-in
`low_cross_return` waypoints. It scored **Tier 3=75 full insertion**, no
scored contact or force penalty, and a **14.25 mm** minimum sampled
cable-center/main-PCB gap. Its separate SC alignment/insertion handoff
can therefore complete at least one route on which the stock diagnostic
controller was inconsistent. Its [sim-time 20 Hz three-wrist video](../../outputs/experiments/2026-09-26_sc_far_return/collector_135_left/wrist_20fps.mp4),
[route plan](../../outputs/experiments/2026-09-26_sc_far_return/collector_135_left/lateral_bypass_plan.json),
[score](../../outputs/experiments/2026-09-26_sc_far_return/collector_135_left/scoring.yaml),
and [candidate-integrity JSON](2026-09-26-sc-low-cross-collector-integrity.json)
are saved. The 1,117 causal rows cover 66.15 simulation seconds, have
complete three-camera RGB, strictly increasing time, exact
teacher/executed targets, and no student or perturbation rows. This is a
single teacher diagnostic, **not BC-admitted or an autonomous policy**;
scene split/duplication, physical grasp, full cable/card collision, and
route visual checks remain.

All six [low-cross variation recordings](../../outputs/experiments/2026-09-26_sc_far_return/low_cross_variations/)
are rendered at synchronized simulator-time 20 Hz. The
[short-forward partial insertion](../../outputs/experiments/2026-09-26_sc_far_return/low_cross_variations/forward_minus10/all_views_20fps.mp4)
is the key negative example. The three
[145 mm margin-stage recordings](../../outputs/experiments/2026-09-26_sc_far_return/forward145_stage1/)
also preserve the 135/145/155 mm outcomes. Their full camera views are
diagnostic only, not model inputs.

To distinguish the geometric route from its final insertion controller,
the opt-in corrective teacher was tested at **115 mm** and **145 mm** beyond
the last card on the same all-left scene (125 mm nominal with −10 or +20 mm).
The 115 mm run scored **Tier 3=75 full insertion**, with no scored excessive
force or off-limit contact; sampled minimum cable-center/main-PCB gap was
**14.17 mm** and peak wrist force **23.21 N**. The 145 mm run reached only
**about 10 mm partial insertion**, Tier 3=46.05, despite the same scored
force/contact categories being clear. Its sampled gap was **12.61 mm** and
peak wrist force **21.71 N**. These box-center gaps do not establish
the physical cable surface clearance. Thus the corrective teacher is not
robust to this forward-distance change; the earlier stock-diagnostic partial
cannot be attributed solely to route clearance. These are distinct simulator
rollouts, so this comparison does not isolate the final handoff controller
perfectly. The exact [115 mm command](../../outputs/experiments/2026-09-26_sc_far_return/collector_115_left/run.json),
[score](../../outputs/experiments/2026-09-26_sc_far_return/collector_115_left/scoring.yaml),
and [wrist video](../../outputs/experiments/2026-09-26_sc_far_return/collector_115_left/wrist_20fps.mp4)
are saved, as are the [145 mm command](../../outputs/experiments/2026-09-26_sc_far_return/collector_145_left/run.json),
[score](../../outputs/experiments/2026-09-26_sc_far_return/collector_145_left/scoring.yaml),
and [wrist video](../../outputs/experiments/2026-09-26_sc_far_return/collector_145_left/wrist_20fps.mp4).
The 145 mm wrist video is resampled to 20 fps from recorded frames; a late
source gap reaches 2.5 s, so repeated images there should not be read as
smooth measured motion. The synchronized fixed-camera diagnostic video is
the better view of the whole cable, although the arm still occludes portions
and the cable may pass close to the last card tops.
None of these teacher episodes is BC-admitted.

Stop the route sweep rather than manufacture a 7/7 claim. Next diagnose the
tip-to-opening and cable state at the first divergence between full and
partial insertions, with tighter overhead cable visibility or named contact
instrumentation; then preregister one repair and retest on new SC ports,
starts, and card layouts. Do not open reserved final scenes or start BC/RL
from this route until admission and generalization pass.

Raw fixed-scene and all-left recordings, including corrective-teacher RGB
and MCAPs for 115, 135, and 145 mm, were copied to
`/home/nas/chmin/aic_archive_20260926/`; source/archive file sizes,
MCAP SHA-256 hashes, and frames indexes were verified. Source files have
not been removed. Saved output paths remain under
`outputs/experiments/2026-09-26_sc_far_return/` (NAS-backed symlink because
`/data1` is nearly full).
