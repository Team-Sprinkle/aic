# Bounded five-card cable-side route repair

## Frozen design before candidate replays

The October 9 [contact diagnosis](2026-10-09-sfp-route-target-and-gazebo-gate.md) found a repeated failure class: cable links 4–5 pressed against a NIC card while the transport route descended, with 98.3 N sampled peak point normal force and 12.2 N peak wrist-force change on one replay. The same retreat route sometimes reached the buffer without that high-force event. The exact original aborted run did not contain contact-pair instrumentation.

This is a **privileged teacher-route diagnosis**, not actor or autonomous insertion training. The two candidates are fixed before Gazebo runs:

1. `side_down_then_center`: retreat to the high axial station, move 60 mm away from the NIC row in the port-frame negative lateral-Y direction, descend while offset to the 120 mm axial buffer, move laterally toward the requested port, then finish at the 85 mm buffer.
2. `side_sweep_high`: retreat to the high axial station, sweep 80 mm in the same lateral direction and return while high, then descend to the 85 mm buffer. This tests whether changing cable slack before descent helps without low-altitude side travel.

These are distinct interventions. The first maintains side clearance during descent; the second only changes the cable's high-path history. Both preserve the measured starting TCP orientation. Port geometry and NIC solids are privileged **planning labels**; neither variant is a deployable observation-only controller.

Before examining candidate outcomes, the per-start acceptance criteria are: final measured TCP within **10 mm** of the 85 mm buffer goal; no wrist-force change above **6 N for three consecutive commanded waypoints**; no sampled cable–NIC-card point normal force above **10 N after route takeover**; and no official prohibited-contact score. Task-board touches are reported separately rather than treated as NIC-card catches. The point force is the existing pinned Bullet impulse/dt diagnostic, not net wrist force. A candidate passes only with at least **2/3** accepted new, nonfinal five-card starts; at most three starts per candidate. The earlier retreat comparison is frozen and will not be re-scored as a candidate.

First plan both candidates from the saved actor state and screen them against the measured cable shape. The rigid cable screen is a ranking diagnostic only because it flagged successful teacher routes. If MoveIt cannot produce a rigid-arm path or the candidate has a clear measured-shape regression, stop that candidate before Gazebo. If neither candidate passes, stop route-actor training and revise route planning instead of scaling data or connecting handoff.

Final scenes remain sealed. Rootless Gazebo runs sequentially on GPU 3; the preexisting Isaac GPU 0 job is left alone.

## Completed bounded result

Both candidates produced rigid-arm MoveIt plans from the saved actor state. Their signed rigid-cable proxy minimum was the same **−3.21 mm at the initial posture** as the retreat baseline; because this proxy also marks known safe motion, it neither endorsed nor rejected the candidates. Live paths were replanned from each actor's measured takeover state. Each run enabled the read-only oracle contact recorder from its start. The predeclared gate was evaluated with `scripts/score_sfp_cable_side_run.py`, using contact events only after route takeover.

| Candidate / run | Buffer error | Wrist force change / longest >6 N streak | Peak cable–NIC-card point force | Gate |
| --- | ---: | ---: | ---: | --- |
| Side descent 1 | 3.23 mm | 8.77 N / 24 waypoints | 19.58 N | Fail: force and contact |
| Side descent 2 | 22.38 mm | 4.43 N / 0 | 46.95 N | Fail: arrival and contact |
| High sweep 1 | 11.01 mm | 8.45 N / 1 | 9.65 N | Fail: arrival |
| High sweep 2 | 4.23 mm | 2.69 N / 0 | 0.43 N | Pass |
| High sweep 3 | 3.61 mm | 7.63 N / 61 | 3.95 N | Fail: sustained force |

The side-descent candidate was stopped at **0/2** because it could no longer reach 2/3. The high-sweep candidate finished at **1/3**. Its third run reduced cable–NIC-card contact yet had a sustained measured wrist-force change beginning early in the route. The captured external contact pairs are small, so the wrench may include cable tension, a different contact or controller effects; this run does not isolate the source. No candidate passes the declared safety/arrival gate. The official scorer reported no prohibited contact in all five runs, which again does not cover cable-card force.

I inspected the fixed side views and contact records. In [side descent 2](</home/nas/chmin/aic_archives/2026-10-09_sfp_side_route_repair_v1/side_down_2/video/all_views_20fps.mp4>), the orange cable crosses a card while the gripper moves toward the side waypoint; the oracle records the cable/NIC-card force reported above. In [high sweep 3](</home/nas/chmin/aic_archives/2026-10-09_sfp_side_route_repair_v1/side_sweep_3/video/all_views_20fps.mp4>), the cable remains visibly drawn over the card row while the wrist-force change persists, but the sampled external cable/NIC-card contact force stays below 4 N. The video supports tension as a possibility, not an isolated cause.

The immutable archive is `/home/nas/chmin/aic_archives/2026-10-09_sfp_side_route_repair_v1/`. It contains the saved-start plan, offline cable proxy, all five rootless Gazebo scene/plan/trace/bag/contact artifacts, per-run `cable_side_gate.json`, videos for representative failures, and [machine decision](</home/nas/chmin/aic_archives/2026-10-09_sfp_side_route_repair_v1/decision.json>). Command pattern:

```bash
AIC_TRANSPORT_EVAL_ORACLE_CONTACTS=true \
AIC_SFP_MOVEIT_VARIANT=side_sweep_high \
bash scripts/run_sfp_moveit_gazebo_recording.sh \
  /home/nas/chmin/aic_archives/2026-10-09_sfp_side_route_repair_v1/case3.json \
  /home/nas/chmin/aic_archives/2026-10-09_sfp_side_route_repair_v1/side_sweep_2 3
.pixi/envs/default/bin/python scripts/analyze_force_oracle.py \
  /home/nas/chmin/aic_archives/2026-10-09_sfp_side_route_repair_v1/side_sweep_2
.pixi/envs/default/bin/python scripts/score_sfp_cable_side_run.py \
  /home/nas/chmin/aic_archives/2026-10-09_sfp_side_route_repair_v1/side_sweep_2
```

## Decision and next bounded hypothesis

**Stop route-actor training from these side-route labels.** The same nominal scene produces different actor takeover/cable states; a rigid arm path to the requested port is insufficient to control cable load. The next useful label feasibility test is to exploit the existing *measured, successful SFP CheatCode continuations from actor takeovers* rather than create another arbitrary side waypoint. Those saved six takeovers had no official contact and five full insertions, but the first second includes controller settling and earlier one-second/20-mm waypoint targets gave 0/6 correct actor-boundary command directions. A stage-aware label must therefore begin at meaningful measured motion after settling and preserve full actor-prefix observation history. Budget at most **30** distinct actor-state queries, six held-out grouped validation takeovers and no actor fit until at least **20 train + six held-out** safe continuations exist and ≥5/6 held-out first corrective directions have cosine ≥0.7 with measured safe movement. Also examine cable/NIC-card oracle contact on a small subset; official no-contact scoring alone is insufficient. If this gate fails, revisit cable-aware state estimation/planning rather than scaling routine teacher data. Automatic handoff, insertion and RL remain disabled.
