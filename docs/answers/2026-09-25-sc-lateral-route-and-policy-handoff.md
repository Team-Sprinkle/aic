I tested a new **lateral route for the SC-to-SC teacher**. It moves away from the cards on the starting side, crosses *behind* the row, travels *beside* it, and approaches the selected port only after passing the cards. Simulator geometry sets these teacher waypoints during data collection; it is not an input to an autonomous policy.

The bounded Gazebo tests show why route quality needs its own check:

| Five-card run | Insertion | Cable clearance from simplified main PCB boxes |
| --- | --- | --- |
| Port 0, with starting-side egress | Full | 11.50 mm |
| Fixed scene with wide cameras | Full | 13.81 mm |
| Port 1, 85 mm side lane | Full | 2.11 mm — too close to accept for BC |
| Same port 1, 110 mm side lane | Failed | 3.25 mm |

The clearance measures sampled **cable-link centers** against simplified PCB boxes; it does not prove that every cable surface avoided contact. The two promising runs are teacher-route evidence, not autonomous policy successes, and still need the normal data-admission checks. I stopped the lane-width sweep after the wider route lost insertion.

You can inspect the route in the [synchronized overhead, side, and wrist video](/data1/chmin/yj/ws_aic/src/aic/outputs/experiments/2026-09-25_sc_lateral_bypass/wide_51500/all_views_20fps.mp4). The [port-0 wrist video](/data1/chmin/yj/ws_aic/src/aic/outputs/experiments/2026-09-25_sc_lateral_bypass/pilot_fivecard_port0_egress_wrist_3view.mp4), [close-clearance port-1 video](/data1/chmin/yj/ws_aic/src/aic/outputs/experiments/2026-09-25_sc_lateral_bypass/pilot_fivecard_port1_wrist_3view.mp4), and [failed wider-lane video](/data1/chmin/yj/ws_aic/src/aic/outputs/experiments/2026-09-25_sc_lateral_bypass/pilot_fivecard_port1_110mm_lane_failed_wrist_3view.mp4) preserve the comparisons.

**On 2-1:** Yes, the earlier *local Isaac SFP-to-NIC* PoseInsert experiment ran an image-based pose estimator and fed its estimated pose to a port-relative actor. It achieved 14/14 short, near-port 8 mm seating events, then 4/5 on an additional same-scene check, without true pose at evaluation. That does not establish normal-start transport, SC-to-SC insertion, or full-depth Gazebo insertion. The later shared Gazebo body-target actor is different: its pose prediction is auxiliary and does **not** feed its action head.

**On 2-2:** I would first test **two actor heads with an explicit handoff**:

1. When the selected target cannot be located reliably, a body-frame transport actor uses images, robot state, and the task code. Its SC demonstrations should use audited lateral routes.
2. A task-conditioned perception model estimates the *selected* port pose, visibility in each camera, and confidence.
3. Once that estimate is stable near the port, a separate PoseInsert-style actor predicts complete actions in the port frame. A confidence drop can return control to transport.

The task code identifies *which* port matters; it cannot locate an offscreen port by itself. Keeping the heads separate makes their action frames explicit. Alternatives are a shared image encoder with two action heads, or one body-frame actor given pose and visibility features. The latter is a useful comparison, but the current shared actor has not used pose effectively. Any handoff must be tested from full starts and keep complete inference p95 below 300 ms.

I documented the commands, scores, failed runs, recordings, limitations, and design options in the [experiment record](/data1/chmin/yj/ws_aic/src/aic/docs/experiments/2026-09-25-sc-lateral-bypass-and-policy-handoff.md) and updated the [SC recovery plan](/data1/chmin/yj/ws_aic/src/aic/docs/SC_CABLE_SNAG_RECOVERY_PLAN.md), [status](/data1/chmin/yj/ws_aic/src/aic/docs/STATUS.md), and [experiment index](/data1/chmin/yj/ws_aic/src/aic/docs/EXPERIMENTS.md). Syntax, JSON, links, and video encoding checks passed. **No BC or RL training was run for this route test.** RL remains after a supervised policy demonstrates reliable autonomous full insertion.
