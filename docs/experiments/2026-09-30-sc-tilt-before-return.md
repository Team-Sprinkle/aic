# SC tilt before return: five-card port-1 pilot

Date: 2026-09-30. These runs use a **privileged Gazebo teacher** on the same development reset, not a trained autonomous actor. The teacher knows the selected port geometry for final alignment and insertion. Reserved final scenes stayed sealed; only GPU 4 was used, one replay at a time. [Machine results](2026-09-30-sc-tilt-before-return-results.json) list every scored run, hashes, gate values, traces, and MCAP paths.

## Change and physical check

The user's hypothesis was that tilting the wrist near the cards swings the plug around the TCP, because the TCP is not at the plug tip. I changed the saved MoveIt route to:

```text
back / clear start side → shift behind row while descending → traverse beside row
→ sweep beyond last card → tilt in the clear beyond-row area
→ return toward selected port → descend and insert
```

`--tilt-before-return` adds `tilt_clear_of_cards` as a distinct, collision-checked MoveIt subgoal at the end of the beyond-row sweep. The preceding sweep keeps the original wrist orientation. The following return starts aligned. Endpoint holds of 8 seconds after sweep and 5 seconds after tilt give the joint controller time to settle; the holds do not replace the MoveIt path. The revised route planned 309 waypoints without Cartesian fallback.

The return is **not perfectly level**: its planned TCP Z rises from 0.1800 to 0.2069 m while it moves about 38 mm toward the selected port. The subsequent insertion command is vertical. This keeps the plug low during the beside-row traverse while reaching the earlier validated insertion standoff. It should not be described as a purely horizontal return.

### Why the new video can look like a shallower descent

The explicit tilt change did **not** raise the low corridor. Its `shift_behind_row_low`, `traverse_beyond_row`, and `sweep_after_row` waypoints are all at **TCP Z = 0.1800 m**, exactly the same as the September 29 port-1 low-cross plan. Both start the down-shift from a 0.2490 m clear-side waypoint, a 69 mm planned drop. The still older high `strict_return` path descended farther visually (0.3380 to 0.2069 m, about 131 mm) because it started **89 mm higher**; its traversal was actually **26.9 mm higher** than the current low corridor. The lower start and the long endpoint hold make the new video look less dramatic without reducing the target depth.

Independent scored `/scoring/tf` measurements agree. In the earlier port-1 low-cross route, plug-tip height relative to the 1.331 m card top was **−21.7 mm at 10 s** and **−32.1 mm at 20 s**. In the held explicit-tilt route, it was **−21.1 mm at 10 s** and **−31.1 mm at 20 s**. The old route first passed below the card top at 5.81 s, the new one at 5.59 s. These are comparable phase samples, not a 27 mm loss of descent. During the new tilt the tip is roughly 33–39 mm below card top; at the end of the held run that stopped at its preinsert gate it was 38.8 mm below. The later TCP rise on return is real and was kept to reach the insertion standoff. The tip-height envelope alone cannot certify cable clearance.

The [sim-time bag analysis](../../scripts/analyze_moveit_tilt_bag.py) confirms the intended physical order. In the held replay, tilt moved the TCP only **1.59 mm** but swept the plug tip **8.78 mm**. The next return moved the TCP **47.37 mm**. This directly supports the concern about rotating around the TCP. The plug tip was about 33–39 mm below the highest card top during this section; the measured first crossing below that top was 5.59 seconds after trial start. These height measurements describe the **tip**, not the entire cable. The [20 fps five-view success video](../../outputs/experiments/2026-09-30_moveit_tilt_before_return/port1_tilt_hold_success_video/all_views_20fps.mp4) shows the cable near and occasionally touching the cards while the gripper continues. Touch is allowed under the revised gate; sustained snag, blocked progress, damaging contact, and failure to get near the opening are not.

An earlier analysis mistakenly matched MCAP wall-storage timestamps to simulation-time waypoints and overstated tilt translation. It was corrected to use the controller and `/scoring/tf` **header stamps**; the analysis file is schema v2. The height script was corrected the same way. Do not use the older wall-time segment values.

## Insertion experiments and diagnosis

An unheld first route reached the port vicinity but failed its preinsert tracking gate. The held route initially did too. The teacher was then tested with a separately configured initial far-field lateral gate, 1.2-second precontact alignment, a later 2.2 mm tracking gate, and guarded lateral servo near the opening. It first produced one official full insertion: Tier 3 **75**, total **83.16**, and no scored off-limit contact or excessive-force penalty. A nominal repeat stopped on a speed gate despite small lateral error. Increasing that speed gate to 8 mm/s exposed variation in precontact force and then near-opening contact.

Two stopped trials reached the mouth with lateral error around **0.33–0.51 mm** and orientation error around **0.01°**, but measured axial progress stalled as force rose to roughly **2.3–3.6 N**. Pushing harder did not resolve it and was stopped. One nominal backoff trial measured only about 9 mm retreat and did not release force. A subsequent diagnostic allowed a longer retreat, but it was **never triggered** in its successful run; its exact repeat also inserted without recovery. Thus there are now **three full insertions** among the targeted tilt-route trials, but no evidence that the larger backoff caused them. The near-mouth stall is a local insertion/contact problem, not proof of a cable snag or failed outside route.

The first long-retreat setting and its repeat scored Tier 3 **75**, totals **83.07** and **83.00**, without scored excessive force or off-limit contact. The route gate passes on this fixed scene under the progress criterion. The full-insertion teacher is still variable: most earlier targeted controls stopped at gates or contact, and all runs share one reset. We do not admit these as broad SC expert data yet.

## Matched control and limits

The older port-1 MoveIt plan changes orientation while sweeping beyond the row, instead of using a separate stationary tilt subgoal after the sweep. With the same updated insertion settings, this old plan inserted in **1/2** repeats; its other repeat contacted near the mouth at a **+6.48 mm commanded Z offset** and **3.50 N**, then failed to release force after retreat. The explicit tilt-before-return plan inserted in **2/2** latest repeats. These tiny counts suggest a possible advantage but cannot attribute it confidently to tilt timing. The new route **does** establish a clear physical tilt-before-return order; the matched control shows improved handoff settings also matter. The comparison is plan-level: endpoint holds and path timing differ, so it is not a pure single-variable test of wrist rotation.

The previously unrendered matched [older-route failure video](../../outputs/experiments/2026-09-30_moveit_tilt_before_return/port1_old_tilt_matched_failure_video/all_views_20fps.mp4) is now available in all five views at 20 fps. The failure occurs near the selected port opening, after the around-card route; the force signal identifies a blockage, but the video does not resolve the exact contact surface. It is not evidence of a cable snag during transport. The [machine results](2026-09-30-sc-tilt-before-return-results.json) now include the video hash.

The official score's “No contact detected” refers to its scored contact category; the video can show harmless cable/card touch and intended plug/port contact. No named cable/card contact stream is available. Wide overhead/side cameras are audit views and are not policy observations.

## Reproduction

The rootless runner validates the scene configuration hash against the saved MoveIt plan and runs the official image. For the two latest successful teacher repeats:

```bash
AIC_MOVEIT_PILOT_GPU=4 \
AIC_MOVEIT_PILOT_SCENE_DIR=/var/tmp/chmin_aic_20260918_act/moveit_sc_port1_generalization_source_20260929 \
AIC_MOVEIT_PILOT_ONLINE_INSERTION=true \
AIC_MOVEIT_PILOT_TRACKING_GATE_FORCE_FRACTION=6.0 \
AIC_MOVEIT_PILOT_FORCE_CONFIRM_THRESHOLD_N=2.0 \
AIC_MOVEIT_PILOT_NOMINAL_INITIAL_LATERAL_GATE_M=0.012 \
AIC_MOVEIT_PILOT_PRECONTACT_PORT_ALIGN_SEC=1.2 \
AIC_MOVEIT_PILOT_SC_TRACKING_GATE_SPEED_MPS=0.008 \
AIC_MOVEIT_PILOT_PRECONTACT_SOFT_FORCE_N=1.5 \
AIC_MOVEIT_PILOT_PRECONTACT_FATAL_FORCE_N=2.0 \
AIC_MOVEIT_PILOT_SC_Z_GATE_MAX_HOLD_SEC=1.5 \
AIC_MOVEIT_PILOT_SC_CONTACT_RECOVERY=true \
AIC_MOVEIT_PILOT_GUARDED_LATERAL_SERVO=true \
AIC_MOVEIT_PILOT_SC_RECOVERY_MAX_BACKOFF_M=0.04 \
AIC_MOVEIT_PILOT_SC_RECOVERY_BACKOFF_SEC=1.2 \
AIC_MOVEIT_PILOT_SC_RECOVERY_REQUIRED_MEASURED_M=0.015 \
bash scripts/run_moveit_sc_route_pilot.sh \
  outputs/experiments/2026-09-30_moveit_tilt_before_return/plan_port1_tilt_hold_insert \
  /var/tmp/chmin_aic_20260918_act/NEW_UNIQUE_AUDIT_DIR
```

Exact source scene, measured start, route plan, MoveIt output, held plan, insertion handoff, raw trace, score, and scored MCAP are retained. [Archive](2026-09-30-sc-tilt-before-return-archive.json) copies compact raw material to NAS and verifies hashes. Videos stay in `outputs/`; captured JPEG frames are excluded from the compact NAS copy.

## Gate and next experiment

Step 3's revised route-progress gate passes for this fixed five-card port-1 start. I also advanced Step 4 with one frozen-plan route-only repeat each on the saved five-card port-0 scene and a shifted three-card port-0 scene. Both executed joint paths with **zero pose-command fallback**, reached standoff with **5.03 and 5.05 mm** measured TCP endpoint error, and had no scored off-limit contact or excessive force. I inspected the [three-card five-view video](../../outputs/experiments/2026-09-30_moveit_tilt_before_return/threecard_route_repeat_video/all_views_20fps.mp4); the cable passes near the cards, but the arm and plug continue to standoff. The [five-card port-0 repeat video](../../outputs/experiments/2026-09-30_moveit_tilt_before_return/port0_route_repeat_video/all_views_20fps.mp4) is saved too. These two repeat plans retain their earlier wrist behavior; they test the lowered around-card corridor, **not** the new explicit tilt timing. Their route-only Tier 3 = 0 is expected because insertion was disabled. Raw directories are `moveit_sc_port0_low18_diagonal_repeat_20260930` and `moveit_sc_threecard_mixed_low18_repeat_20260930` under `/var/tmp/chmin_aic_20260918_act/`.

The [route-repeat JSON](2026-09-30-sc-route-generality-repeats.json) preserves their scores, audits, video hashes, and bag hashes. Step 4 remains open because these are repeats of saved resets, not several new reset configurations or a matched scripted-control comparison. The next bounded test is a frozen-plan comparison on new development starts and both SC ports, reporting route completion separately from full insertion. Stop or revise the planner setup if the cable prevents progress, force becomes excessive, or the route misses the opening on repeated starts. Do not promote to step 5 local BC from these fixed-start privileged successes. Perception, two-actor handoff, RL, and final evaluation remain gated.

## Subsequent force diagnosis

The [oracle follow-up](2026-09-30-force-oracle.md) supersedes the earlier lack of contact attribution for newly instrumented repeats. A repeated older-route failure caught the plug against the selected SC port; this is not proof of the exact historical contact pair. The wrist delta and corrected point forces are distinct. Gazebo contact reporting has a confirmed units/depth bug. All physics collisions remained enabled. Additional new-route outcomes are kept separately from the original frozen comparison.
