# Planned steps: approach history, selected-port handoff, and PoseInsert

The target pipeline is:

```text
task code + cameras + robot state
    → transport toward the requested port
    → save a few clear approach views and measured camera poses
    → transport proposes a stop; hold and capture fresh views
    → verify the *same requested port* and estimate its current pose
    → PoseInsert actor aligns and inserts from a safe standoff
```

The following are **planned experiments**, not completed results. Existing transport and perception results are the starting baselines; the integrated insertion switch remains disabled and the four reserved final IK scenes remain sealed.

1. **Freeze a fair comparison and audit existing traces.** Fix complete development resets, requested-port labels, current single-frame verifier, transport checkpoint, and its stop proposals. Identify clear early views, close ambiguous views, actual absent/wrong-port cases, and whether saved images and measured TCP poses are synchronized. Use existing data first; collect only missing targeted before/after or occlusion views. This prevents a history model from appearing better merely because it saw easier scenes.

2. **Build a sparse, causal view history.** Save synchronized three-camera RGB, measured TCP pose, task code, and distance traveled. Start with a clear identity anchor; retain up to about 6–10 triplets. Trial spacing: 30–50 mm during open transport and about 10 mm during the final 50–100 mm, plus a frame before visibility drops, a frame after a meaningful turn, and a fresh frame at the stop. Tune spacing on fitting scenes for identity recall, memory, and latency; do not treat 10 mm as established. Reset history at episode reset or task change. No future observations or simulator port coordinates enter inference.

3. **Add history-aware identity tracking.** Begin with a small head or tracker over the *frozen* image features and measured camera motion, conditioned on the requested task/port. It should link the early clearly identified opening to the late view and output **requested port / invalid** and a track for the selected opening. Compare with the current single-frame identity head on the *same* final images and grouped development resets. Do not initially retrain the full vision backbone. Training labels may use simulator geometry, but runtime crop selection, tracking, and decisions must use observations only. If a clear anchor never existed or the late image contradicts the track, abstain and seek a better view.

4. **Validate the handoff, separately from insertion.** Keep the existing original-resolution crop refinement, calibrated multi-camera position, board-line orientation, and force checks. Test real port 0/1, one-port absent requests, neighboring-port confusion, partial occlusion, and views before/after actual TCP rotations. Report false accepts per complete reset, reviewed clear-arrival recall, 3D/lateral/axial pose errors, orientation error, and complete ROS-to-decision p50/p95/p99. The primary promotion gate is **zero accepted known absent or wrong ports** on the frozen development challenges, at least **95% recall on reviewed clear requested-port arrivals**, and **p95 below 300 ms**. A small zero-error sample is not a universal safety claim. If this fails, diagnose the exact scene and repair identity or tracking before connecting the actor.

5. **Rebuild PoseInsert training around safe handoff states.** Use verified successful SFP-to-NIC and SC-to-SC teacher insertion segments and correct *recorded teacher targets* converted into the requested port frame. Keep full target trajectories, not delta corrections to a BC action. Add varied but feasible lateral/axial/orientation starts, including 0°, about 3°, 6–7°, and 10° orientation offsets. The actor receives the estimated port pose, task code, current RGB/state, and force/contact information, never true port geometry. Some SC transport demonstrations end before insertion; use their transport prefix only, not a failed insertion tail as expert insertion data. Train and compare a fresh local actor with a warm-started version under the same data and budget.

6. **Measure the actor's actual correction envelope in isolation.** Start from controlled safe standoffs on new non-final development resets, with position and orientation offsets fixed *before* rollout. Evaluate full insertion depth, lateral error ≤0.5 mm, orientation, force, contact, aborts, and latency for each offset bucket and both task families. A 6.7° correction is considered supported only if the actor repeatedly aligns **before** axial contact and inserts without excessive force. If it cannot, keep a narrower posture gate and improve actor data/control; do not simply ask the handoff verifier to permit 6.7°.

7. **Attach the handoff and actor; run full-start Gazebo development episodes.** Transport follows the requested route, pauses, checks selected-port identity and pose using causal history, then hands off at a safe standoff. PoseInsert takes over for alignment/insertion. On invalid identity, unsafe force, or tracking disagreement, hold or resume via a tested rule. Compare against the frozen transport-plus-single-frame baseline on matched development starts across SFP and SC, port choices, and card counts. Save camera videos and per-stage traces so failures can be attributed to route, stop, identity, pose, actor, or contact. Measure complete ROS-to-command p95 below 300 ms.

8. **Only then consider recovery/RL and final evaluation.** If most failures are correctable insertion/contact failures after a verified handoff, test bounded backtracking and later RL from the supervised actor. If failures are wrong-port handoffs or transport routes, repair those stages instead; RL cannot safely compensate for a wrong target. Keep the reserved final scenes sealed until the development gates pass, then run one frozen final evaluation without tuning on its outcomes.

The first immediate experiment is **Steps 1–4: sparse history versus the frozen single-frame verifier**. It answers whether approach memory fixes the current valid-port false rejection without losing absent/wrong-port rejection. PoseInsert data preparation can be audited alongside it, but attaching or claiming insertion success waits for the handoff gate.

Background: [selected-port handoff plan](../SELECTED_PORT_HANDOFF_PLAN.md), [orientation/identity result](../experiments/2026-10-03-sc-orientation-identity-repair.md), and [PoseInsert/historical actor status](../STATUS.md).
