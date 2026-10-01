# Full experiment execution prompt

Continue the AIC project from its current repository and process state. Execute the supervised perception, motion-planning, two-actor BC, and gated RL plan below autonomously. When a step passes its stated gate, move to the next step. If it fails, inspect synchronized videos and causal traces, identify a specific cause, try several distinct targeted fixes, and rerun the same gate. Preserve negative results. Do not replace diagnosis with an open-ended parameter sweep.

First read:

- `docs/experiments/2026-09-29-sc-low-cross-2cm-rescue.md`
- `docs/experiments/2026-09-26-sc-far-return-route-sensitivity.md`
- `docs/POSE_VISIBILITY_HANDOFF.md`
- `docs/HIERARCHICAL_RECOVERY.md`
- `docs/answers/2026-09-29-two-actor-perception-bc-rl-plan.md`
- `docs/STATUS.md`

Inspect current files, active jobs, rootless Docker setup, and GPU allocations before starting. Do not duplicate or interrupt active work. Use at most four GPUs concurrently, beginning the MoveIt/Gazebo pilot on one GPU. Keep reserved final scenes sealed. Record exact commands, scene and code hashes, scores, traces, contact measurements, videos, and failed attempts. Save progress in the relevant experiment docs and update STATUS.md and artifact indexes.

## 1. Establish the fixed route comparison

Use the archived five-card, all-left, SC-port-0 `forward_minus10` case and its exact scene settings. The September 29 study found 4/7 full insertions for repeated unchanged routes, 4/5 for a lane 20 mm farther outside, 2/2 for behind +20 mm, 2/2 for both changes, and 1/2 for forward +20 mm. These are small, noisy samples. Some partial insertions occurred despite fairly large sampled cable-center gaps. Do not claim that a changed insertion score proves a cable snag was fixed.

## 2. Make MoveIt 2 operational for Gazebo teacher planning

The repo already has ROS 2 Kilted MoveIt configuration and a MoveItPy backend. The current backend does not add the cards and fixtures to its planning scene. Fix that and verify collision-object positions, yaw, dimensions, frames, and robot/gripper/plug geometry against the actual Gazebo scene. The host pixi environment has MoveItPy, but the official evaluation image lacks it; an earlier offline smoke test also lacked a usable live joint-state source. Resolve this through a compatible rootless sidecar with joint states or an explicitly validated planning start state. Verify that MoveIt produces an actual collision-checked joint trajectory and that Gazebo executes that trajectory, not a fallback scripted TCP interpolation.

## 3. Run a bounded MoveIt cable-route pilot on the fixed failure scene

Plan from the recorded start to a safe pre-insertion standoff. Let MoveIt select the joint path while supplying broad around-card corridor or keep-out constraints. Test outside-left and, where feasible, outside-right route classes and small changes to corridor clearance suggested by the ±2 cm study. Use the same start, cards, selected port, controller, and logging as the scripted control.

The gate for **this step is route quality, not insertion**. It is acceptable to stop at the standoff without inserting. Judge whether the measured gripper and full cable pass beside the cards without snagging, draping over card tops, excessive force, sustained cable/card contact, or loss of progress. Save synchronized overhead, side, wrist, and full-cable video at useful frame rates. Use physical cable radius and actual contact geometry where possible; the previous cable-link-center-to-simplified-box gap alone cannot certify clearance. Mark a route indeterminate if the cable is hidden and contact evidence is insufficient.

If a route fails, inspect the video at its first obstruction and distinguish arm/card collision, gripper/card collision, cable/card interaction, controller tracking, bad planning-scene geometry, and a final-insertion issue. Try distinct repairs based on that diagnosis—for example correcting scene geometry, changing corridor topology, increasing clearance, or altering the approach orientation. Rerun the same case after each repair. Do not run a large blind waypoint grid.

## 4. Test route generality before using it as SC transport supervision

Freeze the promising planner setup, then test both SC ports, several reset starts, different card counts and positions, and especially crowded layouts. Repeat identical settings enough to see run-to-run variation. Compare against the scripted route with matched scenes. Report route completion, cable-clearance evidence, contact/force, controller tracking, planning failures, and time separately from insertion score. A single clean fixed-scene path is only a pilot.

## 5. Build training data with explicit admission rules

A clear MoveIt route ending at the standoff can train the **transport actor** even if the later insertion fails. Mark it as a route-only episode; never label it a full expert insertion. For **local insertion BC**, obtain separate verified successful SFP and SC insertion segments, using an appropriate final alignment/insertion teacher after a clear route. Exclude old SC transport segments that take the cable over cards. Preserve every failure and its reason.

During teacher collection, enable the same observation-based hierarchical backtracking supervisor intended for deployment. Log proposed, supervisor, and actually executed actions; action owner; recovery mode; checkpoint history; force; measured motion; and terminal observation. Exclude hard-coded retreat actions from actor BC targets. Train the actor on successful normal transport and successful post-retreat route choices. Split action chunks at ownership changes. Keep failed and supervisor-owned transitions for later critic/reward analysis. Do not assume BC will learn post-backtrack choices if the dataset has none.

## 6. Train task-conditioned perception for Gazebo SFP-to-NIC and SC-to-SC

Warm-start compatible existing perception weights. Train selected-target localization and relative pose, actual per-camera identifiable visibility, and calibrated probability that the fused pose is accurate enough for local control. Projection inside an image is not a true visibility label. Include wrong visible ports as negatives, occlusion, offscreen starts, both SC ports, SFP target-card identities, and cable/card variation. Simulator geometry supplies labels only, never autonomous crop selection or actor input. Split by whole episode and reset configuration. Report near-port 3D, lateral, axial, and orientation errors; false confident predictions; coverage; and errors by camera/task/phase. Check whether accuracy resolves the 0.5 mm insertion corridor.

## 7. Train the two supervised actors without starting their visual representations from scratch

Transport actor: warm-start compatible full-start TCP-body BC weights and train on clean SFP and route-valid SC transport, including successful actor-owned post-backtrack choices. It sees three RGB views, robot state, task code, and observable recovery state. It outputs complete TCP-body targets.

Local actor: use the selected SFP PoseInsert/RPDP checkpoint as the main local warm start. Add task/selected-port conditioning and fine-tune on successful SFP and new SC near-port data using observation-only predicted pose inputs. It outputs complete port-frame trajectories. Do not blindly copy the transport action head: its target frame differs. Preserve SFP performance while adding SC. The old Isaac result establishes only local 8 mm seating, so test official full-depth behavior separately.

## 8. Train and test the handoff

Use per-camera selected-port visibility, calibrated fused-pose confidence, predicted distance to the selected opening, and temporal consistency to select transport or local control. Set thresholds from held-out development scenes, with hysteresis and a stricter single-view rule. Handle short insertion-time occlusion only within a validated uncertainty bound. Make commanded targets continuous across switches. Evaluate full-start autonomous Gazebo runs without privileged geometry, guide, or teacher. Measure full insertion, force/contact, cable route, wrong switches, missed switches, and complete inference p50/p95/p99; p95 must remain below 300 ms.

## 9. Gate Isaac adaptation and RL on supervised results

First verify Isaac camera, action timing, contact, and cable behavior against the Gazebo cases that matter. If the Gazebo BC policy degrades in Isaac, do a bounded Isaac BC adaptation before RL. Then warm-start an off-policy SERL/RLPD-style actor-critic from BC and prior replay, using the **same supervisor** during training and deployment. Reward actual full-depth insertion and safe progress while penalizing damaging force, repeated blockage, and wasted retries; prevent retreat alone from earning reward. Log executed action and ownership. Supervisor retreat actions can inform the critic but do not update the actor as imitation targets. Compare no recovery, local backoff, and hierarchical backoff on matched new development starts.

## 10. Attempt limited Gazebo RL adaptation only after Isaac gives an autonomous gain

Audit and adapt the existing Gazebo RL runner: it currently expects a one-step direct visual actor and cannot simply fine-tune the proposed two-actor four-command system. Verify the observation/action contract, replay ownership, terminal state, supervisor, and reward before training. Because Gazebo RL is slow, use a bounded matched adaptation with a frozen BC/Isaac control and explicit stopping rule.

## 11. Evaluate and document

Keep the final split sealed until all development gates pass. Once passed, freeze checkpoints, thresholds, supervisor rules, code, and scenes before a production-style evaluation. Report results for both task families and card counts, including complete failures and confidence limits. Update STATUS.md, EXPERIMENTS.md, the relevant experiment records, artifact map, and machine-readable summaries. Save representative success and failure videos.

At every failed gate: first inspect the existing videos and causal traces; state the leading hypothesis; try at least two meaningfully different repairs when feasible; rerun a matched comparison; and stop that branch if the evidence still cannot support the gate. Continue independent steps that do not depend on the failed gate, but do not silently advance to RL or final evaluation.
