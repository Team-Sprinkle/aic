# SC insertion force: oracle contact diagnosis (2026-09-30)

## Question and frozen controls

The older-route repeat stopped after a 3.495 N wrist-force change at simulation time 43.052 s. Its five-camera video did not show a cable snag. The exact original trial did not record all collision contacts, so its contact surface cannot be recovered retrospectively with certainty.

We repeated its frozen five-card SC-to-SC port-1 development scene, MoveIt plan, controller, insertion settings, official image digest, and collision configuration. Oracle instrumentation adds contact reporting and two close cameras; it does not remove collisions or feed new observations into an actor. These are privileged teacher trials, not learned-policy success claims. Final scenes remain sealed. Rootless Docker uses GPU 7 sequentially; no active training job was interrupted.

## What the 3.5 N number actually means

The teacher computes $\|f_{sensor}(t)-f_{sensor}(t_0)\|_2$. This is a change in the wrist force vector from a saved baseline, not a force measured at one named contact. Raw readings include approximately 20 N from supported load and can change with motion, cable load, and sensor noise. The raw source frame is `ati/tool_link`, not `gripper/tcp`.

The old trace incorrectly rotated that vector as if its frame were TCP. The new v2 trace names the sensor frame and uses its actual TF for base-frame reporting. The scalar raw-baseline trigger is unchanged; no threshold was raised to hide a failure.

## Confirmed simulator reporting bug

The pinned official image contains gz-physics Bullet Featherstone 8.3.0. Its contact report repeats the scalar impulse in X/Y/Z and reports signed separation as depth. Raw Gazebo transport has the same values as ROS, ruling out our JSON logger or bridge as the source.

Gazebo corrected precisely this in [gz-physics #890 / f34d6e388a](https://github.com/gazebosim/gz-physics/commit/f34d6e388a841aa5eddc276acd01413940841ed5), March 20, 2026:

$$f_{normal}=n\,J/\Delta t,\qquad depth=-distance.$$

Our world has $\Delta t=0.002$ s. `analyze_force_oracle.py` applies this reporting correction to immutable raw JSONL. The oracle plugin also offers a separate corrected Gazebo topic when the explicit legacy flag is enabled. This is normal contact force only: tangential friction force is not reconstructed. It is not a replacement for the joint wrist sensor. This contact-report bug does **not** prove the wrist force reading was erroneous.

## Reproduced failure and explanation

The old plan's first oracle repeat fully inserted. Its second repeat reproduced a port blockage: wrist delta 4.217 N at 45.624 s, with a window median 2.073 N. The largest sampled plug/selected-port point force in the 40–48 s interval was 8.828 N. Named pairs identify SC plug housing colliders against selected SC port colliders. Other cable/enclosure contacts exist, including loading during spawn; their episode-wide maxima must not be misread as the insertion cause.

The recent context at the failure has approximately 1.577 mm lateral offset, 13.472 mm axial separation between the recorded tip/port frames, and 0.053 degrees orientation difference. This geometry uses oracle labels solely for diagnosis. Commanded cheatcode Z offset is a different quantity: +7.748 mm. The named contacts and force spike occur together at the port, with TCP motion stopping. This supports a plug/port catch as the explanation of the reproduced failure, and a likely explanation of the older 3.5 N failure; it does not prove that exact historical contact pair.

During retreat, the teacher commanded staged 6/12/18/24 mm withdrawal, but measured motion reached only 0.00623 mm. The old `backoff_distance_achieved_m` field was actually commanded distance. It is renamed `commanded_backoff_distance_m`; admission uses measured distance. The logger now distinguishes these correctly. Release-check samples now update the force-trend history rather than displaying the old trigger window. The configured 5 s release timeout allowed only four stages (24 mm commanded) despite a 40 mm maximum; it did not demonstrate a measured 40 mm retreat. Low recovery stiffness (90 N/m) produces only about 2.16 N restoring force for a 24 mm position error before other dynamics. Insufficient pulling authority is a hypothesis for the failed release, not yet an experimentally isolated cause. Do not claim successful backtracking from commanded distance alone.

The newer explicit tilt-before-return plan passed **2/2** oracle full-insertion repeats, preserving force limits and all collisions. Their maximum sampled plug/port point normal forces were **0.790 and 0.754 N**, compared with **8.828 N** in the reproduced older-route failure. The older plan passed **1/2** instrumented repeats. These are additional trials, separate from the earlier frozen comparison; no single-variable causal or generality claim is made. This establishes a working fixed-scene route, not broad robustness or autonomous BC/RL performance.

## Artifacts and exact reproduction

Root: `outputs/experiments/2026-09-30_force_oracle/`.

- `old_oracle_r1/`: successful frozen older-plan control.
- `old_oracle_r2/`: reproduced blockage, raw contact JSONL, independently stamped wrench/controller snapshots, trace, score, scored MCAP, and all seven camera frame streams.
- `old_oracle_r1/oracle_raw_contact_sample.txt`: raw Gazebo transport independently confirms the reporting defect.
- `old_oracle_r2/oracle_video/port_contacts_20fps.mp4`: two close views with projected contact locations and corrected normal-force labels.
- `old_oracle_r2/oracle_video/sim_45.6s.jpg`: inspected; markers coincide with plug/port interface in both views.
- `old_oracle_r2/force_and_motion.png`: wrist components, contact force, and controller reference versus measured TCP Z. Inspected; withdrawal reference rises while TCP stays still.
- `*/contact_force_summary.json`, `*/corrected_contact_timeline.jsonl`: named pairs, per-point world force vectors and corrected penetration depth. Raw remains untouched.
- `new_oracle_r1/`, `new_oracle_r2/`: explicit tilt-before-return validation. Their close-view videos use the same `oracle_video/port_contacts_20fps.mp4` filename. Successful `new_oracle_r2/oracle_video/terminal_before_reset.jpg` inspected: seated plug visible in both cameras.
- `reference/`: upstream physics source and exact reporting-fix commit metadata.

Commands:

```bash
bash scripts/run_force_oracle_replay.sh \
  outputs/experiments/2026-09-29_moveit_gazebo_pilot/plan_port1_low18_prealign_after_row_insert \
  outputs/experiments/2026-09-30_force_oracle/NEW_OLD_REPEAT
# Swap the plan for the explicit tilt-before-return validation:
# outputs/experiments/2026-09-30_moveit_tilt_before_return/plan_port1_tilt_hold_insert
.pixi/envs/default/bin/python scripts/analyze_force_oracle.py AUDIT_DIR
.pixi/envs/default/bin/python scripts/render_force_oracle.py AUDIT_DIR
.pixi/envs/default/bin/python scripts/plot_force_oracle.py AUDIT_DIR
```

Current contact publication is 100 Hz while physics steps at 500 Hz. All collision pairs are requested, but short impulses between samples can be missed. Reported maxima are sampled maxima, not guaranteed episode maxima. Videos are restricted to scored bag wall-time bounds, excluding post-reset geometry. Camera frames are 20 Hz; plots and projected markers are oracle audit visuals, never deployment observations.

## Latest official toolkit audit

Fetched official `intrinsic-dev/aic` main at e914548 (August 25, 2026). Seven missing commits contain qualification settings, multiple SC ports, lockfile changes, CAD/assembly materials, camera-mount STL, and documentation; none changes the wrist sensor, controller, or solver.

Imported compatible documentation and CAD/camera-mount assets. Retained the official qualification config and remaining migration patches in `artifacts/upstream_toolkit/2026-09-30/`. The multi-port patch moves `sc_port_1` from rail 1 to rail 0 and uses `sc_port_3` for the former rail-1 position. Applying it blindly would change frozen scenes and task labels. It is staged for explicit schema migration, not silently applied to this comparison. The upstream pixi lock patch is retained instead of replacing the fork's customized dependency lock.

## Decision and remaining gate

Use the explicit tilt-before-return route for the next development-start route-generalization checks. Keep old successes and failures. Do not disable plug/port or cable collisions, relax success to contact-only, or promote teacher trials into an autonomous policy claim. Step 4 generalization and Step 5 teacher admission remain open. If force recurs on the new route, inspect named contact geometry and actual retreat motion, then test a bounded recovery-authority or alignment repair; do not collect/training-loop routinely without that evidence.

## Completed checks

The oracle C++ plugin compiled against the pinned image. A ROS-runtime regression with a sensor frame rotated 90 degrees relative to TCP passed: telemetry uses sensor TF and leaves raw-baseline magnitude unchanged. Shell and Python syntax checks and Git whitespace checks pass. Both new-route scores report full insertion (Tier 3 = 75), no excessive insertion force, and no scored off-limit contact. Do not confuse the latter with absence of intended plug/port contact. Final result summary includes scores and artifact SHA-256 hashes. The corrected live-topic extension is compiled; these four recorded trials used the raw-topic plugin, with correction independently applied offline.

Compact evidence is copied to `/home/nas/chmin/aic_archive_20260930/force_oracle/`. The [archive manifest](2026-09-30-force-oracle-archive.json) verifies 30 key raw/derived files with SHA-256; JPEG frames stay local. NAS copying uses `rsync -rt` because the server rejects Unix group changes; no sudo is required.
