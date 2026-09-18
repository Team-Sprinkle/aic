# Live control, reset, dataset, and policy validation

Date: 2026-09-17. Branch: `feat/hybrid-train`, base `7534090` plus working-tree
changes. Follow-up to the [direct visual actor](2026-09-17-direct-visual-policy.md).
**No new training in this validation. At most two GPUs concurrently (physical
0 for Gazebo, 1 for Isaac), below the four-GPU limit. No sudo.**

## Decision

The direct visual actor now runs through official Gazebo evaluation. Both it
and ACT completed a scored trial, with **no insertion**. Isaac cameras and
terminal episode accounting also ran. Control drift, invalid realized resets,
and questionable demonstration labels prevent a useful encoder comparison yet.
Fix those prerequisites before spending a larger training budget.

Evidence root: `outputs/experiments/2026-09-17_live_validation/` (ignored by Git).
Open the [video gallery](../../outputs/experiments/2026-09-17_live_validation/review/index.html),
[run summary](../../outputs/experiments/2026-09-17_live_validation/validation_summary.json),
or [reset geometry plot](../../outputs/experiments/2026-09-17_live_validation/review/isaac_reset_geometry.png).
The gallery includes videos, one-second snapshots, and first/middle/last views.
`reviewed_source_provenance.json`, `reviewed_source.diff`, and
`reviewed_source_files.tar.gz` preserve the final reviewed changes and container
image IDs. Earlier attempts predate some fixes; their raw logs are retained.
`build_review.py` reproduces the gallery/plot from those saved artifacts.

## Official Gazebo policy trials

One SFP task, card 0 / port 0, from `single_sfp.yaml`; engine limit 60 simulation
seconds, policy limit 25 wall seconds. Both use `delta_pose`, frame `gripper/tcp`,
20 Hz requested policy scheduling, and per-axis limits 0.02 m / 0.2 rad. ACT
executes four actions from an eight-action chunk and retains its legacy
0.5 mm / 0.001 rad deadbands. The direct actor executes one step with zero
deadband. It is the earlier 40-update BC smoke checkpoint, not a mature policy.

| Policy | Complete evaluation | Total score | Tier 3 | Insertion | Final distance |
| --- | --- | --- | --- | --- | --- |
| ACT 175k | Yes | 36.5780 | 15.1420 / 75 | No | 0.07 m |
| Direct visual ResNet, 40 BC updates | Yes | 23.2800 | 7.7867 / 75 | No | 0.09 m |

Completion means policy readiness, engine exit 0, and all configured trials
scored. The official Tier 3 messages explicitly report no insertion. These are
integration results; unequal training budgets and execution settings prevent
an architecture ranking. Tier 2 reported a **0.00 m path despite visible
motion** in both trials; the direct trial additionally reported insufficient
pose samples for jerk. Investigate scorer sampling before using aggregate score
to select models. The final control probe recorded a nonzero path, so this is
not evidence that every runtime measurement has the same scoring defect.

Durable raw results, each with `eval_summary.json`, `attempt_0001/` logs,
`scoring.yaml`, and `rollout/`:

- ACT: `outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k/runtime_eval_live_validation_20260917/175000/`.
- Direct: `outputs/experiments/2026-09-17_direct_visual_bc_smoke/runtime_eval_live_validation_20260917/checkpoint_latest/`.
- Review videos: [ACT](../../outputs/experiments/2026-09-17_live_validation/review/act175k.mp4)
  and [direct actor](../../outputs/experiments/2026-09-17_live_validation/review/direct_visual.mp4).

Policy recordings captured 18 ACT and 19 direct snapshots, approximately one
simulation second apart when observations were requested. The rendered MP4s
hold each snapshot until the next recorded timestamp, then hold the last for
one second. They show the policy observation window, not simulator startup or
scoring after the policy returns. Views were inspected; motion is visible and
does not establish seating.

## Gazebo control diagnostic

`validate_control_rollout.py` sends 20 zero commands, then twelve signed axis
phases. Each phase sends four 0.25 mm / 0.0025 rad commands followed by sixteen
zeros. Translation is measured in the TCP frame at the start of each phase.
The commanded total is 1 mm or 0.01 rad per signed phase. Physics advances
while observations/images/IPC are processed; **20 commands are not necessarily
one simulation second**.

| Run | Initial zero-command behavior | Simulation duration of 20 zeros | TCP displacement |
| --- | --- | --- | --- |
| `gazebo_signed_probe_02` | Retarget measured TCP each tick | 4.298 s | 8.284 mm |
| `gazebo_hold_target_03` | Skip zero-command publication | 4.148 s | 0.482 mm |
| `gazebo_signed_probe_04_chunked_ipc` | Retarget measured TCP; final IPC implementation | 2.348 s | 7.126 mm |

All three completed 260 commands; each saved 261 frames per camera, MP4s,
`camera_timestamps.jsonl`, `frames_1s/`, full observations, and phase measurements.
The smaller hold-target displacement is a diagnostic clue: its initial controller
target/mode is inherited from startup, so this is not a matched controller
comparison or a demonstrated fix. Later signed phases still drift. Rotation
responses have the requested sign; translation direction is confounded by
motion larger than the requested pulse. No action-axis calibration is claimed.

Final probe raw videos use 20 observation frames per playback second; use the
timestamps for physical timing. The gallery contains browser-compatible copies.
Probe 01 failed before observations due to rootless TCP connectivity and is
retained as an infrastructure failure.

## Isaac reset and terminal checks

Three generated episode YAMLs request tip depths −40 mm, −2 mm, and +43 mm,
with zero lateral offset. They derive from an existing v1260 episode with the
position shifted along its insertion axis. Names describe requested geometry;
the physics state must be measured independently. The full-depth target is
45.7997 mm. Original collision geometry was retained. This diagnostic uses
absolute IK targets, the existing XY sign option, contact sensing, constant
zero actions, and zero updates/exploration. It does not evaluate learned actions.

The 100-step probe runs five simulation seconds with an eight-second episode
limit: **zero completed episodes**, and zero strict-success rows.

| Requested depth | Depth after step 1 → 100 | Lateral error after step 1 → 100 | Orientation after step 1 → 100 |
| --- | --- | --- | --- |
| −40 mm | −39.321 → −27.437 mm | 6.951 → 10.526 mm | 0.04869 → 0.06004 rad |
| −2 mm | −1.285 → −0.139 mm | 6.986 → 8.088 mm | 0.04873 → 0.05249 rad |
| +43 mm | 48.394 → 49.451 mm | 1.005 → 7.561 mm | 0.03820 → 0.08609 rad |

Strict criteria are 0.5 mm axial/lateral, 0.03 rad orientation, and module
consistency within 1.0 mm axial / 1.5 mm lateral. A tip beyond target depth is
not success when the other checks fail. These resets do not remain aligned.
The already-inserted CheatCode start fix still needs a valid physical reset and
a dedicated expert rollout; these zero-action probes do not verify that fix.

A separate one-second terminal probe stopped at step 20 after **exactly three
completed episodes**. Each outcome has its own environment/config identity,
length 20, `truncated=true`, reason `time_out`, and `success=false`. This verifies
live terminal accounting and `--max_completed_episodes 3`. Post-step geometry on
that final row is already reset; use `episode_outcomes` for terminal results.

Artifacts under the evidence root:

- `isaac_zero_probe/2026-09-17_20-33-20_reset_zero/`: metrics, audit, config,
  nine camera MP4s, and images at steps 0/20/40/60/80/100 (one second apart).
- `isaac_terminal_probe/2026-09-17_20-36-06_reset_zero_terminal/`: terminal
  records, config, and videos.
- `run_isaac_zero_probe.sh`, `run_isaac_terminal_probe.sh`, `isaac_resets/`:
  exact diagnostic commands and input YAMLs. Use fresh output roots when rerunning.

### Rootless runtime and driver limitation

Dedicated containers `aic_eval_validation_20260917` and
`aic_isaac_validation_20260917` were created with one GPU each. The original
`isaac-lab-base` was not started. Both dedicated containers were stopped after
the checks; their writable layers and all mounted artifacts were retained.

The Isaac image needed editable packages rebound to its mounted paths and
`flatdict==4.0.1`, `safetensors==0.6.2` installed in its own environment. Initial
failures and setup logs remain in the evidence root. Host driver **535.104.05**
failed the RTX minimum-version check; the subsequent camera run used the
existing `patch_isaac_rtx_driver_check.sh` with
`AIC_ISAAC_ALLOW_UNSUPPORTED_RTX_DRIVER=1`. This edits only the new container's
driver-requirements file, with a backup. It is an **unsupported-driver
workaround**, not a host driver upgrade. Cameras then produced images/videos
and both probes completed. The result does not validate every RTX feature.

## Demonstration audit

**Subsequent source-location correction:** the [trajectory inventory](../DATASETS.md)
found both CheatCode sources under `outputs/trajectory_datasets/clean/` and
verified their 140 selected trials' historical Tier 3 scores of 75. The counts
below describe the original audit's narrower `outputs/s3_clean/` lookup; they
are not a statement that those source files are unavailable elsewhere.

Source: `outputs/hf_combined/clean_sfp_to_nic_sc_to_sc_task_conditioned_contact_features_h264`,
with the corresponding `raw32/manifests/accepted.csv`. Output:
[dataset_audit_final/summary.json](../../outputs/experiments/2026-09-17_live_validation/dataset_audit_final/summary.json),
`episode_audit.csv`, and `task_balance.csv`. No source data were modified.

- 546 episodes / 389,907 frames; state 82, action 6, three cameras, 20 Hz.
  No nonfinite state/actions, duplicate episode/frame pairs, or nonincreasing
  episode timestamps were found. There is no reward column.
- **201,753 frames (51.74%) have all-zero actions.** Of these, **189,885**
  record TCP speed above 1 mm/s; median TCP speed among zero-action frames is
  25.82 mm/s. Some zero actions can be legitimate, so blanket deletion is wrong.
- The recorder returns zero for missing/stale or unsupported Cartesian
  commands; joint-command transport does not provide a Cartesian label through
  that path. Historical generation enabled joint transport. This is a likely
  contributor to the mismatch, not proof that every moving-zero frame is corrupt.
  Recover command type, target, age, and time alignment before relabeling.
- **23,824 frames (6.11%) exceed the direct actor's configured action limits**,
  all in translation. Maximum recorded Z magnitude is 112.7 mm, versus its
  20 mm output limit. Resolve command semantics/control rate before changing clips.
- 212 episodes explicitly stopped at the near gate and have historical total
  10. Another 194 have matched scalar totals around 82–96; 140 lack matched
  scalar provenance locally. Scalar selection reports do not establish official
  Tier 3 insertion labels. All insertion labels remain unknown in this audit.
- The current last-5%-episodes holdout contains 27 episodes / 14,799 frames,
  **all SFP card 0 / port 1**. It is not a balanced evaluation of task/port
  generalization. Preserve grouping by source trial and scene in a replacement split.

## Code fixes and checks

- Rootless Gazebo bridge uses a Unix socket through the repository bind mount;
  rootless daemon `--network host` did not share the learner's host localhost.
- IPC now reads chunks, retains partial messages across timeouts, and uses
  `sendall`. The final 260-command camera probe verifies the repaired path live.
- Bridge startup locates the installed Pixi `ros2` executable and loads current
  checkout policy/model code. The image's older installed copy lacked a helper.
- Docker cleanup targets the dedicated container; it no longer kills host
  Zenoh routers. `hold_previous_target` remains an explicit diagnostic option.
- Policy-thread exceptions now log the traceback and return failure. Optional
  `--record-rollout` records raw camera snapshots through the policy observation
  callback. [Renderer](../../scripts/render_rollout_snapshots.py) builds review
  MP4s and contact sheets without running a model.
- Fixed vector rotation in the older geometry probe: normalizing a vector as
  a quaternion destroyed translation magnitude. The new live probe uses SciPy
  rotations and did not depend on that bug.
- **85 runtime regression tests passed** (2.88 s), including TCP/Unix messaging,
  a large image-sized message, timeout continuation, cleanup scope, explicit
  zero-action modes, geometry rotation, snapshot cadence, and visible failures.
  See `runtime_regressions_final.txt`. The renderer also encoded and decoded both
  actual policy recordings; contact sheets were visually inspected.

## Next experiments

1. Trace commands, controller targets, observed TCP/plug poses, and simulation
   timestamps together. Establish a stable Cartesian hold, then repeat signed
   pulses at a measured control rate. Investigate stiffness/gravity compensation
   and relative-target semantics using that trace.
2. Validate each reset after settling, including contact force and module
   consistency. Reject invalid starts before counting an evaluation episode.
   Then test the expert's already-inserted start behavior.
3. Recover valid action labels and explicit insertion outcomes, or regenerate
   a small verified dataset. Record command validity/type/age instead of silently
   substituting zero. Build scene-balanced episode holdouts and evaluate the
   full holdout, including action-limit violations and contact phases.
4. Once these pass, compare scratch, ACT-visual, ImageNet, and DINOv2 backbones
   with the same data, full-action head, training budget, runtime settings, and
   held-out resets. Contact history and connector-focused crops are subsequent
   isolated comparisons. Continue the direct visual architecture; no residual
   adapter is needed.
