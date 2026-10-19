# October 19: short transport retreat and lateral route retry

## Question and control change

Before this change, the insertion supervisor could back away locally, but rule transport had only a raw-force stop and an older large route-retreat path. An earlier bug also treated a visual-acquisition timeout as blockage; that bug was corrected separately. This experiment adds the user's requested **local** maneuver to transport for both SC-to-SC and SFP-to-NIC. It does not train an actor or alter perception weights.

At the initial quiet free hold, the controller freezes a gravity-compensated force reference from ten samples. On each transport decision it compares current load with that reference. A local contact is declared only when load change is at least 8 N on three observations spanning roughly 0.45 s **and** measured progress toward the current waypoint is at most 2 mm while its remaining error is at least 15 mm. The existing raw-force emergency stop remains. Visual timeout alone is not a contact signal. The contact gate uses public TCP and wrench observations, never simulator contact or port geometry.

After a trigger, the controller estimates the direction of recent *measured* TCP travel. It retreats 8 mm against that direction, with a 12 mm maximum if load does not clear. It may briefly hold the reached pose to let cable load relax. Once unloaded, it moves horizontally perpendicular to the recent travel direction; it requires at least 4 mm measured sideways movement, at most 3 mm axial drift, and low load. If the first side loads up, it returns to the retreat pose before trying the other side. It then shifts the remaining intermediate route waypoints by the measured sideways displacement and continues the **same stage**, retaining the visual context. It does not shift the final requested-port acquisition goal. Two local retries are allowed. An unsafe or unmeasured move aborts rather than pressing further into contact.

Implementation: `aic_example_policies/aic_example_policies/ros/RunSampleTransportRecovery.py` and `scripts/transport_local_recovery.py`. Four focused tests are in `aic_example_policies/test/test_transport_local_recovery.py`. The old long route-retreat routine remains in source for historical diagnosis but is not called by the current path. No privileged geometry is passed to the runtime policy; oracle contact logs below are post hoc only.

## Tests and development trace

The focused monitor/axis tests pass. A nominal SFP development transport completed with zero false contact triggers. Several bounded SC variants showed the safety gate working: an exact 8 mm sideways-arrival demand was too strict for compliant motion; an exploratory side increased load and aborted; another contact remained loaded after 12 mm retreat and aborted. These failed attempts are retained under the `2026-10-19_transport_local_recovery_{sc_contact_v1,sc_partial_lateral_v2,sc_both_sides_v3}` NAS roots. Accepting at least 4 mm *measured* lateral travel and a stationary unloading hold yielded the selected diagnostic below. This is a targeted controller repair, not a threshold sweep on reserved final scenes.

In `development_sc_n5_p1_r0` (five cards, SC port 1), the detector fired at simulator time 17.45 s on a 14.01 N load change plus stalled route progress. The measured retreat was **8.403 mm**. The measured sideways displacement was **9.492 mm**, with **−0.250 mm** drift along the precontact travel direction. Load during retreat peaked at 14.01 N and had a 1.64 N median; during lateral motion it peaked at 1.97 N. The controller resumed `shift_behind_row_low` and reached a qualified correct-port visual acquisition at 40.45 s. Terminal post-hoc geometry was roughly 154.10 mm before the opening, 3.48 mm lateral, 2.21° orientation; insertion did not run. The independent score recorded a transient 27.13 N maximum, below its one-second force-penalty duration. Offline oracle events around the trigger include cable link 5 against NIC card mount 0. This is one successful *transport recovery*, not a reliable insertion result.

The five-camera 20 fps video is `/home/nas/chmin/aic_archives/2026-10-19_transport_local_recovery_sc_settle_v4/development_sc_n5_p1_r0/five_cameras_20fps_grid.mp4`. Inspected stills are `review_before.jpg` and `review_after.jpg` in the same experiment root. Exact measured motion and event timing are in `development_sc_n5_p1_r0/runtime_trace.jsonl`; `summary.json`, the scene manifest, scoring, source snapshots, hashes, and launch environment are retained there. The visual displacement is subtle; the measured TCP trace establishes the motion.

The earlier released SFP force-stop trace was replayed **offline** through the new detector using its frozen initial force reference. It would have triggered near 24.15–24.35 s at approximately 16.6–16.8 N load change, before the old raw-force stop at 24.5 s. `replay_gate_diagnostic.json` stores this check. It is not a live SFP recovery result.

## Released three-round comparison

A single matched rerun of the already exposed released `eval_config.yaml` changed only `RunSampleTransportRecovery.py` among 160 selected existing inputs and added the new helper. Models and scenes remained frozen. Previous visual-timeout-repair version: **98.856989/300, 1/3 strict insertions**. New transport-local-recovery version: **98.749703/300, 1/3**. Trial 1 SFP inserted to 46.756 mm depth with 0.225 mm lateral and 0.251° orientation error; trial 2 SFP and trial 3 SC timed out on visual acquisition. The small score difference is not a demonstrated degradation or improvement. This rerun had **zero transport contact/recovery events**, three frozen load calibrations, and zero false triggers. It therefore provides no causal evidence that the new maneuver improves insertion on these three starts. No route backtrack occurred. Used-stage inference p95 was below 300 ms; transport active-perception p95 was about 129.8 ms and fresh acquisition p95 about 221.63 ms.

Artifact root: `/home/nas/chmin/aic_archives/2026-10-19_V56_transport_local_recovery_official_v1/`. `command.json` contains the exact rootless launch command and environment; `source_delta.json` and `plan.json` freeze hashes/scope; `transport_local_recovery_comparison.json`, `analysis_summary.json`, `manual_recorded_image_review.json`, and `rollout/results/scoring.yaml` report the outcomes. Videos are `per_trial/trial_{1,2,3}/five_cameras_20fps_grid.mp4`; raw trace and pre-reset scoring remain under `rollout/`. These released scenes are exposed diagnostics. The reserved internal final scenes remain sealed.

## Decision

Keep local transport recovery as a bounded safety mechanism: the SC trace demonstrates a real unloaded short retreat, sideways move, and same-route continuation. Do not claim an insertion gain or promote hierarchical retreat from these data. The current three-round bottleneck is missed visual acquisition on the two failed trials, with no contact event for this maneuver to address. A future matched contact-rich development set should test both lateral signs, repeatability, force/harm, visual reacquisition, and full insertion before claiming transport-recovery value.

## Exact entry points

```bash
.pixi/envs/default/bin/python -m pytest -q aic_example_policies/test/test_transport_local_recovery.py
python3 -m py_compile aic_example_policies/aic_example_policies/ros/RunSampleTransportRecovery.py scripts/transport_local_recovery.py
```

The development launcher and complete environment are in `development_sc_n5_p1_r0.command.json` under the development artifact root. The released comparison launcher and exact environment are in `command.json` under the released comparison root. Preserve those manifests rather than substituting a current image or checkpoint.
