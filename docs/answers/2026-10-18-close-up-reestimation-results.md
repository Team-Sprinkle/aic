# Close-up re-estimation results

Implemented and tested the 25 mm re-estimation idea. **It helps SC insertion**, together with the centering and tracking repairs.

## How the update works

- Hold the last commanded robot pose and collect five fresh, synchronized three-camera observations.
- Require position agreement within 0.2 mm and orientation agreement within 0.5°.
- Accept changes from the original estimate only within 1 mm / 3°. Larger disagreement gets a second independent observation window; persistent disagreement causes a safe stop. Automatic reacquisition in this rejection branch is not implemented yet.
- Keep the port frame unchanged, restart the command generator from the same reference, and freeze the updated plug estimate for that attempt.

## Results

- **Matched SC development: 4/4 passes**, covering both ports and zero/five obstacle cards.
- **Public sample SC round: pass** — 15.510 mm depth, 0.141 mm lateral error, 0.469° orientation error, toolkit completion and no scored harm.
- **Complete public sample: 170.57/300, 2/3 strict insertions.** One SFP trial fails. All used-stage p95 timings pass; transport p99 remains 620 ms.

The failed SFP estimate is already 3.53° wrong before contact. Its actual plug-to-gripper roll is outside the original training range. Re-estimating with the same model at close range remains about 3.57° wrong: agreement between images cannot remove systematic bias. Further identical repeats stopped at that failure. The next targeted repair is SFP orientation coverage.

These are explicit camera/public-state controller results, not BC or RL learning. No simulator geometry enters runtime control. Gazebo fixes the held plug to the gripper, so this demonstrates correction of estimation bias, not real grasp-slip recovery. SC depth margins remain small, and its model-wide accuracy qualification is still open. No evaluation or training jobs are currently running; final scenes remain sealed.

[SC public-sample video](/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1/sample_V34_experimental/per_trial/trial_3/five_cameras_20fps_grid.mp4).

[Detailed experiment, gates and next repair](../experiments/2026-10-18-sc-explicit-controller-continuation.md).

Saved response: [2026-10-18-close-up-reestimation-results.md](/data1/chmin/yj/ws_aic/src/aic/docs/answers/2026-10-18-close-up-reestimation-results.md).
