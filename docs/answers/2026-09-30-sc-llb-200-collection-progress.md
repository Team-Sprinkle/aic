# SC-to-SC Low Lateral Bypass collection — running status

**Snapshot: 2026-09-30, after case `llb_0033`.** The requested collection is **in progress**, not finished. There are **34/200 attempted settings**, **20 official full insertions**, and **33 valid scored replay bags**. No BC training has started from this collection, and the reserved final scenes remain sealed.

The first 30 attempts were the predeclared pilot. They produced **17 full insertions**, exceeding the required ten. Each card count from one through five had six attempts; successes were 4/6, 4/6, 4/6, 2/6 and 3/6. The two target SC ports had 7/15 and 10/15 successes. One attempt (`llb_0024`) failed model startup before executing the teacher and has no bag; it remains in the 30-attempt denominator. The other failed cases mostly reached the port and stopped during guarded insertion/contact recovery. These are privileged teacher rollouts, not autonomous learned-policy successes.

The continuation has 170 deterministic settings: 85 per selected SC port and 34 per card count. It varies board pose, individual card positions and yaw, both SC rail positions, cable grasp offsets, and robot start joints. The pilot varied the scene but **did not vary the actual Gazebo robot joints**: the official first-trial launch used fixed xacro values. I repaired that for the continuation with per-case robot descriptions and verified every scored start against bag `/joint_states`. Cases `llb_0030`–`llb_0032` all fully inserted with truly varied starts. The first run matched its planned start within 0.00041 rad. A calibration check showed elbow and wrist-1 respond to their own xacro offsets; I corrected the predicted MoveIt starts for the 167 unplanned cases before `llb_0033`. The corrected case `llb_0033` matched within 0.00052 rad, although it did not insert.

The collector runs sequentially on **GPU 7** under rootless Docker. Each continuation run is archived to `/home/nas/chmin/aic_llb_collection_20260930/continuation170/` and verified before its local run folder becomes a symlink. The process will stop if a scored robot start fails validation or if fewer than ten of the first 30 *varied-start* continuation settings fully insert. It will otherwise continue toward case `llb_0199`. The detached finalizer will write `collection_final_report.json`, update the experiment record, and create `outputs/trajectory_datasets/sc_llb_score_verified/manifest.json` containing only full insertions with a scored bag and action trace. It will not merge them into the older canonical expert set or start BC training.

Artifacts:

- [Experiment record](../experiments/2026-09-30-sc-llb-collection.md)
- `outputs/experiments/2026-09-30_sc_llb_collection/summary.json` — exact current scores and per-case rows
- `outputs/experiments/2026-09-30_sc_llb_collection/pilot30/collection_progress.json` — first 30 attempts and gate
- `outputs/experiments/2026-09-30_sc_llb_collection/continuation170/collection_progress.json` — continuation and measured-start audits
- `outputs/experiments/2026-09-30_sc_llb_collection/supervisor_status.json` — detached job status
- `outputs/experiments/2026-09-30_sc_llb_collection/continuation170/start_prediction_repair.json` — start-label correction
- [Varied-start success video](../../outputs/experiments/2026-09-30_sc_llb_collection/continuation170/runs/llb_0030/video/all_views_20fps.mp4), [Five-card success video](../../outputs/experiments/2026-09-30_sc_llb_collection/pilot30/runs/llb_0009/video/all_views_20fps.mp4), [two-card success video](../../outputs/experiments/2026-09-30_sc_llb_collection/pilot30/runs/llb_0001/video/all_views_20fps.mp4), [failed near-port video](../../outputs/experiments/2026-09-30_sc_llb_collection/pilot30/runs/llb_0013/video/all_views_20fps.mp4)

The recorded videos use 20 fps H.264. The overhead and side cameras are diagnostic views; they are not inputs to a learned policy.
