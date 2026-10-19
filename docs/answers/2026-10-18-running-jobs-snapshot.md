## Running jobs

Checked at 2026-10-18 05:25 UTC.

**No current training, dataset collection, or Gazebo evaluations are running.** The recent evaluation and recovery jobs have finished.

| Process | Runtime | Resources | Purpose |
|---|---|---|---|
| PID468750, `probe_target_reward.py` |24days17hours|GPU0:1.35GiB,0%GPU utilization; roughly one CPU core|Old Isaac SC collision diagnostic with five cards and gripper-base collisions disabled; not current training|
| PID3174618, waiting shell |25days16hours|No GPU allocation shown|Old waiter for an RPDP BC analysis; no analysis currently executing|

One rootless Docker container remains: `aic_isaac_world_policy_20260920`. GPUs1–7 are idle, with no compute processes shown. GPU0also reports0%utilization at this snapshot, although the legacy diagnostic retains memory.

The legacy diagnostic appears stale given its250-step command and24-day runtime; I have not stopped it. **There is currently no background job advancing the latest experiment plan.**
