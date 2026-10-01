# GPU 1 pose-probe shutdown — 2026-09-30

I stopped the rootless Docker container `aic_sc_fix_gpu1` at **2026-09-30 21:21:10 UTC**. Docker reports `exited`, and the training process PID 1954824 is gone. The saved best checkpoint (47 MB) and metrics remain at `/var/tmp/aic_sc_fix_20260924/sc_pose_probe_visual/`.

The stopped job was an older diagnostic SC visual pose probe on 14 training and 7 validation episodes. It is **not part of the current Low Lateral Bypass data collection or the planned task-conditioned perception and two-actor training**. Its checkpoint might be examined as a compatible initialization, but the current plan does not depend on its final epoch, and its 1.49 mm near-port lateral p95 error does not meet the 0.5 mm insertion corridor.

GPU 2's older visualization container was stopped separately, as recorded in [the prior answer](2026-09-30-gpu1-plan-and-gpu2-stop.md). No current collection job was interrupted.
