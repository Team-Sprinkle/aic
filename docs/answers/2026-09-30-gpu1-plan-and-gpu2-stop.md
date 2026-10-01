# GPU 1's role and GPU 2 shutdown — 2026-09-30

Yes. GPU 1's `train_sc_pose_probe.py` job is an **older diagnostic** trained on 14 Gazebo SC training episodes and validated on 7. It is not part of the current 200-attempt SC Low Lateral Bypass collection or the planned two-actor training. The current plan calls for **new task-conditioned perception training** after suitable data and labels are ready: selected-target pose, actual visibility in each camera, and calibrated confidence, covering both SFP-to-NIC and SC-to-SC. That training has **not started**. We can inspect the old probe's checkpoint as a possible compatible warm start, but its reported near-port error is not good enough to treat it as the finished estimator. I left GPU 1 running because no shutdown was requested for it.

At the user's request, I stopped the old AIC Isaac visualization container `aic_sc_native_gpu2` through rootless Docker. Docker reports it **exited at 2026-09-30 21:20:13 UTC**. Its former GPU process is absent from the post-stop `nvidia-smi` compute-process list. No other container or job was stopped.

Sources: [current project status](../STATUS.md), [two-actor execution plan](2026-09-29-full-experiment-execution-prompt.md), and [earlier GPU provenance audit](2026-09-30-gpu1-gpu2-provenance.md).
