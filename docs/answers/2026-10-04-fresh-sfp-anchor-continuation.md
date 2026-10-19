# Fresh SFP anchor continuation

I resumed the frozen SFP perception check. I fixed ten new, non-final Gazebo reset configurations before evaluation, covering all five cards and both ports. Their reset hashes do not overlap the 50 training scenes, ten earlier development scenes, or the reserved source audit partition. The four perception models were left unchanged.

The first scene, card 0/port 0 with five cards, ran successfully: the **CheatCode teacher** fully inserted and produced 52 synchronized three-camera observations. On those images, all four frozen heads made **zero accepted wrong-port predictions**. Their accepted port-position errors at the 95th percentile were **1.58–4.13 mm**. The combined rule found the requested port on its first sampled image, with no >10 mm disagreement between models. The 20-sample offline image-processing profile took **190.8 ms p95**. This timing excludes ROS, camera transport, the actor and controller.

I also repaired bag extraction so it can run on the host without starting another Docker container. I checked it against a prior bag: the commands, ROS streams, transforms, reset identity, score and timestamps matched the old Docker extraction exactly; only host versus container file paths differed.

**The ten-scene gate is still open.** After the first full Gazebo run, the account's kernel key quota stayed at **200/200**. The remaining nine Gazebo containers cannot start. The runner now checks this before launching, so it preserves the frozen scene plan and does not create partial rollouts. The earlier minimal-container test had returned its key after exit; this full Gazebo run showed that sequential launches are not reliably possible at the current quota. I left the active Isaac job and all final scenes untouched. No actor or RL training was started.

The next action is to raise the UID 1008 kernel key quota or have the administrator safely clear stale keys, then rerun the **same nine planned scenes** and measure complete live inference latency. The promotion gate remains zero wrong ports, at least nine of ten early acquisitions including middle cards, accepted coarse position p95 ≤20 mm, and complete live p95 <300 ms. One successful scene does not meet it.

[Full experiment record, commands and artifacts](../experiments/2026-10-04-sfp-fresh-anchor-shadow.md). The archive is `/home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/`.
