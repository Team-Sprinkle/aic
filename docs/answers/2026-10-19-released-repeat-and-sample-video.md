Yes. **276.34/300** was the best score on the released `eval_config.yaml`, in repeat 10. You stopped the batch after **19 scored runs**; repeat 20 was stopped before scoring. Fifteen of the 19 runs inserted all three connectors. The four misses show that the route is promising but not yet fully reliable.

I also ran the unchanged repaired controller once on the exact public `sample_config.yaml`: **287.90/300, with 3/3 insertions**. Here are the new H.264 five-camera recordings at 20 fps:

- [Full sample run](/home/nas/chmin/aic_archives/2026-10-19_repaired_public_sample_video/rollout/five_cameras_20fps_grid.mp4)
- [SFP trial 1](/home/nas/chmin/aic_archives/2026-10-19_repaired_public_sample_video/rollout/sample_trial_1_five_cameras_20fps.mp4)
- [SFP trial 2](/home/nas/chmin/aic_archives/2026-10-19_repaired_public_sample_video/rollout/sample_trial_2_five_cameras_20fps.mp4)
- [SC trial 3](/home/nas/chmin/aic_archives/2026-10-19_repaired_public_sample_video/rollout/sample_trial_3_five_cameras_20fps.mp4)

The sample and released evaluation configurations are different. [Experiment details](/data1/chmin/yj/ws_aic/src/aic/docs/experiments/2026-10-19-released-sfp-sc-visual-route-repair.md#frozen-repetition-and-public-sample-video).
