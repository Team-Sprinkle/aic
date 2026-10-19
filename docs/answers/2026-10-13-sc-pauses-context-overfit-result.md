# SC pause cleanup, longer action spacing, and overfit test

**The SC training pauses were already cleaned in a separate copy.** The 104 original successful teacher episodes are intact. Their 83 training episodes use a derived dataset that removed 19,570 quiet hold rows; the 21 validation episodes keep the original holds. The scripted waits were 8 and 5 seconds. The actor already looks back up to 60 seconds, so a history window shorter than the pauses does not explain the current failure. Its sparse sampling and mismatch between teacher and live action histories can still matter.

I trained three SC-specific 12-predict/6-execute diffusion actors from the existing checkpoint:

- **100 ms actor:** Targets are 100 ms apart, so it predicts 1.2 seconds and executes 0.6 seconds per decision. It used a new immutable cache derived from the cleaned dataset. Training stopped after a held-out plateau at 12,000 updates.
- **Longer-context 100 ms actor:** Continued the 100 ms actor with 27 observation samples over 70 seconds, versus the prior 17 samples over 60 seconds. Held-out teacher-action error stayed about 6.26 mm p95, and training stopped at a plateau after 5,500 more updates.
- **Intentional-fit 50 ms actor:** Continued fitting the cleaned teacher examples for two 30,000-update stages. Its executed-chunk error on sampled training observations reached **0.91 mm p95**, while held-out teacher-observation error remained **4.46 mm p95**. This tests whether close teacher imitation alone fixes live control.

Both hide the previous teacher action during training and use a zero previous-action input at deployment, since earlier checks found that input encouraged a shortcut. Neither actor receives the true port location or a teacher command while rolling out.

On the **same six nonfinal SC development episodes**, the results were:

| Actor | Reached approach band | Inserted | Median closest TCP-to-port distance | Median inference p95 |
| --- | ---: | ---: | ---: | ---: |
| Existing 50 ms 12/6 | 0/6 | 0/6 | 190 mm | 416 ms |
| New 100 ms 12/6, 60 s context | 0/6 | 0/6 | 162 mm | 393 ms |
| New 100 ms 12/6, dense 70 s context | 0/6 | 0/6 | 135 mm | 394 ms |
| Intentional-fit 50 ms 12/6 | 0/6 | 0/6 | 137 mm | 406 ms |

The new actors generally came closer, but none reached the axial approach band. None of these runs scored contact. The distance is from the **TCP to the opening**, so it is a transport measure rather than plug-tip insertion accuracy. The denser history got closer than the first 100 ms model on five of six layouts, but neither inserted. I also rolled the intentional-fit actor out on **two training scenes**: both missed the approach band and insertion. That points to a closed-loop route/progress problem, not only poor generalization to new scenes. Its later commands repeatedly moved away from the opening, even after it came closer earlier. The 100 ms actor still had decisions about 0.90 seconds apart in Gazebo rather than its nominal 0.60 seconds, so it did not eliminate the inference gap. All new actors remain above the 300 ms p95 inference limit. I did not promote any checkpoint.

Representative recordings: [100 ms one-card](</home/nas/chmin/aic_archives/2026-10-13_sc_longcontext_100ms_eval6_v2/rollouts/llb_0005/five_cameras_10fps.mp4>), [100 ms five-card](</home/nas/chmin/aic_archives/2026-10-13_sc_longcontext_100ms_eval6_v2/rollouts/llb_0009/five_cameras_10fps.mp4>), [dense 70 s five-card](</home/nas/chmin/aic_archives/2026-10-13_sc_dense70_100ms_eval6_v1/rollouts/llb_0009/five_cameras_10fps.mp4>), and [intentional-fit five-card](</home/nas/chmin/aic_archives/2026-10-13_sc_overfit_50ms_eval6_v1/rollouts/llb_0009/five_cameras_10fps.mp4>).

The next experiment should target the first live route divergence at roughly 6–9 seconds: obtain safe correction labels at **actor-visited states** and align training with the real decision timing or overlap inference with execution. More training on the same teacher states or further pause removal is not supported by this result. The [experiment record](../experiments/2026-10-13-sc-pauses-context-and-overfit.md) has data provenance, commands, all artifacts, and per-scene metrics; the [machine comparison](../../outputs/experiments/2026-10-13_sc_pause_models_v1/live_comparison_v2.json) has the six matched results. Final scenes remain sealed.
