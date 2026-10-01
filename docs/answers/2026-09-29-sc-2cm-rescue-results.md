# September 29: the 2 cm route test

I found the exact settings for the one failed run in the earlier seven-run test: SC port 0, five cards at the leftmost positions, seed 51500, and the route's forward margin set to −1 cm. I reran that scene and tested several 2 cm changes. These were Gazebo CheatCode teacher rollouts using the existing scripted route, not a trained policy or a MoveIt path.

| Route change on that same scene | Full insertions |
| --- | ---: |
| Original failed setting, including the archived run | **4/7** |
| 2 cm farther behind the cards | 2/2 |
| 2 cm farther outside the card row | **4/5** |
| Both of those 2 cm changes | 2/2 |
| 2 cm farther forward than the failed setting | 1/2 |

The original setting's mixed results mean its failure was not deterministic. The outside-lane change is the most promising tested margin, but 4/5 on one scene does not establish a reliable route.

I compared a full and a partial insertion with that outside-lane margin. The partial run actually had the **largest sampled cable-to-card gap** in its group, so I cannot attribute that failure to a cable snag. At the port entrance, the plug was about **0.57 mm off laterally**; it then stopped after roughly **3.2 mm** of insertion and recorded no insertion event. The successful run crossed about **0.04 mm off laterally** and reached roughly **15.6 mm**. This points to a near-port alignment or insertion stall for that particular failure. Other partial runs can have a different cause. The sampled cable gap omits cable thickness and protrusions, and the videos do not prove the cable stayed entirely clear of card tops.

Review videos, at 20 fps:

- [Original-route partial](../../outputs/experiments/2026-09-29_sc_low_cross_2cm_rescue/baseline_partial/all_views_20fps.mp4)
- [2 cm outside-lane partial](../../outputs/experiments/2026-09-29_sc_low_cross_2cm_rescue/lane20_partial/all_views_20fps.mp4)
- [2 cm outside-lane full insertion](../../outputs/experiments/2026-09-29_sc_low_cross_2cm_rescue/lane20_full/all_views_20fps.mp4)

The [experiment report](../experiments/2026-09-29-sc-low-cross-2cm-rescue.md) has the commands and interpretation; the [machine-readable results](../experiments/2026-09-29-sc-low-cross-2cm-rescue.json) include all 17 new runs. Scored bags and scene settings are backed up under `/home/nas/chmin/aic_archive_20260929/sc_low_cross_2cm_rescue_raw/`.

I would stop expanding margins on this one scene. The next focused test should separate free-space routing from final insertion: fix the near-port handoff, then compare a planner-generated around-card route with the same insertion controller on new SC ports and starts. MoveIt was investigated but **no MoveIt path was generated or run**: the current backend needs live joint state and the actual card and fixture collision objects in its planning scene. No model training or final-scene evaluation was done here.
