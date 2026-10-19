# SC cable snag: route comparison

I traced the video back to a **development scene made from the predeadline public `sample_config.yaml`**, with randomized card and rail positions. It was not copied from the later released production evaluation configuration. I used only that sample-derived scene family for these route changes. The scene folder's `eval_config.yaml` is a generated config for the development rollout, not the released production file.

The overhead frames show the cable running through the card area as the arm moves. Offline oracle contacts confirm multiple cable links against NIC card mounts, including a PCB contact during the failed tilt. This is consistent with a cable load stopping the arm; it is not evidence that the gripper itself hit a card. I sampled the original and candidate overhead views every second and saved the image sheets.

I named the routes as requested:

- **sc old:** back, sideways/down, along the row, toward the port, tilt.
- **sc skip back:** start with sideways/down, omitting the initial back motion.
- **sc ascend:** lift first, then back at height, sideways/down, and continue.

Neither new route worked with the original left-side corridor on the difficult five-card start. `sc skip back` on the opposite side initially worked, but one of seven development repeats snagged while tilting. More sideways or forward clearance did not fix it. The strongest candidate is **sc skip back with an early tilt**: move to the far-side corridor, rotate the tool while still outside the card row, then sweep toward the port. In the seven SC development cases it achieved **7/7 correct-port visual handoffs**, with no scored harm or local recovery. **6/7 passed the complete gate** because the five-card port-0 run had active perception p95 **391 ms**, over the 300 ms limit. A repeat of that exact case passed at **190 ms p95**. The difficult five-card port-1 case passed twice with early tilt.

This establishes a promising **transport and visual handoff** route for the tested starts, not a full insertion result. The cable can brush cards and the route has not been qualified across the broader independent SC settings. The new route is opt-in; the default remains `sc old` until broader testing. SFP was not changed.

Review:

- [Detailed experiment record](../experiments/2026-10-19-sc-start-side-route-comparison.md)
- Original one-second frames: `/home/nas/chmin/aic_archives/2026-10-19_sc_route_origin_review/frames_1hz/`
- Early-tilt overhead sheet, one frame each second: `/home/nas/chmin/aic_archives/2026-10-19_sc_route_origin_review/sc_skip_back_short_side_early_tilt_v1/overhead_every_second_first35.jpg`
- Early-tilt five-camera 20 fps video: `/home/nas/chmin/aic_archives/2026-10-19_sc_route_origin_review/sc_skip_back_short_side_early_tilt_v1/development_sc_n5_p1_r0/five_cameras_20fps_grid_h264.mp4`
- Seven-case metrics and raw traces: `/home/nas/chmin/aic_archives/2026-10-19_sc_route_origin_review/sc_skip_back_short_side_early_tilt_all7_v1/`
