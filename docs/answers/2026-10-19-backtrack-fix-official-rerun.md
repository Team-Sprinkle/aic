# Backtrack fix and released evaluation rerun

I fixed the SC transport trigger so a **missed visual port lock no longer causes a long route reversal**. This is the correction to the 322-waypoint backtrack you questioned. The short near-port backoff/lateral retry is separate and was unchanged.

I reran the released three-trial `eval_config.yaml` with the selected V56 controller, changing only the transport recovery source. It scored **98.856989/300 with one insertion**, compared with **98.826833/300 with one insertion** before. SFP trial 1 inserted. SFP trial 2 again stopped on force before port acquisition. SC trial 3 now stopped at its visual-acquisition timeout at 82.15 simulated seconds and made **zero route backtracks**. It still did not insert.

The SC controller tried its bounded view search, but none of 114 visual checks accepted the requested port: 47 failed pose geometry, 38 exceeded the crop margin, and 29 lacked the required projection in every view. The saved late wrist-camera frames show blue SC port structures, yet the estimator did not validate a pose. The fix therefore removed wasted movement and time, **not the underlying perception failure**. It did not exercise the local backoff or lateral retry, so this run cannot judge their effectiveness.

Full score, code hash, trace, terminal measurements and videos: [experiment record](../experiments/2026-10-19-visual-timeout-route-retreat-fix.md). Artifact folder: `/home/nas/chmin/aic_archives/2026-10-19_V56_visual_timeout_no_route_retreat_official_match_v1/`. The released scenes are already exposed; I kept the reserved final split sealed and did not tune the controller from this rerun.
