# Repair/evaluation cycle result

**Reached 290.19/300 on the unchanged public sample.** The five initial cycles reached281.05; continuing toward your requested target took18completed full evaluations in total.

| Final round | Score /100 | Physical insertion |
| --- | --- | --- |
| SFP1 | 97.18 | Pass |
| SFP2 | 96.56 | Pass |
| SC | 96.46 | Pass |

The main improvements were shorter visually checked transport, smoother approach/orientation control, correct camera timing, less redundant processing, and retaining enough quiet SCimages for orientation correction. Existing perception weights stayed unchanged; insertion uses the explicit visual controller, not the BCactor or RL.

All three physical checks and every used-stage p95 inference limit pass. SCdepth is15.529mm. A preceding290.09run failed the15.5mm depth gate and remains recorded as a failure.

I stopped the loop and updated the main docs. This is one selected public-sample pass, not proof of reliability across all settings. Final scenes stayed sealed.

[Complete results and changes](../experiments/2026-10-18-five-repair-cycles.md).

Recorded20fps five-camera videos:

- [SFP1](/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1/cycle_18/per_trial/trial_1/five_cameras_20fps_grid.mp4)
- [SFP2](/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1/cycle_18/per_trial/trial_2/five_cameras_20fps_grid.mp4)
- [SC](/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1/cycle_18/per_trial/trial_3/five_cameras_20fps_grid.mp4)

Selected source/model/calibration manifests, commands and metrics are pinned under`/home/nas/chmin/aic_archives/2026-10-18_five_repair_cycles_v1/cycle_18/`. No new jobs are queued. Transportp99remains323.23ms despite passingp95158.10ms; fresh-perception p95is220.44ms. SCdepth margin is only0.028748mm, and the older out-of-support SFPgrasp still needs independent future validation.
