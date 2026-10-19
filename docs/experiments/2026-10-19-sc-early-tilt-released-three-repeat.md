# SC early-tilt route: released-config three-repeat check

This was a **local replay of the publicly released qualification `eval_config.yaml`**, not a cloud submission or sealed final test. The frozen V56 runtime used `sc_skip_back` plus early tilt and the SC-only opposite-side corridor (`outside_board_x_m=+0.12`). The runner verified both route variables inside the container and the released-config hash. SFP settings were unchanged. Three sequential rootless runs used GPU 4. No released-scene geometry, oracle trace or video was used for tuning.

Command: `.pixi/envs/default/bin/python scripts/run_sc_route_official_three_repeats.py`. Complete frozen inputs, hashes, commands, engine logs, scores and `summary.json`: `/home/nas/chmin/aic_archives/2026-10-19_sc_early_tilt_official_three_v1/`.

| Repeat | Score / 300 | SFP 1 | SFP 2 | SC 3 | SC force above 20 N |
| --- | ---: | --- | --- | --- | --- |
| 1 | 98.7311 | Inserted, 96.7311 | Incomplete, 1 | Incomplete, 1 | 0.20 s; peak 26.23 N |
| 2 | 98.7490 | Inserted, 96.7490 | Incomplete, 1 | Incomplete, 1 | 0.15 s; peak 22.69 N |
| 3 | 98.6188 | Inserted, 96.6188 | Incomplete, 1 | Incomplete, 1 | 0.12 s; peak 74.92 N |

All force excursions were below the one-second score-penalty threshold. “No contact detected” is a separate scoring category, not proof that there was no load. The official engine log reports the SC task pending/not completed, but no route-stage or contact-location cause. The earlier V56 plus local-recovery replay was 98.7497/300, 1/3 insertions. **There is no full-task improvement.** The 22/24 sample-derived SC result measured correct-port handoff only; it never established insertion.

The development pilot used 0.055 m/s translation, 0.25 rad/s rotation and a 1.0 s hold. This frozen replay retained V56's 0.1 m/s, 0.5 rad/s and 0.25 s to preserve its baseline. Speed is therefore an unresolved transfer variable, not a proven cause. Different scene distributions and downstream visual/insertion failures also remain possible.

Next diagnostic: on **fresh predeadline-sample-derived development starts**, compare the same SC route at pilot and V56 gains, logging only public force, measured TCP motion and policy stage. Include both ports, multiple card counts and starts amid the row. If transport stalls, test SC-only slower gains or force/progress-triggered alternate route; if visual lock or insertion fails, repair that stage separately. Require representative **full insertions** with acceptable force before another frozen released-config score check. The SFP-2 failure is separate from this SC route.

Official references: [local results/logs](https://github.com/intrinsic-dev/aic/blob/main/docs/getting_started.md), [portal results/logs](https://github.com/intrinsic-dev/aic/blob/main/docs/submission.md), [released-config scope](https://github.com/intrinsic-dev/aic/blob/main/docs/qualification_phase.md).
