Both failed rounds now insert in the released `eval_config.yaml` local replay. Four complete runs scored **272.70, 275.44, 273.14 and 276.26/300**, with **12/12 physical insertions**. The previous configuration inserted only 1/3 per run.

- **SFP:** A valid three-camera port estimate was rejected because the port was 254 mm from the sample-based nominal position. I widened the guarded search, guided transport from consistent visual estimates, and lifted the fallback route.
- **SC:** The actual board was about 150 mm and 85° from the sample-board pose. A new wrist-camera RGB marker check estimates the board layout before routing beside the card row. The reviewed [five-camera video](/home/nas/chmin/aic_archives/2026-10-19_sc_sfp_released_repair_cycle8_sc_marker/repeat_1/rollout/five_cameras_10fps_grid.mp4) shows the detour and insertion.

No model was retrained; these runs use rule transport and the explicit visual insertion controller. Active transport-decision p95 stayed below 300 ms, with one trial at 299.4 ms. These are repeated tests on a **published, exposed configuration**. Broader SC reliability remains open: 2/4 separate sample-derived SC starts inserted. The reserved internal final scenes remain sealed.

The changes and records were committed in three reviewable commits and pushed to `origin/feat/hybrid-train`.

[Experiment record](/data1/chmin/yj/ws_aic/src/aic/docs/experiments/2026-10-19-released-sfp-sc-visual-route-repair.md) · [Current status](/data1/chmin/yj/ws_aic/src/aic/docs/STATUS.md)
