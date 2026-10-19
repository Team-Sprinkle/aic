## Documentation updated

The main docs now explain the major milestones, how they were achieved, the measured results, and their limits:

- [STATUS.md](/data1/chmin/yj/ws_aic/src/aic/docs/STATUS.md): rule-based transport, verified handoff, full-connector-pose insertion, and remaining gaps.
- [EXPERIMENTS.md](/data1/chmin/yj/ws_aic/src/aic/docs/EXPERIMENTS.md): compact milestone/comparison table.
- README, project overview, handoff plan and artifact map now point to the current pipeline and results.

## Complete sample evaluation: **165.42 / 300**

I ran the unchanged `sample_config.yaml` from its original starting pose, including transport, perception, approach and insertion. No retraining or backtracking was enabled; reserved final scenes stayed sealed.

| Trial | Toolkit score /100 | Physical outcome |
| --- | ---: | --- |
| SFP → NIC card0 |48.63|At mouth; 3.50mm lateral error |
| SFP → NIC card1 |35.48|At mouth; 2.40mm lateral error |
| SC → SC port1 |81.31|Simulator completion, but only 1.97mm deep |

**All 3 reached the near-port handoff; none passed our strict full-insertion check.** The toolkit awards proximity points and SC completion credit, so 165points does not mean reliable insertion.

### What worked

- Rule-based transport and requested-port acquisition worked in all three trials.
- Port-position estimates were accurate: approximately 0.08–0.14mm error.
- Both SFP trials handed off about 25mm from the opening, within 0.32mm lateral error.
- No scored harmful-contact or excessive-force penalties occurred.

### What needs fixing

1. **SFP insertion loses lateral alignment.** Good handoff/perception deteriorates under actor control. Commands can request ~40mm forward while the plug remains at the mouth. Actual tip/port-cage contacts were recorded; cable/mount touches also occurred, but this does not establish cable snag.
2. **Controller handover still jumps:** approximately 3/5/10mm across the three trials. Test continuous command handover and progress-aware forward limits.
3. **SC plug orientation remains inaccurate:**3.36° tilt at handoff; grasp estimation is 2.04° off. Repair grasp-angle coverage and the approach's nominal-gripper orientation check. Its simulator completion must not be treated as full physical insertion.
4. **Speed remains poor:** all three lost duration points. Correct alignment/control first, then improve speed and route efficiency.

The earlier successful SFP insertion remains encouraging, but this complete sample confirms it is not yet reliable. Step 5 stays open; no further training was started.

[Detailed results and artifacts](/data1/chmin/yj/ws_aic/src/aic/docs/experiments/2026-10-17-full-connector-sample-configuration.md). All three regular 20 fps videos are under `/home/nas/chmin/aic_archives/2026-10-17_full_connector_sample_configuration_v2/per_trial/trial_{1,2,3}/five_cameras_20fps_grid.mp4`.

Saved response: [milestones and sample results](/data1/chmin/yj/ws_aic/src/aic/docs/answers/2026-10-17-milestones-and-complete-sample-results.md).
