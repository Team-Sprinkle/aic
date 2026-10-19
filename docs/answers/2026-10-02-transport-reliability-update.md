Transport has improved, but **reliable route following and full insertion are still unproven**.

- I fixed the visual handoff’s stop rule. On a five-card development scene, the actor now returned at the pre-insertion pose instead of running to timeout: **0.58 mm position error, 2.08° orientation error, 178 ms inference p95**.
- I tested smaller transport commands. A 30 mm cap reduced route error on one five-card scene from **27.6 to 19.5 mm p95**, but raised it on a different four-card scene from **19.3 to 42.1 mm p95**. I rejected that cap. Route error here means distance from the recorded teacher path, measured afterward; p95 means 95% of sampled positions were closer than that distance.
- Teacher and actor used the same previsual Cartesian gains, so a simple gain mismatch does not explain the drift. I then trained two phase-waypoint probes on **83 training and 21 held-out SC episodes**. The better probe reached **95.0% phase accuracy** but **22.4 mm waypoint error p95** when selecting its own phase. It missed the 20 mm gate, so I did not connect it to the robot.
- Saved wrist-camera frames show that the full five-card row is often outside the view or partly hidden during transport. This is a sampled visual check, not a complete visibility measurement. It points toward a retained scene map or a deliberate observation pose as the next bounded test.

The [experiment record](/data1/chmin/yj/ws_aic/src/aic/docs/experiments/2026-10-01-transport-reliability-continuation.md) has the commands, metrics, checkpoint paths, failed runs, and example frames. [Status](/data1/chmin/yj/ws_aic/src/aic/docs/STATUS.md) and the other main docs are updated. No final scenes were opened, no RL or insertion training was started, and no project training or evaluation job is running now.

Saved copy: [this answer](/data1/chmin/yj/ws_aic/src/aic/docs/answers/2026-10-02-transport-reliability-update.md).
