# Transport backoff and lateral retry — October 19

Yes. Previously the short local backoff was available during insertion; transport could stop on force but did not do the same short sideways retry. I added it to transport for both SFP-to-NIC and SC-to-SC.

The trigger now checks **both** a sustained rise in measured force and loss of progress toward the current transport waypoint. When it fires, the robot backs up along the direction it had just traveled (usually about 8 mm, at most 12 mm), moves sideways relative to that direction, and continues the same route from the new side position. It measures the actual movement and stops if force stays high or clearance is not achieved. A missing camera lock alone cannot trigger backoff.

In one five-card SC development episode, this unloaded a real cable/card contact: the robot retreated 8.4 mm, moved 9.5 mm sideways, and reached the correct visual handoff. The [five-camera recording](</home/nas/chmin/aic_archives/2026-10-19_transport_local_recovery_sc_settle_v4/development_sc_n5_p1_r0/five_cameras_20fps_grid.mp4>) shows the run. This was transport only; it did not test insertion.

I reran the three released evaluation rounds with the fixed models and scenes. The score was **98.75/300, one insertion out of three**, essentially the prior **98.86/300, one out of three**. No transport blockage occurred in that rerun, so the new recovery never activated. The remaining two rounds failed to acquire the port visually. Thus the new behavior is implemented and has one live recovery demonstration, but it has not yet improved full insertion in the three-round test.

[Detailed experiment, exact commands, traces and all artifacts](../experiments/2026-10-19-transport-local-recovery.md)
