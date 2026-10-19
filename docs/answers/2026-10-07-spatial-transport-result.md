# Spatial transport test: result

I tested whether keeping the layout of the camera features helps the
transport actor choose the route for the requested SFP card and port.
The new actor reads a 4×5 feature grid from each of the three current
cameras, uses the request code to attend to those grids, and still predicts
complete TCP-body actions. It retains the earlier temporal history.

I trained it against a no-spatial control from the same checkpoint, data,
split, sampler, seed and 3,000-update budget. On held-out teacher frames,
first-action error improved slightly from **1.863 to 1.828 mm p95**.
But swapping the requested card/port while holding the image fixed changed
the action by only **0.903 mm median**, versus **0.883 mm** for the control.
We required at least twice the control's change, since matched teachers
change their early route action by **12.44 mm median**. The spatial model
failed that gate. I therefore did not run it autonomously or promote it.
The last live transport result is still **0/3 safe buffered arrivals**.

I also checked whether the existing paired-request teacher episodes could
directly train the missing route choice. Only **27 early paired rows across
20 scenes** had both nearly identical observations and at least a 3 mm
difference in teacher actions. None had a 10 mm difference. The larger
teacher route differences happen after the two trajectories have already
changed the camera view and TCP state, so treating those later images as
the same input would give misleading labels. I stopped before another fit.

The next targeted data test would hold one starting observation fixed and
obtain teacher commands for alternate requested ports *before* either route
executes. It should first prove that this yields sizable, valid command
differences across independent resets. Handoff, insertion, RL and final-scene
evaluation remain gated.

Full commands, cache audit, checkpoints and metrics:
[Spatial SFP transport experiment](/data1/chmin/yj/ws_aic/src/aic/docs/experiments/2026-10-07-sfp-spatial-request-transport.md).
