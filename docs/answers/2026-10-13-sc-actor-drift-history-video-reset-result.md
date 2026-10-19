# SC actor drift: what changed and what happened

The cards and board disappearing at the end of the earlier videos was **recording after the trial**, not the actor losing the scene. Gazebo deleted `task_board` and `cable_1` during normal reset after the actor returned, while the camera recorder kept running. I made separate videos that stop at the actor's last command and inspected their last frames; the board and orange cable remain visible. Original recordings and scored bags are preserved.

Your observation about the route was useful: the earlier 27-history model and intentional-fit model looked less prone to the final overshoot than the other examples. I tested a finer 38-request history, a deliberately memorized version of it, extra late-route training weight, and faster inference. All use the same verified SC split of **83 training and 21 validation episodes**, and all were judged by autonomous Gazebo rollouts without teacher/oracle geometry commanding the actor.

| Actor | Matched SC scenes | Median closest TCP-to-port distance | Reached approach band | Full insertion |
| --- | ---: | ---: | ---: | ---: |
| Earlier 27-history dense actor | 6 | **135 mm** | 0/6 | 0/6 |
| Finer 38-request history, trained to validation plateau | 6 | 141 mm | 0/6 | 0/6 |
| Finer history, deliberate overfit stage 2 | 6 | 148 mm | 0/6 | 0/6 |
| Final deliberate overfit plateau stage | 3 | 145 mm | 0/3 | 0/3 |

The final three are a subset of the six; the earlier dense actor's median on that same subset was 141 mm. These are 70-second transport diagnostics, so the insertion counts refer to that bounded window.

The final overfit actor reached **0.49 mm p95 error on recorded training actions**, but still missed on autonomous rollouts. Its first stage also missed the approach band on **two scenes from its own training set**. Extra fitting of teacher frames therefore did not solve the error that accumulates after the actor takes its own actions. A late-phase sampling repair and a faster six-denoising-step inference test each missed 0/3 approach bands too. Faster inference brought median p95 to 296 ms but did not restore the route.

The likely repair target is the first small route departure around **7–9 seconds**. On one five-card scene, between about 3 and 12 seconds the earlier dense actor's measured TCP traveled 238 mm while the successful teacher traveled 304 mm. After this early lag the model sees states unlike its teacher training frames, and its commands become too small or point into the wrong final route. This is a measured pattern, not proof that one issue alone causes every failure. The 38 requested history positions also produced only **14 distinct late training frames** on the median episode because the cleaned cache thinned frames around holds.

I did **not** shorten or regenerate the dataset for speed. First the actor must reliably reach the safe approach region. The official toolkit awards a duration bonus only for a completed task, with the bonus reaching zero at 60 seconds; the six matched successful teachers took about 63–101 seconds, so they got zero duration bonus but still fully inserted. The 70-second actor runs here are route diagnostics and shorter than some teachers. Once transport works, make an immutable copy of the verified SC episodes, shorten holds while keeping every image/action aligned, validate full insertion, and compare speed before replacing training data.

Watch the same five-card scene with the teardown frames removed:

- [Earlier dense-history actor](</home/nas/chmin/aic_archives/2026-10-13_sc_dense70_100ms_eval6_v1/rollouts/llb_0009/five_cameras_10fps_active.mp4>)
- [Finer-history actor](</home/nas/chmin/aic_archives/2026-10-13_sc_fine70_100ms_validation_eval6_v1/rollouts/llb_0009/five_cameras_10fps_active.mp4>)
- [Deliberately overfit, plateau checkpoint](</home/nas/chmin/aic_archives/2026-10-13_sc_fine70_100ms_overfit_plateau_eval3_v1/rollouts/llb_0009/five_cameras_10fps_active.mp4>)

The next distinct experiment should collect **safe LLB teacher continuations from actor-visited training states** just before and after the early drift. Train with those as labels while retaining the original teacher routes; keep the actor's prefix frames as context, not expert action targets. Vary card count, target port, and start geometry, then compare against the frozen dense actor on new episode-grouped development starts. Continue only if early measured route error falls and at least two of three starts reach the safe approach band without contact and with p95 inference below 300 ms. Keep final scenes sealed. [Detailed experiment record](../experiments/2026-10-13-sc-history-drift-followup.md) and [machine-readable scores](../../outputs/experiments/2026-10-13_sc_pause_models_v1/history_drift_followup_summary.json) have the commands and artifacts.
