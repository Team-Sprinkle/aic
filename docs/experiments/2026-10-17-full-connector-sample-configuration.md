# Unchanged public sample: full-connector pipeline score

## Scope and frozen settings

Run the complete pre-deadline `sample_config.yaml` once, byte-for-byte unchanged, from its original robot home. The three trials are SFP/NIC card0 port0 with one NIC card; SFP/NIC card1 port0 with one NIC card; SC port1 with zero NIC obstacles. **Maximum300points**,100pertrial. This small sample does not cover all card counts/ports or establish population reliability. No released evaluation or reserved final scenes are used.

Runtime: measured rule transport → requested-port coarse/native refinement → visual/proprioceptive25mm approach → shared full-connector-pose H4 actor. Predict4x50ms and execute4; preserve1.2srobot history. Image-estimated port/grasp transforms stay fixed per attempt. Recovery remains **none**, matching the preceding diagnostic; no RL, training, teacher control or privileged policy input. Oracle recording is for independent posthoc analysis only.

No model, rule, threshold or task deadline changes during the run. The whole-container watchdog is1800wallseconds to allow all three unchanged180-second task budgets; this does not extend any task. Use physicalGPU5 for simulation/inference with cooperative locks and rootlessDocker. Existing legacyGPU0 remains untouched.

## Artifacts and command

Selected run: `/home/nas/chmin/aic_archives/2026-10-17_full_connector_sample_configuration_v2/`.

```bash
.pixi/envs/default/bin/python scripts/run_full_connector_sample_configuration.py \
  --output /home/nas/chmin/aic_archives/2026-10-17_full_connector_sample_configuration_v2
```

`plan.json` stores source/model hashes, original sample hash, budgets and stopping rule; `command.json` records exact environment; `frozen/` preserves source and actor; `rollout/` stores effective container environment, original YAML, raw per-trial bags, camera frames, runtime trace and toolkit `results/scoring.yaml`. `summary.json` will retain the complete unmodified toolkit score. Separate per-trial analysis must use each bag's wall-time window and its retained terminal observation; never attach geometry from the next trial.

Initial v1 launcher failed **before simulator startup**: a generated script inferred the NAS artifact directory as its checkout because the invoking process omitted explicit checkout/snapshot environment settings. v1 is preserved as an infrastructure failure, not a policy failure. v2 explicitly sets both variables and has passed the effective-environment startup check. No actor/scene change was made for this repair.

## Required review

Report all three toolkit scores and their total, plus strict pre-reset depth, lateral/orientation errors and harmful-contact/force categories. Check whether transport, acquisition,25mm approach and actor actually ran. Separate simulator completion/locking from physical insertion. Save20fps five-camera videos and inspect key frames around stalls and corrections. If physical audit lacks causal labels, flag that rather than substituting post-reset geometry. Stop after this one complete configuration and diagnose failures without tuning on its outcome during evaluation.

## Current state

The full configuration completed: **165.42/300 toolkit points, 3/3 near handoffs, 0/3 strict physical insertions**. No model/rule changed during evaluation. This is one run of three familiar public sample trials, not an unseen-final or broad reliability estimate.

## Completed scores and physical geometry

| Trial | Task | Toolkit total /100 | Tier3 /75 | Depth mm | Lateral mm | Orientation degrees | Strict physical insertion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| trial_1 | SFP, NIC card0 port0 | 48.63 | 38.02 | 0.065 | 3.501 | 0.336 | No |
| trial_2 | SFP, NIC card1 port0 | 35.48 | 24.98 | 0.023 | 2.405 | 0.415 | No |
| trial_3 | SC port1, zero NIC obstacles | 81.31 | 75.00 | 1.970 | 1.488 | 2.433 | No |

No scored harmful-contact or excessive-force penalty occurred. All three duration scores are0; toolkit task durations are114.52/115.83/107.42seconds. SC receives0path-efficiency points for1.38mtravel; SFP paths are0.44/0.47m. Smoothness scores are5.49/5.45/5.31outof6, with scorer jerk4.28/4.61/5.73m/s³. This scorer measure differs from our command-jump metric; do not equate them or attribute the score difference to one model change.

### What worked

- **Complete autonomous runtime:** original-home transport, requested-port acquisition,25mm approach and full-connector actor all ran in all three trials, without pipeline exceptions. No teacher, true port/plug pose, recovery supervisor or RL controls the robot.
- **Port acquisition:** independent terminal-frame opening errors are0.077/0.138/0.122mm; port orientation errors0.110/0.322/0.033degrees. In SFP1, actual near handoff is24.59mm before the opening,0.210mm lateral and0.108degrees; SFP2 is24.64mm,0.317mm and0.302degrees. Thus at least these two insertion failures begin from accurately aligned near-port observations.
- **SFP grasp estimate:** SFP1 image-time grasp error is0.100mm/0.143degrees, terminal0.074mm/0.142degrees. SFP2 acquisition is0.077mm/0.283degrees, terminal0.058mm/0.284degrees. These precise components did not guarantee insertion.
- **SC handoff limitation:** near handoff is24.82mm before the opening,0.316mm lateral but3.363degrees actual plug-orientation error. Its image-time grasp estimate is0.148mm/2.042degrees off (terminal0.161mm/2.042degrees). The25mm controller checks a nominal TCP-in-port orientation, not full estimated connector alignment; a successful near-position handoff is not a certified<1degree plug orientation.

### What failed and what the videos support

- **SFP1/2 remain at the mouth:** actual depths0.065/0.023mm, with final lateral errors3.501/2.405mm. Their orientation errors remain below0.5degrees. The first actor targets already request roughly1mm lateral offsets, and late targets lie~39.5mm inside the port while the plug stays near its mouth. This confirms poor measured forward progress with a distant command reference; it does not prove one unique cause of the lateral divergence.
- **SC simulator completion is shallow:** the plugin publishes completion, but actual depth is1.970mm versus the15.5mm strict threshold, lateral1.488mm versus0.5mm and orientation2.433degrees versus1degree. The later stationary video segment is simulator attachment/locking, not learned stable insertion or successful recovery.
- **Command ownership is discontinuous:** the last visual-approach reference to first actor reference jumps3.06/5.36/10.45mm in trial order, with orientation steps0.046/0.240/0.980degrees. These are requested-position changes, not instantaneous physical displacement.
- **Actual contact evidence:** oracle pairs include SFP-tip/NIC-port-cage contacts in both SFP trials and SC-plug-housing/SC-port contacts. Cable links3/4 also touch NIC mounting geometry. Corrected maximum per-point normal forces for the strongest relevant tip contacts are~10.2N(SFP1),20.4N(SFP2),9.8N(SC); cable/mount peaks are~1.7/2.0N. These use the previously audited legacy scalar-impulse/dt correction at2ms and are not net wrist forces or sustained-force penalties. Contact names/counts and forces alone do not establish cable snag or its causal contribution.

Five-camera contact sheets from all three trials were visually inspected at handoff, early insertion, a later point and pre-return terminal. They support near-mouth stalls and SC attachment, without assigning an unobserved cable-snag cause. Full20fps videos are exported independently; sampled frames are not a claim to have watched every video frame.

### Next repair order, not executed in this measurement

1. **Continuous handover and blocked-progress control:** with frozen actor weights, bridge the last approach reference into the actor's complete targets and bound forward reference lead using measured progress and baseline-compensated force. Preserve compliance compensation and let safety guards override. A bridge alone need not fix lateral alignment.
2. **Diagnose lateral target bias before another broad fit:** use these traces plus fresh sample-derived development starts to distinguish teacher compensation/phase ambiguity from contact/controller tracking. Causal previous-command and tracking-error context is one bounded training hypothesis. Do not silently replace recorded command labels with measured future poses or assume longer history solves it.
3. **SC grasp-angle coverage and approach orientation:** current image-time grasp error is2.04degrees, with actual3.36degree plug tilt at the25mm handoff. Repair estimator coverage and make pre-contact orientation checks use the estimated full connector/grasp relationship, not only nominal TCP posture. Neither is isolated as the sole cause of the final2.43degree error. Validate alignment before the simulator can freeze a shallow contact.
4. **Recovery comparison comes after integration:** integrate measured hierarchical retreat/reacquisition into this25mm runtime and compare none/local/hierarchical on identical actors and new development starts. No backoff benefit was tested here. Keep step5open and defer new RL until supervised control meets its gate.

This sample score is not evidence that the previous SFP1/2 strict success generalized. Keep that genuine earlier success and this0/3failure together; no matched pre/post complete-sample score is available.

### Analysis artifacts and caveats

- `summary.json`, `rollout/results/scoring.yaml`: unmodified complete toolkit score.
- `terminal_window_summary.json`, `terminal_window_audit_v2/`: independent raw pre-return geometry, causal per-bag wall scopes and same-bag past board transforms. The earlier window-only audit failed safely because board transforms can be published near scene creation; its partial files/code remain preserved. No post-reset geometry was substituted. Force statistics in this minimal audit cover only its final one-second window; use toolkit categories for whole-episode penalties.
- `per_trial/`, `analysis_summary.json`: full-stream independent audit, per-trial causal manifests, grasp/near-handoff checks and regular20fps videos. `review/trial_{1,2,3}_contact_sheet.jpg` are regular image files.
- `port_accuracy_and_actor_targets.json`, `handoff_reference_steps.json`, `contact_pairs_by_insertion_trial.json`, `insertion_contact_forces.json`: diagnosis supporting the above.
- `rollout/contact_force_summary.json`, `corrected_contact_timeline.jsonl`: retained normal-force conversion, not policy input. The raw data remain unchanged.
- Video export exposed OpenCV oversubscription on this large CPU server. The original exporter is preserved under `frozen/`; subsequent exports cap OpenCV to2threads. This is an offline export repair, with no policy/scene change and no active job interrupted.

The independent terminal-window audit and full-stream audit agree exactly on reported depth, lateral and orientation for all three trials. Acquisition checks use the raw bag at their own pre-return observation timestamps, not terminal geometry backfilled into an earlier frame. All three regular H.264/yuv420p1920x1080,20fps videos are complete and ffprobe-verified; see `review/video_manifest.json`. No additional simulation, training or analysis job is queued.
