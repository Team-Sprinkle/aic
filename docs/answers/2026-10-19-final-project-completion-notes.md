# Final project completion: method, evidence and documentation

The demonstration milestone is complete. The selected recording uses rule transport, learned requested-port and held-connector perception, geometric consistency checks, an explicit insertion controller and local force/progress recovery. It does not deploy the historical BC insertion actor or an RL actor. This document holds the detail behind the short [project writeup](../robotic_cable_insertion_project.md), [status](../STATUS.md) and [experiment summary](../EXPERIMENTS.md).

## 1. What is completed, and what the score means

The final repo media come from repeat 10 of a fixed local batch using the published qualification `eval_config.yaml`. It scored **276.338304/300**:

| Trial | Task / NIC obstacles | Score | Toolkit insertion award |
| --- | --- | ---: | ---: |
| 1 | SFP → NIC / 1 card | 95.269205 | 75 |
| 2 | SFP → NIC / 2 cards | 95.153295 | 75 |
| 3 | SC → SC / 3 cards | 85.915804 | 75 |

The later fixed batch was planned for 20 replays. The user stopped it after **19 scored runs**; repeat 20 was stopped before scoring and is excluded. Across those 19 runs:

- Best 276.338304, median 275.710201, mean 256.282464/300 including failures.
- All three insertion awards in 15/19 complete runs.
- Individual awards 53/57: SFP 1 19/19, SFP 2 17/19, SC 17/19.
- The four incomplete runs remain in the score distribution. A successful selected video must not be presented as 100% reliability.

The latest exact public `sample_config.yaml` replay scored **287.899393/300**, also with all three insertions. It has one NIC card, one NIC card and no NIC cards, respectively. The earlier **290.19/300** public-sample result used an earlier selected configuration; it is not the score of the final released-evaluation videos.

The exact qualification config was published after the competition deadline and was used for diagnosis with the user's explicit authorization. These are exposed-scene local replays, not a new official leaderboard submission or an unseen test. The separate internal reserved final split was not opened. “Insertion award” here refers to toolkit scoring. Earlier investigations also audited strict depth/lateral/orientation gates because the simulator can lock on shallow contact; do not claim that every one of the 57 trials had an additional independent geometry audit merely from its scoring YAML.

## 2. Runtime from task start to insertion

### Observations and task identity

The controller receives three wrist RGB images, measured TCP/robot state, wrist force and the public task code. The task code selects SFP versus SC, the requested port, and for SFP the requested NIC card. Camera calibration, static asset dimensions and fixture-normal assumptions are explicit geometric priors. Simulator target TF, collision identities and diagnostic overview-camera geometry are excluded from runtime decisions. Those may be recorded externally for offline diagnosis and scoring.

### Transport

SFP uses a lift-and-transfer route. The repaired route can use a three-camera visual waypoint rather than blindly descending at a sample-derived nominal port location. Candidate waypoints need high projection probabilities, bounded reprojection error, a broad workspace envelope and temporal consistency. The precise native pose gate still decides whether handoff is valid.

SC uses a lateral detour because a collision-free arm path can still pull the trailing cable across cards. In the final selected run, it shifts at the starting height and turns the wrist to observe the magenta board marker. Calibrated projection of the marker's static dimensions estimates board x, y and yaw. Two consistent observations trigger a new route outside the **observed** card row, descending outside it, traveling beside it at low height, tilting beyond the row and sweeping toward the selected port. The marker's square symmetry is constrained by a selected-port prior and an ambiguity gate. The six-second missing-marker guard prevents an unverified continuation.

This is a constrained visual routing method, not general flexible-body motion planning. The marker assumes the known board asset and a horizontal fixture. Its initial movement and missing-marker cases remain measured limitations.

### Coarse perception and handoff

The deployed requested-port component answers **one task query**, not a simultaneous claim about every port in the scene. SC has two query identities; SFP has ten card/port combinations. It combines current images from all three cameras with relative robot-motion context. `RequestedPortHistoryObservation` stores an anchor after **40 mm of traveled distance or 10° of rotation**, keeps at most ten anchors, and retains the first plus the latest nine when full. This avoids filling context with seconds of nearly identical stalled images.

The coarse network proposes a location in each current native image and predicts whether that requested projection is in frame. This output is not calibrated pose confidence or proof of an unoccluded port. A learned native refiner crops the **original 1152×1024 image**, then predicts five reference landmarks of the requested opening. Calibrated multi-view fitting solves translation and yaw with a fixture-normal prior. Positive depth, reprojection error, bounded crop shift, temporal agreement and measured stationary standoff determine acceptance. Port identity and geometric validity are both required.

Transport acquisition is around a 150 mm nominal buffer. A fresh native-image lock then supports the explicit approach to about **25 mm** before final contact. The handoff does not by itself assert insertion.

### Connector pose, stable references and insertion

The gripper's measured pose does not determine the mating connector's pose unless its grasp transform is known. The system estimates connector-in-TCP translation and orientation from RGB. It uses the full transform when converting connector alignment goals into TCP commands in `base_link`.

The accepted port frame stays fixed per attempt, so a single noisy frame cannot rotate the controller's coordinate frame. Near 25 mm, the held plug can be re-estimated using five fresh quiet triplets. The update is bounded and needs consistent observations; a moderate far/near disagreement may require two independently consistent windows. Stability alone is not proof of accuracy, so acceptance remains bounded.

The explicit controller aligns lateral position and angle, then advances along the estimated insertion axis with impedance control. Reference continuity and antiwindup prevent targets from running ahead of measured blocked motion. A seating hold satisfies the toolkit's continuous-contact timing before the action returns. The historical class hierarchy sometimes labels the insertion phase `actor`; the final traces also explicitly state `learned_actor_inference=false` and `actor_owned=false` for this controller.

### Recovery

The force reference is estimated during the initial quiet hold and frozen. Sensor force is rotated into the base frame before subtracting the initial load; this is a load-change estimate, not a perfect contact-force decomposition. Force and stalled measured progress jointly indicate blockage.

Transport recovery uses a short roughly 8–12 mm retreat against recent measured travel, an orthogonal horizontal retry and route continuation. Insertion recovery similarly preserves the physical-path/issued-reference distinction and retries locally. The selected recording uses **local mode**. Hierarchical checkpoint escalation exists as a research direction, but the final result does not establish its advantage over local recovery. Broad SAC/SERL recovery training was not part of this final iteration.

## 3. The edge-detection insight: useful result and accurate attribution

The earlier board-rail method used Canny edges and Hough line segments near the requested port, camera calibration and a known horizontal board to infer yaw. Board edges gave an orientation cue over a longer image baseline than a tiny port alone. A useful root-cause finding was **search clipping**: all views could saturate at the same narrow search boundary and appear to agree while being wrong.

Widening the bounded search, filtering lines by orientation, rejecting boundary saturation and requiring two consistent views improved one difficult control from **1.37° to 0.017°**. Two reused controls passed; this was a scoped diagnostic, not a dataset-wide guarantee. [Original evidence](../experiments/2026-10-14-transport-handoff-execution.md#clipped-board-yaw-agreement-defect-and-bounded-repair) and `scripts/evaluate_sc_board_lines_yaw.py` retain the method.

The final recorded component `RequestedNativePortPose` instead obtains the port orientation through five learned native-image landmarks and calibrated constrained pose fitting. SC routing also observes the colored board marker. The old board-line method was an enabling experiment, but the final score must not be attributed solely to that Hough detector.

A different experiment projected the SC plug's CAD silhouette into images and matched Canny edges. It improved two of four estimates and worsened two, took about 1.65–1.67 seconds per triplet, and mishandled occlusion. A narrower housing-line fit was faster but inconsistent. Both were rejected. This is distinct from the successful board/rail orientation insight.

## 4. Training and experiments that informed the final design

- **Data integrity:** exclude no-insertion episodes and successful recordings with unreliable zero action labels. Preserve raw data and separate cleaned derivatives. Observation/command timestamp mismatch could create millimetre-scale geometry-label errors.
- **ACT/world model:** a world-policy comparison improved a near-opening envelope count over ACT, but neither inserted in its frozen 20-scene test, and dynamics did not beat persistence. Do not equate reconstructed images with useful predictive control.
- **Learned transport:** history, diffusion capacity, route representations, longer chunks and intentional overfitting were tested. Denser context visibly reduced drift; clean data and correct labels were necessary. These did not establish reliable all-start transport, so the final approach uses interpretable routes.
- **Perception:** original-resolution crops, requested identity, history and geometric consistency were more useful than asking a coarse image model to infer precise geometry everywhere. The mandatory native-refiner lineage used 72,857 triplets from 497 original sources, followed by targeted SFP acquisition data; these are perception samples rather than all successful action demonstrations.
- **Grasp orientation:** new held-connector estimators and better causal labels improved held-out angle errors, but some new-grasp domains still failed. Unqualified model fits were preserved rather than promoted because of low training loss.
- **Insertion BC:** complete port-relative connector targets, full grasp orientation and reference continuity repaired representation/execution issues. The controlled BC experiments still did not establish the full insertion success of the later explicit controller.
- **RL/recovery:** earlier near-port Isaac tests showed a local benefit of recovery in one small recording screen (1/8 → 3/8). They do not explain the final Gazebo full-route results or prove hierarchical recovery superiority.

[EXPERIMENTS.md](../EXPERIMENTS.md) is the compact comparison index; its linked dated records retain exact commands, failed variants and counts. [DATASETS.md](../DATASETS.md) summarizes reusable sources and label contracts.

## 5. Final repair and observed limits

The nominal sample location was about 254 mm away from the released SFP-2 port. A useful visual estimate was rejected by independent 200 mm transport and fresh-lock envelopes. Wider guarded envelopes, consistent visual waypoints and a safer fallback lift restored that trial. The released SC board differed roughly 150 mm and 85° from the nominal routing frame; marker-derived board geometry repaired the detour. Source hashes, runtime container settings and scene config hashes were frozen per comparison.

The initial four complete replays all inserted, but the 19-run continuation exposed four distinct gaps:

| Repeat | Trial | Observed stop |
| --- | --- | --- |
| 12 | SFP 2 | Initial quiet force reference did not qualify; controller stayed at the start |
| 15 | SFP 2 | Native crop refinement exceeded its 64 px margin, or view projection failed; acquisition timed out |
| 16 | SC | First sideways movement blocked before marker acquisition; scorer recorded force harm |
| 17 | SC | First movement finished, but no acceptable marker estimate arrived within six seconds |

Those are evidence for targeted repairs, not reasons to discard the failures from the score. The first four replays had active-decision p95 below 300 ms, with one trial at 299.4 ms and some higher individual tail calls. Do not claim this certifies every stage across every later reset. Current future work is consolidated in [one file](../FUTURE_WORK.md).

## 6. Final media and reproducibility

The repo now contains the three original successful MP4 clips and three GIF previews in [media/final_evaluation](../media/final_evaluation/README.md). The MP4s are byte-for-byte copies of the selected archived clips, at 1920×1080, H.264/yuv420p, 20 fps. They retain the complete trial, including initialization. No model rollout was regenerated to make the final score video.

The GIF exporter finds three distinct motion samples with TCP speed above 3 mm/s and displacement above 1 mm from the initial command observation, then includes 0.2 s of preroll. The trim starts are **6.648, 7.600 and 7.650 seconds** within the three original clips. GIFs retain motion through the terminal event, play at **2× speed**, and use 720-pixel width, 8 fps and a reduced palette. This removes initialization and reduces file size without cutting out a failure or interior hesitation. `manifest.json` records exact hashes, source, thresholds and playback settings.

```bash
.pixi/envs/default/bin/python scripts/export_final_project_media.py \
  --source /home/nas/chmin/aic_archives/2026-10-19_sc_sfp_released_repair_twenty_frozen/repeat_10/rollout
```

Only this curated media directory is exempted from the global MP4 ignore rule. Datasets, arbitrary generated videos, raw bags and model checkpoints remain external. Three videos plus three GIFs total about 24 MB, with each GIF below 8 MB. [LOCAL_WORKFLOW.md](../WORKFLOW.md) gives the rootless complete-evaluation and sample-recording commands.

## 7. Artifact map

| Artifact | Location |
| --- | --- |
| Selected final recording, source trial videos, trace and scored bags | `/home/nas/chmin/aic_archives/2026-10-19_sc_sfp_released_repair_twenty_frozen/repeat_10/` |
| Stopped 19-run batch, including all failures | Same parent, `summary_stopped.json`, `progress.json`, `repeat_*/score_summary.json` |
| Frozen policy/models/config and actual settings | Same parent, `frozen/`, `plan.json`, `command_environment.json`; each rollout has `container_environment.json` |
| First complete repaired replay and diagnostic review frames | `/home/nas/chmin/aic_archives/2026-10-19_sc_sfp_released_repair_cycle8_sc_marker/` |
| Earlier three-repeat confirmation | `/home/nas/chmin/aic_archives/2026-10-19_sc_sfp_released_repair_cycle9_sc_marker_repeat3/` |
| Latest exact public-sample recording | `/home/nas/chmin/aic_archives/2026-10-19_repaired_public_sample_video/` |
| Curated repo media/provenance | `docs/media/final_evaluation/manifest.json` |
| Runtime entry point | `aic_example_policies/aic_example_policies/ros/RunVisualConnectorServoQuietContextV56.py` and inherited implementation |
| Requested-port perception / geometric fit | `scripts/requested_native_port_pose.py`, `scripts/native_keypoint_pose_geometry.py` |
| Board marker used for SC routing | `scripts/observed_board_marker.py` |

## 8. Documentation cleanup

Nine long main documents were preserved before rewriting in [answers/archive/2026-10-19-documentation](archive/2026-10-19-documentation/README.md), with original file hashes and a relocation manifest. Fifteen superseded top-level project plans were moved to [obsolete/docs/2026-10-19-retired-plans](../../obsolete/docs/2026-10-19-retired-plans/README.md). Relative links were repaired; historical main-document section links point to preserved snapshots when the short current page no longer has that section.

The public toolkit reference docs, datasets, code, checkpoints, dated evidence and failed results were preserved. Historical `experiments/` and `answers/` records are not current work queues. `STATUS.md` is now a small snapshot, `EXPERIMENTS.md` records the compact comparison history, and the project writeup focuses on high-level design ideas and the final GIFs. [FUTURE_WORK.md](../FUTURE_WORK.md) is the sole active future-improvement roadmap.
