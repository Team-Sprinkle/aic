# SC Low Lateral Bypass BC cleanup and continuation — 2026-10-01

## Objective and immutable source

The original `outputs/trajectory_datasets/sc_llb_score_verified/` contains 104 official full-insertion successes from 200 attempted teacher settings and remains unchanged. A separate, versioned BC derivative removes quiet scripted holds while preserving recorded 200 ms action targets, causal images and robot state, contact decisions, and complete episode/reset splits. Transport training and its bounded autonomous comparison are complete; the local handoff and RL gates remain closed.

## Preparation completed

Command:

```bash
.pixi/envs/default/bin/python scripts/prepare_sc_llb_cleaned_dataset.py \
  --source outputs/trajectory_datasets/sc_llb_score_verified/manifest.json \
  --output outputs/trajectory_datasets/sc_llb_cleaned_v3/manifest.json
```

The selected `v3` manifest audits **104/104** scored successes and their bags, plans, and insertion traces. All 104 contain exactly the named 8-second `sweep_after_row` and 5-second `tilt_clear_of_cards` holds, plus explicit local start and guarded insertion-success events. These planned waits total **22.5 minutes** across the source episodes. Median local phase from preinsert alignment to success is **25.5 seconds**; median guarded axial holds is **8** (maximum **58**). Thus local duration cannot safely be removed wholesale.

The split groups complete scene and measured robot-start configurations, then chooses held-out groups within each card-count/selected-port stratum. It has **83 train** (35 port 0, 48 port 1) and **21 development validation** (8 port 0, 13 port 1) episodes. Every card count and selected port appears in both partitions. The reserved final split remains sealed. The `v1` and `v2` metadata manifests were preliminary split audits; use `v3`.

`scripts/filter_sc_llb_bc_chunks.py` retains the original four recorded target commands in every kept chunk. It identifies long command-target plateaus matching the two plan markers and removes only their interiors, retaining 0.4-second edges. During local control, it removes an anchor only when measured TCP pose, target, and force barely change, force remains below 2 N, and the previous kept anchor is less than 0.5 seconds away. Held-out validation rows are never filtered. The thresholds are conservative by design; the 0.5 mm insertion corridor makes aggressive low-motion deletion unsafe.

An integration check on the existing `2026-09-30_v9_paired_ports` materialized SC data found both known hold plateaus in `port1_five`: **171 + 111 repeated training anchors** matched the two marker interiors. Its local examples had **zero** anchors satisfying the strict static criterion. This is a check of the filtering logic on older materialized data, **not** proof of cleaned 104-episode chunks or autonomous performance.

## Resolved access boundary

An earlier workspace-write session could not access `unix:///run/user/1008/docker.sock` or NVIDIA devices. The user switched to a full host-access session, and the rootless socket and GPUs became available. The preflight failure did not change source data. The successful resumed extraction used:

```bash
.pixi/envs/default/bin/python scripts/run_sc_llb_cleaned_materialization.py \
  --spec outputs/trajectory_datasets/sc_llb_cleaned_v3/manifest.json \
  --output outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized
```

The `v3` selection manifest remains immutable and describes the pre-materialization selection. The preferred motion-safe training derivative is `materialized/bc_chunks_motion_safe_v2/manifest.json`; `materialized/bc_chunks/manifest.json` preserves the earlier cleanup ablation.

## Continuation gates

1. Run the command above. It resumes per-case scored-bag extraction, materializes command-derived TCP-body transport and port-frame local chunks, applies the filter, and writes `materialized/bc_chunks/manifest.json`. It validates score/provenance and preserves source episodes. `scripts/materialize_moveit_bc_segments.py` now accepts `--validation-full-success-streams` so the 21 held-out successful episodes do not need route-only audit files.
2. Audit raw-versus-cleaned chunks by episode, phase, card count, selected port, time coverage, held-command detection, force and TCP motion. Inspect several aligned video/trace examples. Require all 104 groups, no split overlap, no changed target commands, and no missing images. If many planned holds are unmatched, diagnose the command/plan timing rather than forcing a deletion.
3. Train a matched cleanly initialized transport BC experiment with verified SFP CheatCode successes plus SC `bc_chunks/transport_chunks.jsonl`. Do not reuse the old mixed-SC action head without an explicit comparator. Train the local port-frame actor from the compatible SFP PoseInsert checkpoint using successful SC local chunks; deployable pose inputs must come from vision. Balance by episode, phase and selected port. Keep update budgets and development scenes fixed for raw-versus-cleaned comparison.
4. Complete task-conditioned perception with true selected-port visibility and calibrated pose confidence. The current SC projection labels are insufficient to promote the two-actor handoff. Evaluate transport, local insertion, then full-start autonomous Gazebo on new development starts with no privileged geometry. Report full insertion, force, contact, cable route, and inference p50/p95/p99 under the 300 ms p95 limit.
5. Keep Isaac RL, Gazebo RL and reserved final-scene evaluation gated on demonstrated autonomous supervised insertion. Preserve failures, exact commands, checkpoints, videos, and machine summaries.

## October 1 host-access continuation (completed)

The session permission profile was changed to full host access. Rootless `docker info` reported `name=rootless`; GPUs 1–7 were idle at 13:45 UTC, while the old probe remained on GPU 0 and was left untouched. The previously active SC collection was complete, and no competing recording container was running at this check. The extraction command above subsequently completed all 104 scored bags with all three camera streams. Its resumable files are under `sc_llb_cleaned_v3/materialized/streams/`, and `materialization.log` records per-case extraction.

The transport trainer has an `imagenet_clean` initialization that loads only the local ImageNet ResNet18 backbone; its policy/context/action modules start fresh and its normalization statistics come solely from training rows. The old mixed-route checkpoint was tested explicitly under `mixed_warm_start`. Both runs completed in the same approximately 13.6M-parameter architecture, using `scripts/run_sc_llb_clean_transport_bc.sh` and the same verified SFP labels plus SC chunks. The extraction and independent dataset audits passed before training.

The step 5 autonomous insertion gate remains open after this completed comparison.

## October 1 extraction and materialization smoke check

To shorten the read-only bag decoding, `scripts/extract_sc_llb_range.py` now processes disjoint later ranges while the original resumable worker handles early cases. Each worker writes only its own temporary stream and atomically renames it after checking the official score and case provenance. They use CPU and no GPU. The original 104-episode manifest and source bags remain untouched.

Before waiting for all 104 bags, this command exercised the real MoveIt FK and command-to-chunk conversion on two scored successes, with one designated as validation solely for this smoke test:

```bash
bash scripts/run_materialize_transport_bc.sh \
  --streams outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized/streams/llb_0001.json \
  --validation-full-success-streams outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized/streams/llb_0002.json \
  --output outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized/smoke_two
```

It completed with **1,750 transport and 546 local chunks**, no rejected episodes. An independent check of both cases found the named transport plateaus: 171+112 redundant anchors in `llb_0001`, and 173+113 in `llb_0002`. Neither case had a local neighbor satisfying the strict static-deletion rule. This is a pipeline check, not a valid 104-episode audit or a trained-policy result. The real grouped split remains the `v3` manifest's 83/21 split.

## Completed 104-episode derivative and independent audit

The full materializer accepted **104/104 official full insertions**, with no rejected episodes, and wrote **89,251 raw TCP-body transport** and **26,455 raw selected-port-frame local** four-target chunks. The 83/21 split is by complete scene/reset group. All three native camera paths and the recorded command targets are retained in each row. Raw chunks and decoded streams are preserved under `materialized/`.

The first full filtering attempt was a no-op: it treated raw wrist force above 2 N as contact, while the static tool contributes about **20 N** of gravity load. This rejected derivative is preserved as `materialized/bc_chunks_unfiltered_wrist_gravity_diagnostic/`. The corrected filter compares force to the median measured wrench **within each named stationary hold** and retains rows with at least 2 N excess. It removed **23,459 quiet hold anchors** and retained **65,792 transport chunks**. It retained **all 26,455 local chunks**: the 0.5 mm insertion corridor and the very small number of clearly static neighbors did not justify more aggressive local trimming. Validation rows were not filtered.

The independent command below passed. It checks all 104 cases, both named holds in all 83 training cases, exact equality of every retained row and its original recorded four-command target, first/last episode anchors, unchanged validation, representative image paths, split groups, and that no transport row outside a quiet named hold was removed. The report is `materialized/audit.json`, including counts by phase, split, card count, and selected port.

```bash
.pixi/envs/default/bin/python scripts/audit_sc_llb_cleaned_bc.py \
  --spec outputs/trajectory_datasets/sc_llb_cleaned_v3/manifest.json \
  --materialized outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized \
  --output outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized/audit.json
```

## Matched transport BC runs (completed)

The following two runs use the same verified SFP source, the same cleaned SC transport rows, the same episode-grouped validation set, seed, 6,000-update ceiling, 2,000-update minimum before plateau stopping, 300-minute wall budget, architecture, camera preprocessing, and loss. The clean arm loads **only** the local ImageNet ResNet18 backbone; the context and full-target action modules start fresh with statistics computed from training rows. The comparator resumes an older mixed-SC-route checkpoint. Neither receives true port geometry. Both use the 10D task code and complete recorded TCP-body target actions. Output manifests contain source hashes and exact validation keys.

```bash
bash scripts/run_sc_llb_clean_transport_bc.sh 1 imagenet_clean \
  outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean
bash scripts/run_sc_llb_clean_transport_bc.sh 2 mixed_warm_start \
  outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_mixed_warm_start
```

The jobs began after the full audit passed. Their logs, histories, checkpoints, stop reasons, and held-out errors live in the two output directories. **Training start is not a performance result.** Autonomous transport, local pose estimation, two-actor handoff, full insertion, and RL remain gated on measured development results; the reserved final scenes remain sealed.

### Motion-safe hold ablation added during training

A second audit found that 537 anchors removed by the first cleaned derivative still had measured TCP displacement above 0.5 mm over one sample. They had a repeated commanded target but represented robot settling. The two existing GPU runs were left intact with their exact source hashes. A separate, preferred `bc_chunks_motion_safe_v2/` derivative preserves any hold row with at least 0.1 mm measured translation, 0.15° rotation, or 2 N force change relative to the stationary hold baseline, and also retains one row per second of every quiet hold. It removes **19,570** anchors, retaining **69,681 transport** and all **26,455 local** chunks. The stricter `audit_motion_safe_v2.json` passes with zero removed measured-motion or excess-force rows, the same exact command and validation checks, and no reset leakage. The original full raw chunks and both earlier derivatives remain intact.

A matched clean-initialization run on this motion-safe derivative completed on GPU 3:

```bash
AIC_LLB_SC_LABELS=outputs/trajectory_datasets/sc_llb_cleaned_v3/materialized/bc_chunks_motion_safe_v2/transport_chunks.jsonl \
  bash scripts/run_sc_llb_clean_transport_bc.sh 3 imagenet_clean \
  outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_motion_safe_v2
```

This isolated the effect of preserving actual measured settling and sparse visual hold coverage, using the same architecture, seed, SFP rows, grouped validation scenes, update and wall budgets. At launch, project GPU use was the old probe on GPU 0 plus these three training arms on GPUs 1–3, respecting the four-GPU cap. Autonomous results are reported below.

**Action-cap discrepancy tested:** The first eight autonomous comparisons (including the elapsed-time arm) used a 20 mm per-component translation cap; the prior accepted transport experiment used 60 mm. In the 83 SC training episodes, **35.8%** of retained first targets exceeded 20 mm in at least one component, while **0%** exceeded 60 mm (`teacher_target_cap_v3.json`; port-frame component proxy). Both frozen caps were tested on identical scenes and checkpoints (`autonomous_comparison_cap20.json` and `autonomous_comparison_cap60.json`). Results and the decision are below. The launcher now requires the cap explicitly, preventing an unnoticed default from changing future comparisons.

## Completed first supervised comparison: offline fit versus autonomous transport

All three original BC arms finished under their predeclared 6,000-update ceiling or four-check held-out plateau rule. `outputs/experiments/2026-10-01_sc_llb_cleaned_bc/offline_comparison.json` records exact source hashes, selected updates, stop reasons, and held-out metrics. The following SC errors are first recorded-target regression on held-out teacher frames; they are **not** live transport errors.

| Arm | Updates / best | SC first target p95 | SC orientation p95 |
| --- | ---: | ---: | ---: |
| ImageNet-clean, earlier cleanup | 6,000 / 6,000 | 4.16 mm | 0.14° |
| Older mixed-route warm start, earlier cleanup | 5,000 / 4,000 | 4.39 mm | 0.26° |
| ImageNet-clean, preferred motion-safe cleanup | 2,250 / 1,250 | 4.75 mm | 0.12° |

Each frozen best checkpoint then ran autonomously, observation-only, on the **same two existing non-final Gazebo development layouts**, with the same scene files, controller gains, 70-second simulation budget, and no teacher/TF/geometry input. Exact eval configs had no byte-identical match among the 104 new teacher configs. `autonomous_comparison.json` verifies the matched controls and lists all six score, path, force, latency, checkpoint hash, and artifact paths. Final scenes remain sealed.

| Arm | Port-0 two-card final/min handoff error | Port-1 three-card final/min handoff error | Full insertions |
| --- | ---: | ---: | ---: |
| ImageNet-clean, earlier cleanup | 103 / 100 mm | 194 / 190 mm | 0/2 |
| Older mixed-route warm start | 75 / 35 mm | 110 / 28 mm | 0/2 |
| ImageNet-clean, motion-safe cleanup | 372 / 202 mm | 193 / 114 mm | 0/2 |

The score's Tier 3 insertion detector reported no insertion in any rollout. Complete actor inference p95 was **139–183 ms** across these six, below 300 ms; isolated cold-start maxima exceeded 300 ms and are retained in the summaries. The rejected motion-safe arm has 20 fps five-view videos at `autonomous_transport/motion_safe_port{0_two,1_three}/video/all_views_20fps.mp4`; the earlier-clean arm has videos at `autonomous_transport/clean_port{0_two,1_three}/video/all_views_20fps.mp4`. Every run keeps raw bag, RGB frames, execution trace, score, posthoc measured-versus-teacher path plot, and terminal observation. These are policy rollouts, not teacher videos.

### Bounded failure diagnosis and one targeted repair

The first motion-safe port-0 rollout diverged from the held-out teacher by 22 mm after one second and 58 mm after two seconds. At one second its lateral/body target lacked most of the teacher's intended sideways movement and commanded downward motion instead of the teacher's upward component. Its measured path moved to the right and down, ending 372 mm from handoff. The side/overhead video shows the cable trailing freely rather than snagging; raw wrist force is dominated by the tool's roughly 20 N weight. The warm-start actor reached 28–35 mm from handoff transiently, then drifted to 75–110 mm by timeout. This suggests missing route-stage/progress representation or poor closed-loop stopping, rather than an insertion or force-recovery failure. The elapsed-time alignment is diagnostic only; it does not equate states after the trajectories diverge.

One bounded test added **causal elapsed simulation time** to the *same* ImageNet-clean actor and earlier-clean dataset. The teacher route is staged, so time might disambiguate images and measured states that otherwise permit multiple route actions. It used the same 83/21 SC split, SFP source, sampling, 6,000-update ceiling, 2,000-update minimum and plateau rule, then ran on the same two development layouts. The completed result is below; it failed the 10 mm handoff gate.

```bash
AIC_LLB_INCLUDE_ELAPSED=1 bash scripts/run_sc_llb_clean_transport_bc.sh 2 imagenet_clean \
  outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_elapsed_v1
```

## Final bounded supervised result and step-5 decision

The elapsed-time arm finished at 5,000 updates (best 4,000) under the same plateau rule. Its held-out SC first recorded-target p95 was **4.55 mm**, versus **4.16 mm** for the clean no-time actor. It was evaluated at both caps, the same two frozen development layouts, and no privileged runtime input.

The 60 mm comparison is complete: eight additional observation-only rollouts, all terminal observations and official scores retained. `autonomous_comparison_cap60.json` verifies identical scene/controller/budget per layout; `cap_comparison.json` joins both caps. Table entries are **final handoff error in mm**; handoff requires at most 10 mm, and all official Tier 3 scores remained below full insertion.

| Actor | 20 mm cap: port 0 / port 1 | 60 mm cap: port 0 / port 1 |
| --- | ---: | ---: |
| Clean ImageNet, earlier cleanup | 103 / 194 | 202 / 239 |
| Old mixed-route warm start | 75 / 110 | 516 / 459 |
| Clean ImageNet, motion-safe cleanup | 372 / 193 | 327 / 265 |
| Clean ImageNet, causal elapsed time | 96 / 226 | 92 / 77 |

**Result: 0/16 full insertions and 0/16 stable 10 mm handoffs.** The best final error was 75.4 mm; the closest transient pass was 27.9 mm and later drifted away. Five of eight pairs had a smaller final error at 20 mm, three at 60 mm. Thus matching the teacher's 60 mm target range alone does not repair autonomous control; a larger cap can amplify a wrong proposal. Complete live inference p95 was at most **182.6 ms** across the 16 runs, below the 300 ms limit. Cold-start outliers above 300 ms remain in each run's raw summary. No final scenes were opened. The selected 20 fps five-view videos and posthoc measured TCP plots show early wrong route choices or drifting past handoff; reviewed examples show no sustained cable snag. The score/force traces preserve any transient contact rather than treating raw wrist gravity as contact force.

The same-architecture clean versus warm-start test did **not** establish a clean-initialization advantage autonomously: warm start came closer transiently on the two 20 mm layouts, but neither inserted. Preserving more settling samples in the motion-safe derivative also did **not** improve autonomous insertion in this bounded test. Causal elapsed time helped one 60 mm layout's final position but failed the gate on both. This evidence supports stopping further routine single-frame full-target BC on this dataset. It does not show that the verified teacher route or local insertion supervision is invalid.

**Step 5 stays open.** The transport actor has not reliably brought the plug to the local actor's handoff region. Therefore local-actor promotion, visibility switching, Isaac/Gazebo RL, and reserved final evaluation do not proceed. Original 104 scored source episodes, both BC derivatives, all four checkpoints, 16 score/bag/trace artifacts, and all failures remain saved.

### Proposed explicit plan revision: route-progress and subgoal control

**October 1 follow-up outcome:** The bounded frozen-vision temporal route-progress/subgoal test was executed on the same 83/21 SC split with a matched no-history control and one direct-state repair. History improved held-out subgoal prediction, but the best full-validation p95 remained 39.50 mm versus the 20 mm gate; no new live Gazebo rollout was launched. The follow-up diagnosis found near-identical visual and measured TCP states at the 5 s tilt-hold/return boundary. See the [complete temporal experiment record](2026-10-01-temporal-three-model-continuation.md) and [machine decision](../../outputs/experiments/2026-10-01_temporal_transport/decision.json). The plan below is the original proposal; the follow-up record contains its measured disposition and next revision.

The next supervised experiment should target the observed failure rather than collect more ordinary successes. Label each existing teacher transport decision with its planned route stage and **next full TCP subgoal in the current TCP frame**, using the recorded MoveIt plan and commands only for training labels. A causal short-history visual model would read the same three RGB cameras, measured robot state, 10D task code, and recent actions; it would predict stage, subgoal pose, and a handoff/stop signal. During autonomous rollout, it must select stages and subgoals from observations and measured TCP only, never the teacher plan, true target pose, or simulator geometry. A bounded controller would drive toward the predicted subgoal and advance stages only after measured progress, with a force-safe abort. This is one shared transport actor across SFP and SC; it is not a privileged scripted route.

First audit whether the existing 83/21 grouped episodes and plan traces provide unambiguous stage transitions and enough examples for both SC ports and 1–5 cards. Then train **one** small route-progress head, initially reusing the frozen visual backbone, on one GPU for at most **6,000 updates or 2 hours**, with a predeclared plateau stop. Require held-out stage accuracy and subgoal error by card count/port; a starting offline gate is stage accuracy at least 90% and next-subgoal position p95 at most 20 mm, with no skipped stage in trace replay. If it fails, stop and inspect specific label ambiguities before more training. If it passes, evaluate the same two development layouts, capped at two 70-second rollouts plus trace/video review. Promote only if **both** reach and stably hold the 10 mm handoff with acceptable measured force, correct cable-side route, and inference p95 below 300 ms. If either fails, stop this route-progress candidate and consider an observation-only scene/waypoint estimator or a planned transport controller as a separate architecture revision. Use new development starts only after the two-layout gate; keep final scenes sealed. This is a proposal, not a claimed trained result.
