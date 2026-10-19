# SFP wider observation and visual identity gate

**Date:** 2026-10-04. **Status:** predeclared diagnostic; final scenes sealed. No policy or perception training in this view test.

## Question

The requested-port history model could not establish the numbered card/port reliably from the existing teacher images. Does a safely captured *wider initial view* show enough of the whole card row to identify the requested numbered slot and its chosen port, before the close-up refiner runs?

The old ±15 mm side sweep at an 80 mm plug-tip standoff did not rescue card 2/port 0. This test is distinct: at a **180 mm standoff**, it holds a baseline, rises **40 mm**, moves **+80 mm along base y** (the card-row direction), holds a wide view, then reverses the moves before the stock CheatCode route continues. Oracle port geometry is used **only to choose this diagnostic view** and to score captured images afterward. It must never become the autonomous crop selector.

## Frozen scope and stopping rules

Use only four already inspected nonfinal reset groups: card 1/port 0, card 2/port 0, card 3/port 0 and card 4/port 1 from the fresh SFP plan. Run card 1/port 0 as a safety pilot. Continue to the other three only if the pilot reaches the wide hold, returns within 5 mm of baseline, and no forbidden contact or force stop occurs. Motion increments are at most 2 mm per 50 ms. The guard aborts if wrist-force rise exceeds 5 N from the first held view or absolute force exceeds 35 N; stale oracle contact or any new pair besides cable/enclosure and light cable/task-board contact ≤1 N also aborts. Do not relax these limits after seeing a failure. Use one rootless Gazebo container at a time on GPU 1; leave the existing Isaac container running.

The pre-run **geometry feasibility calculation** used only simulator port coordinates and fixed camera calibration. For the first saved TCP pose on the four scenes, the best single-camera minimum pixel margin for all installed card port-0 centers was −95, 207, 166 and 147 px respectively. Translating the TCP +80 mm in base y and +40 mm in base z at the same orientation predicts margins of **217, 220, 251 and 286 px**, respectively. This is a field-of-view upper bound, not evidence of actual visibility or physical safety.

For each completed scene, save synchronized original center/left/right RGB at baseline and wide holds, measured TCP, force, all oracle contact pairs and official teacher score. Review **unmarked** images before opening truth overlays or model predictions. Per camera record: can the installed card row be counted, can the requested numbered card be identified, is the selected port face resolved, and is the answer uncertain? Then compare annotations to posthoc simulator projections and any occlusion evidence. A candidate view is worth a new training set only if at least three of four scenes have the correct requested face identifiable in at least two cameras without a safety event. A view that merely projects the opening inside image bounds does not pass. These reused scenes cannot promote a model; any subsequent learned crop requires a new episode-grouped nonfinal validation set and full inference p95 <300 ms.

## Artifacts

The guarded policy is [CheatCodeWideViewDiagnostic.py](../../aic_example_policies/aic_example_policies/ros/CheatCodeWideViewDiagnostic.py). The fixed scene hashes are in `/home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/plan.json`. New rollout artifacts will be under `/home/nas/chmin/aic_archives/2026-10-04_sfp_wide_observation_v1/`. Preserve failed pilot and contact traces. No reserved final configuration is used.

## Pilot and controller settling correction

The first card-1/port-0 pilot safely reached the wide hold at measured **+76.1 mm base y and +37.9 mm base z**, with no unsafe contact and a roughly 20 N wrist load. It returned along y, but after a fixed one-second z hold the measured TCP remained **5.89 mm** above the baseline; the predeclared 5 mm return check correctly stopped the policy. The resulting score 0 reflects this diagnostic abort, not an insertion attempt. The view capture remains useful, but this pilot does not clear the return gate and no other scene was launched.

The v2 pilot keeps the same 5 mm acceptance criterion, force/contact limits, path, offsets and source scene. It resends the same final target for at most two additional seconds, checking the same guards every 50 ms, to accommodate controller settling. If the measured pose still misses by >5 mm or contacts rise, stop this view route. The original failed `card1_port0/` rollout, events, force and oracle traces are preserved.

**V2 pilot passed.** At the wide hold it measured **+76.21 mm y, +37.66 mm z**; peak wrist-force rise was **0.64 N**, with cable/enclosure contact only at the wide view. After 14 extra guarded 50 ms settling ticks, the return pose was **+0.68 mm y, +4.90 mm z** from baseline, inside the unchanged 5 mm gate. The ordinary CheatCode continuation scored tier 3 = **75**; this is a privileged teacher success, not an autonomous model result. I reviewed unmarked baseline and wide center/right RGB before using projections. The wider view clearly shows more of the four-card row in the center image and all four card faces in the right image. Requested card/port identity still needs explicit blind visual annotation and posthoc verification; simple field-of-view projection is insufficient. Saved images and events are under `/home/nas/chmin/aic_archives/2026-10-04_sfp_wide_observation_v1/card1_port0_v2/`.

The remaining fixed cases run sequentially with [run_sfp_wide_observation.py](../../scripts/run_sfp_wide_observation.py); it checks the v2 pilot record, rootless key quota, scene hashes, exact policy hash, and completion state before each launch. The first failed pilot and v2 rerun are separate artifacts.

## Completed four-scene result and decision

The three remaining scenes finished sequentially. Card 2/port 0 and card 3/port 0 reached the wide hold, returned within the unchanged 5 mm tolerance, and their ordinary privileged CheatCode continuation scored tier 3 = 75. Card 4/port 1 reached the wide hold but then **failed** the measured-pose gate: after the full bounded settling period, it was about 6.0 mm short of the requested y offset and 6.4 mm short of the requested z offset. The policy stopped before reversing the wide detour or attempting insertion; tier 3 = 0 is a diagnostic abort. There was no new forbidden contact or force-rise abort. Its images are useful for diagnosis but it is not a safe completed view action. The original failed card-1 pilot remains preserved too.

| Request | Installed slots, from posthoc geometry | Wide-view measured y/z offset | Return and teacher score | Raw-image finding |
|---|---|---|---|---|
| Card 1, port 0 | 1, 2, 3, 4 | +76.2/+37.7 mm | Returned; 75 | The wider pose makes all four card faces countable in three views; the baseline did not. |
| Card 2, port 0 | 0, 1, 2, 3, 4 | +75.7/+36.1 mm | Returned; 75 | All five faces were already visible at baseline. |
| Card 3, port 0 | 0, 1, 2, 3, 4 | +75.8/+36.2 mm | Returned; 75 | All five faces were already visible at baseline. |
| Card 4, port 1 | 1, 2, 4 | +75.6/+34.7 mm | **Abort; 0** | Three installed faces are visible, but counting them cannot yield slot 4 because slots 0 and 3 are empty. |

I first reviewed the **unmarked** original 1152×1024 images and saved per-camera notes in `raw_views_v1/visual_annotations_v1.json`. The wide images resolve individual card faces and paired openings. They do **not** visibly label the physical slot numbers or declare which opening corresponds to the task's port-0/port-1 convention. On this strict review, **0/4** scenes have the requested numbered face confidently identified in two cameras without a learned or calibrated slot mapping. This does **not** show that a model cannot learn the mapping. It shows that seeing more card faces is not, on its own, an identity solution, especially with empty slots. All eight stage/request positions project inside all three image bounds in the *posthoc* simulator join; that projection says nothing about true visibility or correct identity. The selected opening was already in frame at every baseline hold.

The predeclared ≥3/4 identifiable-face gate also fails because one view route aborted. **Do not promote this view action or collect a routine wide-view training set.** This branch provides a sharper next hypothesis: learn or verify the board's fixed slot reference and each occupied slot separately, then select only the requested slot and opening. If the board reference is ambiguous in raw RGB, a different observation pose or external visual marker may be needed. Use existing frames first; any later fit needs new episode-grouped nonfinal evaluation, observation-only crop selection, and full live p95 <300 ms. Keep automatic SFP handoff, insertion and RL disabled.

### Reproduction and saved images

```bash
.pixi/envs/default/bin/python scripts/run_sfp_wide_observation.py
PYTHONPATH=scripts:aic_utils/lerobot_robot_aic LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}" \
  .pixi/envs/default/bin/python scripts/build_sfp_wide_raw_views.py \
  --root /home/nas/chmin/aic_archives/2026-10-04_sfp_wide_observation_v1 \
  --source-plan /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/plan.json \
  --output /home/nas/chmin/aic_archives/2026-10-04_sfp_wide_observation_v1/raw_views_v1
# Only after unmarked RGB review; one extraction per scene:
PYTHONPATH=/var/tmp/chmin_aic_mcap_reader_py312 LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}" \
  .pixi/envs/default/bin/python scripts/extract_moveit_training_stream.py AUDIT_DIR AUDIT_DIR/scored_stream.json
PYTHONPATH=scripts:aic_utils/lerobot_robot_aic LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}" \
  .pixi/envs/default/bin/python scripts/build_sfp_wide_view_labels.py \
  --root /home/nas/chmin/aic_archives/2026-10-04_sfp_wide_observation_v1 \
  --source-plan /home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/plan.json \
  --output /home/nas/chmin/aic_archives/2026-10-04_sfp_wide_observation_v1/posthoc_truth_v1
```

The runner is resumable but these output directories already exist; do not rerun into them. The unmarked manifest is `raw_views_v1/rows.jsonl`; posthoc projection labels are in `posthoc_truth_v1/rows.jsonl`. Representative raw **wide center/right** pairs: `card1_port0_v2/frames/1791138402818540066_16550000000_center.jpg` and `1791138402848643662_16550000000_right.jpg`; `card2_port0/frames/1791138610432446778_16800000000_center.jpg` and `1791138610460509860_16800000000_right.jpg`. All are below `/home/nas/chmin/aic_archives/2026-10-04_sfp_wide_observation_v1/`. [Machine summary](2026-10-04-sfp-wide-observation-gate.summary.json).
