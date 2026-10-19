# SFP board-reference wrist-tilt diagnostic

**Status:** predeclared bounded pilot. This is a privileged, nondeployable camera diagnostic on reused nonfinal scenes. Final scenes remain sealed.

## Hypothesis and offline feasibility

The +80 mm and −80 mm row translations did not reveal the physical slot origin together with the full row. At the existing 180 mm teacher standoff, rotate the wrist **+10 degrees about its local x axis while keeping its TCP position fixed**, hold, then return. This changes all three wrist-camera viewing angles without translating the cable/tool across the cards. The first reused scene is card 1/port 0; a second scene is allowed only if its view safely returns and improves identity.

Using saved baseline TCP orientation, fixed camera calibration, and simulator port coordinates **only for offline feasibility**, a local-x +10 degree rotation kept the selected port center inside all three image bounds on card 1/port 0, card 2/port 0, card 3/port 0 and card 4/port 1. The minimum projected margin of the installed port centers on card 1/port 0 improved to 19/143/42 px in center/left/right. This is a field-of-view upper bound; it does not show a visible board end, occlusion, or collision clearance. The exact calculation is `/tmp/sfp_rotation_feas.py` and should be preserved with the pilot artifacts.

## Frozen safety and visual gates

Use [CheatCodeBoardTiltViewDiagnostic.py](../../aic_example_policies/aic_example_policies/ros/CheatCodeBoardTiltViewDiagnostic.py) in one rootless Gazebo container on GPU 1. Its local-x command changes at most one degree per 50 ms. The inherited oracle-contact guard aborts on stale data, forbidden contact, wrist-force rise >5 N or absolute force >35 N. At rotated hold and return, measured TCP position must be within 5 mm of baseline and orientation within 2 degrees of target. Preserve abort traces. Only proceed to a second reused nonfinal scene if the first safely returns and unmarked current three-camera RGB reveals more of a fixed board/slot reference; stop after the second if identity is not recoverable in at least two cameras. Teacher insertion score after the view is separate from autonomous performance.

Review the baseline/rotated original RGB **without projected targets**. Record whether the fixed board end, absolute numbered card and chosen port side can be established from at least two cameras. A point projected inside an image is not visual visibility. Do not train another association model or transport actor from this diagnostic unless the identity gate passes on new episode-grouped development scenes under the earlier ≥9/10, zero-wrong, p95 and latency requirements.

## Initial implementation abort and correction

The first `card1_port0/` attempt aborted at the measured-pose gate, with no force/contact event. The pilot command had rotated the *future teacher orientation* by 10 degrees. The saved controller-state quaternions show that this future orientation was already about 45 degrees from the measured held wrist; the robot therefore reached about 55 degrees relative to baseline, and the unchanged 2-degree gate stopped it. This was a diagnostic command-frame error, not evidence about the proposed +10-degree view. The corrected policy takes the measured held TCP pose as the rotation origin. Rerun only the same nonfinal scene as `card1_port0_v2/`, preserving the failed attempt, motion guard, threshold and view hypothesis.

The v2 pilot also reached about 55 degrees and aborted without contact. Its recorded command quaternions revealed a second implementation bug: `_target` changed the saved origin quaternion while constructing each pose. The requested 1, 2, …, 10 degree increments accumulated to 55 degrees. The v3 code writes only to the copied pose, leaving the origin fixed. The two failed traces are preserved; repeat the same scene once more under the unchanged guard. Neither aborted run is evidence that a true 10-degree view is unsafe or visually useful.

## Completed +10-degree pilot and stopping decision

The v3 pilot reached its requested orientation: measured error **0.014 degrees** at the held view. Its measured TCP position drifted **6.38 mm** from baseline despite two seconds of guarded target holding, exceeding the unchanged 5 mm limit. The diagnostic aborted before return or insertion. There was no force/contact trigger: wrist force 20.84 N at baseline and 20.81 N at the view; only cable/enclosure contact was reported. The official tier-3 score is 0 because this is an aborted diagnostic, not evidence that the original CheatCode route failed. Machine values are in `/home/nas/chmin/aic_archives/2026-10-06_sfp_board_tilt_view_v1/card1_port0_v3/pilot_summary.json` and `board_tilt_view_events.json`.

The synchronized unmarked original-RGB comparison is `card1_port0_v3/unmarked_baseline_vs_tilt.jpg`; `unmarked_visual_annotation.json` was saved before any target overlay. The tilt showed slightly more board surface in the left camera, but the center still clipped the row end and the right view remained partial. The requested *absolute numbered card and port side* were not established in two cameras. Thus **both** the measured-pose return gate and visual identity gate fail. Stop after this reused-scene pilot: no second scene, training-set collection, identity fit, handoff, transport-actor fit, insertion or RL from this branch. A different view should be proposed only with a new geometric hypothesis and a safe controller strategy; routine angle/offset sweeps are unwarranted.

The first two attempts are preserved as `card1_port0/` and `card1_port0_v2/`, with policy logs and scored bags. Neither was a true 10-degree view. The corrected v3 is the only visual test of the stated hypothesis. It remains **one scene and one reviewer**, not a general visibility estimate. The independent review packet for the earlier 40 original-RGB panels is `/home/nas/chmin/aic_archives/2026-10-06_sfp_requested_card_blind_review_v1/independent_review_packet.md`; it excludes scene geometry. A second independent reviewer was not available during this run, so the 40-panel identity gate remains incomplete.

The corrected v3 command (GPU 1, rootless socket) was:

```bash
DOCKER_HOST=unix:///run/user/1008/docker.sock AIC_TRANSPORT_EVAL_GPU=1 AIC_TRANSPORT_EVAL_POLICY_GPU=1 \
 AIC_TRANSPORT_EVAL_POLICY=aic_example_policies.ros.CheatCodeBoardTiltViewDiagnostic \
 AIC_TRANSPORT_EVAL_TEACHER_PREFIX=true AIC_TRANSPORT_EVAL_ORACLE_CONTACTS=true \
 AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA=.06 AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC=90 \
 AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC=250 \
 AIC_TRANSPORT_EVAL_SCENE_DIR=/home/nas/chmin/aic_archives/2026-10-04_sfp_fresh_anchor_shadow_v1/scenes/card1_port0 \
 bash scripts/run_clean_transport_eval.sh board_tilt_card1_port0_v3 \
 outputs/experiments/2026-10-01_sc_llb_cleaned_bc/transport_imagenet_clean/best.pt \
 /home/nas/chmin/aic_archives/2026-10-06_sfp_board_tilt_view_v1/card1_port0_v3
```

The v3 audit directory has `eval_manifest.json`, launcher snapshot, input hashes, `frame_manifest.csv`, original `frames/`, `board_tilt_view_events.json`, `side_view_events.json`, `results/scoring.yaml`, `unmarked_visual_annotation.json`, `pilot_summary.json` and a byte-for-byte `policy_source_snapshot.py` (SHA256 `12170af88c9a4cceba574244efe490b1e9e9e030e29f05844e8190a30f59943a`). The launcher's general input hash file does not list this new subclass, so use this saved source snapshot and trace together for reproduction. The first two aborted policy variants are documented by their logs and the implementation-error analysis above.
